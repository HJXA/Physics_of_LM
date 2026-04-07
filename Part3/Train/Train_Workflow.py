import os
import time

os.environ["CUDA_VISIBLE_DEVICES"] = "2"  # 请根据实际情况调整 GPU 可见性

from datasets import Dataset, load_dataset
from transformers import AutoTokenizer, TrainingArguments, set_seed, AutoModelForCausalLM

from SwanLabTrainer import SwanLabTrainer
import torch

from tokenization import prepare_pretrain_dataset_from_text, prepare_sft_dataset_from_messages, build_tokenizer
from train_utils import preview_collator_batch, part3_prepare_sft_source_dataset, part3_qa_text_to_messages


IS_TEST = False
TRAIN_TYPE = "SFT"  # 可选: "PT" / "SFT"



# PT
# MODEL_PATH = '/ruilab/jxhe/CoE_Monitor/Physics_of_LM/Part3/checkpoints/llama2'
if TRAIN_TYPE == "PT":
	TRAIN_PARQUET_PATH = "/ruilab/jxhe/CoE_Monitor/Physics_of_LM/Part3/datasets/text/bioS_single/part_1.parquet"
# TRAIN_PARQUET_PATH = "/ruilab/jxhe/CoE_Monitor/Physics_of_LM/Part3/datasets/text/bioS_multi/part_*.parquet"
# TRAIN_PARQUET_PATH = "/ruilab/jxhe/CoE_Monitor/Physics_of_LM/Part3/datasets/text/bioS_multi_permute_fullname/part_*.parquet"

# SFT
MODEL_PATH = '/ruilab/jxhe/CoE_Monitor/Physics_of_LM/Part3/checkpoints/bioS_multi_permute_fullname/llama2_2026_04_06_11_22_54'


if TRAIN_TYPE == "SFT":
	TRAIN_PARQUET_PATH = "/ruilab/jxhe/CoE_Monitor/Physics_of_LM/Part3/datasets/QA/train/*.parquet"

OUTPUT_BASE_DIR = "/ruilab/jxhe/CoE_Monitor/Physics_of_LM/Part3/checkpoints"

# tokenize 配置

MAX_LENGTH = 512 if TRAIN_TYPE == "PT" else 128
DO_CHUNK = True
ADD_EOS_AT_START = True
ADD_BOS_AT_END = True
DROP_LAST_CHUNK = False # 最后一个Padding
APPLY_CHAT_TEMPLATE = False  # 是否应用 chat 模板（仅 SFT 适用，PT 不使用）


# 训练超参（参考论文：AdamW, weight_decay=0.1, ε=1e-6, lr=1e-3, 1000-step warmup, cosine decay）
SEED = 42
PT_TRAIN_CONFIG = {
	"max_steps": 80000,  # 训练的步数，不是前向的步数。前向的步数 = MAX_STEPS * GRADIENT_ACCUMULATION_STEPS
	"learning_rate": 0.001,
	"weight_decay": 0.1,
	"adam_beta1": 0.9,
	"adam_beta2": 0.98,
	"adam_epsilon": 1e-6,
	"warmup_steps": 1000,
	"lr_scheduler_type": "cosine_with_min_lr",
	"lr_scheduler_kwargs": {"min_lr": 0.0001},
	"per_device_train_batch_size": 96,
	"gradient_accumulation_steps": 1,
	"logging_steps": 100,
	"save_steps": 2000,
	"bf16": True,
	"fp16": False,
}

# SFT 全量微调配置：AdamW, epsilon=1e-6, weight_decay=0.01, lr=3e-4, 无 warmup, 余弦衰减到初始 lr 的 10%
SFT_TRAIN_CONFIG = {
	"max_steps": 50000,
	"learning_rate": 0.0003,
	"weight_decay": 0.01,
	"adam_beta1": 0.9,
	"adam_beta2": 0.98,
	"adam_epsilon": 1e-6,
	"warmup_steps": 0,
	"lr_scheduler_type": "cosine_with_min_lr",
	"lr_scheduler_kwargs": {"min_lr": 0.00003},
	"per_device_train_batch_size": 48,
	"gradient_accumulation_steps": 1,
	"logging_steps": 100,
	"save_steps": 2000,
	"bf16": True,
	"fp16": False,
}


def get_train_config(train_type: str, is_test: bool):
	"""根据训练类型返回训练超参；测试模式下覆盖少量参数。"""
	base_config = PT_TRAIN_CONFIG if train_type == "PT" else SFT_TRAIN_CONFIG
	config = dict(base_config)

	if is_test:
		config["max_steps"] = 2
		config["save_steps"] = 30
		config["per_device_train_batch_size"] = 2
		config["gradient_accumulation_steps"] = 2

	return config


def get_sft_max_length_from_dataset(dataset: Dataset) -> int:
	"""根据 dataset 的 text 列，返回最大字符串长度向上最近的 2 的次幂。"""
	if "text" not in dataset.column_names:
		raise ValueError("SFT 动态 MAX_LENGTH 计算要求数据集包含 'text' 列")

	max_text_len = max(len(str(t)) for t in dataset["text"] if t is not None)
	if max_text_len <= 0:
		return 1

	# 计算 >= max_text_len 的最小 2 的次幂
	return 1 << (max_text_len - 1).bit_length()

	
def part3_prepare_train_dataset(tokenizer, train_type: str):
	"""按训练类型准备 tokenized dataset 与 data collator。"""

	raw_dataset = load_dataset(
		"parquet",
		data_files={"train": TRAIN_PARQUET_PATH},
	)["train"]

	if IS_TEST:
		raw_dataset = raw_dataset.select(range(min(2, len(raw_dataset))))

	raw_dataset = raw_dataset.shuffle(seed=42)  # 先打乱原始数据，避免后续 chunk 时样本分布不均

	print("原始数据集加载完成，样例: ", raw_dataset[:10])
	print("长度: ", len(raw_dataset))

	if train_type == "PT":
		if "text" not in raw_dataset.column_names:
			raise ValueError("PT 模式要求数据集包含 'text' 列")
		return prepare_pretrain_dataset_from_text(
			dataset=raw_dataset,
			tokenizer=tokenizer,
			max_length=MAX_LENGTH,
			do_chunk=DO_CHUNK,
			add_bos_at_start=ADD_EOS_AT_START,
			add_eos_at_end=ADD_BOS_AT_END,
			drop_last_chunk=DROP_LAST_CHUNK,
		)

	if train_type == "SFT":
		sft_max_length = get_sft_max_length_from_dataset(raw_dataset)
		print(f"SFT 动态 MAX_LENGTH（text 向上 2 次幂）: {sft_max_length}")
		print("此时的 MAX_LENGTH 配置（优先级高于动态计算）: ", MAX_LENGTH)
		sft_source = part3_prepare_sft_source_dataset(raw_dataset)
		print("SFT 数据预处理完成，样例: ", sft_source[:2])
		return prepare_sft_dataset_from_messages(
			dataset=sft_source,
			tokenizer=tokenizer,
			max_length=sft_max_length if MAX_LENGTH is None else MAX_LENGTH,
			add_bos_at_start=ADD_EOS_AT_START,
			add_eos_at_end=ADD_BOS_AT_END,
			apply_chat_template=APPLY_CHAT_TEMPLATE,  # SFT 不使用 chat 模板，直接拼接 messages
			test=IS_TEST,
		)

	raise ValueError(f"Unsupported TRAIN_TYPE={train_type}, expected PT or SFT")

def main():
	train_type = str(TRAIN_TYPE).upper()
	if train_type not in {"PT", "SFT"}:
		raise ValueError(f"Unsupported TRAIN_TYPE={TRAIN_TYPE}, expected PT or SFT")

	set_seed(SEED)
	train_config = get_train_config(train_type=train_type, is_test=IS_TEST)


	model = AutoModelForCausalLM.from_pretrained(
		MODEL_PATH,
		dtype=torch.bfloat16,
		device_map="auto",
	)
	tokenizer = build_tokenizer(model=model, model_path=MODEL_PATH)

	print(
		f"Tokenizer已加载: pad={tokenizer.pad_token_id}, "
		f"bos={tokenizer.bos_token_id}, eos={tokenizer.eos_token_id}"
	)


	print(f"开始准备训练数据，train_type={train_type}...")
	train_dataset, data_collator = part3_prepare_train_dataset(tokenizer=tokenizer, train_type=train_type)
	print(
		f"训练数据准备完成: train_type={train_type}, samples={len(train_dataset)}, "
		f"columns={train_dataset.column_names}"
	)


	preview_collator_batch(train_dataset=train_dataset, data_collator=data_collator, preview_n=1)

	if TRAIN_TYPE == "PT":
		dataset_tag = TRAIN_PARQUET_PATH.split("/")[-2]
		output_dir = os.path.join(OUTPUT_BASE_DIR, dataset_tag, MODEL_PATH.split("/")[-1]) + f"_{time.strftime('%Y_%m_%d_%H_%M_%S')}"
	elif TRAIN_TYPE == "SFT":
		dataset_tag = TRAIN_PARQUET_PATH.split("/")[-3]
		pt_datasets_type = MODEL_PATH.split("/")[-2]
		output_dir = os.path.join(OUTPUT_BASE_DIR, f"{dataset_tag}", pt_datasets_type, "sft_" + MODEL_PATH.split("/")[-1]) + f"_{time.strftime('%Y_%m_%d_%H_%M_%S')}"
	if IS_TEST:
		output_dir += "_test"


	print(f"训练输出目录: {output_dir}")
	print(
		f"训练配置: MAX_STEPS={train_config['max_steps']}, LEARNING_RATE={train_config['learning_rate']}, "
		f"BATCH_SIZE={train_config['per_device_train_batch_size']}, GRAD_ACC={train_config['gradient_accumulation_steps']}, "
		f"SCHEDULER={train_config['lr_scheduler_type']}, WARMUP_STEPS={train_config['warmup_steps']}, "
		f"BF16={train_config['bf16']}, FP16={train_config['fp16']}"
	)


	training_args = TrainingArguments(
		output_dir=output_dir,
		do_train=True,
		do_eval=False,
		eval_strategy="no",
		max_steps=train_config["max_steps"],
		learning_rate=train_config["learning_rate"],
		weight_decay=train_config["weight_decay"],
		adam_beta1=train_config["adam_beta1"],
		adam_beta2=train_config["adam_beta2"],
		adam_epsilon=train_config["adam_epsilon"],
		warmup_steps=train_config["warmup_steps"],
		lr_scheduler_type=train_config["lr_scheduler_type"],
		lr_scheduler_kwargs=train_config["lr_scheduler_kwargs"],
		per_device_train_batch_size=train_config["per_device_train_batch_size"],
		gradient_accumulation_steps=train_config["gradient_accumulation_steps"],
		logging_steps=train_config["logging_steps"],
		save_steps=train_config["save_steps"],
		gradient_checkpointing=True,
		bf16=train_config["bf16"],
		fp16=train_config["fp16"],
		dataloader_num_workers=4,
		remove_unused_columns=False,
		report_to="none",
		save_only_model=False,
	)

	trainer = SwanLabTrainer(
		model=model,
		args=training_args,
		train_dataset=train_dataset,
		data_collator=data_collator,
		swanlab_project=("BioS_Pretrain" if train_type == "PT" else "BioS_SFT") if not IS_TEST else "BioS_Test",
		swanlab_experiment_name=output_dir.split("/")[-2] + "_" + output_dir.split("/")[-1],
		swanlab_description=f"{train_type} training on parquet dataset",
		test_falg=IS_TEST,
		CoE_Flag=True,
		Train_type=train_type,
	)

	train_result = trainer.train()
	trainer.save_model()
	tokenizer.save_pretrained(output_dir)
	metrics = train_result.metrics
	metrics["train_samples"] = len(train_dataset)
	trainer.log_metrics("train", metrics)
	trainer.save_metrics("train", metrics)
	trainer.save_state()

	try:

		import shutil

		# 自动把 tokenizer 同步到所有检查点
		latest_ckpt = max([os.path.join(output_dir, d) for d in os.listdir(output_dir) if d.startswith("checkpoint-")], key=os.path.getmtime)
		for f in os.listdir(output_dir):
			if "tokenizer" in f or "vocab" in f or "special_tokens" in f:
				shutil.copy(os.path.join(output_dir, f), latest_ckpt)
	except Exception as e:
		print(f"自动同步 tokenizer 文件到最新 checkpoint 失败: {e}")


if __name__ == "__main__":
	main()
