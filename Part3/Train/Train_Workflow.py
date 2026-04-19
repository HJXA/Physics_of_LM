import os
import time

# export PATH="/ruilab/jxhe/miniconda3/envs/PoL/bin:$PATH"

os.environ["CUDA_VISIBLE_DEVICES"] = "5"  # 请根据实际情况调整 GPU 可见性

from datasets import Dataset, load_dataset
from transformers import AutoTokenizer, TrainingArguments, set_seed, AutoModelForCausalLM, Trainer
from peft import LoraConfig, TaskType, get_peft_model, PeftModel

from SwanLabTrainer import SwanLabTrainer
import torch

from tokenization import prepare_pretrain_dataset_from_text, prepare_sft_dataset_from_messages, build_tokenizer
from utils.train_utils import preview_collator_batch, part3_prepare_sft_source_dataset, part3_qa_text_to_messages
from utils.merge_lora_checkpoints import find_checkpoints, merge_single_checkpoint


IS_TEST = False
TRAIN_TYPE = "LORA"  # 可选: "PT" / "SFT" / "LORA"
DATA_MODE = "raw"  # 可选: "no-answer" / "#" / "# #" / "#*10" / "attribute" / "raw" # 我现在tokenizer中的aplly_chat会自动加空格在Answer前
SFT_CURRICULUM_MODE = True  # SFT 课程学习模式：按 TRAIN_QA_FILES 顺序读取，文件内 shuffle，文件间不 shuffle

# SFT/LORA 专用：指定要训练的 QA 属性文件列表，为空或 None 时使用 TRAIN_PARQUET_PATH 的 glob
TRAIN_QA_FILES = [
	# "/ruilab/jxhe/CoE_Monitor/Physics_of_LM/Part3/datasets/QA/train/q1_birth_date.parquet",
	# "/ruilab/jxhe/CoE_Monitor/Physics_of_LM/Part3/datasets/QA/train/q2_birth_city.parquet",
	# "/ruilab/jxhe/CoE_Monitor/Physics_of_LM/Part3/datasets/QA/train/q3_university.parquet",
	# "/ruilab/jxhe/CoE_Monitor/Physics_of_LM/Part3/datasets/QA/train/q4_major.parquet",
	# "/ruilab/jxhe/CoE_Monitor/Physics_of_LM/Part3/datasets/QA/train/q5_company.parquet",
	# "/ruilab/jxhe/CoE_Monitor/Physics_of_LM/Part3/datasets/QA/train/q6_company_city.parquet",
]



# PT
# MODEL_PATH = '/ruilab/jxhe/CoE_Monitor/Physics_of_LM/Part3/checkpoints/llama2'
if TRAIN_TYPE == "PT":
	MODEL_PATH = '/ruilab/jxhe/CoE_Monitor/Physics_of_LM/Part3/checkpoints/llama2'

	# TRAIN_PARQUET_PATH = "/ruilab/jxhe/CoE_Monitor/Physics_of_LM/Part3/datasets/text/bioS_single/part_1.parquet"
	# TRAIN_PARQUET_PATH = "/ruilab/jxhe/CoE_Monitor/Physics_of_LM/Part3/datasets/text/bioS_multi/part_*.parquet"
	TRAIN_PARQUET_PATH = "/ruilab/jxhe/CoE_Monitor/Physics_of_LM/Part3/datasets/text/bioS_multi_permute_fullname/part_*.parquet"




if TRAIN_TYPE in {"SFT", "LORA"}:
	TRAIN_PARQUET_PATH = "/ruilab/jxhe/CoE_Monitor/Physics_of_LM/Part3/datasets/QA/train/*.parquet"

	# MODEL_PATH = '/ruilab/jxhe/CoE_Monitor/Physics_of_LM/Part3/checkpoints/bioS_single/llama2'
	MODEL_PATH = '/ruilab/jxhe/CoE_Monitor/Physics_of_LM/Part3/checkpoints/bioS_multi/llama2'
	# MODEL_PATH = '/ruilab/jxhe/CoE_Monitor/Physics_of_LM/Part3/checkpoints/bioS_multi_permute_fullname/llama2'

LORA_RANK_EMBED = 128
LORA_RANK_QV = 16

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
	"max_steps": 10000,
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
	"save_steps": 400,
	"bf16": True,
	"fp16": False,
}

LORA_TRAIN_CONFIG = dict(SFT_TRAIN_CONFIG)


def get_train_config(train_type: str, is_test: bool):
	"""根据训练类型返回训练超参；测试模式下覆盖少量参数。"""
	if train_type == "PT":
		base_config = PT_TRAIN_CONFIG
	elif train_type == "SFT":
		base_config = SFT_TRAIN_CONFIG
	elif train_type == "LORA":
		base_config = LORA_TRAIN_CONFIG
	else:
		raise ValueError(f"Unsupported TRAIN_TYPE={train_type}, expected PT, SFT or LORA")
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

	# SFT/LORA: 课程学习模式 vs 传统全局 shuffle 模式
	if train_type in {"SFT", "LORA"} and SFT_CURRICULUM_MODE and TRAIN_QA_FILES:
		from datasets import concatenate_datasets

		datasets_parts = []
		for i, qa_file in enumerate(TRAIN_QA_FILES):
			ds = load_dataset("parquet", data_files={"train": qa_file})["train"]
			ds = ds.shuffle(seed=42)
			print(f"  课程学习 [{i+1}/{len(TRAIN_QA_FILES)}] {os.path.basename(qa_file)}: {len(ds)} samples")
			datasets_parts.append(ds)
		raw_dataset = concatenate_datasets(datasets_parts)
		print(f"课程学习模式拼接完成，总样本数: {len(raw_dataset)}")
	elif train_type in {"SFT", "LORA"} and TRAIN_QA_FILES:
		data_path = TRAIN_QA_FILES
		raw_dataset = load_dataset("parquet", data_files={"train": data_path})["train"]
		raw_dataset = raw_dataset.shuffle(seed=42)
	else:
		data_path = TRAIN_PARQUET_PATH
		raw_dataset = load_dataset("parquet", data_files={"train": data_path})["train"]
		raw_dataset = raw_dataset.shuffle(seed=42)

	if IS_TEST:
		raw_dataset = raw_dataset.select(range(min(2, len(raw_dataset))))

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

	if train_type in {"SFT", "LORA"}:
		sft_max_length = get_sft_max_length_from_dataset(raw_dataset)
		print(f"{train_type} 动态 MAX_LENGTH（text 向上 2 次幂）: {sft_max_length}")
		print("此时的 MAX_LENGTH 配置（优先级高于动态计算）: ", MAX_LENGTH)
		sft_source = part3_prepare_sft_source_dataset(raw_dataset, mode=DATA_MODE)
		print(f"{train_type} 数据预处理完成，样例: ", sft_source[:2])
		return prepare_sft_dataset_from_messages(
			dataset=sft_source,
			tokenizer=tokenizer,
			max_length=sft_max_length if MAX_LENGTH is None else MAX_LENGTH,
			add_bos_at_start=ADD_EOS_AT_START,
			add_eos_at_end=ADD_BOS_AT_END,
			apply_chat_template=APPLY_CHAT_TEMPLATE,  # SFT 不使用 chat 模板，直接拼接 messages
			test=IS_TEST,
		)

	raise ValueError(f"Unsupported TRAIN_TYPE={train_type}, expected PT, SFT or LORA")


def apply_lora_to_model(model):
	"""将 LoRA 注入到 embedding / q_proj / v_proj。"""
	if hasattr(model, "config"):
		model.config.use_cache = False
	if hasattr(model, "enable_input_require_grads"):
		model.enable_input_require_grads()

	# 这里只对 3 个模块加 LoRA：输入 embedding、注意力里的 q_proj 和 v_proj。
	# 其中 embedding 使用更大的秩 r'，q/v 使用较小的秩 r。
	lora_config = LoraConfig(
		task_type=TaskType.CAUSAL_LM,
		r=LORA_RANK_QV,
		lora_alpha=LORA_RANK_QV,
		lora_dropout=0.0,
		bias="none",
		target_modules=["embed_tokens", "q_proj", "v_proj"],
		rank_pattern={
			# embedding 层用更大的秩，专门缓解 BIO 到 QA 的分布偏移。
			"embed_tokens": LORA_RANK_EMBED,
			# q/v 矩阵保持较小秩即可。
			"q_proj": LORA_RANK_QV,
			"v_proj": LORA_RANK_QV,
		},
		alpha_pattern={
			# 与秩设置保持一致，避免不同模块的缩放不匹配。
			"embed_tokens": LORA_RANK_EMBED,
			"q_proj": LORA_RANK_QV,
			"v_proj": LORA_RANK_QV,
		},
	)
	model = get_peft_model(model, lora_config)
	model.print_trainable_parameters()
	return model

def main():
	train_type = str(TRAIN_TYPE).upper()
	if train_type not in {"PT", "SFT", "LORA"}:
		raise ValueError(f"Unsupported TRAIN_TYPE={TRAIN_TYPE}, expected PT, SFT or LORA")

	set_seed(SEED)
	train_config = get_train_config(train_type=train_type, is_test=IS_TEST)


	model = AutoModelForCausalLM.from_pretrained(
		MODEL_PATH,
		dtype=torch.bfloat16,
		device_map="auto",
	)
	if train_type == "LORA":
		model = apply_lora_to_model(model)
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

	timestamp_short = time.strftime('%y%m%d%H')  # e.g. 26041916
	if train_type == "PT":
		dataset_tag = TRAIN_PARQUET_PATH.split("/")[-2]
		output_dir = os.path.join(
			OUTPUT_BASE_DIR,
			dataset_tag,
			f"{MODEL_PATH.split('/')[-1]}",
		)
	elif train_type in {"SFT", "LORA"}:
		dataset_tag = TRAIN_PARQUET_PATH.split("/")[-3]
		pt_datasets_type = MODEL_PATH.split("/")[-2]
		mode_prefix = "lora_" if train_type == "LORA" else "sft_"
		output_dir = os.path.join(
			OUTPUT_BASE_DIR,
			f"{dataset_tag}",
			pt_datasets_type,
			f"{mode_prefix}{MODEL_PATH.split('/')[-1]}_{DATA_MODE}_{timestamp_short}",
		)
	if IS_TEST:
		output_dir += "_test"


	print(f"训练输出目录: {output_dir}")
	print(
		f"训练配置: MAX_STEPS={train_config['max_steps']}, LEARNING_RATE={train_config['learning_rate']}, "
		f"BATCH_SIZE={train_config['per_device_train_batch_size']}, GRAD_ACC={train_config['gradient_accumulation_steps']}, "
		f"SCHEDULER={train_config['lr_scheduler_type']}, WARMUP_STEPS={train_config['warmup_steps']}, "
		f"BF16={train_config['bf16']}, FP16={train_config['fp16']}"
	)

	# 课程学习模式: 关闭 Trainer 内部 DataLoader 的 shuffle，按数据集拼接顺序训练
	sampling_strategy = "sequential" if (train_type in {"SFT", "LORA"} and SFT_CURRICULUM_MODE and TRAIN_QA_FILES) else "random"


	# return


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
		save_only_model= False if train_type == "PT" else True,  
		train_sampling_strategy=sampling_strategy,
		# use_liger_kernel = True
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

	if train_type == "LORA":
		# 批量将所有 checkpoint 的 LoRA adapter 合并到 base model
		merged_output_dir = output_dir + "_merged"
		os.makedirs(merged_output_dir, exist_ok=True)

		checkpoints = find_checkpoints(output_dir)
		if not checkpoints:
			print(f"[LoRA Merge] 未找到任何 checkpoint-* 子目录，跳过合并")
		else:
			print(f"\n[LoRA Merge] 开始批量合并 {len(checkpoints)} 个 checkpoint → {merged_output_dir}")
			for i, ckpt_path in enumerate(checkpoints, 1):
				ckpt_name = os.path.basename(ckpt_path)
				out_path = os.path.join(merged_output_dir, ckpt_name)
				print(f"\n[{i}/{len(checkpoints)}] 合并 {ckpt_name} ...")
				merge_single_checkpoint(ckpt_path, MODEL_PATH, out_path)
			print(f"\n[LoRA Merge] 全部 {len(checkpoints)} 个 checkpoint 合并完成！保存至: {merged_output_dir}")


if __name__ == "__main__":
	main()
