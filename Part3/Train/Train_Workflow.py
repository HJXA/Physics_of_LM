import os

os.environ["CUDA_VISIBLE_DEVICES"] = "0"  # 请根据实际情况调整 GPU 可见性

from datasets import Dataset, load_dataset
from transformers import AutoTokenizer, TrainingArguments, set_seed, AutoModelForCausalLM

from SwanLabTrainer import SwanLabTrainer
import torch

from tokenization import prepare_pretrain_dataset_from_text, prepare_sft_dataset_from_messages, build_tokenizer
from train_utils import preview_collator_batch, part3_prepare_sft_source_dataset, part3_qa_text_to_messages


IS_TEST = False
TRAIN_TYPE = "PT"  # 可选: "PT" / "SFT"
MODEL_PATH = 'xxx'
TRAIN_PARQUET_PATH = "/ruilab/jxhe/CoE_Monitor/Physics_of_LM/Part3/datasets/text/bioS_single/part_x.parquet"

# tokenize 配置
MAX_LENGTH = 512
DO_CHUNK = True
ADD_EOS_AT_START = True
ADD_BOS_AT_END = True
DROP_LAST_CHUNK = False

# 训练超参（参考论文：AdamW, weight_decay=0.1, ε=1e-6, lr=1e-3, 1000-step warmup, cosine decay）
SEED = 42
MAX_STEPS = 80000
LEARNING_RATE = 0.001
WEIGHT_DECAY = 0.1
ADAM_BETA1 = 0.9
ADAM_BETA2 = 0.98
ADAM_EPSILON = 1e-6
WARMUP_STEPS = 1000
LR_SCHEDULER_TYPE = "cosine_with_min_lr"
LR_SCHEDULER_KWARGS = {"min_lr": 0.0001}
PER_DEVICE_TRAIN_BATCH_SIZE = 96
GRADIENT_ACCUMULATION_STEPS = 1
LOGGING_STEPS = 100
SAVE_STEPS = 2500
BF16 = True
FP16 = False

if IS_TEST:
	MAX_STEPS = 100
	SAVE_STEPS = 30
	PER_DEVICE_TRAIN_BATCH_SIZE = 2
	
def part3_prepare_train_dataset(tokenizer, train_type: str):
	"""按训练类型准备 tokenized dataset 与 data collator。"""
	raw_dataset = load_dataset(
		"parquet",
		data_files={"train": TRAIN_PARQUET_PATH},
	)["train"]

	if train_type == "PT":
		if "text" not in raw_dataset.column_names:
			raise ValueError("PT 模式要求数据集包含 'text' 列")
		return prepare_pretrain_dataset_from_text(
			dataset=raw_dataset,
			tokenizer=tokenizer,
			max_length=MAX_LENGTH,
			do_chunk=DO_CHUNK,
			add_eos_at_start=ADD_EOS_AT_START,
			add_bos_at_end=ADD_BOS_AT_END,
			drop_last_chunk=DROP_LAST_CHUNK,
		)

	if train_type == "SFT":
		sft_source = part3_prepare_sft_source_dataset(raw_dataset)
		print(f"raw_dataset 样例: {raw_dataset[0]}")
		return prepare_sft_dataset_from_messages(
			dataset=sft_source,
			tokenizer=tokenizer,
			max_length=MAX_LENGTH,
			add_eos_at_start=ADD_EOS_AT_START,
			add_bos_at_end=ADD_BOS_AT_END,
		)

	raise ValueError(f"Unsupported TRAIN_TYPE={train_type}, expected PT or SFT")

def main():
	train_type = str(TRAIN_TYPE).upper()
	if train_type not in {"PT", "SFT"}:
		raise ValueError(f"Unsupported TRAIN_TYPE={TRAIN_TYPE}, expected PT or SFT")

	set_seed(SEED)


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

	train_dataset, data_collator = part3_prepare_train_dataset(tokenizer=tokenizer, train_type=train_type)
	print(
		f"训练数据准备完成: train_type={train_type}, samples={len(train_dataset)}, "
		f"columns={train_dataset.column_names}"
	)

	preview_collator_batch(train_dataset=train_dataset, data_collator=data_collator)

	dataset_tag = TRAIN_PARQUET_PATH.split("/")[-2]
	output_dir = os.path.join(MODEL_PATH.split("/")[-1], dataset_tag, train_type.lower())
	if IS_TEST:
		output_dir += "_test"

	print(f"训练输出目录: {output_dir}")
	print(
		f"训练配置: MAX_STEPS={MAX_STEPS}, LEARNING_RATE={LEARNING_RATE}, "
		f"BATCH_SIZE={PER_DEVICE_TRAIN_BATCH_SIZE}, GRAD_ACC={GRADIENT_ACCUMULATION_STEPS}, "
		f"SCHEDULER={LR_SCHEDULER_TYPE}, WARMUP_STEPS={WARMUP_STEPS}, BF16={BF16}, FP16={FP16}"
	)

	training_args = TrainingArguments(
		output_dir=output_dir,
		do_train=True,
		do_eval=False,
		eval_strategy="no",
		max_steps=MAX_STEPS,
		learning_rate=LEARNING_RATE,
		weight_decay=WEIGHT_DECAY,
		adam_beta1=ADAM_BETA1,
		adam_beta2=ADAM_BETA2,
		adam_epsilon=ADAM_EPSILON,
		warmup_steps=WARMUP_STEPS,
		lr_scheduler_type=LR_SCHEDULER_TYPE,
		lr_scheduler_kwargs=LR_SCHEDULER_KWARGS,
		per_device_train_batch_size=PER_DEVICE_TRAIN_BATCH_SIZE,
		gradient_accumulation_steps=GRADIENT_ACCUMULATION_STEPS,
		logging_steps=LOGGING_STEPS,
		save_steps=SAVE_STEPS,
		gradient_checkpointing=True,
		bf16=BF16,
		fp16=FP16,
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
	metrics = train_result.metrics
	metrics["train_samples"] = len(train_dataset)
	trainer.log_metrics("train", metrics)
	trainer.save_metrics("train", metrics)
	trainer.save_state()


if __name__ == "__main__":
	main()
