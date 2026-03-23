import os 
os.environ["CUDA_VISIBLE_DEVICES"] = "6"  # 请根据实际情况调整 GPU 可见性
# export PATH="/ruilab/jxhe/miniconda3/envs/PoL/bin:$PATH"
import logging
from typing import Dict, List

from datasets import load_dataset
from transformers import (
	AutoConfig,
	AutoModelForCausalLM,
	TrainingArguments,
	set_seed,
)
import sys
sys.path.append("/ruilab/jxhe/CoE_Monitor/Physics_of_LM/Part1/model")
from modeling_gpt2_variants import CustomGPT2LMHeadModel, GPT2Config
from SwanLabTrainer import SwanLabTrainer
import torch

IS_TEST = False


# ============================================================
# Block 1: 配置路径与按论文要求的超参
# ============================================================
MODEL_PATH = "/ruilab/jxhe/CoE_Monitor/Physics_of_LM/Part1/checkpoints/gpt_2_Init/gpt_2_rot" # 请在这里指向您初始化的GPT模型
TRAIN_FROM_SCRATCH = False # 开启后将随机初始化模型权重, 我现在是提前Init了所以不用开
GPT_VARIANT_TYPE = MODEL_PATH.split("/")[-1].split("_")[-1] # 要训练的 GPT 变体类型 ('standard', 'rot', 'rel', 'pos', 'uni')
print(f"准备训练 GPT: [{GPT_VARIANT_TYPE}] | 模型路径: {MODEL_PATH}")
TRAIN_PARQUET_PATH = "/ruilab/jxhe/CoE_Monitor/Physics_of_LM/Part1/datasets/512_Padding/train.parquet"
OUTPUT_DIR = "/ruilab/jxhe/CoE_Monitor/Physics_of_LM/Part1/checkpoints/" + f"gpt_{GPT_VARIANT_TYPE}_Pretrain" # 模型输出目录

# 按论文 GPT 预训练指定的超参配置
SEED = 42
MAX_STEPS = 100000              # “预训练10万次迭代”
LEARNING_RATE = 0.0003          # “GPT学习率为0.0003”
WEIGHT_DECAY = 0.1              # “权重衰减为0.1”
ADAM_BETA1 = 0.9                # “AdamW优化器, beta=(0.9, 0.98)”
ADAM_BETA2 = 0.98

# 论文批次大小为 96。因为现在只有单卡环境：
# 利用梯度累加达到全局 96 的 Batch Size: 8(每次灌入卡内) * 12(步数累加) = 96
PER_DEVICE_TRAIN_BATCH_SIZE = 96
GRADIENT_ACCUMULATION_STEPS = 1

# 采用线性学习率衰减
LR_SCHEDULER_TYPE = "linear"

LOGGING_STEPS = 100
SAVE_STEPS = 2500
# SAVE_TOTAL_LIMIT = 2

BF16 = True
FP16 = False

if IS_TEST:
	# 测试模式下缩短训练以验证流程正确性
	MAX_STEPS = 100
	SAVE_STEPS = 30
	PER_DEVICE_TRAIN_BATCH_SIZE = 2
	OUTPUT_DIR += "_test"


def build_causal_lm_collator(pad_token_id: int):
	"""
	用于已tokenized样本的动态padding。
	输入样本格式：
	{
		'input_ids': [...],
		'labels': [...],
		'attention_mask': [...],
	}
	"""

	def collate_fn(features: List[Dict]):
		input_ids = [torch.tensor(x["input_ids"], dtype=torch.long) for x in features]

		input_ids = torch.nn.utils.rnn.pad_sequence(
			input_ids,
			batch_first=True,
			padding_value=pad_token_id,
		)
		attention_mask = (input_ids != pad_token_id).long()

		# 对于因果语言模型，不使用 -100 遮蔽 pad_token 以外的其他输入
		# 如果之前的数据集中发生了 padding，由于 padding 不参与预测，我们需要将 padding 的位置设置为 -100
		labels = input_ids.clone()
		# 将 padding_token_id 所在位置替换为 -100 以便交叉熵忽略
		labels[labels == pad_token_id] = -100

		return {
			"input_ids": input_ids,
			"labels": labels, # 由 collator 根据 input_ids 动态生成并遮蔽 pad
			"attention_mask": attention_mask,
		}

	return collate_fn


def main():
	# ============================================================
	# Block 2: 初始化日志与随机种子
	# ============================================================
	logging.basicConfig(
		format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
		datefmt="%Y-%m-%d %H:%M:%S",
		level=logging.INFO,
	)
	set_seed(SEED)

	# ============================================================
	# Block 3: 加载模型
	# ============================================================
	attn_impl = "flash_attention_2" if GPT_VARIANT_TYPE in ['standard','rot'] else "eager"

	if TRAIN_FROM_SCRATCH:
		print(f"Initialize {GPT_VARIANT_TYPE} model from scratch using config from: {MODEL_PATH} with {attn_impl}")
		config = AutoConfig.from_pretrained(MODEL_PATH, attn_implementation=attn_impl)
		model = CustomGPT2LMHeadModel(config, gpt_type=GPT_VARIANT_TYPE)
		model.to(torch.bfloat16)  # 转为半精度
	else:
		print(f"Load pretrained {GPT_VARIANT_TYPE} model from: {MODEL_PATH} with {attn_impl}")
		# 警告: 如果变体使用预训练权重，原本必须和保存的结构相对应，否则将加载失败
		model = CustomGPT2LMHeadModel.from_pretrained(
			MODEL_PATH,
			gpt_type=GPT_VARIANT_TYPE,
			attn_implementation=attn_impl,
			dtype=torch.bfloat16
		)
	# 显式设定我们的 pad_token_id，因为生成的 CFG 数据中 eos 是 101
	model.config.bos_token_id = 0 # 开始
	pad_token_id = 5
	model.config.eos_token_id = 4 # 结束

	# ============================================================
	# Block 4: 读取 parquet 训练集（已tokenized）
	# ============================================================
	train_dataset = load_dataset(
		"parquet",
		data_files={"train": TRAIN_PARQUET_PATH},
	)["train"]


	# 仅保留训练会用到的字段；其他字段（如 length/lengths）不参与前向。
	keep_columns = {"input_ids"}
	remove_columns = [x for x in train_dataset.column_names if x not in keep_columns]
	if remove_columns:
		train_dataset = train_dataset.remove_columns(remove_columns)

	data_collator = build_causal_lm_collator(pad_token_id=pad_token_id)

	# 训练开始前，打印经过 data_collator 处理后的样本示例
	if len(train_dataset) > 0:
		preview_n = min(2, len(train_dataset))
		preview_features = [train_dataset[i] for i in range(preview_n)]
		preview_batch = data_collator(preview_features)

		print("\n[Collator Preview]")
		print(
			f"input_ids shape: {tuple(preview_batch['input_ids'].shape)}, "
			f"labels shape: {tuple(preview_batch['labels'].shape)}, "
			f"attention_mask shape: {tuple(preview_batch['attention_mask'].shape)}"
		)
		print(f"input_ids[0]: {preview_batch['input_ids'][0].tolist()}")
		print(f"labels[0]: {preview_batch['labels'][0].tolist()}")
		print(f"attention_mask[0]: {preview_batch['attention_mask'][0].tolist()}")
	else:
		print("[Collator Preview] 训练集为空，请检查 train.parquet")

	# ============================================================
	# Block 5: 根据论文修改后的严谨训练参数
	# ============================================================
	training_args = TrainingArguments(
		output_dir=OUTPUT_DIR,
		do_train=True,
		do_eval=False,
		eval_strategy="no",
		max_steps=MAX_STEPS, # 替换 num_train_epochs，严格按照 10万次迭代 设定
		learning_rate=LEARNING_RATE,
		weight_decay=WEIGHT_DECAY,
		adam_beta1=ADAM_BETA1,
		adam_beta2=ADAM_BETA2,
		lr_scheduler_type=LR_SCHEDULER_TYPE,
		per_device_train_batch_size=PER_DEVICE_TRAIN_BATCH_SIZE,
		gradient_accumulation_steps=GRADIENT_ACCUMULATION_STEPS,
		logging_steps=LOGGING_STEPS,
		save_steps=SAVE_STEPS,
		gradient_checkpointing=True, # 开启梯度检查点以节省显存
		# save_total_limit=SAVE_TOTAL_LIMIT,
		bf16=BF16,
		fp16=FP16,
		dataloader_num_workers=4,
		remove_unused_columns=False,
		report_to="none",
		save_only_model=False,
	)

	# ============================================================
	# Block 6: 初始化 Trainer 并训练
	# ============================================================
	trainer = SwanLabTrainer(
		model=model,
		args=training_args,
		train_dataset=train_dataset,
		data_collator=data_collator,
		swanlab_project="CFG_Pretrain",
		swanlab_experiment_name=f"gpt2_{GPT_VARIANT_TYPE}_pretrain",
		swanlab_description="Pretraining GPT on synthetic CFG dataset",
		test_falg=IS_TEST,
		CoE_Flag=True,
		Train_type = "PT",
	)

	train_result = trainer.train()
	trainer.save_model()
	metrics = train_result.metrics
	metrics["train_samples"] = len(train_dataset)
	trainer.log_metrics("train", metrics)
	trainer.save_metrics("train", metrics)
	trainer.save_state()


# ============================================================
# Block 7: 程序入口
# ============================================================
if __name__ == "__main__":
	main()
