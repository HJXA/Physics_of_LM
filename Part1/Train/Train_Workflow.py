import os 
os.environ["CUDA_VISIBLE_DEVICES"] = "0"  # 请根据实际情况调整 GPU 可见性
# export PATH="/ruilab/jxhe/miniconda3/envs/PoL/bin:$PATH"

from transformers import (
	TrainingArguments,
	set_seed,
)

from SwanLabTrainer import SwanLabTrainer
import torch
from train_utils import (
	load_model,
	prepare_train_dataset,
	preview_collator_batch,
)

IS_TEST = False

MODEL_TYPE = None
# ============================================================
# Block 1: 配置路径与按论文要求的超参
# ============================================================
MODEL_ROOT = "/ruilab/jxhe/CoE_Monitor/Physics_of_LM/Part1/checkpoints/"
MODEL_PATH = MODEL_ROOT + "Init/gpt_2/gpt_2_raw" # 请在这里指向您初始化的GPT模型
if "gpt" in MODEL_PATH.lower():
	MODEL_TYPE = MODEL_PATH.split("/")[-1].split("_")[-1] # 要训练的 GPT 变体类型 ('standard', 'rot', 'rel', 'pos', 'uni')
	print(f"准备训练 GPT: [{MODEL_TYPE}] | 模型路径: {MODEL_PATH}")
elif "llama_wpe" in MODEL_PATH.lower():
	print(f"准备训练 LLaMA_wpe | 模型路径: {MODEL_PATH}")
	MODEL_TYPE = "llama_wpe"

OUTPUT_DIR = MODEL_ROOT + MODEL_PATH.split("/")[-1] # 模型输出目录

TRAIN_PARQUET_PATH = "/ruilab/jxhe/CoE_Monitor/Physics_of_LM/Part1/datasets/512_Chunk/cfg3f_Chunk/train.parquet"

DATASETS = TRAIN_PARQUET_PATH.split("/")[-2]

OUTPUT_DIR = os.path.join(OUTPUT_DIR, DATASETS) # 将数据集名称加入输出目录，方便区分不同数据集的训练结果
print(f"训练输出目录: {OUTPUT_DIR}")

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


print(f"训练配置: MAX_STEPS={MAX_STEPS}, LEARNING_RATE={LEARNING_RATE}, BATCH_SIZE={PER_DEVICE_TRAIN_BATCH_SIZE}, GRADIENT_ACCUMULATION_STEPS={GRADIENT_ACCUMULATION_STEPS}, LR_SCHEDULER_TYPE={LR_SCHEDULER_TYPE}, BF16={BF16}, FP16={FP16}")


def main():
	set_seed(SEED)

	# ============================================================
	# Block 3: 加载模型
	# ============================================================

	model = load_model(
		model_path=MODEL_PATH,
		model_type=MODEL_TYPE,
		dtype=torch.bfloat16,
	)
	# 显式设定我们的 pad_token_id，因为生成的 CFG 数据中 eos 是 101
	model.config.bos_token_id = 0 # 开始
	model.config.pad_token_id = 5
	pad_token_id = model.config.pad_token_id
	model.config.eos_token_id = 4 # 结束

	# ============================================================
	# Block 4: 读取 parquet 训练集（已tokenized）
	# ============================================================
	train_dataset, data_collator = prepare_train_dataset(
		train_parquet_path=TRAIN_PARQUET_PATH,
		pad_token_id=pad_token_id,
	)

	preview_collator_batch(train_dataset=train_dataset, data_collator=data_collator)

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
		swanlab_project="CFG_Pretrain" if not IS_TEST else "CFG_Pretrain_Test",
		swanlab_experiment_name=OUTPUT_DIR.split("/")[-2] + "_" + OUTPUT_DIR.split("/")[-1],
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
