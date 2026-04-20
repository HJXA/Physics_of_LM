"""
Q-probing 训练脚本。

核心思路：冻结预训练 LLaMA 的全部参数，仅训练少量可训练参数来做属性分类。
可训练模块由三部分组成：
  1. rank-r embedding update（低秩分解：vocab_size × r, r × hidden_size）
     - 用两个小矩阵 A 和 B 来近似 embedding 的更新量，而非直接修改整个 embedding 矩阵
     - A: (vocab_size, r) 随机初始化，B: (r, hidden_size) 初始化为 0
     - 最终 embedding = 原始 embedding + A @ B，这样参数量从 V*H 降到 (V+r)*H
  2. BatchNorm1d(hidden_size, affine=True)
     - 对提取的隐藏状态做归一化，affine=True 表示可学习缩放和偏移参数
  3. Linear(hidden_size, n_classes)
     - 最终的分类层，将隐藏状态映射到类别数

参考论文超参设定：AdamW, ε=1e-6, weight_decay=0.3, lr=0.001, no warmup, linear decay to 0,
                 batch_size=200, 30000 steps。
"""

import os
import sys
import time
from transformers import LlamaConfig, LlamaForCausalLM




# 指定使用的 GPU 编号
os.environ["CUDA_VISIBLE_DEVICES"] = "4"

# 让 probing/ 能 import Train/ 下的模块
# 因为项目结构中 Train/ 和 probing/ 是平级目录，需要将项目根目录加入 sys.path
sys.path.append("/ruilab/jxhe/CoE_Monitor/Physics_of_LM")

PART3_ROOT = "/ruilab/jxhe/CoE_Monitor/Physics_of_LM/Part3"

import torch
import swanlab
from transformers import AutoModelForCausalLM, TrainingArguments, set_seed

from Part3.Train.tokenization import build_tokenizer
from _utils import build_output_dir_tag, get_train_config, LabelEncoder, inspect_dataset_samples
from _model import QProbingModel
from _data import prepare_qprobing_dataset, QProbingCollator
from transformers import Trainer


# ======== 配置 ========

# 是否为测试模式（测试模式下会大幅减少训练步数和 batch size，用于快速验证代码能跑通）
IS_TEST = False
DROUPOUT_RATE = 0.1

# 任务名称，对应 Q-probing parquet 数据集中的列名
# 不同任务探测模型对不同属性的记忆程度：
#   birth_date - 出生日期
#   birthcity  - 出生地
#   university - 毕业院校
#   field      - 研究领域
#   company1name - 第一家公司名称
#   company1city  - 第一家公司所在城市
TASK_NAME = "birth_date"

# 预训练模型选择：llama2（基座模型）或 bioS_*（知识增强模型）
# - bioS_single: 单模板知识增强
# - bioS_multi: 多模板知识增强
# - bioS_multi_permute_fullname: 多模板 + 姓名排列增强
MODEL_NAME = "llama2"

_MODEL_PATHS = {
    "bioS_multi": os.path.join(PART3_ROOT, "checkpoints", "bioS_multi", "llama2_lr1e-03_wd1e-01_2026_04_13_22_03_07"),
    "bioS_multi_permute_fullname": os.path.join(PART3_ROOT, "checkpoints", "bioS_multi_permute_fullname", "llama2_lr1e-03_wd1e-01_2026_04_13_22_03_59"),
    "bioS_single": os.path.join(PART3_ROOT, "checkpoints", "bioS_single", "llama2_lr1e-03_wd1e-01_2026_04_13_22_00_10"),
}

MODEL_PATH = _MODEL_PATHS[MODEL_NAME]

# Q-probing 训练和测试数据路径（parquet 格式）
TRAIN_PARQUET_PATH = os.path.join(PART3_ROOT, "datasets", "probing", "Q-probing", "train", "data.parquet")
TEST_PARQUET_PATH = os.path.join(PART3_ROOT, "datasets", "probing", "Q-probing", "test", "data.parquet")

# LoRA rank for embedding update
# rank 越大，embedding 更新的表达能力越强，但可训练参数也越多
# 论文中使用 16
LORA_RANK = 16

# 模型 checkpoint 和日志的输出根目录
OUTPUT_BASE_DIR = os.path.join(PART3_ROOT, "checkpoints", "probing", "Q-probing")

# 随机种子，确保实验可复现
SEED = 42



# ======== Main ========

def main():
    """Q-probing 训练主流程。"""
    # 设置随机种子，确保实验可复现
    set_seed(SEED)
    # 获取训练配置（区分正式/测试模式）
    train_config = get_train_config(IS_TEST)

    # 1. 加载 frozen LLaMA + tokenizer
    # dtype=torch.bfloat16: 使用 bfloat16 精度加载模型，节省显存
    # device_map="auto": 自动将模型分配到可用 GPU
    config = LlamaConfig.from_pretrained(MODEL_PATH)
    config.attention_dropout = DROUPOUT_RATE  # 开启 attention dropout，做正则化
    print(f"Config attention_dropout: {config.attention_dropout}")
    base_model = AutoModelForCausalLM.from_pretrained(
        MODEL_PATH,
        config=config,
        dtype=torch.bfloat16,
        device_map="auto",
    )
    # 构建分词器，与预训练模型保持一致
    tokenizer = build_tokenizer(model=base_model, model_path=MODEL_PATH)
    print(f"Tokenizer: pad={tokenizer.pad_token_id}, bos={tokenizer.bos_token_id}, eos={tokenizer.eos_token_id}")

    # 2. 构建 label encoder（从训练集统计所有类别）
    # 先读取训练集 parquet，收集该任务的所有唯一标签值
    import pandas as pd
    train_df = pd.read_parquet(TRAIN_PARQUET_PATH)
    label_encoder = LabelEncoder(train_df[TASK_NAME].tolist())
    n_classes = label_encoder.n_classes
    print(f"Task: {TASK_NAME}, n_classes: {n_classes}")

    # 3. 准备数据
    print("准备训练数据...")
    train_dataset = prepare_qprobing_dataset(tokenizer, TASK_NAME, TRAIN_PARQUET_PATH, label_encoder, is_test=IS_TEST)
    eval_dataset = prepare_qprobing_dataset(tokenizer, TASK_NAME, TEST_PARQUET_PATH, label_encoder, is_test=IS_TEST)
    print(f"训练集: {len(train_dataset)}, 测试集: {len(eval_dataset)}")

    # 构建 collator，用于动态 padding
    collator = QProbingCollator(pad_token_id=tokenizer.pad_token_id)

    # preview: 打印一个 batch 的形状，验证数据处理是否正确
    preview_features = [train_dataset[i] for i in range(min(2, len(train_dataset)))]
    preview_batch = collator(preview_features)
    print(f"[Preview] input_ids: {preview_batch['input_ids'].shape}, labels: {preview_batch['labels'].shape}")

    # 可视化数据集样本
    inspect_dataset_samples(train_dataset, tokenizer, label_encoder=label_encoder, num_samples=2)

    # 4. 构建 QProbingModel
    vocab_size = base_model.config.vocab_size      # 词表大小（如 32000）
    hidden_size = base_model.config.hidden_size    # 隐藏层维度（如 4096）

    q_model = QProbingModel(
        base_model=base_model,
        vocab_size=vocab_size,
        hidden_size=hidden_size,
        n_classes=n_classes,
        rank=LORA_RANK,
    ).cuda().to(torch.bfloat16)

    # 统计可训练参数和总参数
    trainable = sum(p.numel() for p in q_model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in q_model.parameters())
    print(f"可训练参数: {trainable:,} / 总参数: {total:,} ({100*trainable/total:.2f}%)")

    # 5. 构建输出目录
    # 目录名包含模型类型、学习率、权重衰减、rank 等关键超参信息
    tag = build_output_dir_tag(train_config, LORA_RANK)
    output_dir = os.path.join(
        OUTPUT_BASE_DIR,
        TASK_NAME,
        f"{base_model.config.model_type}_{tag}",
    ) + f"_{time.strftime('%Y_%m_%d_%H_%M_%S')}"
    if IS_TEST:
        output_dir += "/test"
    print(f"输出目录: {output_dir}")

    # 6. 构建 TrainingArguments
    # 将 TRAIN_CONFIG 中的配置传入 HuggingFace TrainingArguments
    training_args = TrainingArguments(
        output_dir=output_dir,
        do_train=True,                     # 执行训练
        do_eval=True,                       # 执行评估
        eval_strategy="steps",              # 按步数间隔评估
        eval_steps=train_config.get("eval_steps", 2000),  # 评估间隔步数
        max_steps=train_config["max_steps"],             # 最大训练步数
        learning_rate=train_config["learning_rate"],     # 学习率
        weight_decay=train_config["weight_decay"],       # 权重衰减
        adam_beta1=train_config["adam_beta1"],          # Adam beta1
        adam_beta2=train_config["adam_beta2"],          # Adam beta2
        adam_epsilon=train_config["adam_epsilon"],      # Adam epsilon
        warmup_steps=train_config["warmup_steps"],      # 预热步数
        lr_scheduler_type=train_config["lr_scheduler_type"],  # 学习率调度器
        per_device_train_batch_size=train_config["per_device_train_batch_size"],  # batch size
        gradient_accumulation_steps=train_config["gradient_accumulation_steps"],  # 梯度累积
        logging_steps=train_config["logging_steps"],     # 日志记录间隔
        save_steps=train_config["save_steps"],          # checkpoint 保存间隔
        bf16=train_config["bf16"],                      # bfloat16 混合精度
        fp16=train_config["fp16"],                      # float16（不使用）
        dataloader_num_workers=4,                        # 数据加载线程数
        remove_unused_columns=False,                    # 不移除未使用的列（自定义模型需要）
        report_to="swanlab",                            # 用 swanlab 记录训练日志
        save_only_model=True,                           # 只保存模型权重，不保存优化器状态
        run_name=f"{TASK_NAME}_{tag}",                  # swanlab 实验名
        project="Q-probing" if not IS_TEST else "Q-probing-test",                                # swanlab 项目名
    )

    # 7. 构建 Trainer
    trainer = Trainer(
        model=q_model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        data_collator=collator,
    )

    # 8. 开始训练
    train_result = trainer.train()

    # 9. 保存训练结果
    trainer.save_model()
    # 额外保存 label encoder 映射（推理时需要将模型输出映射回原始标签）
    import json
    label_map_path = os.path.join(output_dir, "label_map.json")
    with open(label_map_path, "w") as f:
        # 注意：JSON 的 key 必须是字符串，所以 id2label 的整数 key 要转为字符串
        json.dump({"label2id": label_encoder.label2id, "id2label": {str(k): v for k, v in label_encoder.id2label.items()}}, f, indent=2)
    print(f"Label mapping saved to {label_map_path}")

    # 记录和保存训练指标
    metrics = train_result.metrics
    metrics["train_samples"] = len(train_dataset)
    trainer.log_metrics("train", metrics)
    trainer.save_metrics("train", metrics)
    trainer.save_state()  # 保存 Trainer 状态（可用于恢复训练）

    print("Q-probing 训练完成！")


if __name__ == "__main__":
    main()
