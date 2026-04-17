"""
Q-probing 训练脚本。

冻结预训练 LLaMA，仅训练少量参数做属性分类：
  - rank-r embedding update (vocab_size × r, r × hidden_size)
  - BatchNorm1d(hidden_size, affine=True)
  - Linear(hidden_size, n_classes)

参考论文：AdamW, ε=1e-6, weight_decay=0.3, lr=0.001, no warmup, linear decay to 0,
         batch_size=200, 30000 steps。
"""

import os
import sys
import time

os.environ["CUDA_VISIBLE_DEVICES"] = "2"

# 让 probing/ 能 import Train/ 下的模块

sys.path.append("/ruilab/jxhe/CoE_Monitor/Physics_of_LM")

import torch
import torch.nn as nn
from datasets import load_dataset
from transformers import AutoModelForCausalLM, Trainer, TrainingArguments, set_seed
import swanlab

from Part3.Train.tokenization import build_tokenizer
from Part3.Train.train_utils import preview_collator_batch


# ======== 配置 ========

IS_TEST = False

# 任务名称，对应 Q-probing parquet 中的列名
TASK_NAME = "birth_date"  # 可选: birth_date, birthcity, university, field, company1name, company1city

# 预训练模型路径
MODEL_PATH = os.path.join(PART3_ROOT, "checkpoints", "llama2")

# Q-probing 数据路径
TRAIN_PARQUET_PATH = os.path.join(PART3_ROOT, "datasets", "probing", "Q-probing", "train", "data.parquet")
TEST_PARQUET_PATH = os.path.join(PART3_ROOT, "datasets", "probing", "Q-probing", "test", "data.parquet")

# LoRA rank for embedding update
LORA_RANK = 128

OUTPUT_BASE_DIR = os.path.join(PART3_ROOT, "checkpoints", "probing", "Q-probing")

SEED = 42

# 训练超参（论文设定）
TRAIN_CONFIG = {
    "max_steps": 30000,
    "learning_rate": 0.001,
    "weight_decay": 0.3,
    "adam_beta1": 0.9,
    "adam_beta2": 0.98,
    "adam_epsilon": 1e-6,
    "warmup_steps": 0,
    "lr_scheduler_type": "linear",  # linear decay to 0
    "per_device_train_batch_size": 200,
    "gradient_accumulation_steps": 1,
    "logging_steps": 100,
    "save_steps": 2000,
    "eval_steps": 2000,
    "bf16": True,
    "fp16": False,
}


def get_train_config(is_test: bool) -> dict:
    config = dict(TRAIN_CONFIG)
    if is_test:
        config["max_steps"] = 3
        config["save_steps"] = 10
        config["per_device_train_batch_size"] = 4
        config["gradient_accumulation_steps"] = 1
    return config


def format_scientific(value: float) -> str:
    return format(value, ".0e")


def build_output_dir_tag(train_config: dict) -> str:
    return f"lr{format_scientific(train_config['learning_rate'])}_wd{format_scientific(train_config['weight_decay'])}_rank{LORA_RANK}"


# ======== Model ========

class QProbingEmbeddingUpdate(nn.Module):
    """rank-r 可训练 embedding 更新：E + A @ B，其中 A: (V, r), B: (r, H)。"""

    def __init__(self, vocab_size: int, hidden_size: int, rank: int):
        super().__init__()
        self.lora_A = nn.Parameter(torch.randn(vocab_size, rank) * 0.01)
        self.lora_B = nn.Parameter(torch.zeros(rank, hidden_size))

    def forward(self, input_ids: torch.Tensor, base_embedding: nn.Embedding) -> torch.Tensor:
        base = base_embedding(input_ids)
        delta = self.lora_A.to(input_ids.device)[input_ids] @ self.lora_B.to(input_ids.device)
        return base + delta


class QProbingHead(nn.Module):
    """BN + Linear 分类头。"""

    def __init__(self, hidden_size: int, n_classes: int):
        super().__init__()
        self.bn = nn.BatchNorm1d(hidden_size, affine=True)
        self.classifier = nn.Linear(hidden_size, n_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.classifier(self.bn(x))


class QProbingModel(nn.Module):
    """
    Q-probing 模型：frozen LLaMA + trainable rank-r embedding update + probing head。

    Forward 流程：
    1. 用 embedding update 计算合并 embedding
    2. 通过 frozen LLaMA（output_hidden_states=True）
    3. 提取 ending token（eos）位置的 last layer hidden state
    4. BN → Linear → logits
    """

    def __init__(self, base_model, vocab_size: int, hidden_size: int, n_classes: int, rank: int):
        super().__init__()
        self.base_model = base_model
        self.vocab_size = vocab_size
        self.hidden_size = hidden_size

        # 冻结 base model
        for param in self.base_model.parameters():
            param.requires_grad = False
        # 允许 frozen model 使用 dropout
        self.base_model.train()

        self.emb_update = QProbingEmbeddingUpdate(vocab_size, hidden_size, rank)
        self.head = QProbingHead(hidden_size, n_classes)

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        labels: torch.Tensor | None = None,
    ):
        # 1. 计算 embedding（base + update）
        inputs_embeds = self.emb_update(input_ids, self.base_model.model.embed_tokens)

        # 2. 通过 frozen LLaMA（train 模式保留 dropout，但 frozen 参数不更新）
        outputs = self.base_model.model(
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
            output_hidden_states=True,
            return_dict=True,
            use_cache=False,
        )

        # 3. 提取 ending token 位置的 last layer hidden state
        last_hidden = outputs.hidden_states[-1]  # (B, seq_len, H)

        # ending token = 每个样本最后一个非 padding 位置
        # attention_mask: (B, seq_len), 1=有效, 0=padding
        seq_lengths = attention_mask.sum(dim=1) - 1  # (B,)
        batch_idx = torch.arange(last_hidden.size(0), device=last_hidden.device)
        eos_hidden = last_hidden[batch_idx, seq_lengths]  # (B, H)

        # 4. 分类
        logits = self.head(eos_hidden)  # (B, n_classes)

        # 5. 计算 loss
        loss = None
        if labels is not None:
            loss = nn.functional.cross_entropy(logits, labels)

        return {"loss": loss, "logits": logits}


# ======== Data ========

class LabelEncoder:
    """字符串 label → 整数索引。未知 label 映射到 -1。"""

    def __init__(self, labels: list[str]):
        unique = sorted(set(labels))
        self.label2id = {l: i for i, l in enumerate(unique)}
        self.id2label = {i: l for l, i in self.label2id.items()}
        self.n_classes = len(unique)

    def encode(self, label: str) -> int:
        return self.label2id.get(label, -1)

    def decode(self, idx: int) -> str:
        return self.id2label.get(idx, "<UNK>")


def prepare_qprobing_dataset(tokenizer, task_name: str, parquet_path: str, label_encoder: LabelEncoder, filter_unknown: bool = True):
    """读取 parquet，tokenize text 列，编码 label。filter_unknown 时过滤掉未知 label。"""
    raw = load_dataset("parquet", data_files=parquet_path)["train"]

    if IS_TEST:
        raw = raw.select(range(min(8, len(raw))))

    if filter_unknown:
        known = set(label_encoder.label2id.keys())
        before = len(raw)
        raw = raw.filter(lambda x: x[task_name] in known, desc="Filtering unknown labels")
        print(f"  过滤未知 label: {before} -> {len(raw)}")

    def _map_fn(example):
        text = example["text"]
        ids = [tokenizer.bos_token_id] + tokenizer(text, add_special_tokens=False)["input_ids"] + [tokenizer.eos_token_id]
        attention_mask = [1] * len(ids)
        label = label_encoder.encode(example[task_name])
        return {
            "input_ids": ids,
            "attention_mask": attention_mask,
            "labels": label,
        }

    tokenized = raw.map(
        _map_fn,
        remove_columns=raw.column_names,
        load_from_cache_file=False,
        desc=f"Tokenizing Q-probing ({task_name})",
    )
    return tokenized


class QProbingCollator:
    """Dynamic padding collator：按 batch 内最长序列 padding，labels 为分类索引。"""

    def __init__(self, pad_token_id: int):
        self.pad_token_id = pad_token_id

    def __call__(self, features: list[dict]) -> dict[str, torch.Tensor]:
        input_ids = [torch.tensor(f["input_ids"], dtype=torch.long) for f in features]
        attention_mask = [torch.tensor(f["attention_mask"], dtype=torch.long) for f in features]
        labels = torch.tensor([f["labels"] for f in features], dtype=torch.long)

        input_ids = nn.utils.rnn.pad_sequence(input_ids, batch_first=True, padding_value=self.pad_token_id)
        attention_mask = nn.utils.rnn.pad_sequence(attention_mask, batch_first=True, padding_value=0)

        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": labels,
        }


# ======== Trainer ========

class QProbingTrainer(Trainer):
    """Q-probing 专用 Trainer：compute_loss 走分类路径，集成 swanlab 日志。"""

    def __init__(self, model, swanlab_project="Q_Probing", swanlab_experiment_name="q_probing",
                 swanlab_description="", *args, **kwargs):
        super().__init__(model=model, *args, **kwargs)
        swanlab.init(
            project=swanlab_project,
            experiment_name=swanlab_experiment_name,
            description=swanlab_description,
            config=self.args.to_dict(),
        )

    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        outputs = model(
            input_ids=inputs["input_ids"],
            attention_mask=inputs["attention_mask"],
            labels=inputs["labels"],
        )
        loss = outputs["loss"]

        preds = outputs["logits"].argmax(dim=-1)
        acc = (preds == inputs["labels"]).float().mean().item()

        step = self.state.global_step
        swanlab.log({"loss": loss.item(), "acc": acc}, step=step)

        return (loss, outputs) if return_outputs else loss


# ======== Main ========

def main():
    set_seed(SEED)
    train_config = get_train_config(IS_TEST)

    # 1. 加载 frozen LLaMA + tokenizer
    base_model = AutoModelForCausalLM.from_pretrained(
        MODEL_PATH,
        dtype=torch.bfloat16,
        device_map="auto",
    )
    tokenizer = build_tokenizer(model=base_model, model_path=MODEL_PATH)
    print(f"Tokenizer: pad={tokenizer.pad_token_id}, bos={tokenizer.bos_token_id}, eos={tokenizer.eos_token_id}")

    # 2. 构建 label encoder（从训练集统计）
    import pandas as pd
    train_df = pd.read_parquet(TRAIN_PARQUET_PATH)
    label_encoder = LabelEncoder(train_df[TASK_NAME].tolist())
    n_classes = label_encoder.n_classes
    print(f"Task: {TASK_NAME}, n_classes: {n_classes}")

    # 3. 准备数据
    print("准备训练数据...")
    train_dataset = prepare_qprobing_dataset(tokenizer, TASK_NAME, TRAIN_PARQUET_PATH, label_encoder)
    eval_dataset = prepare_qprobing_dataset(tokenizer, TASK_NAME, TEST_PARQUET_PATH, label_encoder)
    print(f"训练集: {len(train_dataset)}, 测试集: {len(eval_dataset)}")

    collator = QProbingCollator(pad_token_id=tokenizer.pad_token_id)

    # preview
    preview_features = [train_dataset[i] for i in range(min(2, len(train_dataset)))]
    preview_batch = collator(preview_features)
    print(f"[Preview] input_ids: {preview_batch['input_ids'].shape}, labels: {preview_batch['labels'].shape}")

    # 4. 构建 QProbingModel
    vocab_size = base_model.config.vocab_size
    hidden_size = base_model.config.hidden_size

    q_model = QProbingModel(
        base_model=base_model,
        vocab_size=vocab_size,
        hidden_size=hidden_size,
        n_classes=n_classes,
        rank=LORA_RANK,
    ).cuda().to(torch.bfloat16)

    # 统计可训练参数
    trainable = sum(p.numel() for p in q_model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in q_model.parameters())
    print(f"可训练参数: {trainable:,} / 总参数: {total:,} ({100*trainable/total:.2f}%)")

    # 5. 输出目录
    tag = build_output_dir_tag(train_config)
    output_dir = os.path.join(
        OUTPUT_BASE_DIR,
        TASK_NAME,
        f"{base_model.config.model_type}_{tag}",
    ) + f"_{time.strftime('%Y_%m_%d_%H_%M_%S')}"
    if IS_TEST:
        output_dir += "_test"
    print(f"输出目录: {output_dir}")

    # 6. TrainingArguments
    training_args = TrainingArguments(
        output_dir=output_dir,
        do_train=True,
        do_eval=True,
        eval_strategy="steps",
        eval_steps=train_config.get("eval_steps", 2000),
        max_steps=train_config["max_steps"],
        learning_rate=train_config["learning_rate"],
        weight_decay=train_config["weight_decay"],
        adam_beta1=train_config["adam_beta1"],
        adam_beta2=train_config["adam_beta2"],
        adam_epsilon=train_config["adam_epsilon"],
        warmup_steps=train_config["warmup_steps"],
        lr_scheduler_type=train_config["lr_scheduler_type"],
        per_device_train_batch_size=train_config["per_device_train_batch_size"],
        gradient_accumulation_steps=train_config["gradient_accumulation_steps"],
        logging_steps=train_config["logging_steps"],
        save_steps=train_config["save_steps"],
        bf16=train_config["bf16"],
        fp16=train_config["fp16"],
        dataloader_num_workers=4,
        remove_unused_columns=False,
        report_to="none",
        save_only_model=True,
    )

    # 7. Trainer
    trainer = QProbingTrainer(
        model=q_model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        data_collator=collator,
        swanlab_project="Q_Probing" if not IS_TEST else "Q_Probing_Test",
        swanlab_experiment_name=f"{TASK_NAME}_{tag}",
        swanlab_description=f"Q-probing on {TASK_NAME} ({n_classes} classes)",
    )

    # 8. 训练
    train_result = trainer.train()

    # 9. 保存
    trainer.save_model()
    # 额外保存 label encoder 映射
    import json
    label_map_path = os.path.join(output_dir, "label_map.json")
    with open(label_map_path, "w") as f:
        json.dump({"label2id": label_encoder.label2id, "id2label": {str(k): v for k, v in label_encoder.id2label.items()}}, f, indent=2)
    print(f"Label mapping saved to {label_map_path}")

    metrics = train_result.metrics
    metrics["train_samples"] = len(train_dataset)
    trainer.log_metrics("train", metrics)
    trainer.save_metrics("train", metrics)
    trainer.save_state()

    print("Q-probing 训练完成！")


if __name__ == "__main__":
    main()
