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



# 指定使用的 GPU 编号
os.environ["CUDA_VISIBLE_DEVICES"] = "4"

# 让 probing/ 能 import Train/ 下的模块
# 因为项目结构中 Train/ 和 probing/ 是平级目录，需要将项目根目录加入 sys.path
sys.path.append("/ruilab/jxhe/CoE_Monitor/Physics_of_LM")

PART3_ROOT = "/ruilab/jxhe/CoE_Monitor/Physics_of_LM/Part3"

import torch
import torch.nn as nn
from datasets import load_dataset
from transformers import AutoModelForCausalLM, Trainer, TrainingArguments, set_seed
import swanlab

from Part3.Train.tokenization import build_tokenizer
from utils import build_output_dir_tag, get_train_config, LabelEncoder


# ======== 配置 ========

# 是否为测试模式（测试模式下会大幅减少训练步数和 batch size，用于快速验证代码能跑通）
IS_TEST = False

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
# - llama2: 基座 LLaMA，无知识增强
# - bioS_single: 单模板知识增强
# - bioS_multi: 多模板知识增强
# - bioS_multi_permute_fullname: 多模板 + 姓名排列增强
MODEL_NAME = "llama2"

_MODEL_PATHS = {
    "llama2": os.path.join(PART3_ROOT, "checkpoints", "llama2"),
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
# 论文中使用 128
LORA_RANK = 128

# 模型 checkpoint 和日志的输出根目录
OUTPUT_BASE_DIR = os.path.join(PART3_ROOT, "checkpoints", "probing", "Q-probing")

# 随机种子，确保实验可复现
SEED = 42


# ======== Model ========

class QProbingEmbeddingUpdate(nn.Module):
    """rank-r 可训练 embedding 更新模块：E + A @ B，其中 A: (V, r), B: (r, H)。

    核心思想：不直接修改原始 embedding 矩阵，而是用低秩分解来学习一个增量。
    - 原始 embedding: base_embedding(input_ids)，形状 (B, seq_len, H)
    - 增量 embedding: A[input_ids] @ B，形状 (B, seq_len, H)
    - 最终 embedding: base + delta

    参数量对比：
    - 直接训练 embedding: V × H（如 32000 × 4096 ≈ 1.3 亿）
    - 低秩分解: V × r + r × H（如 32000 × 128 + 128 × 4096 ≈ 460 万，减少约 97%）

    初始化策略：
    - A: 随机高斯分布 * 0.01（小随机值，避免初始增量过大）
    - B: 全零初始化（训练开始时 delta = A @ B = 0，即从原始 embedding 出发）
    """

    def __init__(self, vocab_size: int, hidden_size: int, rank: int):
        super().__init__()
        # lora_A: (vocab_size, rank)，随机初始化
        self.lora_A = nn.Parameter(torch.randn(vocab_size, rank) * 0.01)
        # lora_B: (rank, hidden_size)，零初始化
        # 零初始化保证训练开始时 delta = 0，模型行为与原始模型一致
        self.lora_B = nn.Parameter(torch.zeros(rank, hidden_size))

    def forward(self, input_ids: torch.Tensor, base_embedding: nn.Embedding) -> torch.Tensor:
        """前向传播：计算 base embedding + 低秩增量。

        Args:
            input_ids: 输入 token id，形状 (B, seq_len)
            base_embedding: 原始 embedding 层 (nn.Embedding)

        Returns:
            更新后的 embedding，形状 (B, seq_len, hidden_size)
        """
        # 原始 embedding: (B, seq_len, H)
        base = base_embedding(input_ids)
        # 增量 embedding:
        #   self.lora_A[input_ids]: 从 A 中按 input_ids 索引，得到 (B, seq_len, r)
        #   @ self.lora_B: 矩阵乘法，(B, seq_len, r) @ (r, H) → (B, seq_len, H)
        # .to(input_ids.device) 确保参数和输入在同一设备上（CPU/GPU）
        delta = self.lora_A.to(input_ids.device)[input_ids] @ self.lora_B.to(input_ids.device)
        return base + delta


class QProbingHead(nn.Module):
    """BN + Linear 分类头。

    结构：BatchNorm → Linear
    - BatchNorm 对提取的隐藏状态做归一化，加速训练收敛
    - affine=True 表示 BN 有可学习的 scale (gamma) 和 shift (beta) 参数
    - Linear 将归一化后的隐藏状态映射到 n_classes 个类别

    为什么用 BatchNorm 而不是 LayerNorm？
    - BatchNorm 按 batch 维度统计均值/方差，适合分类任务
    - LayerNorm 按特征维度统计，更适合序列建模
    - 论文中使用 BatchNorm
    """

    def __init__(self, hidden_size: int, n_classes: int):
        super().__init__()
        # BatchNorm1d: 输入形状 (B, H)，在 batch 维度上归一化
        self.bn = nn.BatchNorm1d(hidden_size, affine=True)
        # 分类线性层: H → n_classes
        self.classifier = nn.Linear(hidden_size, n_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """前向传播：BN → Linear。

        Args:
            x: 输入隐藏状态，形状 (B, hidden_size)

        Returns:
            分类 logits，形状 (B, n_classes)
        """
        return self.classifier(self.bn(x))


class QProbingModel(nn.Module):
    """
    Q-probing 完整模型：frozen LLaMA + trainable rank-r embedding update + probing head。

    Forward 流程：
    1. 用 embedding update 计算合并 embedding（原始 + 低秩增量）
    2. 将合并后的 embedding 送入 frozen LLaMA（output_hidden_states=True）
    3. 提取 ending token（每个样本最后一个非 padding token）位置的 last layer hidden state
    4. 通过 BN → Linear 得到分类 logits
    5. 如有 labels，计算交叉熵损失

    关键设计决策：
    - 冻结 LLaMA 所有参数，仅训练 embedding update 和分类头
    - LLaMA 设为 train 模式（而非 eval 模式），保留 dropout 行为
    - 提取 ending token 位置而非 [CLS] 位置，因为 LLaMA 没有 [CLS] token
    - ending token 的隐藏状态被视为整个输入序列的表示
    """

    def __init__(self, base_model, vocab_size: int, hidden_size: int, n_classes: int, rank: int):
        super().__init__()
        self.base_model = base_model
        self.vocab_size = vocab_size
        self.hidden_size = hidden_size

        # 冻结 base model 的所有参数，不参与梯度更新
        for param in self.base_model.parameters():
            param.requires_grad = False

        # 将 frozen model 设为 train 模式
        # 这样模型中的 dropout 层仍然按训练模式工作（随机丢弃），
        # 但由于参数被冻结，梯度不会回传到 base model
        self.base_model.train()

        # 可训练的 embedding 更新模块
        self.emb_update = QProbingEmbeddingUpdate(vocab_size, hidden_size, rank)
        # 可训练的分类头
        self.head = QProbingHead(hidden_size, n_classes)

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        labels: torch.Tensor | None = None,
    ):
        """前向传播。

        Args:
            input_ids: 输入 token id，形状 (B, seq_len)
            attention_mask: 注意力掩码，1=有效token, 0=padding，形状 (B, seq_len)
            labels: 分类标签索引，形状 (B,)，可选

        Returns:
            dict: {"loss": 交叉熵损失（可能为 None）, "logits": 分类 logits (B, n_classes)}
        """
        # 1. 计算 embedding（base + 低秩更新）
        # inputs_embeds: (B, seq_len, H)
        inputs_embeds = self.emb_update(input_ids, self.base_model.model.embed_tokens)

        # 2. 通过 frozen LLaMA（train 模式保留 dropout，但参数不更新）
        # 调用 base_model.model() 而非 base_model()，因为 base_model 是 CausalLM 包装
        # output_hidden_states=True: 返回所有层的隐藏状态，我们需要最后一层
        # use_cache=False: 不需要 KV cache，节省显存
        outputs = self.base_model.model(
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
            output_hidden_states=True,
            return_dict=True,
            use_cache=False,
        )

        # 3. 提取 ending token 位置的 last layer hidden state
        # hidden_states[-1] 是 Transformer 最后一层的输出
        last_hidden = outputs.hidden_states[-1]  # (B, seq_len, H)

        # ending token = 每个样本最后一个非 padding 位置
        # attention_mask: (B, seq_len), 1=有效, 0=padding
        # sum(dim=1) 得到每个样本的有效 token 数，-1 得到最后一个有效 token 的索引
        seq_lengths = attention_mask.sum(dim=1) - 1  # (B,)
        # 构造 batch 索引 [0, 1, 2, ..., B-1]
        batch_idx = torch.arange(last_hidden.size(0), device=last_hidden.device)
        # 使用高级索引提取每个样本 ending token 的隐藏状态
        # last_hidden[i, seq_lengths[i]] 得到第 i 个样本的 ending token 表示
        eos_hidden = last_hidden[batch_idx, seq_lengths]  # (B, H)

        # 4. 分类：BN 归一化 → Linear 投影
        logits = self.head(eos_hidden)  # (B, n_classes)

        # 5. 计算 loss（仅在训练时提供 labels）
        loss = None
        if labels is not None:
            # 使用交叉熵损失，适用于多分类任务
            # logits: (B, n_classes), labels: (B,)
            loss = nn.functional.cross_entropy(logits, labels)

        return {"loss": loss, "logits": logits}


# ======== Data ========

def prepare_qprobing_dataset(tokenizer, task_name: str, parquet_path: str, label_encoder: LabelEncoder, filter_unknown: bool = True):
    """准备 Q-probing 数据集：读取 parquet → 过滤未知标签 → tokenize → 编码标签。

    Args:
        tokenizer: 分词器，用于将文本转为 token id
        task_name: 任务名（parquet 中的列名），如 "birth_date"
        parquet_path: parquet 数据文件路径
        label_encoder: 标签编码器，将字符串标签转为整数
        filter_unknown: 是否过滤掉训练集中未见过的标签类别
            - True: 过滤测试集中出现但训练集中未出现的标签，避免评估偏差
            - False: 保留所有样本

    Returns:
        tokenized Dataset，包含 input_ids, attention_mask, labels 三列
    """
    # 从 parquet 文件加载数据集
    raw = load_dataset("parquet", data_files=parquet_path)["train"]

    # 测试模式下只取少量样本，快速验证
    if IS_TEST:
        raw = raw.select(range(min(8, len(raw))))

    # 过滤掉 label encoder 中未知的标签
    # 这些标签只出现在测试集中，未出现在训练集中，模型不可能正确分类
    if filter_unknown:
        known = set(label_encoder.label2id.keys())
        before = len(raw)
        raw = raw.filter(lambda x: x[task_name] in known, desc="Filtering unknown labels")
        print(f"  过滤未知 label: {before} -> {len(raw)}")

    def _map_fn(example):
        """对单条样本做 tokenize 和标签编码。"""
        text = example["text"]
        # 手动添加 BOS 和 EOS token：
        # - LLaMA 分词器默认不自动添加特殊 token
        # - BOS (Beginning Of Sequence) 标记序列开始
        # - EOS (End Of Sequence) 标记序列结束，也是 probing 提取表示的位置
        ids = [tokenizer.bos_token_id] + tokenizer(text, add_special_tokens=False)["input_ids"] + [tokenizer.eos_token_id]
        # attention_mask 全为 1（后续 padding 时会被 collator 调整）
        attention_mask = [1] * len(ids)
        # 将字符串标签编码为整数
        label = label_encoder.encode(example[task_name])
        return {
            "input_ids": ids,
            "attention_mask": attention_mask,
            "labels": label,
        }

    # 对整个数据集执行 map 操作
    # remove_columns: 移除原始列，只保留 tokenize 后的列
    # load_from_cache_file=False: 不使用缓存，确保数据最新
    tokenized = raw.map(
        _map_fn,
        remove_columns=raw.column_names,
        load_from_cache_file=False,
        desc=f"Tokenizing Q-probing ({task_name})",
    )
    return tokenized


class QProbingCollator:
    """Dynamic padding collator：按 batch 内最长序列 padding，labels 为分类索引。

    功能：
    - 将不同长度的样本 pad 到同一长度（batch 内最长序列的长度）
    - input_ids 用 pad_token_id 填充
    - attention_mask 用 0 填充（表示 padding 位置）
    - labels 不需要 padding（每个样本只有一个分类标签）

    为什么用 dynamic padding？
    - 不同样本长度差异大时，固定 padding 到 max_length 会浪费大量计算
    - dynamic padding 只 pad 到当前 batch 的最大长度，更高效
    """

    def __init__(self, pad_token_id: int):
        """Args:
            pad_token_id: padding token 的 id
        """
        self.pad_token_id = pad_token_id

    def __call__(self, features: list[dict]) -> dict[str, torch.Tensor]:
        """将一个 batch 的特征列表转为 padded tensor dict。

        Args:
            features: 列表，每个元素是一条样本的 dict（含 input_ids, attention_mask, labels）

        Returns:
            dict: {
                "input_ids": (B, max_seq_len), padded,
                "attention_mask": (B, max_seq_len), padded,
                "labels": (B,), 分类标签索引
            }
        """
        # 将 list 转为 tensor
        input_ids = [torch.tensor(f["input_ids"], dtype=torch.long) for f in features]
        attention_mask = [torch.tensor(f["attention_mask"], dtype=torch.long) for f in features]
        labels = torch.tensor([f["labels"] for f in features], dtype=torch.long)

        # pad_sequence: 将变长序列 pad 到同一长度
        # batch_first=True: 输出形状 (B, max_seq_len)
        # padding_value: input_ids 用 pad_token_id，attention_mask 用 0
        input_ids = nn.utils.rnn.pad_sequence(input_ids, batch_first=True, padding_value=self.pad_token_id)
        attention_mask = nn.utils.rnn.pad_sequence(attention_mask, batch_first=True, padding_value=0)

        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": labels,
        }


# ======== Trainer ========

class QProbingTrainer(Trainer):
    """Q-probing 专用 Trainer：重写 compute_loss 走分类路径，集成 swanlab 实验日志。

    与默认 Trainer 的区别：
    1. compute_loss 中调用模型的 forward，获取 loss 和 logits
    2. 额外计算并记录训练准确率 (accuracy)
    3. 集成 swanlab 做实验跟踪和可视化
    """

    def __init__(self, model, swanlab_project="Q_Probing", swanlab_experiment_name="q_probing",
                 swanlab_description="", *args, **kwargs):
        """Args:
            model: QProbingModel 实例
            swanlab_project: swanlab 项目名
            swanlab_experiment_name: swanlab 实验名
            swanlab_description: swanlab 实验描述
        """
        super().__init__(model=model, *args, **kwargs)
        # 初始化 swanlab 实验跟踪
        swanlab.init(
            project=swanlab_project,
            experiment_name=swanlab_experiment_name,
            description=swanlab_description,
            config=self.args.to_dict(),  # 将训练配置记录到 swanlab
        )

    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        """计算损失并记录训练指标。

        重写 Trainer 的 compute_loss：
        1. 调用模型 forward，获取 loss 和 logits
        2. 计算 batch 准确率
        3. 记录 loss 和 acc 到 swanlab

        Args:
            model: 当前模型
            inputs: batch 输入 dict（含 input_ids, attention_mask, labels）
            return_outputs: 是否返回模型输出（用于某些 Trainer 内部逻辑）
            num_items_in_batch: batch 中的样本数（Trainer 传入，此处未使用）

        Returns:
            loss 或 (loss, outputs)
        """
        outputs = model(
            input_ids=inputs["input_ids"],
            attention_mask=inputs["attention_mask"],
            labels=inputs["labels"],
        )
        loss = outputs["loss"]

        # 计算当前 batch 的准确率
        # argmax(dim=-1) 取每个样本预测概率最大的类别
        preds = outputs["logits"].argmax(dim=-1)
        # 与真实标签比较，计算准确率
        acc = (preds == inputs["labels"]).float().mean().item()

        # 记录到 swanlab，用于训练过程可视化
        step = self.state.global_step
        swanlab.log({"loss": loss.item(), "acc": acc}, step=step)

        return (loss, outputs) if return_outputs else loss


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
    base_model = AutoModelForCausalLM.from_pretrained(
        MODEL_PATH,
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
    train_dataset = prepare_qprobing_dataset(tokenizer, TASK_NAME, TRAIN_PARQUET_PATH, label_encoder)
    eval_dataset = prepare_qprobing_dataset(tokenizer, TASK_NAME, TEST_PARQUET_PATH, label_encoder)
    print(f"训练集: {len(train_dataset)}, 测试集: {len(eval_dataset)}")

    # 构建 collator，用于动态 padding
    collator = QProbingCollator(pad_token_id=tokenizer.pad_token_id)

    # preview: 打印一个 batch 的形状，验证数据处理是否正确
    preview_features = [train_dataset[i] for i in range(min(2, len(train_dataset)))]
    preview_batch = collator(preview_features)
    print(f"[Preview] input_ids: {preview_batch['input_ids'].shape}, labels: {preview_batch['labels'].shape}")

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
        output_dir += "_test"
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
        report_to="none",                                # 不用 HF 自带日志（用 swanlab）
        save_only_model=True,                           # 只保存模型权重，不保存优化器状态
    )

    # 7. 构建 Trainer
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
