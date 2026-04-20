import torch.nn as nn
import torch

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