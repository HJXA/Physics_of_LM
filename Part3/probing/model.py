"""
Probing 模型构建：P-Probing 与 Q-Probing 实验。

将冻结的预训练 GPT 模型包装为：
- Embedding 层上的 LoRA 风格低秩更新
- 每个属性任务独立的 BatchNorm + 线性分类器
- 在指定 token 位置提取隐藏状态
"""

from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from . import config


class LoRAEmbedding(nn.Module):
    """
    Embedding 层上的低秩更新。

    E' = E + A @ B

    其中 A: (vocab_size, r)，B: (r, hidden_size)。
    基础嵌入 E 被冻结；A 和 B 是可训练参数。
    """

    def __init__(self, vocab_size: int, hidden_size: int, rank: int, dtype: torch.dtype = torch.float32):
        super().__init__()
        self.rank = rank
        self.hidden_size = hidden_size
        self.vocab_size = vocab_size

        # LoRA 矩阵：小秩分解
        self.lora_A = nn.Parameter(torch.randn(vocab_size, rank, dtype=dtype) * 0.01)
        self.lora_B = nn.Parameter(torch.zeros(rank, hidden_size, dtype=dtype))

    def forward(self, embedding_weight: torch.Tensor, input_ids: torch.Tensor) -> torch.Tensor:
        """
        Args:
            embedding_weight: (vocab_size, hidden_size) — 基础嵌入（冻结）
            input_ids: (batch, seq_len)
        Returns:
            updated_embeddings: (batch, seq_len, hidden_size)
        """
        # 基础嵌入
        base_embed = F.embedding(input_ids, embedding_weight)

        # LoRA 更新
        lora_update = F.embedding(input_ids, self.lora_A) @ self.lora_B  # (B, T, r) @ (r, D) = (B, T, D)

        return base_embed + lora_update


class AttributeClassifier(nn.Module):
    """
    单个属性任务的 BatchNorm + 线性分类器。

    输入：隐藏状态向量 (hidden_size,)
    输出：属性标签空间上的 logits (num_classes,)
    """

    def __init__(self, hidden_size: int, num_classes: int):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_classes = num_classes

        self.batch_norm = nn.BatchNorm1d(hidden_size, eps=1e-6)
        self.linear = nn.Linear(hidden_size, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (batch, hidden_size) 或 (batch, seq_len, hidden_size)
        Returns:
            logits: (batch, num_classes)
        假设输入至少为 2D。
        """
        if x.dim() == 3:
            x = x.mean(dim=1)  # 如有需要则沿序列维度取平均

        x = self.batch_norm(x)
        logits = self.linear(x)
        return logits


class ProbeModel(nn.Module):
    """
    Probing 实验的模型包装器。

    - 冻结预训练的 GPT 模型
    - 在 Embedding 层上添加 LoRA 风格低秩更新
    - 添加每个属性独立的线性分类器
    - 在指定 token 位置或 END token 处提取隐藏状态

    用法:
        probe = ProbeModel(base_model, lora_rank=2)
        outputs = probe(input_ids, positions=..., mode="p")      # P-Probing
        outputs = probe(input_ids, end_token_pos=..., mode="q")  # Q-Probing
    """

    def __init__(
        self,
        base_model: nn.Module,
        lora_rank: int = 2,
        frozen: bool = True,
    ):
        super().__init__()
        self.base_model = base_model
        self.hidden_size = base_model.config.hidden_size
        self.vocab_size = base_model.config.vocab_size
        self.lora_rank = lora_rank

        # 冻结基础模型
        if frozen:
            for param in base_model.parameters():
                param.requires_grad = False
            base_model.eval()

        # Embedding LoRA 低秩更新
        self.lora_embedding = LoRAEmbedding(
            vocab_size=self.vocab_size,
            hidden_size=self.hidden_size,
            rank=lora_rank,
            dtype=torch.float32,
        )

        # 每个属性独立的分类器
        self.classifiers = nn.ModuleDict()
        for attr_name in config.ATTRIBUTE_NAMES:
            num_classes = len(config.LABEL_SPACES[attr_name])
            self.classifiers[attr_name] = AttributeClassifier(
                hidden_size=self.hidden_size,
                num_classes=num_classes,
            )

        # 损失函数（交叉熵，自动处理 -100 标签）
        self.loss_fn = nn.CrossEntropyLoss()

    def get_embedding_weight(self) -> torch.Tensor:
        """获取基础嵌入权重矩阵（冻结的）"""
        return self.base_model.get_input_embeddings().weight.data

    def get_updated_embeddings(self, input_ids: torch.Tensor) -> torch.Tensor:
        """
        计算带 LoRA 更新后的嵌入。

        返回：(batch, seq_len, hidden_size)
        """
        embed_weight = self.get_embedding_weight()
        return self.lora_embedding(embed_weight, input_ids)

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        positions: Optional[list[dict[str, int]]] = None,
        end_token_pos: Optional[torch.Tensor] = None,
        mode: str = "p",
        labels: Optional[dict[str, torch.Tensor]] = None,
    ) -> dict:
        """
        Probing 前向传播。

        Args:
            input_ids: (batch, seq_len)
            attention_mask: (batch, seq_len)
            positions: 列表 [{属性名: token_idx, ...}, ...]，用于 P-Probing
            end_token_pos: (batch,) 结束 token 位置张量，用于 Q-Probing
            mode: "p" 表示 P-Probing，"q" 表示 Q-Probing
            labels: {属性名: (batch,)} 标签张量字典（可选，用于计算损失）

        Returns:
            {
                "logits": {属性名: (batch, num_classes)},
                "loss": 标量（如果提供了 labels）,
                "hidden_states": 提取的隐藏状态,
                "last_hidden": 最后一层所有 token 的隐藏状态,
            }
        """
        batch_size = input_ids.size(0)

        # 通过 LoRA 更新获取嵌入
        input_embeds = self.get_updated_embeddings(input_ids)

        # 通过基础模型完整前向传播
        # 使用 inputs_embeds 传入自定义嵌入
        outputs = self.base_model(
            inputs_embeds=input_embeds,
            attention_mask=attention_mask,
            output_hidden_states=True,
            return_dict=True,
        )

        # 最后一层隐藏状态：(batch, seq_len, hidden_size)
        last_hidden = outputs.last_hidden_state

        # 在目标位置提取隐藏状态
        if mode == "p" and positions is not None:
            # P-Probing：按样本、按属性提取指定位置的隐藏状态
            extracted = {}
            for attr_name in config.ATTRIBUTE_NAMES:
                vectors = []
                for i in range(batch_size):
                    pos = positions[i].get(attr_name, -1)
                    if pos >= 0:
                        vectors.append(last_hidden[i, pos])
                    else:
                        # 退化方案：使用第一个位置
                        vectors.append(last_hidden[i, 0])
                extracted[attr_name] = torch.stack(vectors)  # (batch, hidden_size)
        elif mode == "q" and end_token_pos is not None:
            # Q-Probing：在所有样本的 END token 位置提取隐藏状态
            end_hidden = last_hidden[torch.arange(batch_size), end_token_pos]  # (batch, hidden_size)
            extracted = {attr_name: end_hidden for attr_name in config.ATTRIBUTE_NAMES}
        else:
            raise ValueError(f"探测模式/位置无效: mode={mode}")

        # 经过分类器
        logits = {}
        for attr_name, hidden_vec in extracted.items():
            logits[attr_name] = self.classifiers[attr_name](hidden_vec)

        # 计算损失：对各属性损失取平均
        loss = None
        if labels is not None:
            total_loss = 0.0
            count = 0
            for attr_name in config.ATTRIBUTE_NAMES:
                if attr_name in labels:
                    attr_labels = labels[attr_name].to(last_hidden.device)
                    valid_mask = attr_labels >= 0
                    if valid_mask.any():
                        attr_logits = logits[attr_name][valid_mask]
                        attr_label_filtered = attr_labels[valid_mask]
                        loss_val = self.loss_fn(attr_logits, attr_label_filtered)
                        total_loss += loss_val
                        count += 1
            if count > 0:
                loss = total_loss / count
            else:
                loss = torch.tensor(0.0, device=last_hidden.device)

        return {
            "logits": logits,
            "loss": loss,
            "hidden_states": extracted,
            "last_hidden": last_hidden,
        }

    def compute_accuracy(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        positions: Optional[list[dict[str, int]]] = None,
        end_token_pos: Optional[torch.Tensor] = None,
        mode: str = "p",
        labels: Optional[dict[str, torch.Tensor]] = None,
    ) -> dict[str, float]:
        """计算每个属性的准确率"""
        with torch.no_grad():
            outputs = self.forward(
                input_ids,
                attention_mask=attention_mask,
                positions=positions,
                end_token_pos=end_token_pos,
                mode=mode,
                labels=None,
            )

        accuracies = {}
        if labels is not None:
            for attr_name in config.ATTRIBUTE_NAMES:
                if attr_name not in labels:
                    continue
                attr_labels = labels[attr_name].to(outputs["logits"][attr_name].device)
                valid_mask = attr_labels >= 0
                if valid_mask.sum().item() == 0:
                    accuracies[attr_name] = 0.0
                    continue

                preds = outputs["logits"][attr_name].argmax(dim=-1)
                correct = (preds[valid_mask] == attr_labels[valid_mask]).float()
                accuracies[attr_name] = correct.mean().item()

        return accuracies
