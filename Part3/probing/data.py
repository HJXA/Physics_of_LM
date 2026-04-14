"""
P-Probing 与 Q-Probing 实验数据集构建。

P-Probing：加载 bioS 预训练数据集，对传记文本分词，
    定位每个属性值前的特殊 token 位置（介词位置）。

Q-Probing：从 QA parquet / dict 数据集中生成仅含人名的句子
    并附带属性标签。
"""

from __future__ import annotations

import os
from typing import Optional

import numpy as np
import pyarrow.parquet as pq
import torch
from torch.utils.data import Dataset

from . import config


# ============================================================================
# 工具函数：加载 parquet 数据集
# ============================================================================

def load_parquet_files(dir_path: str) -> list:
    """加载目录下所有 .parquet 文件并返回列表"""
    files = sorted([f for f in os.listdir(dir_path) if f.endswith(".parquet")])
    tables = []
    for f in files:
        tables.append(pq.read_table(os.path.join(dir_path, f)))
    return tables


def concat_tables(tables) -> dict[str, list]:
    """将多个 pyarrow Table 合并为列名 → 列表 的字典"""
    result = {col: [] for col in tables[0].column_names}
    for t in tables:
        for col in result:
            result[col].extend(t.column(col).to_pylist())
    # 数值列转为 numpy 数组加速访问
    for col in result:
        if all(isinstance(x, (int, float)) for x in result[col]):
            result[col] = np.array(result[col])
    return result


# ============================================================================
# P-Probing 专用：在传记文本中定位属性值的 token 位置
# ============================================================================

def find_attribute_positions(
    text: str,
    tokenizer,
    attribute_values: dict[str, str],
) -> dict[str, int]:
    """
    对传记文本中的每个属性，找到该属性值出现位置之前的最后一个 token 索引。

    P-Probing 在属性值前面的介词 token 处进行探测。

    返回：{属性名 → 在 input_ids 中的 token 索引}

    策略：
    1. 分词获取 token → 原文偏移映射
    2. 查找属性值在原文中的起始偏移
    3. 探测位置为该偏移之前的 token
    """
    tokenized = tokenizer(text, return_offsets_mapping=True, add_special_tokens=False)
    input_ids = tokenized["input_ids"]
    offsets = tokenized["offset_mapping"]

    positions = {}
    for attr_name, attr_value in attribute_values.items():
        # 查找属性值在原文中的首次出现位置
        text_offset = text.find(attr_value)
        if text_offset == -1:
            # 尝试大小写不敏感匹配
            idx = text.lower().find(attr_value.lower())
            if idx != -1:
                text_offset = idx
            else:
                positions[attr_name] = -1
                continue

        # 找到该偏移之前的 token 索引
        probe_pos = None
        for i, (start, end) in enumerate(offsets):
            if start >= text_offset:
                probe_pos = max(0, i - 1)
                break

        if probe_pos is None:
            probe_pos = len(input_ids) - 2

        positions[attr_name] = probe_pos

    return positions


def build_attribute_dict(record: dict[str, str]) -> dict[str, str]:
    """从 dict 记录中提取 6 个属性值"""
    return {
        "birth_date":   record["birthmonth"],              # 仅月份（首 token 预测）
        "birth_city":   record["birthcity"],
        "university":   record["university"],
        "major":        record["field"],
        "company":      record["company1name"].split(";")[0].strip(),
        "company_city": record["company1city"],
    }


# ============================================================================
# Q-Probing 专用：仅含人名的数据集
# ============================================================================

def build_q_probe_sentences(
    dict_data: dict[str, list],
    tokenizer,
    max_length: int = 128,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, list[str]]:
    """
    构建 Q-Probing 输入：仅含人名的句子。

    格式："{full_name}."（人名后加句号作为结束 token）

    返回：
        input_ids: (N, seq_len)
        end_token_pos: (N,) — 人名后 '.' 的位置
        labels_per_attr: 各属性的标签索引数组字典
    """
    names = dict_data["full_name"]
    n = len(names)

    sentences = []
    labels = {attr: [] for attr in config.ATTRIBUTE_NAMES}
    spaces = config.LABEL_SPACES

    for i in range(n):
        full_name = names[i]
        sentence = f"{full_name}."
        sentences.append(sentence)

        # 为每个属性分配标签
        for attr_name in config.ATTRIBUTE_NAMES:
            if attr_name == "birth_date":
                value = dict_data["birthmonth"][i]
            elif attr_name == "birth_city":
                value = dict_data["birthcity"][i]
            elif attr_name == "university":
                value = dict_data["university"][i]
            elif attr_name == "major":
                value = dict_data["field"][i]
            elif attr_name == "company":
                company_full = dict_data["company1name"][i]
                value = company_full.split(";")[0].strip()
            elif attr_name == "company_city":
                value = dict_data["company1city"][i]
            else:
                value = ""

            label_space = spaces.get(attr_name, [])
            if value in label_space:
                label_idx = label_space.index(value)
            else:
                label_idx = -1  # 未知标签
            labels[attr_name].append(label_idx)

    # 批量分词
    tokenized = tokenizer(
        sentences,
        padding=True,
        truncation=True,
        max_length=max_length,
        return_tensors="pt",
    )

    # 找到每句的结束 token 位置（最后一个非 padding token）
    attention_mask = tokenized["attention_mask"]
    end_token_pos = attention_mask.sum(dim=1) - 1

    # 标签列表转数组
    labels_tensors = {}
    for attr_name in config.ATTRIBUTE_NAMES:
        labels_tensors[attr_name] = np.array(labels[attr_name], dtype=np.int64)

    return (
        tokenized["input_ids"],
        tokenized["attention_mask"],
        end_token_pos,
        labels_tensors,
    )


# ============================================================================
# 数据集类
# ============================================================================

class PProbeDataset(Dataset):
    """
    P-Probing 数据集：分词后的传记 + 特殊 token 位置。

    每条样本包含：
    - input_ids：分词后的传记
    - positions：{属性名 → 探测 token 索引}
    - labels：{属性名 → 标签索引}
    """

    def __init__(
        self,
        variant_name: str,
        tokenizer,
        max_seq_length: int = 512,
        max_samples: Optional[int] = None,
    ):
        self.tokenizer = tokenizer
        self.max_seq_length = max_seq_length

        # 加载 dict 数据（包含结构化字段）
        dict_dir = os.path.join(config.DICT_DIR, variant_name)
        if not os.path.isdir(dict_dir):
            raise FileNotFoundError(f"Dict 目录未找到: {dict_dir}")

        files = sorted([f for f in os.listdir(dict_dir) if f.endswith(".parquet")])
        tables = []
        for f in files:
            tables.append(pq.read_table(os.path.join(dict_dir, f)))
        data = concat_tables(tables)

        n = len(data["full_name"])
        if max_samples is not None and max_samples < n:
            n = max_samples

        self.texts = data["text"][:n]
        self.data = {k: v[:n] if isinstance(v, list) else v[:n] for k, v in data.items()}

        self._positions: list[dict[str, int]] = []
        self._label_indices: list[dict[str, int]] = []

        # 预先计算位置和标签
        self._compute_positions_and_labels()

    def _compute_positions_and_labels(self):
        spaces = config.LABEL_SPACES
        for i in range(len(self.texts)):
            text = self.texts[i]
            attr_values = {}
            for attr in config.ATTRIBUTE_NAMES:
                if attr == "birth_date":
                    attr_values[attr] = self.data["birthmonth"][i]
                elif attr == "birth_city":
                    attr_values[attr] = self.data["birthcity"][i]
                elif attr == "university":
                    attr_values[attr] = self.data["university"][i]
                elif attr == "major":
                    attr_values[attr] = self.data["field"][i]
                elif attr == "company":
                    val = self.data["company1name"][i]
                    attr_values[attr] = val.split(";")[0].strip()
                elif attr == "company_city":
                    attr_values[attr] = self.data["company1city"][i]

            positions = find_attribute_positions(text, self.tokenizer, attr_values)

            # 将属性值映射为标签索引
            label_indices = {}
            for attr_name, value in attr_values.items():
                label_space = spaces.get(attr_name, [])
                if value in label_space:
                    label_indices[attr_name] = label_space.index(value)
                else:
                    label_indices[attr_name] = -1

            self._positions.append(positions)
            self._label_indices.append(label_indices)

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text = self.texts[idx]
        tokenized = self.tokenizer(
            text,
            max_length=self.max_seq_length,
            truncation=True,
            padding="max_length",
            return_tensors="pt",
        )
        return {
            "input_ids": tokenized["input_ids"].squeeze(0),
            "attention_mask": tokenized["attention_mask"].squeeze(0),
            "text": text,
            "positions": self._positions[idx],
            "labels": self._label_indices[idx],
        }


class QProbeDataset(Dataset):
    """
    Q-Probing 数据集：仅含人名的句子 + 属性标签。
    """

    def __init__(
        self,
        dict_data_dir: str,
        tokenizer,
        max_seq_length: int = 128,
        max_samples: Optional[int] = None,
    ):
        self.tokenizer = tokenizer
        self.max_seq_length = max_seq_length

        # 加载 dict 数据
        files = sorted([f for f in os.listdir(dict_data_dir) if f.endswith(".parquet")])
        tables = []
        for f in files:
            tables.append(pq.read_table(os.path.join(dict_data_dir, f)))
        self.data = concat_tables(tables)

        n = len(self.data["full_name"])
        if max_samples is not None and max_samples < n:
            n = max_samples

        for k in self.data:
            if isinstance(self.data[k], list):
                self.data[k] = self.data[k][:n]
            else:
                self.data[k] = self.data[k][:n]

        self.n = n
        self.spaces = config.LABEL_SPACES

        self._build_data()

    def _build_data(self):
        self.sentences = []
        self.labels = {attr: [] for attr in config.ATTRIBUTE_NAMES}
        self.end_token_positions = []

        for i in range(self.n):
            full_name = self.data["full_name"][i]
            sentence = f"{full_name}."
            self.sentences.append(sentence)

            for attr in config.ATTRIBUTE_NAMES:
                if attr == "birth_date":
                    value = self.data["birthmonth"][i]
                elif attr == "birth_city":
                    value = self.data["birthcity"][i]
                elif attr == "university":
                    value = self.data["university"][i]
                elif attr == "major":
                    value = self.data["field"][i]
                elif attr == "company":
                    val = self.data["company1name"][i]
                    value = val.split(";")[0].strip()
                elif attr == "company_city":
                    value = self.data["company1city"][i]
                else:
                    value = ""

                label_space = self.spaces.get(attr, [])
                if value in label_space:
                    self.labels[attr].append(label_space.index(value))
                else:
                    self.labels[attr].append(-1)

        # 一次性全部分词
        self.tokenized = self.tokenizer(
            self.sentences,
            padding=True,
            truncation=True,
            max_length=self.max_seq_length,
            return_tensors="pt",
        )

        # 结束 token 位置（每句最后一个有效 token）
        mask = self.tokenized["attention_mask"]
        self.end_token_positions = mask.sum(dim=1) - 1

    def __len__(self):
        return self.n

    def __getitem__(self, idx):
        return {
            "input_ids": self.tokenized["input_ids"][idx],
            "attention_mask": self.tokenized["attention_mask"][idx],
            "end_token_pos": self.end_token_positions[idx],
            "labels": {attr: self.labels[attr][idx] for attr in config.ATTRIBUTE_NAMES},
        }


# ============================================================================
# DataLoader 与 collator
# ============================================================================

def p_probe_collator(batch, positions_to_probe: Optional[list[str]] = None):
    """
    P-Probing 的 collator。

    返回堆叠的张量及位置/标签字典。
    """
    input_ids = torch.stack([item["input_ids"] for item in batch])
    attention_mask = torch.stack([item["attention_mask"] for item in batch])

    # 按属性收集标签
    all_labels = {}
    for attr in config.ATTRIBUTE_NAMES:
        labels = [item["labels"][attr] for item in batch]
        all_labels[attr] = torch.tensor(labels, dtype=torch.long)

    # 收集位置信息
    all_positions = [item["positions"] for item in batch]

    return {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "labels": all_labels,
        "positions": all_positions,
        "texts": [item["text"] for item in batch],
    }


def q_probe_collator(batch):
    """Q-Probing 的 collator"""
    input_ids = torch.stack([item["input_ids"] for item in batch])
    attention_mask = torch.stack([item["attention_mask"] for item in batch])
    end_token_pos = torch.stack([item["end_token_pos"] for item in batch])

    all_labels = {}
    for attr in config.ATTRIBUTE_NAMES:
        labels = [item["labels"][attr] for item in batch]
        all_labels[attr] = torch.tensor(labels, dtype=torch.long)

    return {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "end_token_pos": end_token_pos,
        "labels": all_labels,
    }
