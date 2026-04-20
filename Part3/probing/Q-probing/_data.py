from datasets import load_dataset
from _utils import LabelEncoder
import torch
import torch.nn as nn

# ======== Data ========

def prepare_qprobing_dataset(tokenizer, task_name: str, parquet_path: str, label_encoder: LabelEncoder, filter_unknown: bool = True, is_test: bool = False) -> torch.utils.data.Dataset:
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
    if is_test:
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
