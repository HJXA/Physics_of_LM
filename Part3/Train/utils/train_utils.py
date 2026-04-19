import re
from typing import Dict, List, Tuple
from datasets import Dataset

# 问题关键词 → 属性名映射（与 generate_bios_datasets.py 中6类 QA 模板对齐）
_QA_ATTR_MAP: List[Tuple[str, str]] = [
    (r"What is the birth date of", "birth date"),
    (r"What is the birth city of", "birth city"),
    (r"Which university did", "university"),
    (r"What major did", "major"),
    (r"Which company did", "company"),
    (r"Where did", "company city"),
]


def _extract_attr_name(question: str) -> str:
    """从问句中提取属性名，未匹配时返回 'unknown'。"""
    for pattern, attr_name in _QA_ATTR_MAP:
        if re.search(pattern, question):
            return attr_name
    return "unknown"


def part3_qa_text_to_messages(text: str, mode: str = "no-answer"):
    """
    将 QA 文本转为 messages，支持以下 mode：
    - "no-answer":   assistant content = "January 26, 1980"（纯答案，无前缀）
    - "#":          assistant content = "#January 26, 1980"（前导井号+纯答案）
    - "# #":        assistant content = "# #January 26, 1980"（双井号+纯答案）
    - "#*10":       assistant content = "# # # # # # # # # #January 26, 1980"（10个#号+纯答案）
    - "attribute":  assistant content = "birth date: January 26, 1980"（属性名替代Answer:）
    - "raw":        assistant content = "Answer: January 26, 1980."（保留原文不去除Answer:和句号）
    """
    text = str(text).strip()

    if "Answer:" in text:
        idx = text.index("Answer:")
        question = text[:idx].strip()
        answer_raw = text[idx + len("Answer:"):].strip()
        answer_value = answer_raw.rstrip(".")

        if mode == "no-answer":
            assistant_content = answer_value
        elif mode == "#":
            assistant_content = f"#{answer_value}"
        elif mode == "# #":
            assistant_content = f"# #{answer_value}"
        elif mode == "#*10":
            assistant_content = f"{'# ' * 10}{answer_value}"
        elif mode == "attribute":
            attr_name = _extract_attr_name(question)
            assistant_content = f"{attr_name}: {answer_value}"
        elif mode == "raw":
            assistant_content = text[idx:].strip()
        else:
            raise ValueError(f"Unsupported mode: {mode}, expected 'no_answer'/'#'/'# #'/'#*10'/'attribute'/'raw'")
    else:
        question = text
        assistant_content = ""

    return [
        {"role": "user", "content": question},
        {"role": "assistant", "content": assistant_content},
    ]


def part3_prepare_sft_source_dataset(raw_dataset: Dataset, mode: str = "no-answer") -> Dataset:
    """SFT 数据预处理：支持 messages 原生格式，或由 QA text 动态构造成 messages。"""
    if "messages" in raw_dataset.column_names:
        return raw_dataset

    if "text" not in raw_dataset.column_names:
        raise ValueError("SFT 模式要求数据集包含 'messages' 或 'text' 列")

    def _map_fn(example):
        return {"messages": part3_qa_text_to_messages(example["text"], mode=mode)}

    return raw_dataset.map(
        _map_fn,
        desc="Converting QA text to messages",
        remove_columns=raw_dataset.column_names,
        load_from_cache_file=False,  # 强制重新计算，避免之前的缓存不符合当前逻辑

    )


def preview_collator_batch(train_dataset: Dataset, data_collator, preview_n: int = 2):
    """训练开始前打印经过 data_collator 处理后的样本示例。"""
    if len(train_dataset) > 0:
        actual_preview_n = min(preview_n, len(train_dataset))
        preview_features = [train_dataset[i] for i in range(actual_preview_n)]
        preview_batch = data_collator(preview_features)

        print("\n[Collator Preview]")
        print(
            f"input_ids shape: {tuple(preview_batch['input_ids'].shape)}, "
            f"labels shape: {tuple(preview_batch['labels'].shape)}, "
            f"attention_mask shape: {tuple(preview_batch['attention_mask'].shape)}"
        )
        print(f"input_ids[:{preview_n}]: {preview_batch['input_ids'][:preview_n].tolist()}")
        print(f"labels[:{preview_n}]: {preview_batch['labels'][:preview_n].tolist()}")
        print(f"attention_mask[:{preview_n}]: {preview_batch['attention_mask'][:preview_n].tolist()}")
    else:
        print("[Collator Preview] 训练集为空，请检查 train.parquet")
