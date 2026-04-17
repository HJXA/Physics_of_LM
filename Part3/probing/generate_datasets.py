"""
Probing 数据集生成脚本（P-probing + Q-probing）。

从已有的 bioS_single dict parquet 中读取原始传记和属性，
严格按照 QA 数据集的 train/test 人名划分来构造 probing 数据集。

P-probing：输入为完整传记文本，输出为 6 个属性。
Q-probing：输入仅为人名，输出为 6 个属性。

输出位置：
  datasets/probing/P-probing/train/data.parquet
  datasets/probing/P-probing/test/data.parquet
  datasets/probing/Q-probing/train/data.parquet
  datasets/probing/Q-probing/test/data.parquet
"""

import json
import os
import re

import pandas as pd


# Part3 项目根目录（本脚本的上上级目录）。
PART3_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
# 数据集根目录。
DATASETS_ROOT = os.path.join(PART3_ROOT, "datasets")
# 样本检查目录。
SAMPLES_ROOT = os.path.join(DATASETS_ROOT, "samples")


def extract_fullname_from_qa(text: str) -> str:
    """从 QA 文本中提取全名。"""
    m = re.match(r"(?:What|Which|Where).*?of (.+?)\?", text)
    if m:
        return m.group(1)
    raise ValueError(f"无法从 QA 文本中提取全名: {text[:80]}")


def load_qa_split_names(qa_dir: str) -> tuple[set[str], set[str]]:
    """从 QA train/test parquet 中提取人名集合，用于严格对齐划分。"""
    train_path = os.path.join(qa_dir, "q1_birth_date.parquet")
    test_path = os.path.join(qa_dir, "q1_birth_date.parquet").replace("train", "test")

    train_df = pd.read_parquet(train_path)
    test_df = pd.read_parquet(test_path)

    train_names = set(extract_fullname_from_qa(t) for t in train_df["text"])
    test_names = set(extract_fullname_from_qa(t) for t in test_df["text"])

    return train_names, test_names


def build_p_probing_records(base_records: list[dict]) -> list[dict]:
    """构造 P-probing 记录：输入传记文本，输出 6 个属性。"""
    records = []
    for r in base_records:
        records.append(
            {
                "text": r["text"].strip(),
                "birth_date": f"{r['birthmonth']} {r['birthday']}, {r['birthyear']}",
                "birthcity": r["birthcity"],
                "university": r["university"],
                "field": r["field"],
                "company1name": r["company1name"],
                "company1city": r["company1city"],
            }
        )
    return records


def build_q_probing_records(base_records: list[dict]) -> list[dict]:
    """构造 Q-probing 记录：输入仅为人名，输出 6 个属性。"""
    records = []
    for r in base_records:
        records.append(
            {
                "text": r["full_name"],
                "birth_date": f"{r['birthmonth']} {r['birthday']}, {r['birthyear']}",
                "birthcity": r["birthcity"],
                "university": r["university"],
                "field": r["field"],
                "company1name": r["company1name"],
                "company1city": r["company1city"],
            }
        )
    return records


def split_by_names(records: list[dict], full_names: list[str], train_names: set[str], test_names: set[str]) -> tuple[list[dict], list[dict]]:
    """按人名集合严格划分 train/test。"""
    train_records = [rec for rec, name in zip(records, full_names) if name in train_names]
    test_records = [rec for rec, name in zip(records, full_names) if name in test_names]
    return train_records, test_records


def save_probing_split(train_records: list[dict], test_records: list[dict], probing_dir: str, samples_dir: str, n_samples: int):
    """保存 probing train/test parquet + json 样本。"""
    for split_name, records in [("train", train_records), ("test", test_records)]:
        parquet_path = os.path.join(probing_dir, split_name, "data.parquet")
        os.makedirs(os.path.dirname(parquet_path), exist_ok=True)
        pd.DataFrame(records).to_parquet(parquet_path, index=False)

        sample_path = os.path.join(samples_dir, split_name, "data.json")
        os.makedirs(os.path.dirname(sample_path), exist_ok=True)
        with open(sample_path, "w", encoding="utf-8") as f:
            json.dump(records[:n_samples], f, ensure_ascii=False, indent=2)

    print(f"  train: {len(train_records)} 条 -> {os.path.join(probing_dir, 'train', 'data.parquet')}")
    print(f"  test:  {len(test_records)} 条 -> {os.path.join(probing_dir, 'test', 'data.parquet')}")


def main():
    """主流程。"""
    print("=" * 72)
    print("Probing 数据集生成（P-probing + Q-probing）")
    print(f"数据集根目录: {DATASETS_ROOT}")
    print("=" * 72)

    # 1) 回读 bioS_single base dict。
    base_path = os.path.join(DATASETS_ROOT, "dict", "bioS_single", "part_1.parquet")
    base_records = pd.read_parquet(base_path).to_dict(orient="records")
    print(f"\n回读 base: {base_path} ({len(base_records)} 条)")

    # 2) 从 QA 中读取 train/test 人名集合，严格对齐划分。
    qa_dir = os.path.join(DATASETS_ROOT, "QA", "train")
    train_names, test_names = load_qa_split_names(qa_dir)
    print(f"QA train 人名数: {len(train_names)} | QA test 人名数: {len(test_names)}")

    full_names = [r["full_name"] for r in base_records]
    n_samples = 10

    # 3) P-probing：输入传记文本。
    print("\n--- P-probing ---")
    p_records = build_p_probing_records(base_records)
    p_train, p_test = split_by_names(p_records, full_names, train_names, test_names)
    save_probing_split(
        p_train, p_test,
        os.path.join(DATASETS_ROOT, "probing", "P-probing"),
        os.path.join(SAMPLES_ROOT, "probing", "P-probing"),
        n_samples,
    )

    # 4) Q-probing：输入仅为人名。
    print("\n--- Q-probing ---")
    q_records = build_q_probing_records(base_records)
    q_train, q_test = split_by_names(q_records, full_names, train_names, test_names)
    save_probing_split(
        q_train, q_test,
        os.path.join(DATASETS_ROOT, "probing", "Q-probing"),
        os.path.join(SAMPLES_ROOT, "probing", "Q-probing"),
        n_samples,
    )

    print("\n完成")


if __name__ == "__main__":
    main()
