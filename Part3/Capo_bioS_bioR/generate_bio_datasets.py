"""
生成 Capo 三种数据集（bioS / bioS-aug / bioR-prompt）并保存为 parquet + json 样本。

三种数据集说明：
1. bioS: 纯合成传记，由 get_text_simple3 直接生成结构化文本。
2. bioS-aug: 在 bioS 基础上做句子级重排增强（augmentation_permutation2）。
3. bioR-prompt: 生成用于大模型写传记的提示词（generate_prompt2），不含生成结果。

输出：
- {output_dir}/bioS.parquet / bioS_samples.json
- {output_dir}/bioS_aug.parquet / bioS_aug_samples.json
- {output_dir}/bioR_prompt.parquet / bioR_prompt_samples.json
"""

import os
import sys
import json
import random
import argparse
import numpy as np
import pandas as pd

# 导入同目录下的原始函数
from Capo_bioS_bioR import get_text_simple3, generate_prompt2, augmentation_permutation2

# ──────────────────────────────────────────────
# 字段文件路径
# ──────────────────────────────────────────────
FIELDS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "fields")


def load_lines(filename: str) -> list[str]:
    """从 fields/ 目录加载文本列表，每行一个条目。"""
    path = os.path.join(FIELDS_DIR, filename)
    with open(path, "r", encoding="utf-8") as f:
        return [line.strip() for line in f if line.strip()]


def load_companies(filename: str = "company.txt") -> list[dict]:
    """加载公司列表，每行格式为 '公司名; 城市, 州'。"""
    path = os.path.join(FIELDS_DIR, filename)
    companies = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            parts = line.split("; ", 1)
            if len(parts) == 2:
                companies.append({"name": parts[0], "city": parts[1]})
    return companies


def generate_person(person_id: int) -> dict:
    """随机生成一个人物字典，包含所有传记字段。"""
    first_names = load_lines("first_name.txt")
    middle_names = load_lines("middle_name.txt")
    last_names = load_lines("last_name.txt")
    universities = load_lines("university.txt")
    fields = load_lines("field.txt")
    cities = load_lines("city.txt")
    companies = load_companies("company.txt")

    company = random.choice(companies)

    # 生日字段
    months = [
        "January", "February", "March", "April", "May", "June",
        "July", "August", "September", "October", "November", "December",
    ]
    birthmonth = random.choice(months)
    birthday = str(random.randint(1, 28))
    birthyear = str(random.randint(1960, 2000))

    return {
        "id": person_id,
        "first_name": random.choice(first_names),
        "middle_name": random.choice(middle_names),
        "last_name": random.choice(last_names),
        "university": random.choice(universities),
        "field": random.choice(fields),
        "birthcity": random.choice(cities),
        "company1name": company["name"],
        "company1city": company["city"],
        "birthmonth": birthmonth,
        "birthday": birthday,
        "birthyear": birthyear,
    }


# ──────────────────────────────────────────────
# 简易 mock class，让 generate_prompt2 的 self.xxx 正常工作
# ──────────────────────────────────────────────
class _PersonProxy:
    """把 person dict 包装成对象，使 generate_prompt2(self, ...) 可用。"""
    def __init__(self, person: dict):
        for k, v in person.items():
            setattr(self, k, v)


# ──────────────────────────────────────────────
# 数据集生成函数
# ──────────────────────────────────────────────

def generate_bioS_dataset(n: int, seed: int = 42) -> pd.DataFrame:
    """生成 bioS 数据集：纯合成传记文本。"""
    rng = random.Random(seed)
    random.seed(seed)
    np.random.seed(seed)

    records = []
    for i in range(n):
        person = generate_person(i)
        order = rng.randint(0, 1)
        reverse_md = rng.choice([False, True])
        text = get_text_simple3(person, order=order, reverse_md=reverse_md)
        records.append({
            "id": person["id"],
            "first_name": person["first_name"],
            "middle_name": person["middle_name"],
            "last_name": person["last_name"],
            "full_name": f"{person['first_name']} {person['middle_name']} {person['last_name']}",
            "university": person["university"],
            "field": person["field"],
            "birthcity": person["birthcity"],
            "company1name": person["company1name"],
            "company1city": person["company1city"],
            "birthday": f"{person['birthmonth']} {person['birthday']}, {person['birthyear']}",
            "text": text.strip(),
        })
    return pd.DataFrame(records)


def generate_bioS_aug_dataset(n: int, seed: int = 42) -> pd.DataFrame:
    """生成 bioS-aug 数据集：合成传记 + 句子重排增强。"""
    rng = random.Random(seed)
    random.seed(seed)
    np.random.seed(seed)

    records = []
    for i in range(n):
        person = generate_person(i)
        order = rng.randint(0, 1)
        reverse_md = rng.choice([False, True])
        base_text = get_text_simple3(person, order=order, reverse_md=reverse_md)
        # 对合成文本做重排增强
        aug_text = augmentation_permutation2(person, base_text)
        records.append({
            "id": person["id"],
            "first_name": person["first_name"],
            "middle_name": person["middle_name"],
            "last_name": person["last_name"],
            "full_name": f"{person['first_name']} {person['middle_name']} {person['last_name']}",
            "university": person["university"],
            "field": person["field"],
            "birthcity": person["birthcity"],
            "company1name": person["company1name"],
            "company1city": person["company1city"],
            "birthday": f"{person['birthmonth']} {person['birthday']}, {person['birthyear']}",
            "text_original": base_text.strip(),
            "text_augmented": aug_text.strip(),
        })
    return pd.DataFrame(records)


def generate_bioR_prompt_dataset(n: int, seed: int = 42) -> pd.DataFrame:
    """生成 bioR-prompt 数据集：用于大模型生成传记的提示词。"""
    rng = random.Random(seed)
    random.seed(seed)
    np.random.seed(seed)

    records = []
    for i in range(n):
        person = generate_person(i)
        proxy = _PersonProxy(person)
        mode = rng.randint(0, 1)
        word_count = rng.randint(50, 200)
        prompt = generate_prompt2(proxy, word=word_count, mode=mode)
        records.append({
            "id": person["id"],
            "first_name": person["first_name"],
            "middle_name": person["middle_name"],
            "last_name": person["last_name"],
            "full_name": f"{person['first_name']} {person['middle_name']} {person['last_name']}",
            "university": person["university"],
            "field": person["field"],
            "birthcity": person["birthcity"],
            "company1name": person["company1name"],
            "company1city": person["company1city"],
            "birthday": f"{person['birthmonth']} {person['birthday']}, {person['birthyear']}",
            "prompt": prompt.strip(),
            "mode": mode,
            "word_count": word_count,
        })
    return pd.DataFrame(records)


# ──────────────────────────────────────────────
# 保存函数
# ──────────────────────────────────────────────

def save_dataset(df: pd.DataFrame, parquet_path: str, json_path: str, n_samples: int = 10):
    """保存 parquet 全量数据 + json 样本展示。"""
    os.makedirs(os.path.dirname(parquet_path) or ".", exist_ok=True)

    # 保存 parquet
    df.to_parquet(parquet_path, index=False)
    print(f"  [✓] 已保存 parquet: {parquet_path}  ({len(df)} 条)")

    # 提取样本并保存 json
    samples = df.sample(n=min(n_samples, len(df)), random_state=42).to_dict(orient="records")
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(samples, f, ensure_ascii=False, indent=2)
    print(f"  [✓] 已保存样本:   {json_path}  ({len(samples)} 条)")


# ──────────────────────────────────────────────
# 主入口
# ──────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="生成 Capo bioS / bioS-aug / bioR-prompt 数据集")
    parser.add_argument("-n", "--num_samples", type=int, default=1000, help="每种数据集的样本数量（默认 1000）")
    parser.add_argument("-o", "--output_dir", type=str, default=None,
                        help="输出目录（默认为当前脚本所在目录下的 output/）")
    parser.add_argument("--seed", type=int, default=42, help="随机种子（默认 42）")
    parser.add_argument("--show_samples", type=int, default=10, help="json 样本展示数量（默认 10）")
    args = parser.parse_args()

    output_dir = args.output_dir or os.path.join(os.path.dirname(os.path.abspath(__file__)), "output")
    os.makedirs(output_dir, exist_ok=True)

    print(f"{'='*60}")
    print(f"  Capo 数据集生成器")
    print(f"  样本数: {args.num_samples}  |  种子: {args.seed}  |  输出: {output_dir}")
    print(f"{'='*60}\n")

    # 1. bioS
    print("[1/3] 生成 bioS 数据集 ...")
    df_bioS = generate_bioS_dataset(args.num_samples, seed=args.seed)
    save_dataset(df_bioS,
                 os.path.join(output_dir, "bioS.parquet"),
                 os.path.join(output_dir, "bioS_samples.json"),
                 n_samples=args.show_samples)
    print()

    # 2. bioS-aug
    print("[2/3] 生成 bioS-aug 数据集 ...")
    df_bioS_aug = generate_bioS_aug_dataset(args.num_samples, seed=args.seed + 1)
    save_dataset(df_bioS_aug,
                 os.path.join(output_dir, "bioS_aug.parquet"),
                 os.path.join(output_dir, "bioS_aug_samples.json"),
                 n_samples=args.show_samples)
    print()

    # 3. bioR-prompt
    print("[3/3] 生成 bioR-prompt 数据集 ...")
    df_bioR = generate_bioR_prompt_dataset(args.num_samples, seed=args.seed + 2)
    save_dataset(df_bioR,
                 os.path.join(output_dir, "bioR_prompt.parquet"),
                 os.path.join(output_dir, "bioR_prompt_samples.json"),
                 n_samples=args.show_samples)
    print()

    print(f"{'='*60}")
    print(f"  全部完成！输出目录: {output_dir}")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
