"""
Capo 数据集生成脚本（仅规则流程）。

设计原则：
1) 只依赖已有两个核心函数：`get_text_simple3` 与 `augmentation_permutation2`；
2) 先生成 200K 唯一全名的 base（bioS_single），并保存 dict/text 双格式；
3) 所有增强都从已保存的 base dict 回读后进行（不使用模型重写）；
4) 每个 parquet 同步导出前 N 条检验样本：text->txt, dict->json；
5) 样本目录与原 parquet 相对路径镜像一致。
"""

# 参数解析库，用于定义命令行参数。
import argparse
# JSON 序列化库，用于导出字典样本检查文件。
import json
# 文件系统与路径处理库。
import os
# Python 标准随机库（本脚本用于人物采样、模板采样、置换可复现）。
import random
# 正则库，用于代词替换。
import re
# 类型标注：Callable 用于 QA 模板构造器签名声明。
from typing import Callable

# 数值随机库（这里主要用于和原始流程保持一致地固定随机态）。
import numpy as np
# 表格库，用于读写 parquet。
import pandas as pd

# 复用既有规则函数：文本合成函数 + 句子置换函数。
from Capo_bioS_bioR import augmentation_permutation2, get_text_simple3


# fields 目录的绝对路径（与当前脚本同级）。
FIELDS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "fields")
# 月份候选池（生日月份从这里均匀采样）。
MONTHS = [
    "January", "February", "March", "April", "May", "June",
    "July", "August", "September", "October", "November", "December",
]
# 代词候选池（每个样本随机分配 He/She）。
PRONOUNS = ("He", "She")
# 出生年份下界。
YEAR_START = 1900
# 出生年份上界。
YEAR_END = 2099


def load_lines(filename: str) -> list[str]:
    """读取 fields 文件，每行一个非空项。"""
    # 拼接词表文件绝对路径。
    path = os.path.join(FIELDS_DIR, filename)
    # 以 UTF-8 打开文本文件并读取所有非空行。
    with open(path, "r", encoding="utf-8") as f:
        # strip 去掉首尾空白；if line.strip() 过滤空行。
        return [line.strip() for line in f if line.strip()]


def load_companies(filename: str = "company.txt") -> list[dict]:
    """读取公司词表，每行格式：公司名; 城市。"""
    # 拼接公司词表绝对路径。
    path = os.path.join(FIELDS_DIR, filename)
    # 结果列表中的每个元素形如 {"name": 公司名, "city": 城市}。
    companies: list[dict] = []
    # 逐行读取公司词表。
    with open(path, "r", encoding="utf-8") as f:
        # 循环处理每一行。
        for line in f:
            # 去除行首尾空白。
            line = line.strip()
            # 跳过空行。
            if not line:
                continue
            # 按 "; " 最多切一次，得到公司名与城市。
            parts = line.split("; ", 1)
            # 只有切分成功时才写入，避免脏数据导致索引错误。
            if len(parts) == 2:
                companies.append({"name": parts[0], "city": parts[1]})
    # 返回公司词表。
    return companies


def apply_pronoun(text: str, pronoun: str) -> str:
    """把 get_text_simple3 输出中的主语代词统一为指定代词。"""
    # 先校验代词是否合法，避免误传入其它字符串。
    if pronoun not in PRONOUNS:
        raise ValueError(f"不支持的代词: {pronoun}")

    # 把句中所有首字母大写代词统一替换为目标代词。
    text = re.sub(r"\b(He|She|They)\b", pronoun, text)
    # 把句中所有小写代词也替换为目标代词的小写形式。
    text = re.sub(r"\b(he|she|they)\b", pronoun.lower(), text)

    # 当目标代词是 They 时，做最小语法纠正，减少明显不通顺的组合。
    if pronoun == "They":
        # 这些规则只处理高频的 is/was/has 搭配问题。
        fixes = [
            ("They is", "They are"),
            ("they is", "they are"),
            ("They was", "They were"),
            ("they was", "they were"),
            ("They has", "They have"),
            ("they has", "they have"),
        ]
        # 逐条替换修正规则。
        for old, new in fixes:
            text = text.replace(old, new)
    # 返回统一代词后的文本。
    return text


def replace_pronouns_with_fullname(text: str, full_name: str) -> str:
    """fullname 增强：将代词替换为全名。"""
    # 先替换首字母大写形式的代词。
    text = re.sub(r"\b(He|She|They)\b", full_name, text)
    # 再替换小写形式的代词。
    text = re.sub(r"\b(he|she|they)\b", full_name, text)
    # 返回全名增强后的文本。
    return text


def build_unique_people(num_samples: int, seed: int) -> list[dict]:
    """生成唯一全名人物档案。"""
    # 使用局部随机源，避免污染全局随机状态。
    rng = random.Random(seed)

    # 读取名字与属性词表。
    first_names = load_lines("first_name.txt")
    # 读取 middle_name 词表。
    middle_names = load_lines("middle_name.txt")
    # 读取 last_name 词表。
    last_names = load_lines("last_name.txt")
    # 读取大学词表。
    universities = load_lines("university.txt")
    # 读取专业词表。
    majors = load_lines("field.txt")
    # 读取城市词表。
    cities = load_lines("city.txt")
    # 读取公司+总部城市词表。
    companies = load_companies("company.txt")

    # 理论可生成的唯一全名上限。
    max_unique = len(first_names) * len(middle_names) * len(last_names)
    # 若请求样本数超过上限，直接报错。
    if num_samples > max_unique:
        raise ValueError(f"样本数 {num_samples} 超过唯一全名上限 {max_unique}")

    # 记录已使用全名，保证唯一。
    used_names: set[str] = set()
    # 存放最终人物结构化档案。
    people: list[dict] = []

    # 逐个生成人物。
    for person_id in range(num_samples):
        # 重采样名字直到拿到一个未出现过的全名。
        while True:
            # 随机选 first name。
            first_name = rng.choice(first_names)
            # 随机选 middle name。
            middle_name = rng.choice(middle_names)
            # 随机选 last name。
            last_name = rng.choice(last_names)
            # 拼接全名。
            full_name = f"{first_name} {middle_name} {last_name}"
            # 若全名未出现，则接受本次采样。
            if full_name not in used_names:
                # 记入已用集合，确保后续不重复。
                used_names.add(full_name)
                # 退出 while，进入属性采样。
                break

        # 采样公司（公司名与公司城市是绑定关系）。
        company = rng.choice(companies)
        # 把人物完整属性写入列表。
        people.append(
            {
                # 人物唯一整数 id（后续作为可复现种子偏移量的一部分）。
                "id": person_id,
                # 名字字段：first_name。
                "first_name": first_name,
                # 名字字段：middle_name。
                "middle_name": middle_name,
                # 名字字段：last_name。
                "last_name": last_name,
                # 名字字段：full_name（保证全局唯一）。
                "full_name": full_name,
                # 出生月份，12 个月均匀采样。
                "birthmonth": rng.choice(MONTHS),
                # 出生日，限定 1~28 以避免闰月与月份长度边界问题。
                "birthday": str(rng.randint(1, 28)),
                # 出生年，范围 1900~2099。
                "birthyear": str(rng.randint(YEAR_START, YEAR_END)),
                # 出生城市。
                "birthcity": rng.choice(cities),
                # 毕业院校。
                "university": rng.choice(universities),
                # 专业。
                "field": rng.choice(majors),
                # 公司名（来自绑定的 company 记录）。
                "company1name": company["name"],
                # 公司城市（公司总部城市，和公司名绑定）。
                "company1city": company["city"],
                # 随机代词（He/She/They）。
                "pronoun": rng.choice(PRONOUNS),
            }
        )

    # 返回结构化人物档案。
    return people


def render_single_bio(person: dict, fixed_templates: bool = False) -> str:
    """用 get_text_simple3 生成六句传记。"""
    # 复制一份 person，避免就地修改调用方对象。
    p = dict(person)
    # get_text_simple3 内部通过 id 奇偶决定 He/She，因此这里显式映射。
    p["id"] = 1 if person["pronoun"] == "She" else 0
    # 生成六句传记：order=1 表示先公司名句，再公司城市句。
    text = get_text_simple3(p, order=1, reverse_md=False, fixed_templates=fixed_templates).strip()
    # 再把代词统一成 person 指定代词（含 They）。
    return apply_pronoun(text, person["pronoun"])


def build_base_records(num_samples: int, seed: int) -> list[dict]:
    """生成 bioS_single 基础记录。"""
    # 先生成唯一全名的人物档案。
    people = build_unique_people(num_samples=num_samples, seed=seed)

    # get_text_simple3 会使用 random.choice；固定随机态确保可复现。
    random.seed(seed + 1000)
    # 与主随机态一起固定 numpy 随机态，保持整体实验一致性。
    np.random.seed(seed + 1000)

    # 结果记录：每条均为 dict，后续直接可写入 dict parquet。
    records: list[dict] = []
    # 逐人生成 bioS_single。
    for person in people:
        # 拷贝人物基础字段。
        row = dict(person)
        # 标记数据集名称。
        row["dataset"] = "bioS_single"
        # base 的 variant 固定为 1。
        row["variant_index"] = 1
        # 生成文本字段。
        # base 阶段固定模板，保证每条样本使用同一组句式结构。
        row["text"] = render_single_bio(person, fixed_templates=False)
        # 追加到结果列表。
        records.append(row)
    # 返回基础记录。
    return records


def permute_text(person: dict, source_text: str, seed: int) -> str:
    """用 augmentation_permutation2 做规则置换，并保持可复现。"""
    # 备份当前全局随机状态，避免影响外层流程。
    old_state = random.getstate()
    # 设定本次置换专用种子。
    random.seed(seed)
    try:
        # 执行规则置换（不走模型重写）。
        out = augmentation_permutation2(person, source_text)
    finally:
        # 恢复外层随机状态。
        random.setstate(old_state)

    # 若置换函数返回空，视为异常。
    if not out:
        raise RuntimeError("augmentation_permutation2 返回空结果")
    # 清理前后空白后返回。
    return out.strip()


def make_record(base_record: dict, text: str, dataset: str, variant_index: int) -> dict:
    """基于 base 记录构建增强记录。"""
    # 复制 base 记录，保留人物全部属性字段。
    row = dict(base_record)
    # 覆盖为当前增强的数据集名。
    row["dataset"] = dataset
    # 标记增强编号（part 索引）。
    row["variant_index"] = variant_index
    # 覆盖文本字段为增强后的文本。
    row["text"] = text.strip()
    # 返回增强记录。
    return row


def _sample_file_from_parquet(parquet_path: str, datasets_root: str, samples_root: str, ext: str) -> str:
    """镜像 parquet 相对路径到 samples 目录。"""
    # 先计算 parquet 相对 datasets_root 的路径。
    rel = os.path.relpath(parquet_path, datasets_root)
    # 去掉 .parquet 后缀，得到路径主干。
    stem = os.path.splitext(rel)[0]
    # 拼接到 samples_root 下并替换扩展名（.txt 或 .json）。
    return os.path.join(samples_root, stem + ext)


def save_text_parquet(texts: list[str], parquet_path: str, datasets_root: str, samples_root: str, n_samples: int):
    """保存 text parquet + txt 样本。"""
    # 确保 parquet 目标目录存在。
    os.makedirs(os.path.dirname(parquet_path), exist_ok=True)
    # 只保留 text 列写入 parquet（字符串格式训练输入）。
    pd.DataFrame({"text": texts}).to_parquet(parquet_path, index=False)

    # 计算镜像样本文件路径（同相对路径，后缀改为 .txt）。
    sample_path = _sample_file_from_parquet(parquet_path, datasets_root, samples_root, ".txt")
    # 确保样本目录存在。
    os.makedirs(os.path.dirname(sample_path), exist_ok=True)
    # 导出前 n_samples 条样本到 txt。
    with open(sample_path, "w", encoding="utf-8") as f:
        # 用 min 防止 n_samples 大于文本总数。
        for x in texts[: min(n_samples, len(texts))]:
            # 每条文本占一行，便于人工快速抽查。
            f.write(x.strip() + "\n")


def save_dict_parquet(records: list[dict], parquet_path: str, datasets_root: str, samples_root: str, n_samples: int):
    """保存 dict parquet + json 样本。"""
    # 确保 parquet 目标目录存在。
    os.makedirs(os.path.dirname(parquet_path), exist_ok=True)
    # 直接按所有字段写入 dict parquet（后续增强可回读逐条改写）。
    pd.DataFrame(records).to_parquet(parquet_path, index=False)

    # 计算镜像样本文件路径（同相对路径，后缀改为 .json）。
    sample_path = _sample_file_from_parquet(parquet_path, datasets_root, samples_root, ".json")
    # 确保样本目录存在。
    os.makedirs(os.path.dirname(sample_path), exist_ok=True)
    # 导出前 n_samples 条字典样本到 JSON，便于结构化核验。
    with open(sample_path, "w", encoding="utf-8") as f:
        # ensure_ascii=False 便于直读中文；indent=2 便于 diff/审阅。
        json.dump(records[: min(n_samples, len(records))], f, ensure_ascii=False, indent=2)


def save_dual_dataset(
    records: list[dict],
    datasets_root: str,
    samples_root: str,
    dataset_name: str,
    part_idx: int,
    n_samples: int,
):
    """同一数据同时落地 dict/text 两种格式。"""
    # 当前 part 文件名（例如 part_1.parquet）。
    file_name = f"part_{part_idx}.parquet"
    # dict 版本输出路径：datasets/dict/<dataset_name>/part_i.parquet。
    dict_path = os.path.join(datasets_root, "dict", dataset_name, file_name)
    # text 版本输出路径：datasets/text/<dataset_name>/part_i.parquet。
    text_path = os.path.join(datasets_root, "text", dataset_name, file_name)

    # 写入 dict parquet + json 样本。
    save_dict_parquet(records, dict_path, datasets_root, samples_root, n_samples)
    # 写入 text parquet + txt 样本。
    save_text_parquet([r["text"] for r in records], text_path, datasets_root, samples_root, n_samples)


def load_dict_records(parquet_path: str) -> list[dict]:
    """从 dict parquet 回读记录。"""
    # 读取 parquet 并转回 list[dict]，用于后续规则增强。
    return pd.read_parquet(parquet_path).to_dict(orient="records")


def generate_single_augments(
    base_records: list[dict],
    datasets_root: str,
    samples_root: str,
    seed: int,
    max_permute: int,
    n_samples: int,
):
    """single 系列增强。"""
    # 1) bioS_single_fullname：从 base 一对一替换代词为全名。
    fullname_records = [
        make_record(r, replace_pronouns_with_fullname(r["text"], r["full_name"]), "bioS_single_fullname", 1)
        for r in base_records
    ]
    # 保存 fullname 单文件（part_1）。
    save_dual_dataset(fullname_records, datasets_root, samples_root, "bioS_single_fullname", 1, n_samples)

    # 2) bioS_single_permute + bioS_single_permute_fullname：按 idx 逐个 part 生成。
    for idx in range(1, max_permute + 1):
        # 当前 idx 对应的 permute 记录容器。
        perm_records: list[dict] = []
        # 对每条 base 记录做一次可复现置换。
        for r in base_records:
            # 构造逐样本且逐 part 不同的置换种子。
            pseed = seed + idx * 1_000_003 + int(r["id"])
            # 做规则置换（全名只在首句，后续保持代词行为由原函数保证）。
            ptext = permute_text(r, r["text"], pseed)
            # 写入当前 part 的 permute 记录。
            perm_records.append(make_record(r, ptext, "bioS_single_permute", idx))
        # 保存当前 part 的 single_permute。
        save_dual_dataset(perm_records, datasets_root, samples_root, "bioS_single_permute", idx, n_samples)

        # 在 single_permute 的基础上再做 fullname 版本（一对一）。
        perm_fullname_records = [
            make_record(r, replace_pronouns_with_fullname(r["text"], r["full_name"]), "bioS_single_permute_fullname", idx)
            for r in perm_records
        ]
        # 保存当前 part 的 single_permute_fullname。
        save_dual_dataset(
            perm_fullname_records,
            datasets_root,
            samples_root,
            "bioS_single_permute_fullname",
            idx,
            n_samples,
        )


def generate_multi_augments(
    base_records: list[dict],
    datasets_root: str,
    samples_root: str,
    seed: int,
    max_multi: int,
    n_samples: int,
):
    """multi 系列增强。"""
    # 逐个 multi part 生成（例如 part_1 到 part_5）。
    for idx in range(1, max_multi + 1):
        # 固定当前 part 的随机态，保证可复现。
        random.seed(seed + idx * 20_011)
        # 同步固定 numpy 随机态。
        np.random.seed(seed + idx * 20_011)

        # 1) bioS_multi：对同一人重新采样模板，生成新的六句传记。
        multi_records: list[dict] = []
        # 遍历 base 人物档案。
        for r in base_records:
            # 重新合成一条传记（属性不变、模板重新采样）。
            mtext = render_single_bio(r)
            # 组装 multi 记录。
            multi_records.append(make_record(r, mtext, "bioS_multi", idx))
        # 保存当前 part 的 bioS_multi。
        save_dual_dataset(multi_records, datasets_root, samples_root, "bioS_multi", idx, n_samples)

        # 2) bioS_multi_fullname：在 multi 文本上把代词替换为全名。
        multi_fullname_records = [
            make_record(r, replace_pronouns_with_fullname(r["text"], r["full_name"]), "bioS_multi_fullname", idx)
            for r in multi_records
        ]
        # 保存当前 part 的 bioS_multi_fullname。
        save_dual_dataset(multi_fullname_records, datasets_root, samples_root, "bioS_multi_fullname", idx, n_samples)

        # 3) bioS_multi_permute：在 multi 文本上做规则置换。
        multi_permute_records: list[dict] = []
        # 遍历当前 part 的 multi 记录。
        for r in multi_records:
            # 构造逐样本且逐 part 唯一的置换种子。
            pseed = seed + idx * 3_000_007 + int(r["id"])
            # 执行规则置换。
            ptext = permute_text(r, r["text"], pseed)
            # 组装 multi_permute 记录。
            multi_permute_records.append(make_record(r, ptext, "bioS_multi_permute", idx))
        # 保存当前 part 的 bioS_multi_permute。
        save_dual_dataset(multi_permute_records, datasets_root, samples_root, "bioS_multi_permute", idx, n_samples)

        # 4) bioS_multi_permute_fullname：在 multi_permute 文本上代词替换为全名。
        multi_permute_fullname_records = [
            make_record(
                r,
                replace_pronouns_with_fullname(r["text"], r["full_name"]),
                "bioS_multi_permute_fullname",
                idx,
            )
            for r in multi_permute_records
        ]
        # 保存当前 part 的 bioS_multi_permute_fullname。
        save_dual_dataset(
            multi_permute_fullname_records,
            datasets_root,
            samples_root,
            "bioS_multi_permute_fullname",
            idx,
            n_samples,
        )


def generate_qa(base_records: list[dict], datasets_root: str, samples_root: str, n_samples: int, qa_layout: str):
    """生成六类 QA 数据（字符串格式）。

    qa_layout:
    - raw: 仅生成 QA/<q_name>/data.parquet（旧布局）
    - final: 仅生成 QA/train/*.parquet + QA/test/*.parquet（最终布局）
    - both: 同时生成 raw 与 final
    """
    # 六类 QA 构造器：每个问题单独一个目录与 parquet。
    builders: list[tuple[str, Callable[[dict], str]]] = [
        (
            "q1_birth_date",
            lambda r: (
                f"What is the birth date of {r['full_name']}? "
                f"Answer: {r['birthmonth']} {r['birthday']}, {r['birthyear']}."
            ),
        ),
        ("q2_birth_city", lambda r: f"What is the birth city of {r['full_name']}? Answer: {r['birthcity']}."),
        ("q3_university", lambda r: f"Which university did {r['full_name']} study? Answer: {r['university']}."),
        ("q4_major", lambda r: f"What major did {r['full_name']} study? Answer: {r['field']}."),
        ("q5_company", lambda r: f"Which company did {r['full_name']} work for? Answer: {r['company1name']}."),
        ("q6_company_city", lambda r: f"Where did {r['full_name']} work? Answer: {r['company1city']}."),
    ]

    # 逐类问题生成 200K（或 num_samples）条 QA 文本。
    for qa_name, fn in builders:
        # 按当前问题模板映射所有人物。
        texts = [fn(r) for r in base_records]

        # 旧布局：datasets/QA/<qa_name>/data.parquet。
        if qa_layout in {"raw", "both"}:
            raw_parquet_path = os.path.join(datasets_root, "QA", qa_name, "data.parquet")
            save_text_parquet(texts, raw_parquet_path, datasets_root, samples_root, n_samples)

        # 最终布局：datasets/QA/train/<q_name>.parquet 与 datasets/QA/test/<q_name>.parquet。
        if qa_layout in {"final", "both"}:
            split_idx = len(texts) // 2
            train_texts = texts[:split_idx]
            test_texts = texts[split_idx:]

            train_parquet_path = os.path.join(datasets_root, "QA", "train", f"{qa_name}.parquet")
            test_parquet_path = os.path.join(datasets_root, "QA", "test", f"{qa_name}.parquet")

            save_text_parquet(train_texts, train_parquet_path, datasets_root, samples_root, n_samples)
            save_text_parquet(test_texts, test_parquet_path, datasets_root, samples_root, n_samples)


def build_parser() -> argparse.ArgumentParser:
    """参数解析。"""
    # 初始化命令行参数解析器。
    parser = argparse.ArgumentParser(description="生成 Capo 规则数据集（仅 paper 流程）")
    # 样本总数（默认 200000）。
    parser.add_argument("-n", "--num_samples", type=int, default=200000, help="基础样本数（默认 200000）")
    # 输出目录（未指定时默认脚本同级 datasets/）。
    parser.add_argument(
        "-o",
        "--output_dir",
        type=str,
        default=None,
        help="输出目录（默认脚本同级 datasets/）",
    )
    # 全局随机种子。
    parser.add_argument("--seed", type=int, default=42, help="随机种子")
    # 每个 parquet 导出的检查样本条数。
    parser.add_argument("--show_samples", type=int, default=10, help="每个 parquet 导出的检验样本条数")
    # single_permute 的 part 文件数量（默认 5）。
    parser.add_argument("--max_permute", type=int, default=5, help="single permute 文件数（默认 5）")
    # multi 系列的 part 文件数量（默认 5）。
    parser.add_argument("--max_multi", type=int, default=5, help="multi 文件数（默认 5）")
    # QA 输出布局：默认直接产出最终 train/test 结构。
    parser.add_argument(
        "--qa_layout",
        type=str,
        choices=["raw", "final", "both"],
        default="final",
        help="QA 输出布局：raw=旧布局, final=最终布局, both=两者都生成（默认 final）",
    )
    # 返回解析器对象。
    return parser


def main():
    """主流程。"""
    # 解析命令行参数。
    args = build_parser().parse_args()

    # 校验样本数必须为正。
    if args.num_samples <= 0:
        raise ValueError("--num_samples 必须为正整数")
    # 校验增强 part 数量必须为正。
    if args.max_permute <= 0 or args.max_multi <= 0:
        raise ValueError("--max_permute / --max_multi 必须为正整数")

    # 确定数据输出根目录。
    datasets_root = args.output_dir or os.path.join(os.path.dirname(os.path.abspath(__file__)), "datasets")
    # 确定样本检查目录（镜像存放 txt/json）。
    samples_root = os.path.join(datasets_root, "samples")
    # 创建数据根目录。
    os.makedirs(datasets_root, exist_ok=True)
    # 创建样本目录。
    os.makedirs(samples_root, exist_ok=True)

    # 打印运行头信息。
    print("=" * 72)
    # 打印任务名称。
    print("Capo 规则数据集生成")
    # 打印样本数与种子。
    print(f"样本数: {args.num_samples} | seed: {args.seed}")
    # 打印输出目录。
    print(f"输出目录: {datasets_root}")
    # 打印分隔线。
    print("=" * 72)

    # 1) 先生成 base：bioS_single。
    print("\n[1/4] 生成 bioS_single(base)")
    # 合成基础记录。
    base_records = build_base_records(args.num_samples, args.seed)
    # 同步保存 dict/text 两个版本。
    save_dual_dataset(base_records, datasets_root, samples_root, "bioS_single", 1, args.show_samples)

    # 2) 从刚保存的 dict 回读，保证后续增强严格基于落盘数据。
    print("\n[2/4] 回读 base dict")
    # base dict parquet 路径。
    base_path = os.path.join(datasets_root, "dict", "bioS_single", "part_1.parquet")
    # 回读为 list[dict]。
    base_records = load_dict_records(base_path)
    # 打印回读统计。
    print(f"已回读: {base_path} ({len(base_records)} 条)")

    # 3) 生成 single 与 multi 全部规则增强。
    print("\n[3/4] 生成 single/multi 增强")
    # 生成 single 系列增强。
    generate_single_augments(
        base_records=base_records,
        datasets_root=datasets_root,
        samples_root=samples_root,
        seed=args.seed + 10_000,
        max_permute=args.max_permute,
        n_samples=args.show_samples,
    )
    # 生成 multi 系列增强。
    generate_multi_augments(
        base_records=base_records,
        datasets_root=datasets_root,
        samples_root=samples_root,
        seed=args.seed + 20_000,
        max_multi=args.max_multi,
        n_samples=args.show_samples,
    )

    # 4) 从原始 base 信息出发生成六类 QA。
    print("\n[4/4] 生成 QA")
    # 保存 QA 数据与对应样本。
    generate_qa(base_records, datasets_root, samples_root, args.show_samples, qa_layout=args.qa_layout)

    # 打印完成信息。
    print("\n" + "=" * 72)
    # 打印完成标志。
    print("完成")
    # 打印目录说明。
    print("dict: datasets/dict/... | text: datasets/text/... | QA: datasets/QA/... | samples: datasets/samples/...")
    # 打印收尾分隔线。
    print("=" * 72)


# 仅当脚本被直接运行时执行主流程。
if __name__ == "__main__":
    # 启动生成任务。
    main()
