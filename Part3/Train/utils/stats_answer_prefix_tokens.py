"""
统计 part3_qa_text_to_messages 各 mode 下，? 后第 0~4 个位置的 token_id、text 及其频次。

用法：
    cd Part3/Train/utils
    python stats_answer_prefix_tokens.py [--tokenizer_path PATH] [--n_samples N] [--qa_files ...]
"""

import argparse
import sys
import os
from collections import Counter
from typing import Dict, List

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from datasets import load_dataset

from train_utils import part3_qa_text_to_messages
from tokenization import build_tokenizer, SFTMessagesTokenizerBuilder

SAMPLE_FILES = [
    "/ruilab/jxhe/CoE_Monitor/Physics_of_LM/Part3/datasets/QA/train/q1_birth_date.parquet",
    "/ruilab/jxhe/CoE_Monitor/Physics_of_LM/Part3/datasets/QA/train/q2_birth_city.parquet",
    "/ruilab/jxhe/CoE_Monitor/Physics_of_LM/Part3/datasets/QA/train/q3_university.parquet",
    "/ruilab/jxhe/CoE_Monitor/Physics_of_LM/Part3/datasets/QA/train/q4_major.parquet",
    "/ruilab/jxhe/CoE_Monitor/Physics_of_LM/Part3/datasets/QA/train/q5_company.parquet",
    "/ruilab/jxhe/CoE_Monitor/Physics_of_LM/Part3/datasets/QA/train/q6_company_city.parquet",
]

MODES = ["no-answer", "#", "# #", "#*10", "attribute", "raw"]
N_POS = 5


def _find_after_q(tokenizer, full_text: str) -> List[int]:
    """在 full_text 中找到 ? 之后的 token_id 列表。"""
    encodings = tokenizer(full_text, add_special_tokens=False, return_offsets_mapping=True)
    full_ids = encodings["input_ids"]
    offsets = encodings.encodings[0].offsets

    q_pos = full_text.find("?")
    if q_pos == -1:
        return []

    q_tok_pos = None
    for ti in range(len(full_ids)):
        tok_start, tok_end = offsets[ti]
        if tok_start <= q_pos < tok_end:
            q_tok_pos = ti
            break
    if q_tok_pos is None:
        return []

    return full_ids[q_tok_pos + 1: q_tok_pos + 1 + N_POS]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--tokenizer_path", default="/ruilab/jxhe/CoE_Monitor/Physics_of_LM/Part3/checkpoints/llama2")
    parser.add_argument("--n_samples", type=int, default=10000)
    parser.add_argument("--qa_files", nargs="*", default=SAMPLE_FILES)
    args = parser.parse_args()

    print(f"加载 Tokenizer: {args.tokenizer_path}")
    tokenizer = build_tokenizer(None, args.tokenizer_path)
    sft_builder = SFTMessagesTokenizerBuilder(tokenizer, max_length=4096, apply_chat_template=False, test=False)
    print(f"pad_token_id={tokenizer.pad_token_id}, bos_token_id={tokenizer.bos_token_id}, eos_token_id={tokenizer.eos_token_id}")

    all_texts: List[str] = []
    for fpath in args.qa_files:
        ds = load_dataset("parquet", data_files={"train": fpath})["train"]
        for i in range(min(args.n_samples, len(ds))):
            all_texts.append(str(ds[i]["text"]))
    print(f"\n已加载 {len(all_texts)} 条样本，来自 {len(args.qa_files)} 个文件")

    for mode in MODES:
        print(f"\n{'='*90}")
        print(f"mode: {mode}")
        print(f"{'='*90}")

        pos_counter: Dict[int, Counter] = {p: Counter() for p in range(N_POS)}
        pos_count: Dict[int, int] = {p: 0 for p in range(N_POS)}
        examples: List[dict] = []

        for i, text in enumerate(all_texts):
            messages = part3_qa_text_to_messages(text, mode=mode)
            full_text = sft_builder._apply_chat_template_text(messages)

            after_q = _find_after_q(tokenizer, full_text)

            for p in range(min(len(after_q), N_POS)):
                pos_counter[p][after_q[p]] += 1
                pos_count[p] += 1

            if i < 3:
                full_ids = tokenizer(full_text, add_special_tokens=False)["input_ids"]
                user_text = sft_builder._apply_chat_template_text(messages[:1])
                user_ids = tokenizer(user_text, add_special_tokens=False)["input_ids"]
                labels = [-100] * len(user_ids) + full_ids[len(user_ids):]
                examples.append({
                    "text": text[:150],
                    "full_text": repr(full_text[:200]),
                    "full_input_ids": full_ids,
                    "full_labels": labels,
                    "?后token": [tokenizer.decode([tid]) for tid in after_q],
                })

        for pos in range(N_POS):
            covered = pos_count[pos]
            print(f"\n'?' 后第 {pos} 个位置（覆盖 {covered}/{len(all_texts)} 条）:")
            for tid, freq in pos_counter[pos].most_common(20):
                tstr = tokenizer.decode([tid], skip_special_tokens=False)
                print(f"  token_id={tid:>6}, text={repr(tstr):30s} → {freq}")

        print(f"\n示例（前 {len(examples)} 条）:")
        for i, ex in enumerate(examples):
            print(f"\n  --- 示例 {i} ---")
            print(f"  原始: {ex['text']}")
            print(f"  拼接: {ex['full_text']}")
            print(f"  ?后  : {ex['?后token']}")
            print(f"  Full input_ids: {ex['full_input_ids']}")
            print(f"  Full labels: {[x for x in ex['full_labels'] if x != -100]}")


if __name__ == "__main__":
    main()
