import os
import json
import argparse
from concurrent.futures import ProcessPoolExecutor
from functools import partial

from tqdm import tqdm

import sys
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
sys.path.append("/ruilab/jxhe/CoE_Monitor/Physics_of_LM/Part1/Lano-cfg")
from data_cfg import CFG_Config


_WORKER_CONFIG = None
_WORKER_BOS_TOKEN = None
_WORKER_EOS_TOKEN = None


def _init_worker(config_path, bos_token, eos_token):
    """每个子进程初始化一次：加载 CFG 配置并缓存 token。"""
    global _WORKER_CONFIG, _WORKER_BOS_TOKEN, _WORKER_EOS_TOKEN
    _WORKER_CONFIG = CFG_Config.from_graph(config_path)
    _WORKER_BOS_TOKEN = bos_token
    _WORKER_EOS_TOKEN = eos_token


def _eval_accuracy_in_worker(generated_seq):
    """子进程内判分入口（避免在 map 中反复传递 config 大对象）。"""
    return eval_accuracy(
        _WORKER_CONFIG,
        generated_seq,
        bos_token=_WORKER_BOS_TOKEN,
        eos_token=_WORKER_EOS_TOKEN,
    )


def eval_accuracy(config, generated_seq, bos_token, eos_token):
    """
    判断模型生成 token 序列是否满足 CFG。
    评测逻辑保持不变：
    1) 必须以 EOS 正常结束；
    2) 去除 BOS/EOS；
    3) 使用 solve_dp_noneq_fast 做合法性判定；
    4) count == 0 记为正确。
    """
    if len(generated_seq) < 2 or generated_seq[-1] != eos_token:
        return False

    # 新增：如果序列中有token_id大于eos_token，直接返回False
    if any(token_id > eos_token for token_id in generated_seq):
        return False

    if generated_seq[0] == bos_token:
        pure_seq = generated_seq[1:-1]
    else:
        pure_seq = generated_seq[:-1]

    if len(pure_seq) == 0:
        return False

    count, *_ = config.solve_dp_noneq_fast(pure_seq, no_debug=True)
    return count == 0


def parallel_eval_accuracy(config, config_path, sequences, bos_token, eos_token, num_workers, desc):
    """并发（或单进程）执行准确率判定，返回逐样本 bool 列表。"""
    if num_workers <= 1:
        return [eval_accuracy(config, seq, bos_token, eos_token) for seq in tqdm(sequences, desc=desc)]

    with ProcessPoolExecutor(
        max_workers=num_workers,
        initializer=_init_worker,
        initargs=(config_path, bos_token, eos_token),
    ) as executor:
        return list(tqdm(executor.map(_eval_accuracy_in_worker, sequences), total=len(sequences), desc=desc))


def write_jsonl(path, records):
    dir_path = os.path.dirname(path)
    if dir_path:
        os.makedirs(dir_path, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for item in records:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")


def write_text(path, text):
    dir_path = os.path.dirname(path)
    if dir_path:
        os.makedirs(dir_path, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        f.write(text)


def read_jsonl(path):
    records = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            records.append(json.loads(line))
    return records


def extract_generated_ids(record):
    """仅读取生成脚本落盘字段 generate_id。"""
    return [int(x) for x in record["generate_id"]]


def main():
    parser = argparse.ArgumentParser(description="只做 CFG 判分（读取已生成样本）")
    parser.add_argument("--config_path", type=str, default="/ruilab/jxhe/CoE_Monitor/Physics_of_LM/Part1/Lano-cfg/configs/cfg3f.json")
    parser.add_argument("--p", type=str, default="")
    parser.add_argument("--judge_workers", type=int, default=max(1, (os.cpu_count() or 1) // 2))
    parser.add_argument("--save_judged_results_path", type=str, default="")
    parser.add_argument("--save_bad_cases_path", type=str, default="")
    parser.add_argument("--save_good_cases_path", type=str, default="")
    parser.add_argument("--save_report_path", type=str, default="")
    args = parser.parse_args()

    args.generated_path = args.p


    args.save_judged_results_path = os.path.join(os.path.dirname(args.generated_path), "judged_results.jsonl") if not args.save_judged_results_path else args.save_judged_results_path
    args.save_bad_cases_path = os.path.join(os.path.dirname(args.generated_path), "bad_cases.jsonl") if not args.save_bad_cases_path else args.save_bad_cases_path
    args.save_good_cases_path = os.path.join(os.path.dirname(args.generated_path), "good_cases.jsonl") if not args.save_good_cases_path else args.save_good_cases_path
    args.save_report_path = os.path.join(os.path.dirname(args.generated_path), "final_report.txt") if not args.save_report_path else args.save_report_path


    BOS_TOKEN = 0
    EOS_TOKEN = 4

    if not os.path.exists(args.generated_path):
        print(f"找不到生成文件 {args.generated_path}，请先运行 generate_cfg_samples.py")
        return

    config = CFG_Config.from_graph(args.config_path)
    records = read_jsonl(args.generated_path)
    if len(records) == 0:
        print("生成文件为空，无法判分。")
        return

    uncond_indices = []
    cond_indices = []
    uncond_sequences = []
    cond_sequences = []

    for idx, record in enumerate(records):
        seq = extract_generated_ids(record)
        scenario = record.get("scenario", "")

        if scenario == "unconditional":
            uncond_indices.append(idx)
            uncond_sequences.append(seq)
        elif scenario == "conditional":
            cond_indices.append(idx)
            cond_sequences.append(seq)

    if len(uncond_sequences) == 0 and len(cond_sequences) == 0:
        print("未找到有效场景数据（scenario 需为 unconditional 或 conditional）。")
        return

    print(f"读取到 {len(records)} 条记录，开始判分...")

    print(f"num_uncond={len(uncond_sequences)}, num_cond={len(cond_sequences)}, judge_workers={args.judge_workers}")

    uncond_results = parallel_eval_accuracy(
        config,
        args.config_path,
        uncond_sequences,
        BOS_TOKEN,
        EOS_TOKEN,
        args.judge_workers if args.judge_workers < len(uncond_sequences) else len(uncond_sequences),  # 小样本时不使用多线程
        desc="Unconditional Eval",
    ) if len(uncond_sequences) > 0 else []

    cond_results = parallel_eval_accuracy(
        config,
        args.config_path,
        cond_sequences,
        BOS_TOKEN,
        EOS_TOKEN,
        args.judge_workers if args.judge_workers < len(cond_sequences) else len(cond_sequences),  # 小样本时不使用多线程
        desc="Conditional Eval",
    ) if len(cond_sequences) > 0 else []

    for pos, ok in enumerate(uncond_results):
        records[uncond_indices[pos]]["is_correct"] = bool(ok)
    for pos, ok in enumerate(cond_results):
        records[cond_indices[pos]]["is_correct"] = bool(ok)

    correct_unconditional = sum(uncond_results)
    correct_conditional = sum(cond_results)

    total_uncond = len(uncond_results)
    total_cond = len(cond_results)

    acc_uncond = (correct_unconditional / total_uncond) if total_uncond > 0 else 0.0
    acc_cond = (correct_conditional / total_cond) if total_cond > 0 else 0.0

    if args.save_judged_results_path:
        write_jsonl(args.save_judged_results_path, records)
        print(f"已保存判分结果到: {args.save_judged_results_path}")

    if args.save_bad_cases_path:
        bad_cases = [item for item in records if item.get("is_correct") is False]
        write_jsonl(args.save_bad_cases_path, bad_cases)
        print(f"已保存 bad case 到: {args.save_bad_cases_path} (共 {len(bad_cases)} 条)")

    if args.save_good_cases_path:
        good_cases = [item for item in records if item.get("is_correct") is True]
        write_jsonl(args.save_good_cases_path, good_cases)
        print(f"已保存 good case 到: {args.save_good_cases_path} (共 {len(good_cases)} 条)")

    report_lines = [
        "-" * 50,
        "【评测最终报告】",
        f"场景一 (无条件, 即生成完整句子) Accuracy: {acc_uncond * 100:.4f}%  (n={total_uncond})",
        f"场景二 (有条件, 馈送前缀后续写) Accuracy: {acc_cond * 100:.4f}%  (n={total_cond})",
        "-" * 50,
    ]
    report_text = "\n".join(report_lines)
    print(report_text)

    if args.save_report_path:
        write_text(args.save_report_path, report_text + "\n")
        print(f"已保存评测报告到: {args.save_report_path}")


if __name__ == "__main__":
    main()
