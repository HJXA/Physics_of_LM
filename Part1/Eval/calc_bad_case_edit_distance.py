import os
import json
import argparse
import statistics
from concurrent.futures import ProcessPoolExecutor

from tqdm import tqdm

import sys
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
sys.path.append("/ruilab/jxhe/CoE_Monitor/Physics_of_LM/Part1/Lano-cfg")
from data_cfg import CFG_Config


_WORKER_CFG = None
_WORKER_BOS_TOKEN = None
_WORKER_EOS_TOKEN = None


def read_jsonl(path):
    records = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            records.append(json.loads(line))
    return records


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


def extract_generated_ids(record):
    return [int(x) for x in record["generate_id"]]


def strip_bos_eos(seq, bos_token, eos_token):
    if len(seq) == 0:
        return []
    if seq[0] == bos_token:
        seq = seq[1:]
    if len(seq) > 0 and seq[-1] == eos_token:
        seq = seq[:-1]
    return seq


def _init_worker(config_path, bos_token, eos_token):
    """每个子进程初始化一次 CFG，避免反复加载。"""
    global _WORKER_CFG, _WORKER_BOS_TOKEN, _WORKER_EOS_TOKEN
    _WORKER_CFG = CFG_Config.from_graph(config_path)
    _WORKER_BOS_TOKEN = bos_token
    _WORKER_EOS_TOKEN = eos_token


def _compute_edit_distance_in_worker(task):
    """子进程任务入口。"""
    return compute_edit_distance(
        task,
        cfg=_WORKER_CFG,
        bos_token=_WORKER_BOS_TOKEN,
        eos_token=_WORKER_EOS_TOKEN,
    )


def compute_edit_distance(task, cfg, bos_token, eos_token):
    pos, record = task
    scenario = record.get("scenario", "")
    global_index = record.get("index", None)

    result = {
        "file_pos": pos,
        "record_index": global_index,
        "scenario": scenario,
        "edit_distance": None,
        "num_possibility": None,
        "layer_edit_counts": None,
        "raw_len": None,
        "pure_len": None,
        "status": "ok",
        "error": "",
    }

    try:
        generated_ids = extract_generated_ids(record)
        result["raw_len"] = len(generated_ids)
        pure_seq = strip_bos_eos(generated_ids, bos_token=bos_token, eos_token=eos_token)
        result["pure_len"] = len(pure_seq)

        if len(pure_seq) == 0:
            result["status"] = "empty_after_strip"
            return result

        best, _, counts, num_poss = cfg.solve_dp_noneq(pure_seq, no_debug=True)
        result["edit_distance"] = int(best)
        result["num_possibility"] = int(num_poss) if num_poss is not None else None
        result["layer_edit_counts"] = [int(x) for x in counts] if counts is not None else None
        if int(best) >= 10000:
            result["status"] = "unsatisfied_or_unreachable"
        return result
    except Exception as e:
        result["status"] = "error"
        result["error"] = str(e)
        return result


def format_summary(results, bad_case_path, config_path, num_workers, topk):
    total = len(results)
    error_count = sum(1 for item in results if item["status"] == "error")
    empty_count = sum(1 for item in results if item["status"] == "empty_after_strip")
    unreachable_count = sum(1 for item in results if item["status"] == "unsatisfied_or_unreachable")

    valid_dist = [item["edit_distance"] for item in results if isinstance(item["edit_distance"], int)]
    finite_dist = [x for x in valid_dist if x < 10000]

    by_scenario = {}
    for item in results:
        sc = item.get("scenario", "") or "unknown"
        by_scenario.setdefault(sc, []).append(item)

    lines = []
    lines.append("=" * 72)
    lines.append("Bad Case 最小编辑距离报告")
    lines.append("=" * 72)
    lines.append(f"输入 bad_case 文件: {bad_case_path}")
    lines.append(f"CFG 配置文件: {config_path}")
    lines.append(f"并发进程数: {num_workers}")
    lines.append(f"总样本数: {total}")
    lines.append(f"可计算样本数: {len(valid_dist)}")
    lines.append(f"空序列样本数: {empty_count}")
    lines.append(f"无法满足/不可达样本数(=10000): {unreachable_count}")
    lines.append(f"异常样本数: {error_count}")

    if finite_dist:
        lines.append("-" * 72)
        lines.append("全局统计（仅统计 edit_distance < 10000）")
        lines.append(f"最小值: {min(finite_dist)}")
        lines.append(f"最大值: {max(finite_dist)}")
        lines.append(f"均值: {statistics.mean(finite_dist):.4f}")
        lines.append(f"中位数: {statistics.median(finite_dist):.4f}")

    lines.append("-" * 72)
    lines.append("分场景统计")
    for scenario, items in by_scenario.items():
        dists = [x["edit_distance"] for x in items if isinstance(x["edit_distance"], int) and x["edit_distance"] < 10000]
        unreach = sum(1 for x in items if x["status"] == "unsatisfied_or_unreachable")
        err = sum(1 for x in items if x["status"] == "error")
        emp = sum(1 for x in items if x["status"] == "empty_after_strip")
        lines.append(f"- {scenario}: total={len(items)}, finite={len(dists)}, unreachable={unreach}, empty={emp}, error={err}")
        if dists:
            lines.append(
                f"  mean={statistics.mean(dists):.4f}, median={statistics.median(dists):.4f}, min={min(dists)}, max={max(dists)}"
            )

    if topk > 0:
        lines.append("-" * 72)
        lines.append(f"Top-{topk} 最难样本（按 edit_distance 降序，排除 10000）")
        ranked = [item for item in results if isinstance(item["edit_distance"], int) and item["edit_distance"] < 10000]
        ranked.sort(key=lambda x: x["edit_distance"], reverse=True)
        for item in ranked[:topk]:
            lines.append(
                f"file_pos={item['file_pos']}, record_index={item['record_index']}, "
                f"scenario={item['scenario']}, pure_len={item['pure_len']}, edit_distance={item['edit_distance']}"
            )

    lines.append("=" * 72)
    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(description="多进程计算 bad case 的最小编辑距离")
    parser.add_argument("--config_path", type=str, default="/ruilab/jxhe/CoE_Monitor/Physics_of_LM/Part1/Lano-cfg/configs/cfg3f.json")
    parser.add_argument("--p", type=str, default="/ruilab/jxhe/CoE_Monitor/Physics_of_LM/Part1/Eval/result/gpt_2_rot/checkpoint-100000/bad_cases.jsonl")
    parser.add_argument("--num_workers", type=int, default=max(1, (os.cpu_count() or 1) // 4))
    parser.add_argument("--bos_token", type=int, default=0)
    parser.add_argument("--eos_token", type=int, default=4)
    parser.add_argument("--topk", type=int, default=20)
    parser.add_argument("--save_results_path", type=str, default="")
    parser.add_argument("--save_report_path", type=str, default="")
    args = parser.parse_args()

    args.bad_case_path = args.p

    if not os.path.exists(args.bad_case_path):
        print(f"找不到 bad_case 文件: {args.bad_case_path}")
        return
    if not os.path.exists(args.config_path):
        print(f"找不到 CFG 配置文件: {args.config_path}")
        return

    base_dir = os.path.dirname(args.bad_case_path)
    save_results_path = args.save_results_path or os.path.join(base_dir, "bad_case_edit_distance.jsonl")
    save_report_path = args.save_report_path or os.path.join(base_dir, "bad_case_edit_distance_report.txt")

    records = read_jsonl(args.bad_case_path)
    if len(records) == 0:
        print("bad_case 文件为空，无法计算。")
        return

    tasks = list(enumerate(records))
    workers = min(max(1, args.num_workers), len(tasks))

    with ProcessPoolExecutor(
        max_workers=workers,
        initializer=_init_worker,
        initargs=(args.config_path, args.bos_token, args.eos_token),
    ) as executor:
        results = list(
            tqdm(
                executor.map(_compute_edit_distance_in_worker, tasks),
                total=len(tasks),
                desc="Computing Min Edit Distance",
            )
        )

    report_text = format_summary(
        results=results,
        bad_case_path=args.bad_case_path,
        config_path=args.config_path,
        num_workers=workers,
        topk=max(0, args.topk),
    )

    print(report_text)
    write_jsonl(save_results_path, results)
    write_text(save_report_path, report_text + "\n")

    print(f"\n已保存明细到: {save_results_path}")
    print(f"已保存报告到: {save_report_path}")


if __name__ == "__main__":
    main()
