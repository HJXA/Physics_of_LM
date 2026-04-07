import argparse
import glob
import json
import os
import re
import subprocess
from datetime import datetime
from typing import Dict, List, Optional, Tuple

import matplotlib.pyplot as plt


def parse_checkpoint_step(checkpoint_path: str) -> int:
    name = os.path.basename(checkpoint_path.rstrip("/"))
    m = re.match(r"checkpoint-(\d+)$", name)
    if not m:
        return -1
    return int(m.group(1))


def find_checkpoints(model_dir: str) -> List[Tuple[int, str]]:
    pattern = os.path.join(model_dir, "checkpoint-*")
    candidates = glob.glob(pattern)
    parsed: List[Tuple[int, str]] = []
    for p in candidates:
        if not os.path.isdir(p):
            continue
        step = parse_checkpoint_step(p)
        if step >= 0:
            parsed.append((step, p))
    parsed.sort(key=lambda x: x[0])
    return parsed


def parse_metrics_file(metrics_path: str) -> Dict[str, str]:
    out: Dict[str, str] = {}
    with open(metrics_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line or "=" not in line:
                continue
            k, v = line.split("=", 1)
            out[k.strip()] = v.strip()
    return out


def get_latest_metrics_file(checkpoint_result_dir: str) -> Optional[str]:
    metrics_pattern = os.path.join(checkpoint_result_dir, "metrics_*.txt")
    files = glob.glob(metrics_pattern)
    if not files:
        return None
    files.sort(key=os.path.getmtime)
    return files[-1]


def run_single_eval(
    eval_script: str,
    checkpoint_path: str,
    test_dir: str,
    result_dir: str,
    batch_size: int,
    max_samples: int,
    max_input_length: int,
    max_new_tokens: int,
    verbose_errors: int,
    python_bin: str,
) -> int:
    cmd = [
        python_bin,
        eval_script,
        "--model_path",
        checkpoint_path,
        "--test_dir",
        test_dir,
        "--result_dir",
        result_dir,
        "--batch_size",
        str(batch_size),
        "--max_samples",
        str(max_samples),
        "--max_input_length",
        str(max_input_length),
        "--max_new_tokens",
        str(max_new_tokens),
        "--verbose_errors",
        str(verbose_errors),
    ]
    print("[Run]", " ".join(cmd))
    ret = subprocess.run(cmd, check=False)
    return ret.returncode


def main() -> None:
    parser = argparse.ArgumentParser(
        description="批量评测一个模型目录下全部 checkpoint，并绘制 step-accuracy 曲线。"
    )
    parser.add_argument(
        "--model_dir",
        type=str,
        required=True,
        help="形如 .../checkpoints/QA/bioS_single/sft_xxx 的目录，目录内应包含 checkpoint-* 子目录",
    )
    parser.add_argument(
        "--eval_script",
        type=str,
        default=os.path.join(os.path.dirname(__file__), "eval.py"),
        help="单 checkpoint 评测脚本路径（默认同目录 eval.py）",
    )
    parser.add_argument(
        "--test_dir",
        type=str,
        default="/ruilab/jxhe/CoE_Monitor/Physics_of_LM/Part3/datasets/QA/test",
        help="测试集目录",
    )
    parser.add_argument(
        "--result_dir",
        type=str,
        default="/ruilab/jxhe/CoE_Monitor/Physics_of_LM/Part3/eval/result",
        help="结果根目录（与 eval.py 的 --result_dir 一致）",
    )
    parser.add_argument("--batch_size", type=int, default=8192)
    parser.add_argument("--max_samples", type=int, default=0)
    parser.add_argument("--max_input_length", type=int, default=128)
    parser.add_argument("--max_new_tokens", type=int, default=32)
    parser.add_argument("--verbose_errors", type=int, default=0)
    parser.add_argument(
        "--python_bin",
        type=str,
        default="python",
        help="用于调用 eval.py 的 Python 可执行文件",
    )
    parser.add_argument(
        "--continue_on_error",
        action="store_true",
        help="某个 checkpoint 失败时是否继续后续 checkpoint",
    )
    args = parser.parse_args()

    model_dir = os.path.abspath(args.model_dir)
    eval_script = os.path.abspath(args.eval_script)

    if not os.path.isdir(model_dir):
        raise FileNotFoundError(f"model_dir 不存在: {model_dir}")
    if not os.path.isfile(eval_script):
        raise FileNotFoundError(f"eval_script 不存在: {eval_script}")

    checkpoints = find_checkpoints(model_dir)
    if not checkpoints:
        raise ValueError(f"未找到 checkpoint-* 目录: {model_dir}")

    model_type = model_dir.split("/")[-2]
    model_name = model_dir.split("/")[-1]
    model_result_dir = os.path.join(args.result_dir, model_type, model_name)
    os.makedirs(model_result_dir, exist_ok=True)

    print(f"[Info] model_dir={model_dir}")
    print(f"[Info] checkpoints={len(checkpoints)}")
    print(f"[Info] model_result_dir={model_result_dir}")

    records: List[Dict[str, object]] = []

    for step, checkpoint_path in checkpoints:
        ret = run_single_eval(
            eval_script=eval_script,
            checkpoint_path=checkpoint_path,
            test_dir=args.test_dir,
            result_dir=args.result_dir,
            batch_size=args.batch_size,
            max_samples=args.max_samples,
            max_input_length=args.max_input_length,
            max_new_tokens=args.max_new_tokens,
            verbose_errors=args.verbose_errors,
            python_bin=args.python_bin,
        )

        checkpoint_name = os.path.basename(checkpoint_path)
        checkpoint_result_dir = os.path.join(model_result_dir, checkpoint_name)
        metrics_path = get_latest_metrics_file(checkpoint_result_dir)

        if ret != 0:
            msg = f"checkpoint 评测失败: {checkpoint_path} (ret={ret})"
            if not args.continue_on_error:
                raise RuntimeError(msg)
            print(f"[Warn] {msg}")

        if metrics_path is None:
            msg = f"未找到 metrics 文件: {checkpoint_result_dir}"
            if not args.continue_on_error:
                raise FileNotFoundError(msg)
            print(f"[Warn] {msg}")
            continue

        metrics = parse_metrics_file(metrics_path)
        acc = float(metrics.get("accuracy", "nan"))
        n = int(metrics.get("n", "0"))

        item = {
            "step": step,
            "checkpoint": checkpoint_name,
            "accuracy": acc,
            "n": n,
            "metrics_path": metrics_path,
        }
        records.append(item)
        print(f"[Done] step={step}, accuracy={acc:.6f}, n={n}")

    if not records:
        raise RuntimeError("没有可用评测结果，无法绘图。")

    records.sort(key=lambda x: int(x["step"]))
    steps = [int(x["step"]) for x in records]
    accs = [float(x["accuracy"]) for x in records]

    plt.figure(figsize=(10, 6))
    plt.plot(steps, accs, marker="o", linewidth=2)
    plt.xlabel("Checkpoint Step")
    plt.ylabel("Accuracy")
    plt.title(f"Checkpoint Accuracy Curve: {model_name}")
    plt.grid(True, linestyle="--", alpha=0.4)

    for s, a in zip(steps, accs):
        plt.annotate(f"{a:.4f}", (s, a), textcoords="offset points", xytext=(0, 8), ha="center", fontsize=8)

    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    plot_path = os.path.join(model_result_dir, f"accuracy_curve_{ts}.png")
    summary_path = os.path.join(model_result_dir, f"accuracy_summary_{ts}.jsonl")
    metric_txt_path = os.path.join(model_result_dir, f"accuracy_summary_{ts}.txt")

    plt.tight_layout()
    plt.savefig(plot_path, dpi=200)
    plt.close()

    with open(summary_path, "w", encoding="utf-8") as f:
        for x in records:
            f.write(json.dumps(x, ensure_ascii=False) + "\n")

    best = max(records, key=lambda x: float(x["accuracy"]))
    lines = [
        f"time={ts}",
        f"model_dir={model_dir}",
        f"test_dir={args.test_dir}",
        f"num_checkpoints={len(records)}",
        f"best_step={best['step']}",
        f"best_checkpoint={best['checkpoint']}",
        f"best_accuracy={float(best['accuracy']):.6f}",
        f"plot_path={plot_path}",
        f"summary_jsonl={summary_path}",
    ]
    with open(metric_txt_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")

    print(f"[Summary] plot saved to: {plot_path}")
    print(f"[Summary] jsonl saved to: {summary_path}")
    print(f"[Summary] txt saved to: {metric_txt_path}")


if __name__ == "__main__":
    main()
