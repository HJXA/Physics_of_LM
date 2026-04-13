import os
import sys
import glob
import subprocess
import re
import argparse
from collections import defaultdict
from multiprocessing import Pool, Manager

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


CHECKPOINT_BASE = "/ruilab/jxhe/CoE_Monitor/Physics_of_LM/Part3/checkpoints/QA"
RESULT_BASE = "/ruilab/jxhe/CoE_Monitor/Physics_of_LM/Part3/eval/result"
EVAL_SCRIPT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "eval.py")
PLOT_DIR = "/ruilab/jxhe/CoE_Monitor/Physics_of_LM/Part3/eval/plots"
NUM_GPUS = 4


def get_checkpoints_to_eval():
    """
    扫描 checkpoints/QA/*/*/checkpoint* 目录，
    如果某实验目录存在 _merged 版本，则跳过原目录，只评测 _merged 版本。
    返回: {experiment_dir: [checkpoint_path, ...], ...}
    """
    pattern = os.path.join(CHECKPOINT_BASE, "*", "*", "checkpoint*")
    all_ckpt_dirs = sorted(glob.glob(pattern))

    # 按实验目录分组
    experiments = defaultdict(list)
    for ckpt_path in all_ckpt_dirs:
        exp_dir = os.path.dirname(ckpt_path)
        experiments[exp_dir].append(ckpt_path)

    # 筛选：有 _merged 的跳过原目录
    eval_experiments = {}
    for exp_dir, ckpts in sorted(experiments.items()):
        exp_name = os.path.basename(exp_dir)
        if exp_name.endswith("_merged"):
            eval_experiments[exp_dir] = sorted(ckpts)
        else:
            merged_dir = exp_dir + "_merged"
            if os.path.isdir(merged_dir):
                print(f"[跳过] {exp_dir}  (存在 _merged 版本: {merged_dir})")
            else:
                eval_experiments[exp_dir] = sorted(ckpts)

    return eval_experiments


def _run_single_eval(task):
    """子进程 worker：从共享 GPU 队列中获取空闲 GPU，执行评测后归还。"""
    model_path, test_dir, batch_size, max_input_length, max_new_tokens, gpu_queue = task

    # 从队列中获取一个空闲 GPU（阻塞等待）
    gpu_id = gpu_queue.get()
    try:
        cmd = [
            sys.executable, EVAL_SCRIPT,
            "--model_path", model_path,
            "--gpu_id", str(gpu_id),
            "--test_dir", test_dir,
            "--batch_size", str(batch_size),
            "--max_input_length", str(max_input_length),
            "--max_new_tokens", str(max_new_tokens),
        ]

        tag = f"[GPU{gpu_id}] {os.path.basename(os.path.dirname(model_path))}/{os.path.basename(model_path)}"
        print(f"  {tag} 开始评测...")

        result = subprocess.run(cmd, capture_output=True, text=True)
        if result.returncode != 0:
            print(f"  {tag} [ERROR]")
            print(result.stderr[-500:] if len(result.stderr) > 500 else result.stderr)
            return (model_path, False, None)

        acc = None
        for line in result.stdout.strip().split("\n"):
            if "accuracy=" in line:
                print(f"  {tag} {line.strip()}")
                m = re.search(r"accuracy=([\d.]+)", line)
                if m:
                    acc = float(m.group(1))
        return (model_path, True, acc)
    finally:
        # 评测完成后归还 GPU，让下一个任务可以使用
        gpu_queue.put(gpu_id)


def collect_results():
    """
    从 result 目录收集所有 metrics 文件中的 accuracy。
    返回: {model_type: {model_name: {step: accuracy}}}
    """
    results = defaultdict(lambda: defaultdict(dict))

    pattern = os.path.join(RESULT_BASE, "*", "*", "checkpoint*", "metrics_*.txt")
    for metrics_file in sorted(glob.glob(pattern)):
        parts = metrics_file.replace("\\", "/").split("/")
        # .../result/model_type/model_name/checkpoint-xxx/metrics_xxx.txt
        model_type = parts[-4]
        model_name = parts[-3]
        checkpoint_name = parts[-2]

        step_match = re.search(r"checkpoint-(\d+)", checkpoint_name)
        if not step_match:
            continue
        step = int(step_match.group(1))

        with open(metrics_file, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line.startswith("accuracy="):
                    acc = float(line.split("=", 1)[1])
                    results[model_type][model_name][step] = acc
                    break

    return dict(results)


def _map_legend_name(model_name):
    """将复杂的模型名称映射为图例中简洁的标签。

    LoRA qv8 → lora_llama2_rank_qv8
    LoRA qv16_04_09_11 merged → lora_llama2_rank_qv16
    LoRA qv16_04_09_14 merged → lora_llama2_rank_qv16_no_answer
    SFT answer → sft_llama2_answer
    SFT no_answer → sft_llama2_no_answer
    """
    # LoRA qv8
    if model_name.startswith("lora_llama2"):
        if "_rank_qv8_" in model_name:
            return "lora_llama2_rank_qv8"
        elif "_rank_qv16_" in model_name:
            label = "lora_llama2_rank_qv16"
            # 04_09_14xx → no_answer
            if "_2026_04_09_14" in model_name:
                label += "_no_answer"
            return label
        else:
            return model_name

    # SFT: extract sft_llama2 + answer flag
    if model_name.startswith("sft_llama2"):
        if "_no_answer_" in model_name:
            return "sft_llama2_no_answer"
        else:
            return "sft_llama2_answer"

    return model_name


def plot_results(results, plot_dir):
    """每个 model_type 一张图，横轴 step，纵轴 accuracy，不同实验为不同曲线。"""
    os.makedirs(plot_dir, exist_ok=True)

    for model_type, experiments in sorted(results.items()):
        if not experiments:
            continue

        fig, ax = plt.subplots(figsize=(12, 7))

        for model_name, steps_dict in sorted(experiments.items()):
            if not steps_dict:
                continue
            steps = sorted(steps_dict.keys())
            accs = [steps_dict[s] for s in steps]

            label = _map_legend_name(model_name)

            ax.plot(steps, accs, marker="o", label=label, linewidth=2, markersize=6)

        ax.set_xlabel("Training Steps", fontsize=14)
        ax.set_ylabel("Accuracy", fontsize=14)
        ax.set_title(f"{model_type} — Accuracy vs Training Steps", fontsize=16)
        ax.legend(fontsize=9, bbox_to_anchor=(1.02, 1), loc="upper left", borderaxespad=0)
        ax.grid(True, alpha=0.3)
        ax.tick_params(axis="both", labelsize=12)

        fig.tight_layout()
        plot_path = os.path.join(plot_dir, f"{model_type}_accuracy.png")
        fig.savefig(plot_path, dpi=150, bbox_inches="tight")
        plt.close(fig)
        print(f"[Plot] {plot_path}")


def main():
    parser = argparse.ArgumentParser(description="批量评测 QA checkpoints 并画图")
    parser.add_argument("--skip_eval", action="store_true", help="跳过评测，仅对已有结果画图")
    parser.add_argument("--skip_plot", action="store_true", help="跳过画图，仅执行评测")
    parser.add_argument("--test_dir", type=str,
                        default="/ruilab/jxhe/CoE_Monitor/Physics_of_LM/Part3/datasets/QA/test",
                        help="测试集目录")
    parser.add_argument("--batch_size", type=int, default=8192)
    parser.add_argument("--max_input_length", type=int, default=128)
    parser.add_argument("--max_new_tokens", type=int, default=32)
    args = parser.parse_args()

    # ---- 评测阶段 ----
    if not args.skip_eval:
        eval_experiments = get_checkpoints_to_eval()
        total_ckpts = sum(len(v) for v in eval_experiments.values())
        print(f"\n{'='*70}")
        print(f"共发现 {len(eval_experiments)} 个实验，{total_ckpts} 个 checkpoint 待评测")
        print(f"使用 {NUM_GPUS} 个 GPU 并行评测")
        print(f"{'='*70}\n")

        # 构建任务列表（GPU 动态分配，无需预分配）
        all_tasks = []
        skipped = 0
        for exp_dir, ckpts in sorted(eval_experiments.items()):
            for ckpt in sorted(ckpts):
                # 检查是否已有评测结果
                model_type = os.path.basename(os.path.dirname(exp_dir))
                model_name = os.path.basename(exp_dir)
                checkpoint_name = os.path.basename(ckpt)
                result_subdir = os.path.join(RESULT_BASE, model_type, model_name, checkpoint_name)
                existing_metrics = glob.glob(os.path.join(result_subdir, "metrics_*.txt"))
                if existing_metrics:
                    skipped += 1
                    continue

                all_tasks.append((
                    ckpt,
                    args.test_dir,
                    args.batch_size,
                    args.max_input_length,
                    args.max_new_tokens,
                    None,  # gpu_queue 占位，下面统一填入
                ))

        if skipped > 0:
            print(f"跳过 {skipped} 个已有评测结果的 checkpoint")
        print(f"实际待评测: {len(all_tasks)} 个 checkpoint\n")

        # 创建共享 GPU 队列，初始放入所有可用 GPU ID
        manager = Manager()
        gpu_queue = manager.Queue()
        for g in range(NUM_GPUS):
            gpu_queue.put(g)

        # 将 gpu_queue 填入每个任务
        all_tasks = [(ckpt, td, bs, mil, mnt, gpu_queue)
                     for ckpt, td, bs, mil, mnt, _ in all_tasks]

        # 使用进程池并行执行，GPU 动态调度：哪个先完成就先领下一个
        with Pool(processes=NUM_GPUS) as pool:
            results_list = pool.map(_run_single_eval, all_tasks)

        failed = [r[0] for r in results_list if not r[1]]
        if failed:
            print(f"\n[WARNING] {len(failed)} 个 checkpoint 评测失败:")
            for p in failed:
                print(f"  - {p}")

    # ---- 画图阶段 ----
    if not args.skip_plot:
        print(f"\n{'='*70}")
        print("收集结果并画图...")
        print(f"{'='*70}\n")
        results = collect_results()
        for model_type, experiments in results.items():
            print(f"  {model_type}: {len(experiments)} 条实验曲线")
        plot_results(results, PLOT_DIR)
        print(f"\n所有图片已保存至: {PLOT_DIR}")


if __name__ == "__main__":
    main()
