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
import matplotlib.gridspec as gridspec
import matplotlib.colors as mcolors


import colorsys
import hashlib

CHECKPOINT_BASE = "/ruilab/jxhe/CoE_Monitor/Physics_of_LM/Part3/checkpoints/QA"
# 线型和 marker 池（与颜色正交组合）
_LINESTYLES = ["-", "--", "-.", ":"]
_MARKERS = ["o", "s", "^", "d", "v", "p", "P", "D", "x", "*", "1", "2", "3", "4", "|", "h", ">", "<"]


def _get_style_for(label):
    """基于 label hash 动态分配 (color, linestyle, marker)，同一 label 永远同一样式。"""
    h = hashlib.sha256(label.encode()).hexdigest()
    # 颜色：hls 均匀分布，固定 lightness 和 saturation 保可读性
    hue = int(h[:6], 16) / 0xFFFFFF
    color = mcolors.to_hex(colorsys.hls_to_rgb(hue, 0.45, 0.75))
    # 线型和 marker 用相隔较远的 hex 位，避免相似字符串分配相同样式
    linestyle = _LINESTYLES[int(h[8], 16) % len(_LINESTYLES)]
    marker = _MARKERS[int(h[20], 16) * len(_MARKERS) // 16]
    return {"color": color, "linestyle": linestyle, "marker": marker}


IS_0419 = False
RESULT_BASE = "/ruilab/jxhe/CoE_Monitor/Physics_of_LM/Part3/eval/result"
EVAL_SCRIPT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "eval.py")
PLOT_DIR = "/ruilab/jxhe/CoE_Monitor/Physics_of_LM/Part3/eval/plots"
if IS_0419:
    RESULT_BASE += "/0419"
    PLOT_DIR += "/0419"
    CHECKPOINT_BASE += "/0419"
GPU_IDS = [4,5]  # 指定可用的 GPU ID 列表，而非数量


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
    model_path, test_dir, batch_size, max_input_length, max_new_tokens, result_dir, gpu_queue = task

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
            "--result_dir", result_dir,
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


def collect_results(result_dir=RESULT_BASE):
    """
    从 result 目录收集所有 metrics 文件中的 accuracy（含 per-attribute）。
    返回: {model_type: {model_name: {step: {"overall": X, "birth_date": Y, ...}}}}
    """
    results = defaultdict(lambda: defaultdict(dict))

    pattern = os.path.join(result_dir, "*", "*", "checkpoint*", "metrics_*.txt")
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

        step_data = {}
        with open(metrics_file, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line.startswith("accuracy="):
                    step_data["overall"] = float(line.split("=", 1)[1])
                elif line.startswith("accuracy_"):
                    key, val = line.split("=", 1)
                    attr_name = key[len("accuracy_"):]
                    step_data[attr_name] = float(val)

        if step_data:
            results[model_type][model_name][step] = step_data

    return dict(results)


def _map_legend_name(model_name):
    """将复杂的模型名称映射为图例中简洁的标签。"""
    # 提取 mode 后缀（如 _mode_no_answer → _no_answer）
    mode_suffix = ""
    m = re.search(r"_mode_([^_]+)", model_name)
    if m:
        mode_suffix = f"_{m.group(1)}"

    if 'mode' not in model_name and 'no_answer' in model_name:
        mode_suffix = "_no_answer"

    if model_name.startswith("lora_llama2"):
        if "_rank_qv8_" in model_name:
            return "lora_llama2_rank_qv8" + mode_suffix
        elif "_rank_qv16_" in model_name:
            return "lora_llama2_rank_qv16" + mode_suffix
        else:
            return "lora_llama2" + mode_suffix

    if model_name.startswith("sft_llama2"):
        return "sft_llama2" + mode_suffix

    return model_name


KNOWN_ATTRS = ["birth_date", "birth_city", "university", "field", "company1name", "company1city"]


def plot_results(results, plot_dir):
    """每个 model_type 一张大图：上方总图，下方6个属性子图。"""
    os.makedirs(plot_dir, exist_ok=True)

    for model_type, experiments in sorted(results.items()):
        if not experiments:
            continue

        # 收集所有出现过的属性
        all_attrs = set()
        for model_name, steps_dict in experiments.items():
            for step_data in steps_dict.values():
                all_attrs.update(k for k in step_data.keys() if k != "overall")
        # 排序：已知属性优先，其余按字母序
        attr_order = [a for a in KNOWN_ATTRS if a in all_attrs]
        attr_order += sorted(a for a in all_attrs if a not in KNOWN_ATTRS)

        n_attrs = len(attr_order)
        if n_attrs == 0:
            # 没有 per-attribute 数据，退回单图模式
            fig, ax = plt.subplots(figsize=(12, 7))
            for model_name, steps_dict in sorted(experiments.items()):
                if not steps_dict:
                    continue
                steps = sorted(steps_dict.keys())
                accs = [steps_dict[s].get("overall", 0) for s in steps]
                label = _map_legend_name(model_name)
                s = _get_style_for(label)
                ax.plot(steps, accs, label=label,
                        color=s["color"], linestyle=s["linestyle"],
                        marker=s["marker"], linewidth=2, markersize=6, alpha=0.85)
            ax.set_xlabel("Training Steps", fontsize=14)
            ax.set_ylabel("Accuracy", fontsize=14)
            ax.set_title(f"{model_type} — Accuracy vs Training Steps", fontsize=16)
            ax.legend(fontsize=9, bbox_to_anchor=(1.02, 1), loc="upper left", borderaxespad=0)
            ax.grid(True, alpha=0.3)
            ax.tick_params(axis="both", labelsize=12)
            fig.subplots_adjust(right=0.75)
            plot_path = os.path.join(plot_dir, f"{model_type}_accuracy.png")
            fig.savefig(plot_path, dpi=150, bbox_inches="tight")
            plt.close(fig)
            print(f"[Plot] {plot_path}")
            continue

        # 筛选有数据的属性
        def _attr_has_data(attr_name):
            for model_name, steps_dict in experiments.items():
                if not steps_dict:
                    continue
                for step_data in steps_dict.values():
                    if attr_name in step_data and isinstance(step_data[attr_name], (int, float)):
                        return True
            return False

        has_data_attrs = [a for a in attr_order if _attr_has_data(a)]
        n_data_attrs = len(has_data_attrs)

        # 布局：上方总图跨全宽，下方属性子图
        n_cols = min(3, n_data_attrs) if n_data_attrs > 0 else 1
        n_rows_attrs = (n_data_attrs + n_cols - 1) // n_cols if n_data_attrs > 0 else 0
        total_rows = 1 + n_rows_attrs

        fig = plt.figure(figsize=(6 * n_cols, 5 * total_rows))
        gs = gridspec.GridSpec(total_rows, n_cols, figure=fig, hspace=0.4, wspace=0.3)

        # 总图：占第一整行
        ax_overall = fig.add_subplot(gs[0, :])
        for model_name, steps_dict in sorted(experiments.items()):
            if not steps_dict:
                continue
            steps = sorted(steps_dict.keys())
            accs = [steps_dict[s].get("overall", 0) for s in steps]
            label = _map_legend_name(model_name)
            s = _get_style_for(label)
            ax_overall.plot(steps, accs, label=label,
                            color=s["color"], linestyle=s["linestyle"],
                            marker=s["marker"], linewidth=2, markersize=6, alpha=0.85)
        ax_overall.set_xlabel("Training Steps", fontsize=12)
        ax_overall.set_ylabel("Accuracy", fontsize=12)
        ax_overall.set_title(f"{model_type} — Overall Accuracy", fontsize=14)
        ax_overall.legend(fontsize=8, bbox_to_anchor=(1.02, 1), loc="upper left", borderaxespad=0)
        ax_overall.grid(True, alpha=0.3)

        # 属性子图（均为有数据的属性）
        for idx, attr_name in enumerate(has_data_attrs):
            row = 1 + idx // n_cols
            col = idx % n_cols
            ax_attr = fig.add_subplot(gs[row, col])

            for model_name, steps_dict in sorted(experiments.items()):
                if not steps_dict:
                    continue
                steps = sorted(steps_dict.keys())
                accs = [steps_dict[s].get(attr_name, float("nan")) for s in steps]
                if all(isinstance(a, float) and a != a for a in accs):
                    continue
                label = _map_legend_name(model_name)
                s = _get_style_for(label)
                ax_attr.plot(steps, accs, label=label,
                             color=s["color"], linestyle=s["linestyle"],
                             marker=s["marker"], linewidth=1.5, markersize=4, alpha=0.85)

            ax_attr.set_xlabel("Steps", fontsize=10)
            ax_attr.set_ylabel("Accuracy", fontsize=10)
            ax_attr.set_title(attr_name, fontsize=11)
            ax_attr.grid(True, alpha=0.3)
            ax_attr.tick_params(axis="both", labelsize=9)

        # 图例放在最后一个属性子图上
        if has_data_attrs:
            fig.axes[-1].legend(fontsize=7, bbox_to_anchor=(1.02, 1), loc="upper left", borderaxespad=0)

        fig.subplots_adjust(hspace=0.5, wspace=0.3, right=0.75)
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
    parser.add_argument("--result_dir", type=str, default=RESULT_BASE, help="评测结果输出目录")
    args = parser.parse_args()

    # ---- 评测阶段 ----
    if not args.skip_eval:
        eval_experiments = get_checkpoints_to_eval()
        total_ckpts = sum(len(v) for v in eval_experiments.values())
        print(f"\n{'='*70}")
        print(f"共发现 {len(eval_experiments)} 个实验，{total_ckpts} 个 checkpoint 待评测")
        print(f"使用 GPU {GPU_IDS} 并行评测")
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
                result_subdir = os.path.join(args.result_dir, model_type, model_name, checkpoint_name)
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
                    args.result_dir,
                    None,  # gpu_queue 占位，下面统一填入
                ))

        if skipped > 0:
            print(f"跳过 {skipped} 个已有评测结果的 checkpoint")
        print(f"实际待评测: {len(all_tasks)} 个 checkpoint\n")

        # 创建共享 GPU 队列，初始放入所有可用 GPU ID
        manager = Manager()
        gpu_queue = manager.Queue()
        for g in GPU_IDS:
            gpu_queue.put(g)

        # 将 gpu_queue 填入每个任务
        all_tasks = [(ckpt, td, bs, mil, mnt, rd, gpu_queue)
                     for ckpt, td, bs, mil, mnt, rd, _ in all_tasks]

        # 使用进程池并行执行，GPU 动态调度：哪个先完成就先领下一个
        with Pool(processes=len(GPU_IDS)) as pool:
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
        results = collect_results(args.result_dir)
        for model_type, experiments in results.items():
            print(f"  {model_type}: {len(experiments)} 条实验曲线")
        plot_results(results, PLOT_DIR)
        print(f"\n所有图片已保存至: {PLOT_DIR}")


if __name__ == "__main__":
    main()
