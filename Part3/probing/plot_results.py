"""
Probing 实验结果可视化工具。

复现论文中的准确率矩阵和对比图：
- 图 5：P-Probing 各属性准确率
- 图 7：Q-Probing 准确率 vs QA 微调准确率对比
"""

from __future__ import annotations

import argparse
import json
import os
import sys

import numpy as np

try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    HAS_MATPLOTLIB = True
except ImportError:
    HAS_MATPLOTLIB = False

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

from probing import config


PLOT_DIR = config.PLOT_OUTPUT_DIR


def set_rc_params():
    """设置 matplotlib 的 rcParams，用于出版质量的图表"""
    if not HAS_MATPLOTLIB:
        return
    plt.rcParams.update({
        "font.size": 12,
        "font.family": "sans-serif",
        "axes.linewidth": 1.5,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "xtick.major.width": 1.5,
        "ytick.major.width": 1.5,
        "xtick.labelsize": 10,
        "ytick.labelsize": 10,
    })


def plot_p_probe_accuracy_matrix(
    results: dict[str, dict],
    title: str = "P-Probing 准确率",
    save_path: str = None,
) -> None:
    """
    绘制 P-Probing 各属性的准确率柱状图。

    复现论文图 5 的风格，展示 6 个属性的准确率。
    """
    if not HAS_MATPLOTLIB:
        print("[画图] matplotlib 不可用。安装方法: pip install matplotlib")
        return

    set_rc_params()

    attributes = config.ATTRIBUTE_NAMES
    accuracies = [results[attr]["accuracy"] for attr in attributes]

    # 中文显示名称
    short_names = [
        "出生日期", "出生城市", "大学",
        "专业", "公司", "公司城市",
    ]

    colors = ["#4C72B0", "#DD8452", "#55A868", "#C44E52", "#8172B3", "#937860"]

    fig, ax = plt.subplots(figsize=(10, 5))

    bars = ax.bar(range(len(attributes)), accuracies, color=colors, width=0.6, edgecolor="white", linewidth=0.5)

    # 在柱子上方添加数值标签
    for i, (bar, acc) in enumerate(zip(bars, accuracies)):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 0.01,
            f"{acc:.2f}",
            ha="center",
            va="bottom",
            fontsize=10,
        )

    ax.set_xticks(range(len(attributes)))
    ax.set_xticklabels(short_names, rotation=30, ha="right")
    ax.set_ylabel("准确率", fontsize=13)
    ax.set_title(title, fontsize=14, fontweight="bold")
    ax.set_ylim(0, 1.15)
    ax.axhline(y=1/12, color="red", linestyle="--", alpha=0.5, label="随机猜测 (1/12)")
    ax.legend(fontsize=10)

    plt.tight_layout()

    if save_path is None:
        save_path = os.path.join(PLOT_DIR, "p_probe_accuracy.png")
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, dpi=200, bbox_inches="tight")
    print(f"[画图] 已保存到: {save_path}")
    plt.close()


def plot_q_probe_vs_qa(
    q_probe_results: dict[str, dict],
    qa_results: dict[str, float],
    title: str = "Q-Probing 与 QA 微调准确率对比",
    save_path: str = None,
) -> None:
    """
    绘制 Q-Probing 准确率与 QA 微调准确率的对比。

    复现论文图 7 的风格。
    """
    if not HAS_MATPLOTLIB:
        print("[画图] matplotlib 不可用。")
        return

    set_rc_params()

    attributes = config.ATTRIBUTE_NAMES
    short_names = [
        "出生日期", "出生城市", "大学",
        "专业", "公司", "公司城市",
    ]

    q_probe_acc = [q_probe_results.get(attr, {}).get("accuracy", 0) for attr in attributes]
    qa_acc = [qa_results.get(attr, 0) for attr in attributes]

    x = np.arange(len(attributes))
    width = 0.35

    fig, ax = plt.subplots(figsize=(10, 5))

    bars1 = ax.bar(x - width/2, q_probe_acc, width, label="Q-Probing", color="#4C72B0", alpha=0.85)
    bars2 = ax.bar(x + width/2, qa_acc, width, label="QA 微调", color="#DD8452", alpha=0.85)

    ax.set_xticks(x)
    ax.set_xticklabels(short_names, rotation=30, ha="right")
    ax.set_ylabel("准确率", fontsize=13)
    ax.set_title(title, fontsize=14, fontweight="bold")
    ax.set_ylim(0, 1.15)
    ax.legend(fontsize=12)

    # 数值标签
    for bar in bars1 + bars2:
        height = bar.get_height()
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            height + 0.01,
            f"{height:.2f}",
            ha="center",
            va="bottom",
            fontsize=9,
        )

    plt.tight_layout()

    if save_path is None:
        save_path = os.path.join(PLOT_DIR, "q_probe_vs_qa.png")
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, dpi=200, bbox_inches="tight")
    print(f"[画图] 已保存到: {save_path}")
    plt.close()


def plot_multi_variant_comparison(
    all_results: dict[str, dict[str, dict]],
    mode: str = "q",
    title: str = None,
    save_path: str = None,
) -> None:
    """
    绘制多变体之间的准确率对比。

    Args:
        all_results: {变体名: {属性名: {"accuracy": float}}}
        mode: "p" 或 "q"
    """
    if not HAS_MATPLOTLIB:
        print("[画图] matplotlib 不可用。")
        return

    set_rc_params()

    attributes = config.ATTRIBUTE_NAMES
    short_names = [
        "出生日期", "出生城市", "大学",
        "专业", "公司", "公司城市",
    ]

    variants = list(all_results.keys())
    n_variants = len(variants)
    n_attrs = len(attributes)

    x = np.arange(n_attrs)
    group_width = 0.8
    bar_width = group_width / max(n_variants, 1)

    # 颜色调色板
    colors = [
        "#4C72B0", "#DD8452", "#55A868", "#C44E52",
        "#8172B3", "#937860", "#CCB974", "#64B5CD",
    ]

    fig, ax = plt.subplots(figsize=(12, 5))

    for i, (variant, color) in enumerate(zip(variants, colors)):
        results = all_results[variant]
        accs = [results.get(attr, {}).get("accuracy", 0) if isinstance(results.get(attr), dict)
                else results.get(attr, 0) for attr in attributes]
        offset = (i - (n_variants - 1) / 2) * bar_width
        ax.bar(x + offset, accs, bar_width, label=variant, color=color, alpha=0.85)

    ax.set_xticks(x)
    ax.set_xticklabels(short_names, rotation=30, ha="right")
    ax.set_ylabel("准确率", fontsize=13)
    ax.set_title(title or f"{mode.upper()}-Probing 多变体准确率对比", fontsize=14, fontweight="bold")
    ax.set_ylim(0, 1.15)
    ax.legend(fontsize=10, loc="upper right")

    plt.tight_layout()

    if save_path is None:
        save_path = os.path.join(PLOT_DIR, f"multi_variant_{mode}.png")
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, dpi=200, bbox_inches="tight")
    print(f"[画图] 已保存到: {save_path}")
    plt.close()


def load_results_from_files(result_dir: str) -> dict:
    """从目录加载所有 results JSON 文件"""
    results = {}
    for f in sorted(os.listdir(result_dir)):
        if f.endswith(".json"):
            path = os.path.join(result_dir, f)
            with open(path) as fp:
                data = json.load(fp)
            variant = data.get("variant", f.replace(".json", ""))
            results[variant] = data.get("results", data)
    return results


def main():
    parser = argparse.ArgumentParser(description="Probing 结果可视化")
    parser.add_argument(
        "--results_dir",
        type=str,
        default=None,
        help="包含 results JSON 文件的目录",
    )
    parser.add_argument(
        "--result_file",
        type=str,
        default=None,
        help="单个 results JSON 文件",
    )
    parser.add_argument(
        "--mode",
        type=str,
        choices=["p", "q"],
        default="q",
        help="用于绘图的 probe 模式",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default=None,
        help="图表输出目录",
    )
    args = parser.parse_args()

    output_dir = args.output_dir or PLOT_DIR
    os.makedirs(output_dir, exist_ok=True)

    if not HAS_MATPLOTLIB:
        print("错误：绘图需要 matplotlib，请安装: pip install matplotlib")
        sys.exit(1)

    if args.result_file:
        with open(args.result_file) as f:
            data = json.load(f)
        results = data.get("results", data)
        variant = data.get("variant", "unknown")
        mode_label = "P" if args.mode == "p" else "Q"
        title = f"{mode_label}-Probing 准确率: {variant}"
        plot_p_probe_accuracy_matrix(results, title=title)

    elif args.results_dir:
        all_results = load_results_from_files(args.results_dir)
        if not all_results:
            print(f"在 {args.results_dir} 中未找到 results JSON 文件")
            return

        # 单独绘图
        for variant, results in all_results.items():
            mode_label = "P" if args.mode == "p" else "Q"
            title = f"{mode_label}-Probing 准确率: {variant}"
            save_path = os.path.join(output_dir, f"{args.mode}_probe_{variant}.png")
            plot_p_probe_accuracy_matrix(results, title=title, save_path=save_path)

        # 综合对比图
        plot_multi_variant_comparison(
            all_results,
            mode=args.mode,
            save_path=os.path.join(output_dir, f"multi_variant_{args.mode}.png"),
        )

    else:
        print("未提供结果数据。请指定 --result_file 或 --results_dir。")


if __name__ == "__main__":
    main()
