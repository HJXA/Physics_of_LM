"""
Probing 模型评估脚本。

评估已保存的 probe 检查点在测试数据上的表现，输出
各属性的准确率矩阵。支持 P-Probing 和 Q-Probing。

用法:
    # 评估已训练的 P-Probe:
    python -m probing.evaluate --probe_mode p \
        --probe_path probing/probe_weights/p_probe_bioS_single/final.pt \
        --checkpoint /path/to/base/model

    # 评估 Q-Probe:
    python -m probing.evaluate --probe_mode q \
        --probe_path probing/probe_weights/q_probe_bioS_single/final.pt \
        --eval_variant bioS_single

    # 使用预训练 checkpoint 从头训练并评估 Q-Probe:
    python -m probing.evaluate --probe_mode q \
        --checkpoint checkpoints/bioS_multi/step-00005000/lit_model.pth \
        --train_variant bioS_multi
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

from transformers import AutoModelForCausalLM, AutoTokenizer

from probing import config
from probing.data import PProbeDataset, QProbeDataset, p_probe_collator, q_probe_collator
from probing.model import ProbeModel


def evaluate_probe(
    probe: ProbeModel,
    dataloader: DataLoader,
    mode: str,
    device: torch.device,
    max_batches: int = None,
) -> dict[str, dict]:
    """
    评估已训练的 probe。

    返回：{属性名: {"accuracy": float, "total": int, "correct": int}}
    """
    probe.eval()

    per_attr = {attr: {"correct": 0, "total": 0} for attr in config.ATTRIBUTE_NAMES}

    num_batches = 0
    with torch.no_grad():
        for batch in dataloader:
            if max_batches is not None and num_batches >= max_batches:
                break

            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["labels"]

            if mode == "p":
                positions = batch["positions"]
                outputs = probe.forward(
                    input_ids,
                    attention_mask=attention_mask,
                    positions=positions,
                    mode="p",
                    labels=None,
                )
            else:
                end_token_pos = batch["end_token_pos"].to(device)
                outputs = probe.forward(
                    input_ids,
                    attention_mask=attention_mask,
                    end_token_pos=end_token_pos,
                    mode="q",
                    labels=None,
                )

            for attr in config.ATTRIBUTE_NAMES:
                if attr not in labels:
                    continue

                attr_labels = labels[attr].to(device)
                valid_mask = attr_labels >= 0
                valid_count = valid_mask.sum().item()

                if valid_count == 0:
                    continue

                preds = outputs["logits"][attr][valid_mask].argmax(dim=-1)
                correct = (preds == attr_labels[valid_mask]).sum().item()

                per_attr[attr]["correct"] += correct
                per_attr[attr]["total"] += valid_count

            num_batches += 1

    # 计算准确率
    results = {}
    for attr in config.ATTRIBUTE_NAMES:
        total = per_attr[attr]["total"]
        correct = per_attr[attr]["correct"]
        acc = correct / total if total > 0 else 0.0
        results[attr] = {
            "accuracy": acc,
            "correct": correct,
            "total": total,
        }

    return results


def run_evaluation(args: argparse.Namespace) -> dict:
    """执行给定参数的评估"""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 加载模型
    checkpoint_path = args.checkpoint or config.BASE_MODEL_DIR
    print(f"[评估] 正在加载模型: {checkpoint_path}")
    base_model = AutoModelForCausalLM.from_pretrained(
        checkpoint_path,
        torch_dtype=torch.bfloat16,
    )
    base_model.to(device)

    tokenizer = AutoTokenizer.from_pretrained(checkpoint_path, use_fast=True)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token

    # 创建 probe 模型
    probe_mode = args.probe_mode if hasattr(args, "probe_mode") else "q"
    lora_rank = config.P_PROBE["lora_rank"] if probe_mode == "p" else config.Q_PROBE["lora_rank"]

    probe = ProbeModel(
        base_model=base_model,
        lora_rank=lora_rank,
        frozen=True,
    ).to(device)

    # 加载 probe 权重（如果提供）
    if args.probe_path:
        state = torch.load(args.probe_path, map_location=device, weights_only=False)
        probe.load_state_dict(state["state_dict"])
        print(f"[评估] 已加载 probe 权重: {args.probe_path}")
    elif args.probe_dir:
        probe_path = os.path.join(args.probe_dir, "final.pt")
        if os.path.exists(probe_path):
            state = torch.load(probe_path, map_location=device, weights_only=False)
            probe.load_state_dict(state["state_dict"])
            print(f"[评估] 已加载 probe 权重: {probe_path}")
        else:
            print(f"[评估] 未在 {args.probe_dir} 中找到 probe 权重，使用随机初始化")

    # 加载评估数据集
    variant = args.eval_variant or "bioS_single"

    if probe_mode == "p":
        print(f"[评估] 正在加载 P-Probe 数据集: {variant}")
        dataset = PProbeDataset(
            variant_name=variant,
            tokenizer=tokenizer,
            max_seq_length=config.P_PROBE["max_seq_length"],
            max_samples=args.max_samples,
        )
        dataloader = DataLoader(
            dataset,
            batch_size=config.P_PROBE["batch_size"],
            collate_fn=p_probe_collator,
            num_workers=2,
            pin_memory=True,
        )
    else:
        dict_dir = os.path.join(config.DICT_DIR, variant)
        print(f"[评估] 正在加载 Q-Probe 数据集: {dict_dir}")
        dataset = QProbeDataset(
            dict_data_dir=dict_dir,
            tokenizer=tokenizer,
            max_seq_length=config.Q_PROBE["max_seq_length"],
            max_samples=args.max_samples,
        )
        dataloader = DataLoader(
            dataset,
            batch_size=config.Q_PROBE["batch_size"],
            collate_fn=q_probe_collator,
            num_workers=2,
            pin_memory=True,
        )

    # 执行评估
    results = evaluate_probe(
        probe=probe,
        dataloader=dataloader,
        mode=probe_mode,
        device=device,
        max_batches=args.max_batches,
    )

    # 打印结果
    print("\n" + "=" * 60)
    print(f"探测模式: {probe_mode.upper()}-Probing")
    print(f"数据集: {variant}")
    print(f"Probe: {args.probe_path or '随机初始化'}")
    print("=" * 60)
    for attr in config.ATTRIBUTE_NAMES:
        r = results[attr]
        print(f"  {attr:15s}: {r['accuracy']:.4f} ({r['correct']}/{r['total']})")

    avg_acc = np.mean([results[attr]["accuracy"] for attr in config.ATTRIBUTE_NAMES])
    print(f"  {'平均':15s}: {avg_acc:.4f}")
    print("=" * 60)

    # 保存结果
    if args.output_file:
        output = {
            "mode": probe_mode,
            "variant": variant,
            "probe_path": args.probe_path or "",
            "results": results,
            "average": avg_acc,
        }
        os.makedirs(os.path.dirname(os.path.abspath(args.output_file)), exist_ok=True)
        with open(args.output_file, "w") as f:
            json.dump(output, f, indent=2)
        print(f"结果已保存到: {args.output_file}")

    return results


def parse_args():
    parser = argparse.ArgumentParser(description="Probe 评估")
    parser.add_argument(
        "--probe_mode",
        type=str,
        choices=["p", "q"],
        default="q",
        help="P-Probing 或 Q-Probing 评估",
    )
    parser.add_argument(
        "--probe_path",
        type=str,
        default=None,
        help="已保存的 probe 检查点路径",
    )
    parser.add_argument(
        "--probe_dir",
        type=str,
        default=None,
        help="包含 probe 检查点的目录（自动查找 final.pt）",
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        default=None,
        help="基础模型检查点路径",
    )
    parser.add_argument(
        "--eval_variant",
        type=str,
        default=None,
        help="用于评估的数据集变体",
    )
    parser.add_argument(
        "--max_samples",
        type=int,
        default=None,
        help="评估的最大样本数",
    )
    parser.add_argument(
        "--max_batches",
        type=int,
        default=None,
        help="最大处理的 batch 数",
    )
    parser.add_argument(
        "--output_file",
        type=str,
        default=None,
        help="将结果保存为 JSON 到该路径",
    )
    args = parser.parse_args()

    if args.checkpoint is None:
        args.checkpoint = config.BASE_MODEL_DIR
    if args.eval_variant is None and args.probe_path:
        # 从 probe 路径推断变体
        variant_from_path = os.path.basename(os.path.dirname(args.probe_path))
        for prefix in ["p_probe_", "q_probe_"]:
            if variant_from_path.startswith(prefix):
                args.eval_variant = variant_from_path[len(prefix):]
                break
    if args.eval_variant is None:
        args.eval_variant = "bioS_single"

    return args


if __name__ == "__main__":
    args = parse_args()
    run_evaluation(args)
