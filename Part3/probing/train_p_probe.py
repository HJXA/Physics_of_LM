"""
P-Probing 训练脚本。

在预训练语言模型上训练线性探针，从传记条目中特殊 token 位置
的隐藏状态预测属性值。

用法:
    python -m probing.train_p_probe \
        --variant bioS_single \
        --checkpoint /path/to/checkpoint \
        --output_dir /path/to/output

    # 或运行所有变体:
    python -m probing.train_p_probe --all_variants
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time

import numpy as np
import torch
from torch.optim import AdamW
from torch.utils.data import DataLoader

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

from transformers import AutoModelForCausalLM, AutoTokenizer

from probing import config
from probing.data import PProbeDataset, p_probe_collator
from probing.model import ProbeModel

try:
    import swanlab
    HAS_SWANLAB = True
except ImportError:
    HAS_SWANLAB = False


def main(args: argparse.Namespace) -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    cfg = config.P_PROBE

    os.makedirs(args.output_dir, exist_ok=True)

    print(f"[P-Probe] 正在加载分词器和模型...")
    tokenizer = AutoTokenizer.from_pretrained(args.checkpoint, use_fast=True)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token

    base_model = AutoModelForCausalLM.from_pretrained(
        args.checkpoint,
        torch_dtype=torch.bfloat16,
    )
    base_model.to(device)
    print(f"[P-Probe] 模型已加载: {args.checkpoint}")
    print(f"[P-Probe] 隐藏层维度: {base_model.config.hidden_size}")

    # 创建 probe 模型
    probe = ProbeModel(
        base_model=base_model,
        lora_rank=cfg["lora_rank"],
        frozen=True,
    ).to(device)
    print(f"[P-Probe] LoRA 秩: {cfg['lora_rank']}")
    print(f"[P-Probe] 参数量: {sum(p.numel() for p in probe.parameters()) / 1e6:.2f}M "
          f"(可训练: {sum(p.numel() for p in probe.parameters() if p.requires_grad) / 1e6:.2f}M)")

    # 加载数据集
    print(f"[P-Probe] 正在加载数据集: {args.variant}")
    dataset = PProbeDataset(
        variant_name=args.variant,
        tokenizer=tokenizer,
        max_seq_length=cfg["max_seq_length"],
        max_samples=args.max_samples,
    )
    print(f"[P-Probe] 数据集大小: {len(dataset)}")

    # DataLoader
    sampler = torch.utils.data.RandomSampler(dataset)
    dataloader = DataLoader(
        dataset,
        batch_size=cfg["batch_size"],
        sampler=sampler,
        collate_fn=p_probe_collator,
        num_workers=2,
        pin_memory=True,
    )

    # 优化器 — 仅 LoRA 和分类器参数可训练
    optimizer = AdamW(
        [p for p in probe.parameters() if p.requires_grad],
        lr=cfg["learning_rate"],
        weight_decay=cfg["weight_decay"],
        eps=cfg["adam_epsilon"],
    )

    # 学习率调度器：线性衰减至 0
    scheduler = torch.optim.lr_scheduler.LinearLR(
        optimizer,
        start_factor=1.0,
        end_factor=1e-6,
        total_iters=cfg["max_steps"],
    )

    # SwanLab 初始化
    if HAS_SWANLAB and args.swanlab:
        swanlab.init(
            project=f"probing_p_{args.variant}",
            experiment_name=f"{args.variant}_{time.strftime('%H%M%S')}",
        )

    # === 训练循环 ===
    print(f"[P-Probe] 开始训练，共 {cfg['max_steps']} 步...")
    probe.train()

    step = 0
    running_loss = 0.0
    best_acc = {attr: 0.0 for attr in config.ATTRIBUTE_NAMES}

    while step < cfg["max_steps"]:
        for batch in dataloader:
            if step >= cfg["max_steps"]:
                break

            input_ids = batch["input_ids"].to(device, non_blocking=True)
            attention_mask = batch["attention_mask"].to(device, non_blocking=True)
            positions = batch["positions"]
            labels = {
                attr_name: lab.to(device)
                for attr_name, lab in batch["labels"].items()
            }

            outputs = probe(
                input_ids=input_ids,
                attention_mask=attention_mask,
                positions=positions,
                mode="p",
                labels=labels,
            )

            loss = outputs["loss"]
            loss.backward()
            running_loss += loss.item()

            # 梯度裁剪
            torch.nn.utils.clip_grad_norm_(
                [p for p in probe.parameters() if p.requires_grad],
                max_norm=1.0,
            )

            optimizer.step()
            optimizer.zero_grad()
            scheduler.step()
            step += 1

            # 日志
            if step % cfg["log_interval"] == 0:
                avg_loss = running_loss / cfg["log_interval"]
                print(f"  第 {step} 步: loss={avg_loss:.4f}")

                # 在训练数据子集上快速评估
                probe.eval()
                eval_size = min(1000, len(dataset))
                eval_indices = torch.randperm(len(dataset))[:eval_size]
                eval_subset = torch.utils.data.Subset(dataset, eval_indices.tolist())
                eval_loader = DataLoader(
                    eval_subset,
                    batch_size=cfg["batch_size"],
                    collate_fn=p_probe_collator,
                )

                all_labels_eval = {attr: [] for attr in config.ATTRIBUTE_NAMES}
                all_preds_eval = {attr: [] for attr in config.ATTRIBUTE_NAMES}

                with torch.no_grad():
                    for ebatch in eval_loader:
                        e_input_ids = ebatch["input_ids"].to(device)
                        e_attention_mask = ebatch["attention_mask"].to(device)
                        e_positions = ebatch["positions"]
                        e_labels = ebatch["labels"]

                        e_outputs = probe.forward(
                            e_input_ids,
                            attention_mask=e_attention_mask,
                            positions=e_positions,
                            mode="p",
                            labels=None,
                        )

                        for attr in config.ATTRIBUTE_NAMES:
                            if attr in e_labels:
                                lbl = e_labels[attr]
                                valid = lbl >= 0
                                if valid.any():
                                    pred = e_outputs["logits"][attr][valid].argmax(dim=-1)
                                    all_preds_eval[attr].append(pred.cpu())
                                    all_labels_eval[attr].append(lbl[valid])

                accuracies = {}
                with torch.no_grad():
                    for attr in config.ATTRIBUTE_NAMES:
                        if all_labels_eval[attr]:
                            preds_t = torch.cat(all_preds_eval[attr])
                            labels_t = torch.cat(all_labels_eval[attr])
                            acc = (preds_t == labels_t).float().mean().item()
                            accuracies[attr] = acc
                            if acc > best_acc[attr]:
                                best_acc[attr] = acc

                avg_eval_acc = np.mean(list(accuracies.values())) if accuracies else 0.0
                print(f"  第 {step} 步: 评估准确率={avg_eval_acc:.4f} | {json.dumps(accuracies, indent=2)}")

                if HAS_SWANLAB and args.swanlab:
                    swanlab.log(
                        {
                            "loss": avg_loss,
                            "eval_acc_mean": avg_eval_acc,
                            **{f"eval_acc/{k}": v for k, v in accuracies.items()},
                            **{f"best_acc/{k}": v for k, v in best_acc.items()},
                            "learning_rate": scheduler.get_last_lr()[0],
                        },
                        step=step,
                    )

                running_loss = 0.0
                probe.train()

            # 保存检查点
            if step % cfg["save_interval"] == 0:
                save_path = os.path.join(args.output_dir, f"step_{step}.pt")
                torch.save(
                    {
                        "step": step,
                        "state_dict": probe.state_dict(),
                        "variant": args.variant,
                        "best_acc": best_acc,
                    },
                    save_path,
                )
                print(f"  已保存 probe 到 {save_path}")

    # 最终保存
    final_path = os.path.join(args.output_dir, "final.pt")
    torch.save(
        {
            "step": step,
            "state_dict": probe.state_dict(),
            "variant": args.variant,
            "best_acc": best_acc,
        },
        final_path,
    )
    print(f"\n[P-Probe] 训练完成。最终 probe 已保存到 {final_path}")
    print(f"[P-Probe] 最佳准确率: {json.dumps(best_acc, indent=2)}")


def parse_args():
    parser = argparse.ArgumentParser(description="P-Probing 训练")
    parser.add_argument(
        "--variant",
        type=str,
        default="bioS_single",
        help="数据集变体（如 bioS_single, bioS_multi 等）",
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        default=None,
        help="预训练模型检查点路径",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default=None,
        help="保存 probe 权重的目录",
    )
    parser.add_argument(
        "--max_samples",
        type=int,
        default=None,
        help="从数据集中使用的最大样本数（None = 全部）",
    )
    parser.add_argument(
        "--swanlab",
        action="store_true",
        help="启用 SwanLab 日志记录",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=None,
        help="覆盖默认 batch size",
    )
    parser.add_argument(
        "--max_steps",
        type=int,
        default=None,
        help="覆盖默认最大训练步数",
    )
    args = parser.parse_args()

    if args.checkpoint is None:
        args.checkpoint = config.BASE_MODEL_DIR
    if args.output_dir is None:
        args.output_dir = os.path.join(config.PROBE_OUTPUT_DIR, f"p_probe_{args.variant}")
    if args.batch_size is not None:
        config.P_PROBE["batch_size"] = args.batch_size
    if args.max_steps is not None:
        config.P_PROBE["max_steps"] = args.max_steps

    return args


if __name__ == "__main__":
    args = parse_args()
    main(args)
