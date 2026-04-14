# Probing 实验：位置探测与查询探测

本模块复现了 *Physics of Language Models* 论文（第 5.1–5.2 节）中的**位置探测（P-Probing）**和**查询探测（Q-Probing）**实验。这些实验探究了语言模型在合成传记（bioS）数据上预训练后，知识是如何编码在隐藏状态中的。

## 概述

| 方面 | P-Probing（位置探测） | Q-Probing（查询探测） |
|------|----------------------|----------------------|
| **输入** | 完整的传记条目（6 句话） | 仅人名（如 "Alexandria Evan Martin."） |
| **探测位置** | 6 个特殊 token 位置（每个属性值前的介词） | 人名后的结束 token（`.`） |
| **Embedding LoRA 秩** | 2 | 16 |
| **测量目标** | 属性在文本的哪些位置被编码 | 属性是否直接关联到人名 |

两种方法都**冻结预训练的 GPT 模型**，仅训练一个低秩嵌入层更新 + 每个属性的线性分类器（带 BatchNorm）。

## 文件结构

```
probing/
├── __init__.py
├── config.py              # 共享常量：属性定义、标签空间、超参数、路径
├── data.py                # PProbeDataset 与 QProbeDataset 数据集构建、分词、collator
├── model.py               # ProbeModel：冻结 GPT + LoRA embedding + 线性分类器
├── train_p_probe.py       # P-Probing 训练脚本（单次运行单个 bioS 变体）
├── train_q_probe.py       # Q-Probing 训练脚本（单次运行单个 bioS 变体）
├── evaluate.py            # 评估训练好的 probe，输出各属性准确率
├── plot_results.py        # 生成准确率柱状图与多变体对比图
└── run_all.sh             # 一键脚本：训练 → 评估 → 画图（单个变体）
```

## 快速开始

### 单个变体（推荐用于测试）

```bash
cd /ruilab/jxhe/CoE_Monitor/Physics_of_LM/Part3

# 使用基础模型（未预训练的 Llama2，42M 参数）
python -m probing.train_q_probe \
    --variant bioS_single \
    --checkpoint checkpoints/llama2 \
    --max_steps 5000

# 使用预训练后的 checkpoint
python -m probing.train_q_probe \
    --variant bioS_single \
    --checkpoint checkpoints/bioS_single/step-00005000/lit_model.pth \
    --max_steps 30000
```

### 完整流程（训练 + 评估 + 画图）

```bash
cd Part3/probing
bash run_all.sh bioS_single checkpoints/llama2
```

### 单独评估

```bash
python -m probing.evaluate \
    --probe_mode q \
    --probe_path probing/probe_weights/q_probe_bioS_single/final.pt \
    --checkpoint checkpoints/llama2 \
    --eval_variant bioS_single \
    --output_file results/q_probe_bioS_single.json
```

### 生成可视化

```bash
python -m probing.plot_results --result_file results/q_probe_bioS_single.json --mode q
```

## 运行后生成的目录结构

```
probing/
├── probe_weights/
│   ├── q_probe_bioS_single/    # 每个变体的 probe 检查点
│   │   ├── step_5000.pt
│   │   ├── ...
│   │   └── final.pt
│   └── p_probe_bioS_single/
├── results/
│   ├── q_probe_bioS_single.json
│   └── p_probe_bioS_single.json
└── plots/
    ├── q_probe_bioS_single.png
    └── p_probe_bioS_single.png
```

## 核心超参数

| 参数 | P-Probing | Q-Probing |
|------|-----------|-----------|
| Embedding LoRA 秩 | 2 | 16 |
| Batch size | 200 | 200 |
| 最大训练步数 | 30,000 | 30,000 |
| 学习率 | 1e-3 | 1e-3 |
| Weight decay | 0.3 | 0.3 |
| LR 调度器 | 线性衰减至 0 | 线性衰减至 0 |
| 优化器 | AdamW | AdamW |

## 架构细节

```
冻结的 GPT 模型                      可训练组件
┌──────────────────────┐            ┌──────────────────────────┐
│ Embedding（冻结）     │            │ LoRA-A: (V, r)           │
│ Transformer（冻结）   │  ──────►   │ LoRA-B: (r, D)           │
│ LM Head（不使用）     │            └──────────┬───────────────┘
└──────────────────────┘                       │
                                               ▼
                                ┌──────────────────────────┐
                                │  隐藏状态提取              │
                                │  （在指定位置或 END token）│
                                └──────────┬───────────────┘
                                           │
                                ┌──────────▼───────────────┐
                                │  BatchNorm1d(D)          │
                                │  Linear(D, num_classes)  │  × 6（每个属性一个）
                                └──────────────────────────┘
```

## 6 个属性

| 属性 | 问题示例 | 标签空间大小 |
|------|----------|-------------|
| Birth Date（出生日期） | "What is the birth date of {name}?" | 12（月份） |
| Birth City（出生城市） | "What is the birth city of {name}?" | 201（城市） |
| University（大学） | "Which university did {name} study?" | 301（大学） |
| Major（专业） | "What major did {name} study?" | 101（专业领域） |
| Company（公司） | "Which company did {name} work for?" | 263（公司） |
| Company City（公司城市） | "Where did {name} work?" | 201（城市） |

## 论文中的预期结果

在不同 bioS 变体上预训练后，Q-Probing 准确率应呈现以下趋势：

- **bioS_single**（无增强）：~2-10% 准确率 —— 模型不直接将属性关联到人名
- **bioS_single+permute**（句子重排）：~70% —— 一定程度的重排有帮助
- **bioS_multi**（模板重采样）：~41-60% —— 模板多样性有帮助
- **bioS_multi+permute**（完整增强）：~90%+ —— 模型学会将属性直接关联到人名

## 使用 SwanLab 记录实验

在训练命令中加上 `--swanlab` 即可自动记录 loss、各属性准确率和最佳准确率：

```bash
python -m probing.train_q_probe \
    --variant bioS_single \
    --checkpoint checkpoints/llama2 \
    --swanlab
```
