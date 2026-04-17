# Probing 数据集

## 概述

生成两种 probing 数据集，用于训练和评估探针模型从模型隐状态中提取属性信息的能力。

- **P-probing**：输入为完整传记文本，探针需要从模型对该传记的隐状态中提取属性。
- **Q-probing**：输入仅为人名，探针需要从模型对人名的隐状态中提取属性。

两者的输出均为 6 个属性标签：`birth_date`, `birthcity`, `university`, `field`, `company1name`, `company1city`。

## 使用方法

确保已运行过 `Capo_bioS_bioR/generate_bios_datasets.py` 生成了 `datasets/dict/bioS_single/` 和 `datasets/QA/`，然后：

```bash
cd /ruilab/jxhe/CoE_Monitor/Physics_of_LM/Part3
python probing/generate_datasets.py
```

## 输出结构

```
datasets/probing/
├── P-probing/
│   ├── train/data.parquet   # 50K 条，text=传记原文
│   └── test/data.parquet    # 50K 条
└── Q-probing/
    ├── train/data.parquet   # 50K 条，text=人名
    └── test/data.parquet    # 50K 条
```

## 数据格式

每条记录 7 个字段：

| 字段 | 说明 | P-probing 示例 | Q-probing 示例 |
|------|------|----------------|----------------|
| `text` | 输入 | "Alexandria Evan Martin celebrates another year of life on April 8, 1935..." | "Alexandria Evan Martin" |
| `birth_date` | 出生日期 | "April 8, 1935" | 同左 |
| `birthcity` | 出生城市 | "Miramar, FL" | 同左 |
| `university` | 毕业院校 | "University of Utah" | 同左 |
| `field` | 专业 | "Analytics" | 同左 |
| `company1name` | 公司名 | "Macy's" | 同左 |
| `company1city` | 公司城市 | "New York, NY" | 同左 |

## 划分规则

train/test 划分严格与 QA 数据集一致：从 `datasets/QA/train/` 和 `datasets/QA/test/` 中提取人名集合，同一人名在所有数据集中归属同一 split。