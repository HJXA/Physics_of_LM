"""Q-probing 工具函数模块。

包含与训练核心逻辑无关的通用工具函数和类，供 q_probing.py 调用。
"""

# 训练超参默认值（论文设定）
DEFAULT_TRAIN_CONFIG = {
    "max_steps": 30000,               # 最大训练步数（论文设定）
    "learning_rate": 0.001,           # 初始学习率
    "weight_decay": 0.3,              # 权重衰减，防止过拟合
    "adam_beta1": 0.9,               # Adam 一阶矩估计的衰减率
    "adam_beta2": 0.98,              # Adam 二阶矩估计的衰减率（论文推荐值）
    "adam_epsilon": 1e-6,            # Adam 数值稳定项
    "warmup_steps": 0,                # 无预热步数（论文设定）
    "lr_scheduler_type": "linear",   # 学习率线性衰减到 0
    "per_device_train_batch_size": 200,  # 每 GPU 的 batch size（论文设定）
    "gradient_accumulation_steps": 1,    # 梯度累积步数
    "logging_steps": 100,             # 每 100 步记录一次日志
    "save_steps": 2000,               # 每 2000 步保存一次 checkpoint
    "eval_steps": 2000,                # 每 2000 步做一次评估
    "bf16": True,                      # 使用 bfloat16 混合精度训练
    "fp16": False,                     # 不使用 float16（LLaMA 适合用 bf16）
}


def get_train_config(is_test: bool, base_config: dict | None = None) -> dict:
    """根据是否为测试模式返回训练配置。

    Args:
        is_test: 是否为测试模式
        base_config: 基础配置字典，默认使用 DEFAULT_TRAIN_CONFIG

    测试模式下大幅缩减训练规模，方便快速验证代码逻辑：
    - max_steps: 30000 → 3
    - batch_size: 200 → 4
    - save_steps: 2000 → 10
    """
    if base_config is None:
        base_config = DEFAULT_TRAIN_CONFIG
    config = dict(base_config)
    if is_test:
        config["max_steps"] = 3
        config["save_steps"] = 10
        config["per_device_train_batch_size"] = 4
        config["gradient_accumulation_steps"] = 1
    return config


def format_scientific(value: float) -> str:
    """将浮点数格式化为科学计数法字符串（如 0.001 → '1e-03'），用于输出目录命名。"""
    return format(value, ".0e")


def build_output_dir_tag(train_config: dict, lora_rank: int) -> str:
    """构建输出目录的标签，包含关键超参信息，便于实验管理和对比。

    Args:
        train_config: 训练配置字典
        lora_rank: LoRA rank 值

    格式示例: lr1e-03_wd3e-01_rank128
    """
    return f"lr{format_scientific(train_config['learning_rate'])}_wd{format_scientific(train_config['weight_decay'])}_rank{lora_rank}"


class LabelEncoder:
    """字符串 label → 整数索引的编码器。

    功能：
    - 将类别字符串映射为 0 ~ n_classes-1 的整数索引
    - 支持反向解码（整数 → 字符串）
    - 未知 label 映射到 -1（用于后续过滤）

    为什么需要 label encoder？
    - 分类模型需要数值标签，而原始数据中的标签是字符串（如 "1970-01-01"）
    - sorted(set(labels)) 保证映射关系稳定可复现
    """

    def __init__(self, labels: list[str]):
        """从标签列表构建映射。

        Args:
            labels: 所有标签的列表（可含重复），会自动去重并排序
        """
        unique = sorted(set(labels))  # 排序保证映射稳定
        self.label2id = {l: i for i, l in enumerate(unique)}  # 字符串 → 整数
        self.id2label = {i: l for l, i in self.label2id.items()}  # 整数 → 字符串
        self.n_classes = len(unique)  # 类别总数

    def encode(self, label: str) -> int:
        """将字符串标签编码为整数索引，未知标签返回 -1。"""
        return self.label2id.get(label, -1)

    def decode(self, idx: int) -> str:
        """将整数索引解码为字符串标签，未知索引返回 '<UNK>'。"""
        return self.id2label.get(idx, "<UNK>")
