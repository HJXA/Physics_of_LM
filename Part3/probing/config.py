"""
P-Probing 与 Q-Probing 实验共享配置
"""

import os

# === 路径 ===
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# 基础模型 / 分词器
BASE_MODEL_DIR = os.path.join(BASE_DIR, "checkpoints", "llama2")
TOKENIZER_PATH = BASE_MODEL_DIR

# 数据集路径
QA_TEST_DIR = os.path.join(BASE_DIR, "datasets", "QA", "test")
QA_TRAIN_DIR = os.path.join(BASE_DIR, "datasets", "QA", "train")
TEXT_DIR = os.path.join(BASE_DIR, "datasets", "text")
DICT_DIR = os.path.join(BASE_DIR, "datasets", "dict")

# 输出目录
PROBE_OUTPUT_DIR = os.path.join(BASE_DIR, "probing", "probe_weights")
PLOT_OUTPUT_DIR = os.path.join(BASE_DIR, "probing", "plots")

# === 标签 ===
MONTHS = [
    "January", "February", "March", "April", "May", "June",
    "July", "August", "September", "October", "November", "December",
]

FIELDS_DIR = os.path.join(BASE_DIR, "Capo_bioS_bioR", "fields")


def load_labels(filename: str) -> list[str]:
    """从 fields/*.txt 文件加载标签列表"""
    filepath = os.path.join(FIELDS_DIR, filename)
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"标签文件未找到: {filepath}")
    with open(filepath, "r", encoding="utf-8") as f:
        return [line.strip() for line in f if line.strip()]


# 各属性对应的标签空间
LABEL_SPACES = {
    "birth_date":     MONTHS,
    "birth_city":     load_labels("city.txt"),
    "university":     load_labels("university.txt"),
    "major":          load_labels("field.txt"),
    "company":        load_labels("company.txt"),  # 仅取公司名称（不含城市）
    "company_city":   load_labels("city.txt"),
}

# QA parquet 文件名称映射
QA_FILES = {
    "birth_date":     "q1_birth_date",
    "birth_city":     "q2_birth_city",
    "university":     "q3_university",
    "major":          "q4_major",
    "company":        "q5_company",
    "company_city":   "q6_company_city",
}

ATTRIBUTE_NAMES = list(LABEL_SPACES.keys())

# === 超参数 ===

P_PROBE = {
    "lora_rank": 2,
    "batch_size": 200,
    "max_steps": 30000,
    "learning_rate": 1e-3,
    "weight_decay": 0.3,
    "adam_epsilon": 1e-6,
    "warmup_steps": 0,
    "max_seq_length": 512,
    "log_interval": 500,
    "eval_interval": 2000,
    "save_interval": 5000,
}

Q_PROBE = {
    "lora_rank": 16,
    "batch_size": 200,
    "max_steps": 30000,
    "learning_rate": 1e-3,
    "weight_decay": 0.3,
    "adam_epsilon": 1e-6,
    "warmup_steps": 0,
    "max_seq_length": 128,
    "log_interval": 500,
    "eval_interval": 2000,
    "save_interval": 5000,
}

# === bioS 数据集变体 ===
BIOS_VARIANTS = [
    "bioS_single",
    "bioS_single_fullname",
    "bioS_single_permute",
    "bioS_single_permute_fullname",
    "bioS_multi",
    "bioS_multi_fullname",
    "bioS_multi_permute",
    "bioS_multi_permute_fullname",
]

# 公司列存储格式为 "名称; 城市"，分类时只需名称部分
COMPANY_LABELS = None


def get_company_labels() -> list[str]:
    """获取纯公司名称标签列表（去掉城市部分）"""
    global COMPANY_LABELS
    if COMPANY_LABELS is None:
        raw = load_labels("company.txt")
        # company.txt 格式: "Company Name; City"
        COMPANY_LABELS = [x.split(";")[0].strip() for x in raw]
    return COMPANY_LABELS
