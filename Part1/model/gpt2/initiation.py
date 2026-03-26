import os
import sys

# 将所需包所在目录加入系统路径以确保顺利导入
current_dir = os.path.dirname(os.path.abspath(__file__))
if current_dir not in sys.path:
    sys.path.append(current_dir)

from transformers import AutoConfig, AutoModelForCausalLM
from modeling_gpt2_variants import CustomGPT2LMHeadModel,GPT2Config

# 指向您基于标准的模型配置路径，以保留特定的 Vocab Size & Pad TokenID 设置
BASE_CONFIG_PATH = "/ruilab/jxhe/CoE_Monitor/checkpoints/GPT_2_Small"
SAVE_BASE_DIR = "/ruilab/jxhe/CoE_Monitor/Physics_of_LM/Part1/checkpoints/gpt_2_Init"

def compare_standard_with_base():
    print("=" * 50)
    print("开始比较基础模型 (Base Model) 与 Standard 变体模型的参数情况...")
    try:
        base_model = AutoModelForCausalLM.from_pretrained(BASE_CONFIG_PATH)
        config = AutoConfig.from_pretrained(BASE_CONFIG_PATH)
        standard_model = CustomGPT2LMHeadModel(config, gpt_type='standard')
    except Exception as e:
        print(f"模型加载失败: {e}")
        return

    base_params = dict(base_model.named_parameters())
    std_params = dict(standard_model.named_parameters())

    base_count = sum(p.numel() for p in base_params.values() if p.requires_grad)
    std_count = sum(p.numel() for p in std_params.values() if p.requires_grad)

    print(f"Base 模型可训练参数量:     {base_count}")
    print(f"Standard 模型可训练参数量: {std_count}")

    if base_count == std_count:
        print("✅ 参数总量完全一致！")
    else:
        print("❌ 参数总量不一致！")

    base_keys = set(base_params.keys())
    std_keys = set(std_params.keys())

    missing_in_std = base_keys - std_keys
    extra_in_std = std_keys - base_keys

    if not missing_in_std and not extra_in_std:
        print("✅ 参数层名称完全匹配！")
        shape_mismatch = False
        for k in base_keys:
            if base_params[k].shape != std_params[k].shape:
                print(f"   ❌ 形状不一致: {k} | Base shape: {base_params[k].shape} | Std shape: {std_params[k].shape}")
                shape_mismatch = True
        if not shape_mismatch:
            print("✅ 所有参数层形状完全对应一致！")
    else:
        print("❌ 参数层名称存在差异！")
        if missing_in_std:
            print(f"    Standard 变体中缺失的层: {missing_in_std}")
        if extra_in_std:
            print(f"    Standard 变体中多出来的层: {extra_in_std}")
    
    print("=" * 50 + "\n")


def init_all_variants():
    # 读入基础模型的配置
    try:
        config = GPT2Config.from_pretrained(BASE_CONFIG_PATH)
        # config.vocab_size = 6  # 强制设置 vocab_size=6 来匹配我们 CFG 数据集的特殊需求
    except Exception as e:
        print(f"Error loading config from {BASE_CONFIG_PATH}: {e}")
        print("请检查基础路径是否正确且含有 config.json 文件。")
        return

    # 涵盖所有的指定模型变体
    variants = ['standard']
    
    for v_type in variants:
        print("=" * 50)
        print(f"开始执行随机初始化变体: [{v_type}] ...")
        
        # 实例化时自带的 post_init() 机制会自动进行一套完整的随机初始化
        model = CustomGPT2LMHeadModel(config, gpt_type=v_type)
        
        # 检查参数及构建模型大小情况
        params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print(f"可训练参数量: {params}")
        
        # 建立保存的文件夹并落地模型文件
        save_path = os.path.join(SAVE_BASE_DIR, f"gpt_2_{v_type}")
        os.makedirs(save_path, exist_ok=True)
        
        # 导出为通用的预训练格式 (.bin / .safetensors 和 config.json)
        model.save_pretrained(save_path)
        config.save_pretrained(save_path)
        
        print(f"✅ 模型 [{v_type}] 初始化与保存成功！路径: {save_path}")
        print("=" * 50 + "\n")

if __name__ == '__main__':
    # 比较原始模型与标准版
    compare_standard_with_base()
    
    # 批量初始化并保存所有变体预训练权重
    init_all_variants()
