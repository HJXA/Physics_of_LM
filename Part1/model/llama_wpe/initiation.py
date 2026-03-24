import os
import sys

import torch
from transformers import AutoConfig

current_dir = os.path.dirname(os.path.abspath(__file__))
if current_dir not in sys.path:
    sys.path.append(current_dir)

from modeling_llama_wpe import LlamaForCausalLMWPE
sys.path.append("/ruilab/jxhe/CoE_Monitor/Physics_of_LM/Part1/model/llama")  # 将上一级目录加入路径，以便导入 configuration_llama

from configuration_llama import LlamaConfig


SAVE_BASE_DIR = "/ruilab/jxhe/CoE_Monitor/Physics_of_LM/Part1/checkpoints/llama_Init"


def _fallback_config() -> LlamaConfig:
    return LlamaConfig(
        vocab_size=6,
        hidden_size=768,
        intermediate_size=768*4,
        num_hidden_layers=12,
        num_attention_heads=12,
        num_key_value_heads=12,
        max_position_embeddings=1024,
        pad_token_id=5,
        bos_token_id=0,
        eos_token_id=4,
    )


def init_all_variants():
    try:
        config = AutoConfig.from_pretrained(BASE_CONFIG_PATH)
    except Exception:
        config = _fallback_config()

    variants = ["pos"]

    for v_type in variants:
        print("=" * 50)
        print(f"开始执行随机初始化变体: [{v_type}] ...")

        torch.manual_seed(42)
        model = LlamaForCausalLMWPE(config)

        params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print(f"可训练参数量: {params}")

        save_path = os.path.join(SAVE_BASE_DIR, f"llama_{v_type}")
        os.makedirs(save_path, exist_ok=True)

        model.save_pretrained(save_path)
        config.save_pretrained(save_path)

        print(f"✅ 模型 [{v_type}] 初始化与保存成功！路径: {save_path}")
        print("=" * 50 + "\n")


if __name__ == "__main__":
    init_all_variants()
