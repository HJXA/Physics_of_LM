"""
批量将 LoRA adapter checkpoint 合并到原模型中。

用法:
    python merge_lora_checkpoints.py <lora_checkpoint_dir>

示例:
    python merge_lora_checkpoints.py \
        /ruilab/jxhe/CoE_Monitor/Physics_of_LM/Part3/checkpoints/QA/bioS_multi/lora_llama2_2026_04_06_11_18_57_lr3e-04_wd1e-02_rank_embed128_rank_qv8_2026_04_08_22_18_35

逻辑:
    1. 读取 lora_checkpoint_dir 下的 adapter_config.json，获取 base_model_name_or_path
    2. 扫描所有 checkpoint-*/ 子目录
    3. 对每个 checkpoint，加载 adapter + base_model，合并后保存到 <lora_checkpoint_dir>_merged/checkpoint-*/
"""

import os
import sys
import re
import glob
import argparse
from pathlib import Path


def find_checkpoints(lora_dir: str) -> list[str]:
    """扫描目录下所有 checkpoint-* 子文件夹，按 step 数字排序。"""
    pattern = os.path.join(lora_dir, "checkpoint-*")
    dirs = glob.glob(pattern)
    dirs = [d for d in dirs if os.path.isdir(d)]

    def extract_step(d):
        name = os.path.basename(d)
        m = re.match(r"checkpoint-(\d+)", name)
        return int(m.group(1)) if m else 0

    dirs.sort(key=extract_step)
    return dirs


def merge_single_checkpoint(adapter_path: str, base_model_path: str, output_path: str):
    """将单个 adapter checkpoint 合并到 base model 并保存。"""
    from peft import PeftModel
    from transformers import AutoModelForCausalLM, AutoTokenizer

    print(f"  加载 base model: {base_model_path}")
    base_model = AutoModelForCausalLM.from_pretrained(
        base_model_path,
        torch_dtype="auto",
        device_map="cpu",
    )

    print(f"  加载 adapter: {adapter_path}")
    model = PeftModel.from_pretrained(base_model, adapter_path)

    print(f"  合并权重...")
    merged_model = model.merge_and_unload()

    os.makedirs(output_path, exist_ok=True)
    print(f"  保存到: {output_path}")
    merged_model.save_pretrained(output_path, safe_serialization=True)

    # 复制 tokenizer 文件
    tokenizer_files = ["tokenizer.json", "tokenizer_config.json", "special_tokens_map.json"]
    for fname in tokenizer_files:
        src = os.path.join(base_model_path, fname)
        dst = os.path.join(output_path, fname)
        if os.path.exists(src) and not os.path.exists(dst):
            import shutil
            shutil.copy2(src, dst)

    # 释放显存/内存
    del merged_model
    del model
    del base_model
    import torch
    torch.cuda.empty_cache()

    print(f"  ✓ 完成")


def main():
    parser = argparse.ArgumentParser(description="批量合并 LoRA adapter checkpoints 到原模型")
    parser.add_argument("--lora_dir", type=str, help="包含多个 checkpoint-* 子目录的 LoRA 训练输出目录")
    parser.add_argument("--base_model", type=str, default=None,
                        help="原模型路径（默认从 adapter_config.json 中读取 base_model_name_or_path）")
    parser.add_argument("--output_dir", type=str, default=None,
                        help="合并后保存的根目录（默认为 <lora_dir>_merged）")
    parser.add_argument("--skip_existing", action="store_true",
                        help="跳过已存在的 checkpoint 目录")
    args = parser.parse_args()

    lora_dir = os.path.abspath(args.lora_dir)
    if not os.path.isdir(lora_dir):
        print(f"错误: 目录不存在: {lora_dir}")
        sys.exit(1)

    # 读取 base_model_name_or_path
    adapter_config_path = os.path.join(lora_dir, "adapter_config.json")
    if not os.path.exists(adapter_config_path):
        print(f"错误: 未找到 adapter_config.json: {adapter_config_path}")
        sys.exit(1)

    import json
    with open(adapter_config_path, "r") as f:
        adapter_config = json.load(f)

    base_model_path = args.base_model or adapter_config.get("base_model_name_or_path")
    if not base_model_path:
        print("错误: 无法确定 base model 路径，请通过 --base_model 指定")
        sys.exit(1)

    if not os.path.isdir(base_model_path):
        print(f"错误: base model 目录不存在: {base_model_path}")
        sys.exit(1)

    output_dir = args.output_dir or lora_dir + "_merged"
    os.makedirs(output_dir, exist_ok=True)

    # 扫描 checkpoints
    checkpoints = find_checkpoints(lora_dir)
    if not checkpoints:
        print(f"错误: 在 {lora_dir} 中未找到任何 checkpoint-* 子目录")
        sys.exit(1)

    print(f"LoRA 目录:       {lora_dir}")
    print(f"Base Model:      {base_model_path}")
    print(f"输出目录:        {output_dir}")
    print(f"待合并 checkpoints: {len(checkpoints)} 个")
    for cp in checkpoints:
        print(f"  - {os.path.basename(cp)}")
    print()

    # 逐个合并
    for i, ckpt_path in enumerate(checkpoints, 1):
        ckpt_name = os.path.basename(ckpt_path)
        out_path = os.path.join(output_dir, ckpt_name)

        if args.skip_existing and os.path.exists(out_path) and os.listdir(out_path):
            print(f"[{i}/{len(checkpoints)}] 跳过已存在: {ckpt_name}")
            continue

        print(f"[{i}/{len(checkpoints)}] 合并 {ckpt_name} ...")
        merge_single_checkpoint(ckpt_path, base_model_path, out_path)
        print()

    print("全部完成！")


if __name__ == "__main__":
    main()
