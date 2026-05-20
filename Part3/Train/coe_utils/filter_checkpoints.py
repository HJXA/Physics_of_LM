import json
import os
import shutil
import sys

import torch
from transformers import AutoConfig, AutoModelForCausalLM

TARGET_TOKENIZER_CLASS = "PreTrainedTokenizerFast"

CHECKPOINT_ROOT = ""
SAVE_ROOT = ""
MODEL_NAME = ""
STEP_START = 0
STEP_SIZE = 0
MODIFY_TOKENIZER_CONFIG = False
SKIP_IF_EXISTS = True
DEST_PREFIX = ""
CHECK_AND_FIX_TIE_WORD_EMBEDDINGS = False


def check_and_fix_tied_embeddings(model_dir: str) -> None:
    try:
        config = AutoConfig.from_pretrained(model_dir, trust_remote_code=True)
    except Exception as e:
        print(f"  [Warn] 读取 config 失败，跳过 tie 检查: {e}")
        return

    if not getattr(config, "tie_word_embeddings", False):
        print("  [Skip Tie Check] config.tie_word_embeddings=False")
        return

    try:
        model = AutoModelForCausalLM.from_pretrained(
            model_dir,
            trust_remote_code=True,
            torch_dtype=torch.float32,
            device_map="cpu",
        )
    except Exception as e:
        print(f"  [Warn] 加载模型失败，跳过 tie 检查: {e}")
        return

    embed_weight = model.get_input_embeddings().weight
    lm_head = model.get_output_embeddings()
    if lm_head is None:
        print("  [Warn] lm_head is None; cannot compare weights.")
        return

    lm_head_weight = lm_head.weight
    same_object = embed_weight is lm_head_weight
    all_close = torch.allclose(embed_weight, lm_head_weight)

    print(f"  [Tie Check] embed_tokens.weight is lm_head.weight: {same_object}")
    print(f"  [Tie Check] embed_tokens.weight allclose lm_head.weight: {all_close}")

    if all_close and not same_object:
        print("  [Fix] 数值相同但对象不同，执行 tie_weights 并重存，仅保留一份共享参数")
        model.tie_weights()

        embed_weight_post = model.get_input_embeddings().weight
        lm_head_post = model.get_output_embeddings()
        if lm_head_post is not None and embed_weight_post is lm_head_post.weight and torch.allclose(embed_weight_post, embed_weight) and torch.allclose(lm_head_post.weight, lm_head_weight):
            model.save_pretrained(model_dir)
            print("  [Saved] 已重存为共享参数")
        else:
            print("  [Warn] tie_weights 后仍未共享，请检查模型实现")


def modify_tokenizer_config(dst_path: str, target_class: str) -> None:
    tok_config_path = os.path.join(dst_path, "tokenizer_config.json")
    if not os.path.exists(tok_config_path):
        print(f"  [Warning] 未找到 tokenizer_config.json")
        return

    try:
        with open(tok_config_path, "r", encoding="utf-8") as f:
            config = json.load(f)

        original_class = config.get("tokenizer_class")
        if original_class != target_class:
            print(f"  [Modify] 更新 tokenizer_class: {original_class} -> {target_class}")
            config["tokenizer_class"] = target_class
            with open(tok_config_path, "w", encoding="utf-8") as f:
                json.dump(config, f, indent=2, ensure_ascii=False)
        else:
            print(f"  [Skip Modify] tokenizer_class 已经是 {target_class}")
    except Exception as e:
        print(f"  [Error] 修改 tokenizer_config.json 失败: {e}")


def run_pt_mode():
   

    source_model_dir = os.path.join(CHECKPOINT_ROOT, MODEL_NAME)
    target_model_dir = os.path.join(SAVE_ROOT, MODEL_NAME)
    little_sets_dir = os.path.join(target_model_dir, "little_sets")

    if not os.path.exists(source_model_dir):
        print(f"错误: 找不到源模型目录 {source_model_dir}")
        return

    version_dirs = [
        d
        for d in os.listdir(source_model_dir)
        if d.startswith("v") and os.path.isdir(os.path.join(source_model_dir, d))
    ]

    if not version_dirs:
        print(f"错误: 在 {source_model_dir} 下未找到以 'v' 开头的版本文件夹")
        return

    version_dirs.sort()

    print(f"找到版本文件夹: {version_dirs}")
    print(f"目标目录 (Target): {little_sets_dir}")

    if not os.path.exists(little_sets_dir):
        os.makedirs(little_sets_dir)
        print(f"创建目标目录 {little_sets_dir}")

    checkpoints = []

    for v_dir in version_dirs:
        full_v_path = os.path.join(source_model_dir, v_dir)
        if not os.path.isdir(full_v_path):
            continue

        print(f"正在扫描: {full_v_path}")

        for d in os.listdir(full_v_path):
            if d.startswith("checkpoint-") and os.path.isdir(os.path.join(full_v_path, d)):
                try:
                    step = int(d.split("-")[1])
                    checkpoints.append((step, d, full_v_path))
                except ValueError:
                    pass

    if not checkpoints:
        print("源目录下没有发现 checkpoint 文件夹。")
        return

    checkpoints.sort(key=lambda x: x[0])
    all_steps = [c[0] for c in checkpoints]

    min_step = all_steps[0]
    max_step = all_steps[-1]

    print(f"共发现 {len(checkpoints)} 个检查点。Step 范围: {min_step} ~ {max_step}")

    target_steps = set()
    target_steps.add(min_step)
    target_steps.add(max_step)

    curr = STEP_START
    while curr < max_step:
        if curr in all_steps:
            target_steps.add(curr)
        curr += STEP_SIZE

    sorted_targets = sorted(list(target_steps))
    print(f"筛选出 {len(sorted_targets)} 个检查点需要移动: {sorted_targets}")

    count_moved = 0
    for step, dirname, source_dir in checkpoints:
        if step in target_steps:
            src_path = os.path.join(source_dir, dirname)
            dst_path = os.path.join(little_sets_dir, dirname)

            if SKIP_IF_EXISTS and os.path.exists(dst_path):
                print(f"[Skip] {dirname} 已存在于目标位置")
                if CHECK_AND_FIX_TIE_WORD_EMBEDDINGS:
                    check_and_fix_tied_embeddings(dst_path)
                continue

            print(f"[Copy] 复制 {dirname} (from {os.path.basename(source_dir)}) ...")
            shutil.copytree(src_path, dst_path)
            count_moved += 1

            global_dir_name = f"global_step{step}"
            dst_global_path = os.path.join(dst_path, global_dir_name)

            if os.path.exists(dst_global_path) and os.path.isdir(dst_global_path):
                print(f"  [Delete] 删除冗余文件夹: {global_dir_name}")
                shutil.rmtree(dst_global_path)

            # if MODIFY_TOKENIZER_CONFIG:
            #     modify_tokenizer_config(dst_path, TARGET_TOKENIZER_CLASS)

            if CHECK_AND_FIX_TIE_WORD_EMBEDDINGS:
                check_and_fix_tied_embeddings(dst_path)

    print(f"完成。共复制了 {count_moved} 个检查点。")
    print(f"保留在 little_sets 中的检查点: {os.listdir(little_sets_dir)}")


def run_sft_mode():
    

    source_model_dir = os.path.join(CHECKPOINT_ROOT, MODEL_NAME)
    target_model_dir = os.path.join(SAVE_ROOT, MODEL_NAME)

    if not os.path.exists(source_model_dir):
        print(f"错误: 找不到源模型目录 {source_model_dir}")
        return

    if not os.path.exists(target_model_dir):
        os.makedirs(target_model_dir)
        print(f"创建目标目录 {target_model_dir}")

    checkpoint_paths = []

    # 收集所有名为 checkpoint-* 的目录
    for dirpath, dirnames, _ in os.walk(source_model_dir, topdown=True):
        base = os.path.basename(dirpath)
        if base.startswith("checkpoint-"):
            checkpoint_paths.append(dirpath)

    if not checkpoint_paths:
        print("未找到可复制的 checkpoint 目录。")
        return

    # 只保留最深的 checkpoint 目录（不作为其他 checkpoint 的祖先）
    leaf_checkpoints = []
    checkpoint_set = set(checkpoint_paths)
    for path in checkpoint_paths:
        is_ancestor = any(
            other != path and other.startswith(path.rstrip(os.sep) + os.sep)
            for other in checkpoint_set
        )
        if not is_ancestor:
            leaf_checkpoints.append(path)

    # 按路径中的顶部 checkpoint step 排序，保证输出顺序稳定
    def top_checkpoint_step(path: str):
        parts = os.path.normpath(path).split(os.sep)
        for p in parts:
            if p.startswith("checkpoint-"):
                try:
                    return int(p.split("-")[1])
                except Exception:
                    return float("inf")
        return float("inf")

    leaf_checkpoints.sort(key=top_checkpoint_step)
    print(f"发现 {len(leaf_checkpoints)} 个叶子 checkpoint: {leaf_checkpoints}")

    count_moved = 0
    for ckpt_dir in leaf_checkpoints:
        # 目标名称使用路径里最外层的 checkpoint step
        step_for_name = top_checkpoint_step(ckpt_dir)
        if step_for_name != float("inf"):
            dest_name = f"{DEST_PREFIX}{step_for_name}"
        else:
            dest_name = f"{DEST_PREFIX}{os.path.basename(ckpt_dir)}"

        dst_path = os.path.join(target_model_dir, dest_name)
        if SKIP_IF_EXISTS and os.path.exists(dst_path):
            print(f"[Skip] {dest_name} 已存在于目标位置")
            if CHECK_AND_FIX_TIE_WORD_EMBEDDINGS:
                check_and_fix_tied_embeddings(dst_path)
            continue

        print(f"[Copy] 复制最底层 {ckpt_dir} -> {dst_path}")
        shutil.copytree(ckpt_dir, dst_path)
        count_moved += 1

        # 删除复制后目录里的 global_step* 以节省空间
        for name in os.listdir(dst_path):
            if name.startswith("global_step"):
                gpath = os.path.join(dst_path, name)
                if os.path.isdir(gpath):
                    print(f"  [Delete] 删除冗余文件夹: {name}")
                    shutil.rmtree(gpath)

        if MODIFY_TOKENIZER_CONFIG:
            modify_tokenizer_config(dst_path, TARGET_TOKENIZER_CLASS)

        if CHECK_AND_FIX_TIE_WORD_EMBEDDINGS:
            check_and_fix_tied_embeddings(dst_path)

    print(f"完成。共复制了 {count_moved} 个 SFT checkpoint。")
    print(f"目标目录当前内容: {os.listdir(target_model_dir)}")


def main():
    global CHECKPOINT_ROOT
    global SAVE_ROOT
    global MODEL_NAME
    global STEP_START
    global STEP_SIZE
    global MODIFY_TOKENIZER_CONFIG
    global SKIP_IF_EXISTS
    global DEST_PREFIX
    global CHECK_AND_FIX_TIE_WORD_EMBEDDINGS

    mode = os.environ.get("CKPT_MODE", "pt").lower()
    if len(sys.argv) > 1:
        mode = sys.argv[1].lower()

    if mode == "sft":
        print("运行 SFT 模式")
        # === 配置区域: SFT ===
        CHECKPOINT_ROOT = "/ruilab/jxhe/CoE_Monitor/ms-swift/output/SFT"
        SAVE_ROOT = "/ruilab/jxhe/CoE_Monitor/checkpoints/sft_models"
        MODEL_NAME = "General_PT_HJXA_Llama_104M_Minimind_no_packing_no_padding_free"
        MODIFY_TOKENIZER_CONFIG = True
        SKIP_IF_EXISTS = True
        DEST_PREFIX = "sft_checkpoint-"
        CHECK_AND_FIX_TIE_WORD_EMBEDDINGS = True
        run_sft_mode()
    else:
        print("运行 PT 模式")
         # === 配置区域: PT ===
        CHECKPOINT_ROOT = "/ruilab/jxhe/CoE_Monitor/ms-swift/output/"
        SAVE_ROOT = "/ruilab/jxhe/CoE_Monitor/checkpoints/pt_models/"
        MODEL_NAME = "PT_Pythia_14M"
        STEP_START = 2000
        STEP_SIZE = 2000
        MODIFY_TOKENIZER_CONFIG = True
        SKIP_IF_EXISTS = True
        CHECK_AND_FIX_TIE_WORD_EMBEDDINGS = True
        run_pt_mode()


if __name__ == "__main__":
    main()
