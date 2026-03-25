import os
# os.environ["CUDA_VISIBLE_DEVICES"] = "2"  # 请根据实际情况调整 GPU 可见性

import json
import argparse

import torch
from tqdm import tqdm

import sys
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
    

def normalize_token_sequence(seq):
    """当前链路只接受 list / ndarray / tensor，统一为 list[int]。"""
    if hasattr(seq, "tolist"):
        seq = seq.tolist()
    return [int(x) for x in seq]


def write_jsonl(path, records):
    dir_path = os.path.dirname(path)
    if dir_path:
        os.makedirs(dir_path, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for item in records:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")


def sample_sequence(model, prompt_ids, max_new_tokens=512, eos_token=0, temperature=1.0):
    """自回归生成（带 KV cache）。"""
    device = model.device
    prompt_ids = normalize_token_sequence(prompt_ids)
    input_ids = torch.tensor([prompt_ids], dtype=torch.long, device=device)
    generated = list(prompt_ids)
    if max_new_tokens <= 0:
        return generated

    with torch.inference_mode():
        outputs = model(input_ids, use_cache=True)
        logits = outputs.logits
        past_key_values = outputs.past_key_values

        for x in range(max_new_tokens):
            
            next_token_logits = logits[0, -1, :]

            if temperature > 0:
                probs = torch.softmax(next_token_logits / temperature, dim=-1)
                next_token = torch.multinomial(probs, num_samples=1).view(1, 1)
            else:
                next_token = torch.argmax(next_token_logits).view(1, 1)

            next_token_id = int(next_token.item())
            generated.append(next_token_id)

            if next_token_id == eos_token:
                break

            outputs = model(next_token, use_cache=True, past_key_values=past_key_values)
            logits = outputs.logits
            past_key_values = outputs.past_key_values

    return generated


def sample_sequence_batch(model, prompt_ids_batch, max_new_tokens=512, eos_token=0, temperature=1.0, pad_token=0):
    """批量自回归生成（带 KV cache，左填充对齐）。"""
    if len(prompt_ids_batch) == 0:
        return []

    device = model.device
    prompt_ids_batch = [normalize_token_sequence(ids) for ids in prompt_ids_batch]
    generated_batch = [list(ids) for ids in prompt_ids_batch]

    if max_new_tokens <= 0:
        return generated_batch

    max_prompt_len = max(len(ids) for ids in prompt_ids_batch)
    input_ids_list = []
    attention_mask_list = []
    for ids in prompt_ids_batch:
        pad_len = max_prompt_len - len(ids)
        input_ids_list.append([pad_token] * pad_len + ids)
        attention_mask_list.append([0] * pad_len + [1] * len(ids))

    input_ids = torch.tensor(input_ids_list, dtype=torch.long, device=device)
    attention_mask = torch.tensor(attention_mask_list, dtype=torch.long, device=device)
    finished = [False] * len(prompt_ids_batch)

    with torch.inference_mode():
        outputs = model(input_ids, attention_mask=attention_mask, use_cache=True)
        logits = outputs.logits
        past_key_values = outputs.past_key_values

        for _ in range(max_new_tokens):
            next_token_logits = logits[:, -1, :]

            if temperature > 0:
                probs = torch.softmax(next_token_logits / temperature, dim=-1)
                next_tokens = torch.multinomial(probs, num_samples=1)
            else:
                next_tokens = torch.argmax(next_token_logits, dim=-1, keepdim=True)

            for idx in range(len(generated_batch)):
                if finished[idx]:
                    continue
                next_token_id = int(next_tokens[idx, 0].item())
                generated_batch[idx].append(next_token_id)
                if next_token_id == eos_token:
                    finished[idx] = True

            if all(finished):
                break

            outputs = model(next_tokens, use_cache=True, past_key_values=past_key_values)
            logits = outputs.logits
            past_key_values = outputs.past_key_values

    return generated_batch


def build_saved_record(index, scenario, prompt_ids, generated_seq, bos_token, eos_token):
    generated_ids = normalize_token_sequence(generated_seq)
    return {
        "index": index,
        "judge_input_len": len(generated_ids),
        "scenario": scenario,
        "prompt_ids": normalize_token_sequence(prompt_ids),
        "generate_id": generated_ids,
    }


def main():
    parser = argparse.ArgumentParser(description="只生成样本并保存（不做正确性判断）")
    parser.add_argument("--test_data_path", type=str, default="/ruilab/jxhe/CoE_Monitor/Physics_of_LM/Part1/datasets/512_Padding/cfg3f/test.parquet")
    parser.add_argument("--model_path", type=str, default="/ruilab/jxhe/CoE_Monitor/Physics_of_LM/Part1/checkpoints/gpt_rot_Pretrain/checkpoint-100000")
    parser.add_argument("--num_eval_samples", type=int, default=200)
    parser.add_argument("--prefix_len", type=int, default=50)
    parser.add_argument("--batch_size", type=int, default=192)
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--max_new_tokens", type=int, default=512)
    parser.add_argument("--save_generated_path", type=str, default="/ruilab/jxhe/CoE_Monitor/Physics_of_LM/Part1/Eval/result")
    args = parser.parse_args()

    

    BOS_TOKEN = 0
    PAD_TOKEN = 5
    EOS_TOKEN = 4

    import pandas as pd
    if not os.path.exists(args.test_data_path):
        print(f"找不到测试集文件 {args.test_data_path}，请先运行 build_parquet_dataset.py 生成。")
        return

    df_test = pd.read_parquet(args.test_data_path)
    test_samples = df_test["input_ids"].tolist()
    num_eval = min(args.num_eval_samples, len(test_samples))
    print(f"已加载 {len(test_samples)} 条测试数据。将生成 {num_eval} 条样本（每条含 uncond+cond 两个场景）。")

    sys.path.append("/ruilab/jxhe/CoE_Monitor/Physics_of_LM")
    from Part1.Train.train_utils import load_model

    path_parts = [p for p in os.path.normpath(args.model_path).split(os.sep) if p]
    model_base = path_parts[-1] if len(path_parts) >= 1 else "unknown_model"
    parent_base = path_parts[-2] if len(path_parts) >= 2 else "unknown_dataset"
    grandparent_base = path_parts[-3] if len(path_parts) >= 3 else "unknown_model_type"

    if model_base.startswith("checkpoint"):
        checkpoints = model_base
        dataset_name = parent_base
        model_type = grandparent_base
    else:
        checkpoints = "End"
        dataset_name = parent_base
        model_type = model_base

    args.save_generated_path = os.path.join(
        args.save_generated_path,
        f"{model_type}",
        f"{dataset_name}",
        f"{checkpoints}",
        f"samples_{args.num_eval_samples}",
        f"generated_temp{args.temperature}_all.jsonl",
    )

    if os.path.exists(args.save_generated_path):
        print(f"生成文件 {args.save_generated_path} 已存在，跳过生成步骤。")
        return
    gpt_type = model_type.split("_")[-2] if "gpt" in model_type.lower() else None
    if gpt_type is None and "llama_wpe" in path_parts:
        gpt_type = "llama_wpe"
    print(f"模型类型: {model_type}, 数据集: {dataset_name}, GPT 变体: {gpt_type}, 检查点: {checkpoints}")
    model = load_model(
        model_path=args.model_path,
        model_type=gpt_type,
        dtype=torch.bfloat16,
    )
    print(model)
    model.config.bos_token_id = BOS_TOKEN
    model.config.pad_token_id = PAD_TOKEN
    model.config.eos_token_id = EOS_TOKEN
    model.to("cuda")
    model.eval()

    records = []
    print("开始生成...")
    if args.batch_size <= 1:
        for i in tqdm(range(num_eval)):
            gen_seq_uncond = sample_sequence(
                model=model,
                prompt_ids=[BOS_TOKEN],
                max_new_tokens=args.max_new_tokens,
                eos_token=EOS_TOKEN,
                temperature=args.temperature,
            )
            records.append(build_saved_record(
                index=i,
                scenario="unconditional",
                prompt_ids=[BOS_TOKEN],
                generated_seq=gen_seq_uncond,
                bos_token=BOS_TOKEN,
                eos_token=EOS_TOKEN,
            ))

            sample_ids = normalize_token_sequence(test_samples[i])
            gt_seq = sample_ids[:]
            if len(gt_seq) > 0 and gt_seq[-1] == EOS_TOKEN:
                gt_seq = gt_seq[:-1]

            prefix = gt_seq[:args.prefix_len] if len(gt_seq) > args.prefix_len else gt_seq

            gen_seq_cond = sample_sequence(
                model=model,
                prompt_ids=prefix,
                max_new_tokens=max(0, args.max_new_tokens - len(prefix)),
                eos_token=EOS_TOKEN,
                temperature=args.temperature,
            )
            records.append(build_saved_record(
                index=i,
                scenario="conditional",
                prompt_ids=prefix,
                generated_seq=gen_seq_cond,
                bos_token=BOS_TOKEN,
                eos_token=EOS_TOKEN,
            ))
    else:
        for start in tqdm(range(0, num_eval, args.batch_size)):
            end = min(start + args.batch_size, num_eval)
            batch_indices = list(range(start, end))

            uncond_prompts = [[BOS_TOKEN] for _ in batch_indices]
            uncond_generated = sample_sequence_batch(
                model=model,
                prompt_ids_batch=uncond_prompts,
                max_new_tokens=args.max_new_tokens,
                eos_token=EOS_TOKEN,
                temperature=args.temperature,
                pad_token=PAD_TOKEN,
            )
            for pos, i in enumerate(batch_indices):
                records.append(build_saved_record(
                    index=i,
                    scenario="unconditional",
                    prompt_ids=uncond_prompts[pos],
                    generated_seq=uncond_generated[pos],
                    bos_token=BOS_TOKEN,
                    eos_token=EOS_TOKEN,
                ))

            cond_prompts = []
            cond_max_new_tokens = []
            for i in batch_indices:
                sample_ids = normalize_token_sequence(test_samples[i])
                gt_seq = sample_ids[:]
                if len(gt_seq) > 0 and gt_seq[-1] == EOS_TOKEN:
                    gt_seq = gt_seq[:-1]
                prefix = gt_seq[:args.prefix_len] if len(gt_seq) > args.prefix_len else gt_seq
                cond_prompts.append(prefix)
                cond_max_new_tokens.append(max(0, args.max_new_tokens - len(prefix)))

            batch_max_new_tokens = max(cond_max_new_tokens) if cond_max_new_tokens else 0
            cond_generated = sample_sequence_batch(
                model=model,
                prompt_ids_batch=cond_prompts,
                max_new_tokens=batch_max_new_tokens,
                eos_token=EOS_TOKEN,
                temperature=args.temperature,
                pad_token=PAD_TOKEN,
            )
            for pos, i in enumerate(batch_indices):
                records.append(build_saved_record(
                    index=i,
                    scenario="conditional",
                    prompt_ids=cond_prompts[pos],
                    generated_seq=cond_generated[pos],
                    bos_token=BOS_TOKEN,
                    eos_token=EOS_TOKEN,
                ))

    write_jsonl(args.save_generated_path, records)
    print(f"生成完成，已保存到: {args.save_generated_path}")


if __name__ == "__main__":
    main()
