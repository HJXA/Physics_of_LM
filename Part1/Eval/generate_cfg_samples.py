import os
os.environ["CUDA_VISIBLE_DEVICES"] = "2"  # 请根据实际情况调整 GPU 可见性

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





def strip_bos_eos_for_judge(seq, bos_token, eos_token):
    seq = normalize_token_sequence(seq)
    if len(seq) > 0 and seq[0] == bos_token:
        seq = seq[1:]
    if len(seq) > 0 and seq[-1] == eos_token:
        seq = seq[:-1]
    return seq


def write_jsonl(path, records):
    dir_path = os.path.dirname(path)
    if dir_path:
        os.makedirs(dir_path, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for item in records:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")


def apply_allowed_token_mask(next_token_logits, allowed_token_ids):
    """在采样前对 logits 做掩码：仅允许 allowed_token_ids 出现。"""
    vocab_size = next_token_logits.shape[-1]
    valid_allowed = [token_id for token_id in allowed_token_ids if 0 <= token_id < vocab_size]
    if not valid_allowed:
        raise ValueError(f"allowed_token_ids 与模型词表不匹配，词表大小={vocab_size}，allowed={allowed_token_ids}")

    masked_logits = torch.full_like(next_token_logits, float("-inf"))
    index_tensor = torch.tensor(valid_allowed, dtype=torch.long, device=next_token_logits.device)
    masked_logits[index_tensor] = next_token_logits[index_tensor]
    return masked_logits


def sample_sequence(model, prompt_ids, max_new_tokens=512, eos_token=0, temperature=1.0, allowed_token_ids=None):
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
            if allowed_token_ids is not None:
                next_token_logits = apply_allowed_token_mask(next_token_logits, allowed_token_ids)

            if x == 0 and allowed_token_ids is not None:
                allowed_token_ids.append(eos_token)

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
        if allowed_token_ids is not None:
            allowed_token_ids.remove(eos_token)

    return generated


def build_saved_record(index, scenario, prompt_ids, generated_seq, bos_token, eos_token):
    generated_ids = normalize_token_sequence(generated_seq)
    judge_input_ids = strip_bos_eos_for_judge(generated_ids, bos_token, eos_token)
    return {
        "index": index,
        "judge_input_len": len(generated_ids),
        "scenario": scenario,
        "prompt_ids": normalize_token_sequence(prompt_ids),
        "generate_id": generated_ids,
        "judge_input_text": " ".join(map(str, judge_input_ids)),
    }


def main():
    parser = argparse.ArgumentParser(description="只生成样本并保存（不做正确性判断）")
    parser.add_argument("--test_data_path", type=str, default="/ruilab/jxhe/CoE_Monitor/Physics_of_LM/Part1/datasets/test.parquet")
    parser.add_argument("--model_path", type=str, default="/ruilab/jxhe/CoE_Monitor/Physics_of_LM/Part1/checkpoints/gpt_standard_pretrain/checkpoint-65000")
    parser.add_argument("--num_eval_samples", type=int, default=20)
    parser.add_argument("--prefix_len", type=int, default=25)
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--max_new_tokens", type=int, default=512)
    parser.add_argument("--allowed_token_ids", type=str, default="")
    parser.add_argument("--save_generated_path", type=str, default="/ruilab/jxhe/CoE_Monitor/Physics_of_LM/Part1/Eval/")
    args = parser.parse_args()

    

    BOS_TOKEN = 100
    EOS_TOKEN = 101
    
    allowed_token_ids = [int(x.strip()) for x in args.allowed_token_ids.split(",") if x.strip()]
    allowed_token_set = set(allowed_token_ids)

    import pandas as pd
    if not os.path.exists(args.test_data_path):
        print(f"找不到测试集文件 {args.test_data_path}，请先运行 build_parquet_dataset.py 生成。")
        return

    df_test = pd.read_parquet(args.test_data_path)
    test_samples = df_test["input_ids"].tolist()
    num_eval = min(args.num_eval_samples, len(test_samples))
    print(f"已加载 {len(test_samples)} 条测试数据。将生成 {num_eval} 条样本（每条含 uncond+cond 两个场景）。")

    sys.path.append("/ruilab/jxhe/CoE_Monitor/Physics_of_LM/Part1/model")
    from modeling_gpt2_variants import CustomGPT2LMHeadModel

    gpt_type = args.model_path.split("/")[-1].split("_")[1] if "checkpoint" not in args.model_path else args.model_path.split("/")[-2].split("_")[1]
    checkpoints = args.model_path.split("/")[-1] if "checkpoint" in args.model_path else "End"
    args.save_generated_path = os.path.join(args.save_generated_path, f"generated_samples_{args.num_eval_samples}_temp{args.temperature}_type_{gpt_type}_{checkpoints}.jsonl")

    model = CustomGPT2LMHeadModel.from_pretrained(
        args.model_path,
        gpt_type=gpt_type,
        device_map="auto",
    )
    model.eval()

    records = []
    print("开始生成...")
    for i in tqdm(range(num_eval)):
        gen_seq_uncond = sample_sequence(
            model=model,
            prompt_ids=[BOS_TOKEN],
            max_new_tokens=args.max_new_tokens,
            eos_token=EOS_TOKEN,
            temperature=args.temperature,
            allowed_token_ids=allowed_token_ids,
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
            allowed_token_ids=allowed_token_ids,
        )
        records.append(build_saved_record(
            index=i,
            scenario="conditional",
            prompt_ids=prefix,
            generated_seq=gen_seq_cond,
            bos_token=BOS_TOKEN,
            eos_token=EOS_TOKEN,
        ))

    write_jsonl(args.save_generated_path, records)
    print(f"生成完成，已保存到: {args.save_generated_path}")


if __name__ == "__main__":
    main()
