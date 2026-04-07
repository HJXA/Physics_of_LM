import os
# os.environ["CUDA_VISIBLE_DEVICES"] = "3"  # 请根据实际情况调整 GPU 可见性
import re
import json
import argparse
from datetime import datetime
from typing import List, Dict, Tuple

import torch
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForCausalLM

DEBUG = False


def qa_text_to_messages(text: str) -> List[Dict[str, str]]:
    text = str(text).strip()
    if "Answer:" in text:
        idx = text.index("Answer:")
        question = text[:idx].strip()
        answer = text[idx:].strip().strip(".")
    else:
        question = text
        answer = ""
    return [
        {"role": "user", "content": question},
        {"role": "assistant", "content": answer},
    ]


def extract_qa_from_example(example: Dict) -> Tuple[str, str]:
    if "messages" in example and example["messages"] is not None:
        messages = example["messages"]
    elif "text" in example:
        messages = qa_text_to_messages(example["text"])
    else:
        raise ValueError("样本缺少 messages 或 text 字段，无法评测。")

    user_text = ""
    assistant_text = ""
    for m in messages:
        role = m.get("role", "")
        content = str(m.get("content", "")).strip()
        if role == "user" and not user_text:
            user_text = content
        elif role == "assistant" and not assistant_text:
            assistant_text = content

    if not user_text:
        raise ValueError(f"样本中未找到 user 内容: {example}")
    return user_text, assistant_text


def normalize_answer(s: str) -> str:
    s = str(s).strip()
    s = re.sub(r"^\s*Answer:\s*", "", s, flags=re.IGNORECASE)
    s = s.strip().strip(".").strip()
    s = re.sub(r"\s+", " ", s)
    return s


def save_jsonl(records: List[Dict], output_path: str) -> None:
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        for r in records:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


def save_metrics_txt(output_path: str, lines: List[str]) -> None:
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")


@torch.no_grad()
def evaluate_accuracy(
    model,
    tokenizer,
    examples: List[Dict],
    batch_size: int,
    max_input_length: int,
    max_new_tokens: int,
    device: torch.device,
    verbose_errors: int = 0,
) -> Tuple[float, List[Dict[str, object]]]:
    total = 0
    correct = 0
    shown = 0
    per_sample_records: List[Dict[str, object]] = []

    for i in range(0, len(examples), batch_size):
        batch = examples[i:i + batch_size]

        questions = []
        gold_answers = []
        for ex in batch:
            q, a = extract_qa_from_example(ex)
            questions.append(q)
            gold_answers.append(a)

        # 贴合当前 SFT 训练格式：训练时是 Question + Answer，评测时只给 Question 让模型续写答案
        prompts = questions
        if DEBUG: print("prompts: ", prompts)

        enc = tokenizer(
            prompts,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=max_input_length,
        )
        if DEBUG: print("enc: ", enc)
        enc = {k: v.to(device) for k, v in enc.items()}

        outputs = model.generate(
            **enc,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            num_beams=1,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id,
        )

        if DEBUG: print("outputs: ", outputs)

        prompt_len = enc["input_ids"].shape[1]
        gen_ids_batch = outputs[:, prompt_len:]
        pred_texts = tokenizer.batch_decode(gen_ids_batch, skip_special_tokens=True)

        for j in range(len(batch)):
            gen_ids = gen_ids_batch[j]
            if DEBUG: print(f"Sample {j + 1} - gen_ids: {gen_ids}")
            pred_text = pred_texts[j]
            if DEBUG: print(f"Sample {j + 1} - Predicted answer: '{pred_text}', Gold answer: '{gold_answers[j]}'")

            pred_norm = normalize_answer(pred_text)
            gold_norm = normalize_answer(gold_answers[j])

            ok = pred_norm == gold_norm
            total += 1
            if ok:
                correct += 1

            per_sample_records.append(
                {
                    "sample_id": total,
                    "is_correct": ok,
                    "question": questions[j],
                    "gold_answer": gold_answers[j],
                    "pred_answer": pred_text,
                    # "gold_answer_norm": gold_norm,
                    # "pred_answer_norm": pred_norm,
                }
            )

            if (not ok) and shown < verbose_errors:
                shown += 1
                print(f"\n[样本 {total} 错误示例]")
                print(f"Q: {questions[j]}")
                print(f"Gold(raw): {gold_answers[j]}")
                print(f"Pred(raw): {pred_text}")
                print(f"Gold(norm): {gold_norm}")
                print(f"Pred(norm): {pred_norm}")

    acc = correct / total if total > 0 else 0.0
    return acc, per_sample_records


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model_path",
        type=str,
        default="/ruilab/jxhe/CoE_Monitor/Physics_of_LM/Part3/checkpoints/QA/bioS_multi_permute_fullname/sft_llama2_2026_04_06_11_22_54_2026_04_07_22_48_17/checkpoint-38000",
        help="SFT checkpoint 路径",
    )
    parser.add_argument(
        "--test_dir",
        type=str,
        default="/ruilab/jxhe/CoE_Monitor/Physics_of_LM/Part3/datasets/QA/test",
        help="测试集目录（里面是 parquet 文件）",
    )
    parser.add_argument("--batch_size", type=int, default=8192)
    parser.add_argument("--max_samples", type=int, default=0, help="最多评测样本数，<=0 表示全量")
    parser.add_argument("--max_input_length", type=int, default=128)
    parser.add_argument("--max_new_tokens", type=int, default=32)
    parser.add_argument("--verbose_errors", type=int, default=5)
    parser.add_argument(
        "--result_dir",
        type=str,
        default="/ruilab/jxhe/CoE_Monitor/Physics_of_LM/Part3/eval/result",
        help="评测结果输出目录",
    )
    args = parser.parse_args()

    parquet_glob = os.path.join(args.test_dir, "*.parquet")
    ds = load_dataset("parquet", data_files={"test": parquet_glob})["test"]

    examples = [ds[i] for i in range(len(ds))]
    if len(examples) == 0:
        raise ValueError(f"测试集为空: {parquet_glob}")
    if args.max_samples > 0:
        examples = examples[: min(args.max_samples, len(examples))]

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dtype = torch.bfloat16 if device.type == "cuda" else torch.float32

    tokenizer = AutoTokenizer.from_pretrained("/ruilab/jxhe/CoE_Monitor/Physics_of_LM/Part3/checkpoints/llama2", use_fast=True, trust_remote_code=True)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"

    model = AutoModelForCausalLM.from_pretrained(
        args.model_path,
        torch_dtype=dtype,
        device_map="auto" if device.type == "cuda" else None,
        trust_remote_code=True,
    )
    if device.type != "cuda":
        model.to(device)
    model.eval()

    full_acc, records = evaluate_accuracy(
        model=model,
        tokenizer=tokenizer,
        examples=examples,
        batch_size=args.batch_size,
        max_input_length=args.max_input_length,
        max_new_tokens=args.max_new_tokens,
        device=device,
        verbose_errors=args.verbose_errors,
    )

    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    model_type = args.model_path.split("/")[-3]
    model_name = args.model_path.split("/")[-2]
    checkpoint_name = args.model_path.split("/")[-1]
    jsonl_path = os.path.join(args.result_dir,model_type,model_name, checkpoint_name,f"predictions_{ts}.jsonl")
    txt_path = os.path.join(args.result_dir, model_type, model_name, checkpoint_name, f"metrics_{ts}.txt")

    save_jsonl(records, jsonl_path)
    metric_lines = [
        f"time={ts}",
        f"model_path={args.model_path}",
        f"test_dir={args.test_dir}",
        f"n={len(examples)}",
        f"accuracy={full_acc:.6f}",
        f"batch_size={args.batch_size}",
        f"max_input_length={args.max_input_length}",
        f"max_new_tokens={args.max_new_tokens}",
    ]
    save_metrics_txt(txt_path, metric_lines)

    print(f"[Eval] n={len(examples)}, accuracy={full_acc:.4f}")
    print(f"[Eval] per-sample outputs saved to: {jsonl_path}")
    print(f"[Eval] final metrics saved to: {txt_path}")


if __name__ == "__main__":
    main()
