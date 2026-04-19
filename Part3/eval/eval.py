import os

import re
import json
import argparse
from datetime import datetime
from typing import List, Dict, Tuple
from collections import defaultdict



DEBUG = False

# 问题关键词 → 属性名映射（与 generate_bios_datasets.py 中6类 QA 模板对齐）
_QA_ATTR_PATTERNS = [
    (r"What is the birth date of", "birth_date"),
    (r"What is the birth city of", "birth_city"),
    (r"Which university did", "university"),
    (r"What major did", "field"),
    (r"Which company did", "company1name"),
    (r"Where did", "company1city"),
]

# 属性名 → 问题关键词反查，用于从答案中剥离属性前缀
_ATTR_PREFIXES = [
    "birth date: ",
    "birth city: ",
    "university: ",
    "field: ",
    "major: ",
    "company: ",
    "company city: ",
]


def _extract_attribute_from_question(question: str) -> str:
    """从问句中提取属性名，未匹配时返回 'unknown'。"""
    for pattern, attr_name in _QA_ATTR_PATTERNS:
        if re.search(pattern, question):
            return attr_name
    return "unknown"


def _strip_all_prefixes(s: str) -> str:
    """剥离所有已知的答案前缀（Answer: / 属性名: / # / 前导空格），返回纯答案文本。"""
    s = s.strip()
    # 去掉 "Answer: " 或 "Answer:" 前缀
    s = re.sub(r'^Answer:\s*', '', s, flags=re.IGNORECASE)
    s = s.strip()
    # 去掉属性名前缀（如 "birth date: "）
    for prefix in _ATTR_PREFIXES:
        if s.lower().startswith(prefix.lower()):
            s = s[len(prefix):]
            break
    s = s.strip()
    # 去掉开头所有 # 号（#January / # #January / # # # ...January → January）
    s = re.sub(r'^#(?:\s*#)*\s*', '', s)
    s = s.strip()
    # 去掉末尾句号
    s = s.rstrip(".")
    s = s.strip()
    # 合并多余空白
    s = re.sub(r'\s+', ' ', s)
    return s


def qa_text_to_messages(text: str) -> List[Dict[str, str]]:
    """兼容原始 text 格式，直接提取 question / assistant answer。"""
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
    """从 sample 中提取 (question, gold_answer_content)。"""
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
    """统一规范化答案文本：剥离所有已知前缀，仅保留核心答案值。"""
    return _strip_all_prefixes(s)


def save_jsonl(records: List[Dict], output_path: str) -> None:
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        for r in records:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


def save_metrics_txt(output_path: str, lines: List[str]) -> None:
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")





def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model_path",
        type=str,
        default=None,
        help="SFT checkpoint 路径",
    )
    parser.add_argument("--gpu_id", type=int, default=0, help="使用的 GPU 编号")


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
    parser.add_argument("--result_dir", type=str,
                        default="/ruilab/jxhe/CoE_Monitor/Physics_of_LM/Part3/eval/result",
                        help="评测结果输出目录")
    args = parser.parse_args()


    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu_id)
    import torch
    from datasets import load_dataset
    from transformers import AutoTokenizer, AutoModelForCausalLM

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
    ) -> Tuple[float, List[Dict[str, object]], Dict[str, Tuple[int, int]]]:
        total = 0
        correct = 0
        shown = 0
        per_sample_records: List[Dict[str, object]] = []
        # per-attribute 统计: {attr_name: (correct, total)}
        attr_stats: Dict[str, Tuple[int, int]] = defaultdict(lambda: [0, 0])

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

            if DEBUG: print("pred_texts: ", pred_texts)

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

                # 提取属性并统计
                attr = _extract_attribute_from_question(questions[j])
                attr_stats[attr][0] += int(ok)
                attr_stats[attr][1] += 1

                per_sample_records.append(
                    {
                        "sample_id": total,
                        "is_correct": ok,
                        "attribute": attr,
                        "question": questions[j],
                        "gold_answer": gold_answers[j],
                        "pred_answer": pred_text,
                        "gold_answer_norm": gold_norm,
                        "pred_answer_norm": pred_norm,
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
        # 转换 defaultdict 为普通 dict
        attr_stats = dict(attr_stats)
        return acc, per_sample_records, attr_stats

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

    if DEBUG:
        examples = examples[:2]
        args.batch_size = 1


    full_acc, records, attr_stats = evaluate_accuracy(
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

    if not DEBUG:

        save_jsonl(records, jsonl_path)
        metric_lines = [
            f"time={ts}",
            f"model_path={args.model_path}",
            f"test_dir={args.test_dir}",
            f"n={len(examples)}",
            f"accuracy={full_acc:.6f}",
        ]
        # 追加 per-attribute accuracy
        for attr_name in sorted(attr_stats.keys()):
            c, t = attr_stats[attr_name]
            acc_attr = c / t if t > 0 else 0.0
            metric_lines.append(f"accuracy_{attr_name}={acc_attr:.6f}")
        metric_lines += [
            f"batch_size={args.batch_size}",
            f"max_input_length={args.max_input_length}",
            f"max_new_tokens={args.max_new_tokens}",
        ]
        save_metrics_txt(txt_path, metric_lines)

    print(f"[Eval] n={len(examples)}, accuracy={full_acc:.4f}")
    for attr_name in sorted(attr_stats.keys()):
        c, t = attr_stats[attr_name]
        print(f"  {attr_name}: {c}/{t} = {c/t:.4f}")
    print(f"[Eval] per-sample outputs saved to: {jsonl_path}")
    print(f"[Eval] final metrics saved to: {txt_path}")


if __name__ == "__main__":
    main()
