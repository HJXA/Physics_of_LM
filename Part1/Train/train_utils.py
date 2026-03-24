from typing import Dict, List, Tuple

import torch
from datasets import Dataset, load_dataset
from transformers import AutoModelForCausalLM

import sys
sys.path.append("/ruilab/jxhe/CoE_Monitor/Physics_of_LM/Part1/model")
from gpt2.modeling_gpt2_variants import CustomGPT2LMHeadModel
from llama_wpe.modeling_llama_wpe import LlamaForCausalLMWPE


GPT2_VARIANTS = {"standard", "rot", "rel", "pos", "uni"}


def _resolve_attn_impl(model_path: str, variant: str) -> str:
	"""按模型类型选择注意力实现。"""
	if "gpt" not in model_path.lower():
		return "flash_attention_2"
	return "flash_attention_2" if variant in {"standard", "rot"} else "eager"


def build_causal_lm_collator(pad_token_id: int):
	"""
	用于已tokenized样本的动态padding。
	输入样本格式：
	{
		'input_ids': [...],
		'labels': [...],
		'attention_mask': [...],
	}
	"""

	def collate_fn(features: List[Dict]):
		input_ids = [torch.tensor(x["input_ids"], dtype=torch.long) for x in features]

		input_ids = torch.nn.utils.rnn.pad_sequence(
			input_ids,
			batch_first=True,
			padding_value=pad_token_id,
		)
		attention_mask = (input_ids != pad_token_id).long()

		labels = input_ids.clone()
		labels[labels == pad_token_id] = -100

		return {
			"input_ids": input_ids,
			"labels": labels,
			"attention_mask": attention_mask,
		}

	return collate_fn


def load_model(
	model_path: str,
	model_type=None,
	dtype=torch.bfloat16,
):
	if model_type in GPT2_VARIANTS:
		variant = (model_type or "").lower()
		attn_impl = _resolve_attn_impl(model_path=model_path, variant=variant)

		if variant in GPT2_VARIANTS:
			print(f"Load pretrained {model_type} model from: {model_path} with {attn_impl}")
			return CustomGPT2LMHeadModel.from_pretrained(
				model_path,
				gpt_type=model_type,
				attn_implementation=attn_impl,
				dtype=dtype,
			)
	elif model_type is "llama_wpe":
		print(f"Load pretrained LLaMA_wpe model from: {model_path} with {dtype}")
		return LlamaForCausalLMWPE.from_pretrained(
			model_path,
			torch_dtype=dtype,
		)

	attn_impl = "flash_attention_2"

	print(f"Load pretrained generic CausalLM model from: {model_path} with {attn_impl}")
	return AutoModelForCausalLM.from_pretrained(
		model_path,
		attn_implementation=attn_impl,
		torch_dtype=dtype,
	)


def prepare_train_dataset(
	train_parquet_path: str,
	pad_token_id: int,
	keep_columns=frozenset({"input_ids"}),
) -> Tuple[Dataset, callable]:
	"""读取 parquet 训练集并完成列裁剪与 collator 构建。"""
	train_dataset = load_dataset(
		"parquet",
		data_files={"train": train_parquet_path},
	)["train"]

	remove_columns = [x for x in train_dataset.column_names if x not in keep_columns]
	if remove_columns:
		train_dataset = train_dataset.remove_columns(remove_columns)

	data_collator = build_causal_lm_collator(pad_token_id=pad_token_id)
	return train_dataset, data_collator


def preview_collator_batch(train_dataset: Dataset, data_collator, preview_n: int = 2):
	"""训练开始前打印经过 data_collator 处理后的样本示例。"""
	if len(train_dataset) > 0:
		actual_preview_n = min(preview_n, len(train_dataset))
		preview_features = [train_dataset[i] for i in range(actual_preview_n)]
		preview_batch = data_collator(preview_features)

		print("\n[Collator Preview]")
		print(
			f"input_ids shape: {tuple(preview_batch['input_ids'].shape)}, "
			f"labels shape: {tuple(preview_batch['labels'].shape)}, "
			f"attention_mask shape: {tuple(preview_batch['attention_mask'].shape)}"
		)
		print(f"input_ids[0]: {preview_batch['input_ids'][0].tolist()}")
		print(f"labels[0]: {preview_batch['labels'][0].tolist()}")
		print(f"attention_mask[0]: {preview_batch['attention_mask'][0].tolist()}")
	else:
		print("[Collator Preview] 训练集为空，请检查 train.parquet")
