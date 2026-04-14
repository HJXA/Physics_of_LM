from typing import Dict, List, Tuple
from datasets import Dataset

def part3_qa_text_to_messages(text: str):
	"""
	将 QA 文本转为 messages。
	示例输入：What is the birth date of Ellie Blair Morgan? Answer: January 26, 1980.
	输出：
	[
		{"role": "user", "content": "What is ...?"},
		{"role": "assistant", "content": "Answer: January 26, 1980"}
	]
	"""
	text = str(text).strip()

	if "Answer:" in text:
		question, answer = text.split("Answer: ", 1) # 去除Answer字样 # 会导致结果下降
		question = question.strip() 
		answer = answer.strip().strip(".")
		# idx = text.index("Answer:") # 不去除
		# question = text[:idx].strip()
		# answer = text[idx:].strip().strip('.')


		# print(f"Parsed QA - Question: '{question}', Answer: '{answer}'")
	else:
		question = text
		answer = ""

	return [
		{"role": "user", "content": question},
		{"role": "assistant", "content": answer},
	]


def part3_prepare_sft_source_dataset(raw_dataset: Dataset) -> Dataset:
	"""SFT 数据预处理：支持 messages 原生格式，或由 QA text 动态构造成 messages。"""
	if "messages" in raw_dataset.column_names:
		return raw_dataset

	if "text" not in raw_dataset.column_names:
		raise ValueError("SFT 模式要求数据集包含 'messages' 或 'text' 列")

	def _map_fn(example):
		return {"messages": part3_qa_text_to_messages(example["text"])}

	return raw_dataset.map(
		_map_fn,
		desc="Converting QA text to messages",
		remove_columns=raw_dataset.column_names,
		load_from_cache_file=False,  # 强制重新计算，避免之前的缓存不符合当前逻辑
		
	)


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
		print(f"input_ids[:{preview_n}]: {preview_batch['input_ids'][:preview_n].tolist()}")
		print(f"labels[:{preview_n}]: {preview_batch['labels'][:preview_n].tolist()}")
		print(f"attention_mask[:{preview_n}]: {preview_batch['attention_mask'][:preview_n].tolist()}")
	else:
		print("[Collator Preview] 训练集为空，请检查 train.parquet")
