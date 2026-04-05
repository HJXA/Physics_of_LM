"""
本文件提供两套数据集 tokenize 构建能力：
1) 预训练文本数据（每条样本形如 {"text": "..."}）
2) SFT 对话数据（每条样本形如 {"messages": [...]}）

并统一返回：
- tokenized_dataset：可直接传给 Trainer 的 Dataset
- data_collator：可直接传给 Trainer 的 batch 组装函数
"""

# 导入类型标注工具，用于明确函数输入输出类型，方便 IDE 与静态检查。
from typing import Callable, Dict, List, Sequence, Tuple

# 导入 PyTorch，用于在 collator 内将 Python 列表转为 Tensor 并执行 padding。
import torch
# 导入 HuggingFace Dataset 类型，作为本文件主要输入输出的数据结构。
from datasets import Dataset

# 以下用于测试
from transformers import AutoTokenizer
from train_utils import preview_collator_batch

def build_tokenizer(model, model_path: str):
	"""加载 tokenizer，保证pad_token_id不为None，至少为eos_token_id。"""
	tokenizer = AutoTokenizer.from_pretrained(model_path, use_fast=True, trust_remote_code=True)

	if tokenizer.pad_token_id is None:
		if tokenizer.eos_token is not None:
			tokenizer.pad_token = tokenizer.eos_token
			model.config.pad_token_id = tokenizer.eos_token_id

	return tokenizer


class PretrainTextTokenizerBuilder:
    """
    用途：将文本预训练数据集转换为可训练的 tokenized Dataset。

    输入数据格式：
    - Dataset，每条样本至少包含一个字段：
      {"text": "..."}

    关键能力：
    - 支持是否 chunk。
      - do_chunk=False：逐条样本独立 tokenize，截断/补齐到 max_length。
      - do_chunk=True：样本 token 首尾拼接后按 max_length 切块。
    - 支持在每条样本 token 序列首部添加 bos、尾部添加 eos。

    输出：
    - tokenized_dataset(Dataset)：至少包含 input_ids 与 attention_mask。
    - data_collator(Callable)：用于训练阶段组成 batch。
    """

    def __init__(
        self,
        tokenizer,
        max_length: int,
        do_chunk: bool,
        add_bos_at_start: bool = False,
        add_eos_at_end: bool = False,
        drop_last_chunk: bool = False,
    ):
        """
        用途：初始化预训练 tokenize 构建器。

        输入：
        - tokenizer：HuggingFace tokenizer 实例。
        - max_length(int)：目标序列长度。
        - do_chunk(bool)：是否执行“首尾拼接后分块”。
        - add_bos_at_start(bool)：是否在序列开头插入 bos。
        - add_eos_at_end(bool)：是否在序列末尾插入 eos。
        - drop_last_chunk(bool)：chunk 模式下是否丢弃最后不足 max_length 的残块。

        输出：
        - 无显式返回值，初始化实例内部状态。
        """
        # 保存 tokenizer 以供后续 tokenize 调用。
        self.tokenizer = tokenizer
        # 保存目标最大长度。
        self.max_length = max_length
        # 保存是否使用 chunk 的开关。
        self.do_chunk = do_chunk
        # 保存是否在开头添加 bos 的开关。
        self.add_bos_at_start = add_bos_at_start
        # 保存是否在末尾添加 eos 的开关。
        self.add_eos_at_end = add_eos_at_end
        # 保存 chunk 模式下是否丢弃尾块的开关。
        self.drop_last_chunk = drop_last_chunk

        # 校验 tokenizer 是否可用于 padding；若无 pad_token_id 则无法构建统一长度 batch。
        if self.tokenizer.pad_token_id is None:
            raise ValueError("tokenizer.pad_token_id is required")
        # 校验 max_length 的合法性，必须为正整数。
        if self.max_length <= 0:
            raise ValueError("max_length must be > 0")

    def _decorate_ids(self, ids: List[int]) -> List[int]:
        """
        用途：根据配置在 token 序列开头/结尾插入特殊 token。

        输入：
        - ids(List[int])：原始 token 序列。

        输出：
        - List[int]：插入可选特殊 token 后的新序列。
        """
        # 复制一份序列，避免原地修改外部传入列表。
        out = list(ids)
        # 若配置要求在开头加 bos，则执行插入。
        if self.add_bos_at_start:
            # 若 tokenizer 无 bos_token_id，直接报错提示配置不兼容。
            if self.tokenizer.bos_token_id is None:
                raise ValueError("add_bos_at_start=True but tokenizer.bos_token_id is None")
            # 在序列首部插入 bos。
            out = [self.tokenizer.bos_token_id] + out

        # 若配置要求在结尾加 eos，则执行追加。
        if self.add_eos_at_end:
            # 若 tokenizer 无 eos_token_id，直接报错提示配置不兼容。
            if self.tokenizer.eos_token_id is None:
                raise ValueError("add_eos_at_end=True but tokenizer.eos_token_id is None")
            # 在序列尾部追加 eos。
            out = out + [self.tokenizer.eos_token_id]

        # 返回装饰后的 token 序列。
        return out

    def _tokenize_text(self, text: str) -> List[int]:
        """
        用途：对单条文本进行 tokenize，并应用特殊 token 装饰。

        输入：
        - text(str)：单条文本。

        输出：
        - List[int]：token id 序列。
        """
        # 不使用 tokenizer 自动 special token，以便完全由本类配置控制首尾 token。
        ids = self.tokenizer(text, add_special_tokens=False)["input_ids"]
        # 应用可选的 bos/eos 规则后返回。
        return self._decorate_ids(ids)

    def _tokenize_no_chunk(self, dataset: Dataset) -> Dataset:
        """
        用途：在不 chunk 的模式下逐样本 tokenize，并对每条样本做定长补齐。

        输入：
        - dataset(Dataset)：包含 text 列的数据集。

        输出：
        - Dataset：仅保留 input_ids 与 attention_mask。
        """
        # 读取 pad token id，后续用于定长补齐。
        pad_token_id = self.tokenizer.pad_token_id

        # 定义 batched map 函数，批量处理一组文本。
        def _map_fn(batch: Dict[str, List[str]]) -> Dict[str, List[List[int]]]:
            # 保存一批样本的 input_ids 结果。
            all_ids: List[List[int]] = []
            # 保存一批样本的 attention_mask 结果。
            all_attention: List[List[int]] = []

            # 遍历当前批次中每条文本。
            for text in batch["text"]:
                # 先 tokenize，再截断到 max_length。
                ids = self._tokenize_text(str(text))[: self.max_length]
                # 当前有效 token 全部标记为 1。
                attention = [1] * len(ids)

                # 若长度不足 max_length，则进行右侧 padding。
                if len(ids) < self.max_length:
                    # 计算需要补齐的 token 数。
                    pad_len = self.max_length - len(ids)
                    # 对 input_ids 追加 pad token。
                    ids = ids + [pad_token_id] * pad_len
                    # attention_mask 对应 padding 位置标记为 0。
                    attention = attention + [0] * pad_len

                # 将当前样本的 input_ids 放入批结果。
                all_ids.append(ids)
                # 将当前样本的 attention_mask 放入批结果。
                all_attention.append(attention)

            # 返回 map 后新增的两列。
            return {
                "input_ids": all_ids,
                "attention_mask": all_attention,
            }

        # 执行 dataset.map 完成全量数据转换，并移除原始列（例如 text）。
        tokenized = dataset.map(
            _map_fn,
            batched=True,
            desc="Tokenizing pretraining dataset (no chunk)",
            remove_columns=dataset.column_names,
        )

        # 返回 tokenized 后的数据集。
        return tokenized

    def _tokenize_with_chunk(self, dataset: Dataset) -> Dataset:
        """
        用途：在 chunk 模式下，将样本 token 序列首尾拼接后切成定长块。

        输入：
        - dataset(Dataset)：包含 text 列的数据集。

        输出：
        - Dataset：仅保留 input_ids 与 attention_mask，其中每条样本长度均为 max_length。
        """
        # 读取 pad token id，用于处理最后一段不足长度的残块。
        pad_token_id = self.tokenizer.pad_token_id

        # 存放最终切好的固定长度 token 块。
        chunks: List[List[int]] = []
        # 作为滚动缓冲区，持续累积 token，直到可切出一个完整块。
        buffer: List[int] = []

        # 遍历原始数据集中的每条样本。
        for sample in dataset:
            # 对单条文本 tokenize 并应用首尾 token 规则。
            token_ids = self._tokenize_text(str(sample["text"]))
            # 若为空序列，则跳过该样本。
            if not token_ids:
                continue

            # 将当前样本 token 追加到总缓冲区，形成“首尾相接”的效果。
            buffer.extend(token_ids)

            # 只要缓冲区长度达到 max_length，就持续切出完整块。
            while len(buffer) >= self.max_length:
                # 取前 max_length 个 token 作为一个训练样本。
                chunks.append(buffer[: self.max_length])
                # 从缓冲区删除已切出的前缀，保留剩余部分继续累积。
                del buffer[: self.max_length]

        # 循环结束后，若仍有残留 token，按配置决定是否保留。
        if buffer:
            # 若不丢弃尾块，则进行右侧 padding 后作为最后一个样本。
            if not self.drop_last_chunk:
                # 将尾块补齐到 max_length。
                buffer = buffer + [pad_token_id] * (self.max_length - len(buffer))
                # 把补齐后的尾块加入 chunks。
                chunks.append(buffer)

        # 为每个 chunk 生成 attention_mask：非 pad 为 1，pad 为 0。
        attention_mask = [
            [1 if token_id != pad_token_id else 0 for token_id in chunk]
            for chunk in chunks
        ]

        # 将 Python 列表结构转换为 HuggingFace Dataset。
        return Dataset.from_dict(
            {
                "input_ids": chunks,
                "attention_mask": attention_mask,
            }
        )

    
    @staticmethod
    def build_causal_lm_collator(pad_token_id: int) -> Callable[[List[Dict]], Dict[str, torch.Tensor]]:
        """
        用途：构建预训练(Causal LM)用的 data_collator。

        输入：
        - pad_token_id(int)：padding 的 token id。

        输出：
        - collate_fn(Callable)：接收一个样本列表，返回包含 input_ids / labels / attention_mask 的 Tensor 字典。

        说明：
        - labels 直接由 input_ids 拷贝得到。
        - labels 中 padding 位置会被置为 -100，避免参与 loss。
        """

        # 定义真正被 Trainer 按 batch 调用的组装函数。
        def collate_fn(features: List[Dict]) -> Dict[str, torch.Tensor]:
            # 将每个样本里的 input_ids 列表转成 long Tensor，便于后续 pad_sequence 处理。
            input_ids = [torch.tensor(x["input_ids"], dtype=torch.long) for x in features]
            # 对当前 batch 动态补齐到同一长度，补齐值使用传入的 pad_token_id。
            input_ids = torch.nn.utils.rnn.pad_sequence(
                input_ids,
                batch_first=True,
                padding_value=pad_token_id,
            )
            # 根据 input_ids 中是否等于 pad_token_id 生成 attention_mask。
            attention_mask = (input_ids != pad_token_id).long()

            # labels 先完整复制 input_ids，做标准自回归语言模型监督。
            labels = input_ids.clone()
            # 将 labels 中 padding 的位置置为 -100，确保这些位置不参与损失计算。
            labels[labels == pad_token_id] = -100

            # 返回 Trainer 期望的标准字段字典。
            return {
                "input_ids": input_ids,
                "labels": labels,
                "attention_mask": attention_mask,
            }

        # 返回构建好的 collate_fn，供 Trainer 在训练阶段调用。
        return collate_fn    
    
    def build(self, dataset: Dataset) -> Tuple[Dataset, Callable]:
        """
        用途：根据 do_chunk 配置构建预训练 tokenized_dataset 和 data_collator。

        输入：
        - dataset(Dataset)：原始数据集，要求包含 text 列。

        输出：
        - Tuple[Dataset, Callable]
          - 第1项：tokenized_dataset
          - 第2项：data_collator
        """
        # 检查输入数据列是否满足要求。
        if "text" not in dataset.column_names:
            raise ValueError("Input dataset must contain a 'text' column")

        # 若开启 chunk，则走拼接切块流程。
        if self.do_chunk:
            tokenized_dataset = self._tokenize_with_chunk(dataset)
        # 否则走逐样本独立定长编码流程。
        else:
            tokenized_dataset = self._tokenize_no_chunk(dataset)

        # 构建预训练用 collator（labels 由 input_ids 自动复制并处理 padding）。
        data_collator = self.build_causal_lm_collator(pad_token_id=self.tokenizer.pad_token_id)
        # 返回数据集与 collator，供 Trainer 直接使用。
        return tokenized_dataset, data_collator


class SFTMessagesTokenizerBuilder:
    """
    用途：将 SFT 对话数据转换为可训练的 tokenized Dataset。

    输入数据格式：
    - Dataset，每条样本至少包含一个字段：
      {
        "messages": [
          {"role": "system", "content": "..."},
          {"role": "user", "content": "..."},
          {"role": "assistant", "content": "..."}
        ]
      }

    关键规则：
    - 仅 assistant 对应 token 参与监督（labels 为 token id）。
    - system/user 及 padding 位置 labels 均为 -100。
    - 不做 chunk，仅按 max_length 截断并补齐。

    输出：
    - tokenized_dataset(Dataset)：包含 input_ids / labels / attention_mask。
    - data_collator(Callable)：SFT 训练用 collator。
    """

    def __init__(
        self,
        tokenizer,
        max_length: int,
        add_bos_at_start: bool = False,
        add_eos_at_end: bool = False,
        apply_chat_template: bool = True,
        test = False,
    ):
        """
        用途：初始化 SFT tokenize 构建器。

        输入：
        - tokenizer：HuggingFace tokenizer 实例。
        - max_length(int)：目标序列长度。
        - add_bos_at_start(bool)：是否在序列开头插入 bos。
        - add_eos_at_end(bool)：是否在序列末尾插入 eos。

        输出：
        - 无显式返回值，初始化实例状态。
        """
        # 保存 tokenizer 供后续对话模板编码。
        self.tokenizer = tokenizer
        # 保存最大长度设置。
        self.max_length = max_length
        # 保存开头 bos 配置。
        self.add_bos_at_start = add_bos_at_start
        # 保存末尾 eos 配置。
        self.add_eos_at_end = add_eos_at_end

        self.apply_chat_template = apply_chat_template
        self.test = test

        # 校验 tokenizer 是否支持 padding。
        if self.tokenizer.pad_token_id is None:
            raise ValueError("tokenizer.pad_token_id is required")
        # 校验 max_length 合法性。
        if self.max_length <= 0:
            raise ValueError("max_length must be > 0")

    def _apply_chat_template_text(self, messages: Sequence[Dict[str, str]]) -> str:
        """
        用途：将 messages 渲染成一段对话文本。

        输入：
        - messages(Sequence[Dict[str, str]])：多轮消息列表。

        输出：
        - str：渲染后的完整对话字符串。

        说明：
        - 优先使用 tokenizer 自带 chat template。
        - 若 tokenizer 不支持 chat template，则使用简单回退格式。
        """
        # 如果 tokenizer 提供 apply_chat_template，优先使用官方模板以保持格式一致性。

        if self.apply_chat_template:
            if hasattr(self.tokenizer, "apply_chat_template"):
                chat_template = getattr(self.tokenizer, "chat_template", None)
                if chat_template is not None:
                    try:
                        # 仅返回文本，不在这里直接 tokenize。
                        return self.tokenizer.apply_chat_template(
                            messages,
                            tokenize=False,
                            add_generation_prompt=False,
                        )
                    except Exception:
                        # chat_template 不可用时，自动回退到手工模板。
                        raise "指定了 apply_chat_template=True，但 tokenizer 无法正确应用 chat template，可能缺乏相关实现或参数支持，请检查 tokenizer 兼容性。"
                else:
                    raise ValueError("apply_chat_template=True 但是 tokenizer 没有 chat_template")

        else: # 不用模版
            """
            不使用任何模板，直接拼接为：
            QuestionAnswer
            """
            lines: List[str] = []
            # 逐条消息构建回退文本。
            for msg in messages:
                # 读取消息内容，若缺失则视为空字符串。
                content = msg.get("content", "")
                # 生成一条简单的“角色+内容”片段。
                lines.append(f"{content}")
            # 将所有片段直接连接成完整文本。
            return "".join(lines)


    def _tokenize_with_assistant_mask(
        self,
        messages: Sequence[Dict[str, str]],
    ) -> Tuple[List[int], List[int]]:
        """
        用途：对单条 messages 执行 tokenize，并构建 assistant-only 的 labels。

        输入：
        - messages(Sequence[Dict[str, str]])：对话消息序列。

        输出：
        - Tuple[List[int], List[int]]
          - 第1项：input_ids
          - 第2项：labels（assistant token 为真实 id，其余为 -100）
        """
        # 若 tokenizer 支持 chat template，则优先尝试直接拿 assistant token mask，精度更高。
        if hasattr(self.tokenizer, "apply_chat_template") and self.apply_chat_template:
            try:
                # 让 tokenizer 同时返回 input_ids 与 assistant 掩码。
                encoded = self.tokenizer.apply_chat_template(
                    messages,
                    tokenize=True,
                    add_generation_prompt=False,
                    return_dict=True,
                    return_assistant_tokens_mask=True,
                )
                # 将编码结果转成标准 Python 列表。
                input_ids = list(encoded["input_ids"])
                # 读取 assistant 位置掩码（1 表示 assistant token）。
                assistant_mask = list(encoded["assistant_masks"])
                # assistant 位置保留 token id，其余位置统一置为 -100。
                labels = [token if mask == 1 else -100 for token, mask in zip(input_ids, assistant_mask)]
                # 对长度进行截断/补齐，并处理可选 bos/eos。
                return self._finalize_sft_lengths(input_ids, labels)
            except Exception:
                # 如果某些 tokenizer 不支持上述参数或返回字段，则自动回退到长度差分方案。
                pass

        # 先将完整多轮消息渲染为文本。
        full_text = self._apply_chat_template_text(messages)

        if self.test: print(f"full_text: {full_text}, len: {len(full_text)}")

        # 对完整文本进行 tokenize。
        input_ids = self.tokenizer(full_text, add_special_tokens=False)["input_ids"]

        if self.test: print(f"input_ids: {input_ids}")
        # 默认全部位置先设为 -100，后续只打开 assistant 对应区间。
        labels = [-100] * len(input_ids)

        # 遍历每一轮消息，定位 assistant 片段对应的 token 区间。
        for idx, msg in enumerate(messages):
            # 非 assistant 消息不参与监督，直接跳过。
            if msg.get("role") != "assistant":
                continue

            # 渲染 assistant 消息之前的前缀文本。
            prefix_text = self._apply_chat_template_text(messages[:idx])
            # 渲染包含当前 assistant 消息在内的前缀文本。
            upto_text = self._apply_chat_template_text(messages[: idx + 1])


            # 计算当前 assistant 起始 token 下标（前缀长度）。
            start = len(self.tokenizer(prefix_text, add_special_tokens=False)["input_ids"])
            # 计算当前 assistant 结束 token 下标（含当前轮后的前缀长度）。
            end = len(self.tokenizer(upto_text, add_special_tokens=False)["input_ids"])

            # 防御式截断，避免越界。
            start = min(start, len(input_ids))
            # 防御式截断，避免越界。
            end = min(end, len(input_ids))
            # 将 assistant 区间对应位置写入真实 token id。
            for pos in range(start, end):
                labels[pos] = input_ids[pos]     

            if self.test: print(f"labels: {labels}")

        # 最后统一执行可选特殊 token 注入 + 定长截断/补齐。
        return self._finalize_sft_lengths(input_ids, labels)

    def _finalize_sft_lengths(self, input_ids: List[int], labels: List[int]) -> Tuple[List[int], List[int]]:
        """
        用途：对 SFT 的 input_ids/labels 统一做首尾 token 注入、截断与补齐。

        输入：
        - input_ids(List[int])：原始输入 token 序列。
        - labels(List[int])：与 input_ids 对齐的监督标签序列。

        输出：
        - Tuple[List[int], List[int]]：处理后的定长 input_ids 与 labels。
        """
        # 若开启开头 bos，则在 input_ids 首部插入 bos，同时 labels 对应位置为 -100。
        if self.add_bos_at_start:
            # 若 tokenizer 无 bos_token_id，则报错提示配置不兼容。
            if self.tokenizer.bos_token_id is None:
                raise ValueError("add_bos_at_start=True but tokenizer.bos_token_id is None")
            # 在输入首部加入 bos。
            input_ids = [self.tokenizer.bos_token_id] + input_ids
            # labels 对应首部置为 -100，避免此位置参与监督。
            labels = [-100] + labels

        # 若开启末尾 eos，则在 input_ids 尾部追加 eos，同时 labels 对应位置为 -100。
        if self.add_eos_at_end:
            # 若 tokenizer 无 eos_token_id，则报错提示配置不兼容。
            if self.tokenizer.eos_token_id is None:
                raise ValueError("add_eos_at_end=True but tokenizer.eos_token_id is None")
            # 在输入尾部追加 eos。
            input_ids = input_ids + [self.tokenizer.eos_token_id]
            # labels 尾部对应位置置为 -100。
            labels = labels + [-100]

        # 先按 max_length 截断 input_ids。
        input_ids = input_ids[: self.max_length]
        # 再按同样长度截断 labels，保持两者严格对齐。
        labels = labels[: self.max_length]

        # 计算当前序列距离 max_length 还差多少。
        pad_len = self.max_length - len(input_ids)
        # 如果不足 max_length，就做右侧补齐。
        if pad_len > 0:
            # input_ids 用 pad_token_id 补齐。
            input_ids = input_ids + [self.tokenizer.pad_token_id] * pad_len
            # labels 用 -100 补齐，保证 padding 不参与损失。
            labels = labels + [-100] * pad_len

        # 返回定长、已对齐的 input_ids 与 labels。
        return input_ids, labels

    
    @staticmethod
    def build_sft_collator(
        pad_token_id: int,
        label_pad_token_id: int = -100,
    ) -> Callable[[List[Dict]], Dict[str, torch.Tensor]]:
        """
        用途：构建 SFT 训练用 data_collator。

        输入：
        - pad_token_id(int)：input_ids 的 padding token id。
        - label_pad_token_id(int)：labels 的 padding 值，默认 -100。

        输出：
        - collate_fn(Callable)：接收样本列表，返回 input_ids / labels / attention_mask 三个 Tensor。

        说明：
        - SFT 的 labels 通常在 tokenize 阶段已构建好（assistant 位置为 token，其余位置 -100）。
        - collator 只负责按 batch 维度补齐，不再改写监督掩码语义。
        """

        # 定义 SFT 的 batch 组装函数。
        def collate_fn(features: List[Dict]) -> Dict[str, torch.Tensor]:
            # 将每条样本的 input_ids 转为 long Tensor。
            input_ids = [torch.tensor(x["input_ids"], dtype=torch.long) for x in features]
            # 将每条样本的 labels 转为 long Tensor。
            labels = [torch.tensor(x["labels"], dtype=torch.long) for x in features]

            # 对 input_ids 做动态 padding，padding 值为 pad_token_id。
            input_ids = torch.nn.utils.rnn.pad_sequence(
                input_ids,
                batch_first=True,
                padding_value=pad_token_id,
            )
            # 对 labels 做动态 padding，padding 值使用 label_pad_token_id（默认 -100）。
            labels = torch.nn.utils.rnn.pad_sequence(
                labels,
                batch_first=True,
                padding_value=label_pad_token_id,
            )
            # attention_mask 由 input_ids 的 padding 位置推导。
            attention_mask = (input_ids != pad_token_id).long()

            # 返回 Trainer 可直接消费的 batch 字段。
            return {
                "input_ids": input_ids,
                "labels": labels,
                "attention_mask": attention_mask,
            }

        # 返回构建好的 collate_fn。
        return collate_fn
    

    def build(self, dataset: Dataset) -> Tuple[Dataset, Callable]:
        """
        用途：构建 SFT 训练所需 tokenized_dataset 与 data_collator。

        输入：
        - dataset(Dataset)：原始 SFT 数据集，要求包含 messages 列。

        输出：
        - Tuple[Dataset, Callable]
          - 第1项：tokenized_dataset（含 input_ids/labels/attention_mask）
          - 第2项：data_collator
        """
        # 检查输入列完整性。
        if "messages" not in dataset.column_names:
            raise ValueError("Input dataset must contain a 'messages' column")

        # 定义逐样本 map 函数，把 messages 转成训练字段。
        def _map_fn(example: Dict) -> Dict[str, List[int]]:
            # 对单条 messages 执行 tokenize 与 assistant-only 标签构建。
            input_ids, labels = self._tokenize_with_assistant_mask(example["messages"])
            # 生成 attention_mask：非 pad 为 1，pad 为 0。
            attention_mask = [1 if x != self.tokenizer.pad_token_id else 0 for x in input_ids]
            # 返回当前样本的训练字段。
            return {
                "input_ids": input_ids,
                "labels": labels,
                "attention_mask": attention_mask,
            }

        # 对数据集执行逐样本映射，并移除原始列。
        tokenized_dataset = dataset.map(
            _map_fn,
            desc="Tokenizing SFT dataset",
            remove_columns=dataset.column_names,
        )

        # 构建 SFT collator（labels 在 tokenize 阶段已完成语义化构建）。
        data_collator = self.build_sft_collator(pad_token_id=self.tokenizer.pad_token_id)
        # 返回 tokenized_dataset 与 data_collator。
        return tokenized_dataset, data_collator


def prepare_pretrain_dataset_from_text(
    dataset: Dataset,
    tokenizer,
    max_length: int,
    do_chunk: bool,
    add_bos_at_start: bool = False,
    add_eos_at_end: bool = False,
    drop_last_chunk: bool = False,
) -> Tuple[Dataset, Callable]:
    """
    用途：预训练数据的便捷封装函数，一行拿到 tokenized_dataset 与 collator。

    输入：
    - dataset(Dataset)：原始文本数据集，样本格式 {"text": "..."}。
    - tokenizer：HuggingFace tokenizer。
    - max_length(int)：目标长度。
    - do_chunk(bool)：是否首尾拼接切块。
    - add_bos_at_start(bool)：是否在开头加 bos。
    - add_eos_at_end(bool)：是否在末尾加 eos。
    - drop_last_chunk(bool)：chunk 时是否丢弃不足长度的尾块。

    输出：
    - Tuple[Dataset, Callable]
      - 第1项：tokenized_dataset
      - 第2项：data_collator
    """
    # 实例化预训练构建器并传入全部配置。
    builder = PretrainTextTokenizerBuilder(
        tokenizer=tokenizer,
        max_length=max_length,
        do_chunk=do_chunk,
        add_bos_at_start=add_bos_at_start,
        add_eos_at_end=add_eos_at_end,
        drop_last_chunk=drop_last_chunk,
    )
    # 调用 build 返回训练可直接使用的 dataset 与 collator。
    return builder.build(dataset)


def prepare_sft_dataset_from_messages(
    dataset: Dataset,
    tokenizer,
    max_length: int,
    add_bos_at_start: bool = False,
    add_eos_at_end: bool = False,
    apply_chat_template: bool = True,
) -> Tuple[Dataset, Callable]:
    """
    用途：SFT 数据的便捷封装函数，一行拿到 tokenized_dataset 与 collator。

    输入：
    - dataset(Dataset)：原始对话数据集，样本格式 {"messages": [...]}。
    - tokenizer：HuggingFace tokenizer。
    - max_length(int)：目标长度。
    - add_bos_at_start(bool)：是否在开头加 bos。
    - add_eos_at_end(bool)：是否在末尾加 eos。
    - apply_chat_template(bool)：是否使用 tokenizer 的 chat template。

    输出：
    - Tuple[Dataset, Callable]
      - 第1项：tokenized_dataset
      - 第2项：data_collator
    """
    # 实例化 SFT 构建器并注入配置。
    builder = SFTMessagesTokenizerBuilder(
        tokenizer=tokenizer,
        max_length=max_length,
        add_bos_at_start=add_bos_at_start,
        add_eos_at_end=add_eos_at_end,
        apply_chat_template=apply_chat_template,
    )
    # 调用 build 返回训练可直接使用的 dataset 与 collator。
    return builder.build(dataset)














# =====================================================================以下用于测试============================================================================================

if __name__ == "__main__":

    def _print_title(title: str) -> None:
        print("\n" + "=" * 90)
        print(title)
        print("=" * 90)


    def _print_token_decode(tokenizer, token_ids: List[int], title: str) -> None:
        print(f"{title} token_ids: {token_ids}")
        decoded = tokenizer.decode(token_ids, skip_special_tokens=False)
        print(f"{title} decoded: {repr(decoded)}")


    def run_pretrain_feature_tests(tokenizer) -> None:
        _print_title("[PT] 构造测试数据")
        pt_raw_chunk_eosbos = Dataset.from_dict(
            {
                "text": [
                    "第一条，短句。",
                    "第二条，稍微长一点用于形成多个 chunk 的边界测试。",
                    "第三条，继续追加内容以确保缓冲区超过 max_length 并产生若干 chunk。",
                    "第四条，最后。",
                ]
            }
        )
        for i, sample in enumerate(pt_raw_chunk_eosbos):
            print(f"pt6_raw[{i}] text: {sample['text']}")
        _print_title("[PT] do_chunk=True + add_bos_at_start=True + add_eos_at_end=True")
        

        pt_builder_6 = PretrainTextTokenizerBuilder(
            tokenizer=tokenizer,
            max_length=16,
            do_chunk=True,
            add_bos_at_start=True,
            add_eos_at_end=True,
            drop_last_chunk=False,
        )

        # 打印每条样本单独 tokenize 后（含 bos/eos）的 token 序列
        for i, sample in enumerate(pt_raw_chunk_eosbos):
            ids = pt_builder_6._tokenize_text(sample["text"])
            _print_token_decode(tokenizer, ids, f"pt6_each_decorated[{i}]")

        # 合并所有样本以查看拼接后缓冲区内容（chunk 前）
        merged = []
        for sample in pt_raw_chunk_eosbos:
            merged.extend(pt_builder_6._tokenize_text(sample["text"]))
        _print_token_decode(tokenizer, merged, "pt6_merged_before_chunk")

        pt_ds_6, pt_collator_6 = pt_builder_6.build(pt_raw_chunk_eosbos)
        print(f"pt6_chunk_count: {len(pt_ds_6)}")
        for idx in range(min(10, len(pt_ds_6))):
            print(f"pt6_chunk[{idx}]: {pt_ds_6[idx]}")
            pad_count = sum(1 for x in pt_ds_6[idx]["input_ids"] if x == tokenizer.pad_token_id)
            print(f"pt6_chunk[{idx}] pad_count: {pad_count}")

        preview_collator_batch(pt_ds_6, pt_collator_6, preview_n=2)


    def run_sft_feature_tests(tokenizer) -> None:
        _print_title("[SFT] 构造测试数据")
        sft_raw = Dataset.from_dict(
            {
                "messages": [
                    [
                        {"role": "user", "content": "你好。"},
                        {"role": "assistant", "content": "你好。"},
                    ],
                    [
                        {"role": "user", "content": "1+1 = 2"},
                        {"role": "assistant", "content": "不对。"},
                        {"role": "user", "content": "你在解释一下。"},
                        {"role": "assistant", "content": "不对。"},
                    ],
                ]
            }
        )
        for i, sample in enumerate(sft_raw):
            print(f"raw_sft[{i}] messages: {sample['messages']}")

        _print_title("[SFT] add_bos_at_start=True + add_eos_at_end=True + apply_chat_template=False")
        sft_builder = SFTMessagesTokenizerBuilder(
            tokenizer=tokenizer,
            max_length=64,
            add_bos_at_start=True,
            add_eos_at_end=True,
            apply_chat_template=False,
            test=True,
        )
        sft_ds, sft_collator = sft_builder.build(sft_raw)
        preview_collator_batch(sft_ds, sft_collator, preview_n=2)


    def run_all_feature_tests() -> None:
        tokenizer_path = "/ruilab/jxhe/CoE_Monitor/Physics_of_LM/Part3/checkpoints/llama2_init_162_42M"
        _print_title("加载 tokenizer")
        tokenizer = AutoTokenizer.from_pretrained(tokenizer_path, trust_remote_code=True)
        print(f"tokenizer是否有template: {bool(getattr(tokenizer, 'chat_template', None))}")
        if tokenizer.pad_token_id is None:
            tokenizer.pad_token = tokenizer.eos_token
        print(f"tokenizer_path: {tokenizer_path}")
        print(f"pad_token_id: {tokenizer.pad_token_id}")
        print(f"eos_token_id: {tokenizer.eos_token_id}")
        print(f"bos_token_id: {tokenizer.bos_token_id}")

        run_pretrain_feature_tests(tokenizer)
        run_sft_feature_tests(tokenizer)

        _print_title("全部测试完成")
    run_all_feature_tests()
