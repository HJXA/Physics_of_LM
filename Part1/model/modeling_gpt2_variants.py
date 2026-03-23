import math
import os 

import torch
from torch import nn
from typing import Optional
from collections.abc import Callable
from modeling_gpt2 import GPT2Model, GPT2Attention, GPT2Block, GPT2Config
from transformers.cache_utils import Cache, DynamicCache, EncoderDecoderCache
from transformers.masking_utils import create_bidirectional_mask, create_causal_mask
from transformers.modeling_outputs import BaseModelOutputWithPastAndCrossAttentions
from transformers.utils import auto_docstring
from transformers.utils.generic import maybe_autocast, merge_with_config_defaults
from transformers.utils.output_capturing import capture_outputs
from transformers.modeling_utils import ALL_ATTENTION_FUNCTIONS
from transformers.modeling_rope_utils import ROPE_INIT_FUNCTIONS, dynamic_rope_update

# ---------------------------------------------------------
# GPT_rot (GPT with Rotary Positional Embeddings - RoPE)
# 对应论文中的 GPT_rot 架构。
# 其与标准 GPT2 主要改变：
# 去除并停用了原始的绝对位置编码层 (wpe)。在 Attention 计算的 Q (Query) 和 K (Key) 的表征中，
# 直接注入了旋转位置嵌入(RoPE)，以引入相对位置信息。被认为比 relative embeddings 训练更快。
# ---------------------------------------------------------

# =============== RoPE相关 ===================
def precompute_freqs_cis(dim: int, end: int = int(32 * 1024), rope_base: float = 1e6,
                         rope_scaling: Optional[dict] = None):
    freqs, attn_factor = 1.0 / (rope_base ** (torch.arange(0, dim, 2)[: (dim // 2)].float() / dim)), 1.0
    if rope_scaling is not None:
        orig_max, factor, beta_fast, beta_slow, attn_factor = (
            rope_scaling.get("original_max_position_embeddings", 2048), rope_scaling.get("factor", 16),
            rope_scaling.get("beta_fast", 32.0), rope_scaling.get("beta_slow", 1.0), rope_scaling.get("attention_factor", 1.0)
        )
        if end / orig_max > 1.0:
            # YaRN: f'(i) = f(i)((1-γ) + γ/s), where γ∈[0,1] is linear ramp
            inv_dim = lambda b: (dim * math.log(orig_max / (b * 2 * math.pi))) / (2 * math.log(rope_base))
            low, high = max(math.floor(inv_dim(beta_fast)), 0), min(math.ceil(inv_dim(beta_slow)), dim // 2 - 1)
            ramp = torch.clamp((torch.arange(dim // 2, device=freqs.device).float() - low) / max(high - low, 0.001), 0, 1)
            freqs = freqs * (1 - ramp + ramp / factor)

    t = torch.arange(end, device=freqs.device)
    freqs = torch.outer(t, freqs).float()
    freqs_cos = torch.cat([torch.cos(freqs), torch.cos(freqs)], dim=-1) * attn_factor
    freqs_sin = torch.cat([torch.sin(freqs), torch.sin(freqs)], dim=-1) * attn_factor
    return freqs_cos, freqs_sin


def apply_rotary_pos_emb(q, k, cos, sin):
    def rotate_half(x):
        return torch.cat((-x[..., x.shape[-1] // 2:], x[..., : x.shape[-1] // 2]), dim=-1)

    # print("cos:", cos.shape)
    # print("sin:", sin.shape)
    # print("q:", q.shape)
    # print("k:", k.shape)
    cos = cos.unsqueeze(1)
    sin = sin.unsqueeze(1)

    q_embed = (q * cos) + (rotate_half(q) * sin)
    k_embed = (k * cos) + (rotate_half(k) * sin)
    return q_embed, k_embed

class GPT2RotaryEmbedding(nn.Module):
    inv_freq: torch.Tensor  # fix linting for `register_buffer`

    def __init__(self, config, device=None):
        super().__init__()
        self.max_seq_len_cached = config.max_position_embeddings if config.max_position_embeddings is not None else 512
        self.original_max_seq_len = config.max_position_embeddings if config.max_position_embeddings is not None else 512

        self.config = config

        self.rope_type = self.config.rope_parameters["rope_type"] if hasattr(self.config, 'rope_parameters') and self.config.rope_parameters is not None else "default"
        rope_init_fn: Callable = self.compute_default_rope_parameters
        if self.rope_type != "default":
            rope_init_fn = ROPE_INIT_FUNCTIONS[self.rope_type]
        inv_freq, self.attention_scaling = rope_init_fn(self.config, device)

        self.register_buffer("inv_freq", inv_freq, persistent=False)
        self.register_buffer("original_inv_freq", inv_freq.clone(), persistent=False)

    @staticmethod
    def compute_default_rope_parameters(
        config = None,
        device: Optional["torch.device"] = None,
        seq_len: int | None = None,
    ) -> tuple["torch.Tensor", float]:
        """
        Computes the inverse frequencies according to the original RoPE implementation
        Args:
            config ([`~transformers.PreTrainedConfig`]):
                The model configuration.
            device (`torch.device`):
                The device to use for initialization of the inverse frequencies.
            seq_len (`int`, *optional*):
                The current sequence length. Unused for this type of RoPE.
        Returns:
            Tuple of (`torch.Tensor`, `float`), containing the inverse frequencies for the RoPE embeddings and the
            post-processing scaling factor applied to the computed cos/sin (unused in this type of RoPE).
        """
        base = config.rope_parameters["rope_theta"] if config.rope_parameters is not None and "rope_theta" in config.rope_parameters else 10_000.0
        dim = config.n_embd // config.n_head

        attention_factor = 1.0  # Unused in this type of RoPE

        # Compute the inverse frequencies
        inv_freq = 1.0 / (
            base ** (torch.arange(0, dim, 2, dtype=torch.int64).to(device=device, dtype=torch.float) / dim)
        )
        return inv_freq, attention_factor

    @torch.no_grad()
    @dynamic_rope_update  # power user: used with advanced RoPE types (e.g. dynamic rope)
    def forward(self, x, position_ids):
        inv_freq_expanded = self.inv_freq[None, :, None].float().expand(position_ids.shape[0], -1, 1).to(x.device)
        position_ids_expanded = position_ids[:, None, :].float()

        device_type = x.device.type if isinstance(x.device.type, str) and x.device.type != "mps" else "cpu"
        with maybe_autocast(device_type=device_type, enabled=False):  # Force float32
            freqs = (inv_freq_expanded.float() @ position_ids_expanded.float()).transpose(1, 2)
            emb = torch.cat((freqs, freqs), dim=-1)
            cos = emb.cos() * self.attention_scaling
            sin = emb.sin() * self.attention_scaling

        return cos.to(dtype=x.dtype), sin.to(dtype=x.dtype)



def eager_attention_forward(module, query, key, value, attention_mask, **kwargs):
    attn_weights = torch.matmul(query, key.transpose(-1, -2))

    if module.scale_attn_weights:
        attn_weights = attn_weights / torch.full(
            [], value.size(-1) ** 0.5, dtype=attn_weights.dtype, device=attn_weights.device
        )

    # Layer-wise attention scaling
    if module.scale_attn_by_inverse_layer_idx:
        attn_weights = attn_weights / float(module.layer_idx + 1)


    if attention_mask is not None:
        # Apply the attention mask
        attn_weights = attn_weights + attention_mask

    attn_weights = nn.functional.softmax(attn_weights, dim=-1)

    # Downcast (if necessary) back to V's dtype (if in mixed-precision) -- No-Op otherwise
    attn_weights = attn_weights.type(value.dtype)
    attn_weights = module.attn_dropout(attn_weights)

    attn_output = torch.matmul(attn_weights, value)
    attn_output = attn_output.transpose(1, 2)

    return attn_output, attn_weights

class GPTRotAttention(GPT2Attention):
    def __init__(self, config, is_cross_attention=False, layer_idx=None):
        super().__init__(config, is_cross_attention, layer_idx)
        


    def forward(
        self,
        hidden_states,
        position_embeddings:tuple[torch.Tensor, torch.Tensor],
        past_key_values: Cache | None = None,
        cache_position: torch.LongTensor | None = None,
        attention_mask=None,
        **kwargs,
    ):
        # 取代了标准架构 Q 和 K 的单纯点积：
        # 将输入拆分为 Q、K、V 后，将 RoPE (旋转编码的sin和cos计算分量) 应用于 Q 和 K。
        query_states, key_states, value_states = self.c_attn(hidden_states).split(self.split_size, dim=2)
    
        shape_kv = (*key_states.shape[:-1], -1, self.head_dim)
        query_states = query_states.view(shape_kv).transpose(1, 2)
        key_states = key_states.view(shape_kv).transpose(1, 2)
        value_states = value_states.view(shape_kv).transpose(1, 2)

        cos, sin = position_embeddings

        query_states, key_states = apply_rotary_pos_emb(
            query_states, key_states, cos, sin
        )

        if past_key_values is not None:
            # sin and cos are specific to RoPE models; cache_position needed for the static cache
            cache_kwargs = {"sin": sin, "cos": cos, "cache_position": cache_position}
            key_states, value_states = past_key_values.update(key_states, value_states, self.layer_idx, cache_kwargs)


        attention_interface: Callable = ALL_ATTENTION_FUNCTIONS.get_interface(
            self.config._attn_implementation, eager_attention_forward
        )


        # 其余情况：走配置指定的 attention 后端（如 eager 默认路径/SDPA/flash 等）。
        attn_output, attn_weights = attention_interface(
            self,
            query_states,
            key_states,
            value_states,
            attention_mask,
            dropout=self.attn_dropout.p if self.training else 0.0,
            **kwargs,
        )

        attn_output = attn_output.reshape(*attn_output.shape[:-2], -1).contiguous()
        attn_output = self.c_proj(attn_output)
        attn_output = self.resid_dropout(attn_output)

        return attn_output, attn_weights


class DummyZeroEmbedding(nn.Module):
    """
    一个 0 参数的占位 Embedding 层。
    接收 position_ids，返回全 0 张量，彻底消除绝对位置编码的影响和参数开销。
    """
    def __init__(self, embedding_dim):
        super().__init__()
        self.embedding_dim = embedding_dim
        # 注册一个标量 buffer，用于自动追踪模型的 device 和 dtype
        self.register_buffer("dummy_tracker", torch.tensor(0.0))

    def forward(self, position_ids):
        # position_ids shape: (batch_size, seq_len)
        # 返回形状: (batch_size, seq_len, embedding_dim)
        return torch.zeros(
            *position_ids.shape, 
            self.embedding_dim, 
            device=self.dummy_tracker.device, 
            dtype=self.dummy_tracker.dtype
        )

class CustomGPT2Model(GPT2Model):
    """
    通用 GPT 集成入口。
    gpt_type 参数选择包括: 
      - 'standard': 原始 GPT-2 / DeBERTa (带有绝对位置编码，或保持原样)
      - 'rot'     : GPT_rot (旋转位置编码)
      - 'rel'     : GPT_rel (基于内容的 DeBERTa 式相对位置偏置注意力)
      - 'pos'     : GPT_pos (仅包含位置偏置信息，无 Q-K 乘积的自注意力)
      - 'uni'     : GPT_uni (完全固定并取移动窗口指数平均的纯位置注意力)
      - 'rand'    : 如需实现 GPT_rand，调用 'rel'，并在训练过程中冻结模型使其不发生梯度下降，发挥随机构建 Kernel 特征的能力。
    """
    def __init__(self, config, gpt_type='standard'):
        super().__init__(config)
        self.gpt_type = gpt_type

        # 根据选定的 gpt_type 改写原始模型中的 Transformer Block 注意力替换实现
        self.h = nn.ModuleList([create_custom_gpt_block(config, i, gpt_type) for i in range(config.num_hidden_layers)])
        
        if gpt_type in ['rot', 'rel', 'pos', 'uni']:
            # 这些类型要么使用的是 RoPE，要么是通过 Relative Position Bias。
            self.wpe = DummyZeroEmbedding(config.n_embd)

        if gpt_type == 'rot':
            # RoPE 相关：预计算频率表，注册为 buffer 以便在 forward 中使用
            self.rotary_emb = GPT2RotaryEmbedding(config=config)

    @merge_with_config_defaults
    @capture_outputs
    @auto_docstring
    def forward(
        self,
        input_ids: torch.LongTensor | None = None,
        past_key_values: Cache | None = None,
        cache_position: torch.LongTensor | None = None,
        attention_mask: torch.FloatTensor | None = None,
        token_type_ids: torch.LongTensor | None = None,
        position_ids: torch.LongTensor | None = None,
        inputs_embeds: torch.FloatTensor | None = None,
        use_cache: bool | None = None,
        **kwargs,
    ) -> BaseModelOutputWithPastAndCrossAttentions:
        r"""
        input_ids (`torch.LongTensor` of shape `(batch_size, input_ids_length)`):
            `input_ids_length` = `sequence_length` if `past_key_values` is `None` else
            `past_key_values.get_seq_length()` (`sequence_length` of input past key value states). Indices of input
            sequence tokens in the vocabulary.

            If `past_key_values` is used, only `input_ids` that do not have their past calculated should be passed as
            `input_ids`.

            Indices can be obtained using [`AutoTokenizer`]. See [`PreTrainedTokenizer.encode`] and
            [`PreTrainedTokenizer.__call__`] for details.

            [What are input IDs?](../glossary#input-ids)
        """
        if self.gpt_type != 'rot':
            return super().forward(
                input_ids=input_ids,
                past_key_values=past_key_values,
                cache_position=cache_position,
                attention_mask=attention_mask,
                token_type_ids=token_type_ids,
                position_ids=position_ids,
                inputs_embeds=inputs_embeds,
                use_cache=use_cache,
                **kwargs,
            )
        kwargs.pop("output_attentions", None)
        kwargs.pop("output_hidden_states", None)

        # 输入检查

        if input_ids is not None and inputs_embeds is not None:
            raise ValueError("You cannot specify both input_ids and inputs_embeds at the same time")
        elif input_ids is not None:
            self.warn_if_padding_and_no_attention_mask(input_ids, attention_mask)
            input_shape = input_ids.size()
            input_ids = input_ids.view(-1, input_shape[-1])
            batch_size = input_ids.shape[0]
        elif inputs_embeds is not None:
            input_shape = inputs_embeds.size()[:-1]
            batch_size = inputs_embeds.shape[0]
        else:
            raise ValueError("You have to specify either input_ids or inputs_embeds")

        # KV Cache
        if use_cache and past_key_values is None:
            past_key_values = DynamicCache(config=self.config)

        # 嵌入层输入：如果外部没有直接传入 inputs_embeds，则通过词嵌入层 wte 将 input_ids 转换为嵌入表示。

        if inputs_embeds is None:
            inputs_embeds = self.wte(input_ids)

        # 位置编码输入：GPT_rot 内部通过 cache_position 定位当前输入序列的起始位置，从而正确切片预计算的 RoPE 频率表，确保位置编码与输入序列对齐。

        if cache_position is None:
            past_seen_tokens = past_key_values.get_seq_length() if past_key_values is not None else 0
            cache_position = torch.arange(
                past_seen_tokens, past_seen_tokens + inputs_embeds.shape[1], device=inputs_embeds.device
            )
        # 如果外部没有传入 position_ids，这里也会在内部构造一个基于 cache_position 的位置 ID 序列，供 RoPE 计算使用。

        if position_ids is None:
            position_ids = cache_position.unsqueeze(0)

        hidden_states = inputs_embeds

        # Attention mask 构建：GPT_rot 内部会根据输入的 attention_mask 构建一个联合了因果 mask 和 padding mask 的最终掩码，供每层 block.attn 使用。
        if attention_mask is not None and attention_mask.ndim < 4:
            attention_mask = attention_mask.view(batch_size, -1)

        # 关键点：即使外部只传了 2D padding attention_mask，这里也会在内部构造因果 mask（下三角） # (seq_len × seq_len)
        # 并与 padding 信息融合，得到最终用于注意力计算的 4D 掩码。 # (batch_size, num_heads, query_len, key_len)
        causal_mask = create_causal_mask(
            config=self.config,
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
            cache_position=cache_position,
            past_key_values=past_key_values,
            position_ids=position_ids,
        )

        hidden_states = self.drop(hidden_states)
        # position_embeddings 的计算：GPT_rot 内部通过 cache_position 定位当前输入序列的起始位置，从而正确切片预计算的 RoPE 频率表，确保位置编码与输入序列对齐。
        if self.gpt_type == 'rot':
            position_embeddings = self.rotary_emb(hidden_states, position_ids)
        else:
            position_embeddings = None

        output_shape = (-1,) + input_shape[1:] + (hidden_states.size(-1),) # (-1, seq_len, hidden_dim)

        for i, block in enumerate(self.h):
            hidden_states = block(
                hidden_states,
                past_key_values if not (self.gradient_checkpointing and self.training) else None,
                cache_position=cache_position,
                attention_mask=causal_mask,  # 这里把“内部生成好的因果+padding联合mask”传给每一层 block.attn
                use_cache=use_cache,
                position_embeddings=position_embeddings,
                position_ids=position_ids,
                **kwargs,
            )

        hidden_states = self.ln_f(hidden_states)

        hidden_states = hidden_states.view(output_shape)

        past_key_values = past_key_values if use_cache else None
        return BaseModelOutputWithPastAndCrossAttentions(
            last_hidden_state=hidden_states,
            past_key_values=past_key_values,
        )


# ---------------------------------------------------------
# GPT_rel 与 GPT_pos (基于 DeBERTa 框架的相对位置注意力)
#
# 对应论文中的 GPT_rel 与 GPT_pos 架构，这两者通过 mode='rel' / 'pos' 区分：
# 共通改变：去除了标准的绝对位置编码层。增加了一个可供训练的相对位置偏置 (Relative Position Bias)。
#
# GPT_rel (mode='rel'):
# - 标准注意力机制下附加该可供学习的“相对距离位置嵌入(|j - i|)”。这是在算好了 Q @ K^T 之后，加上偏置矩阵。
# 
# GPT_pos (mode='pos', Position-Only 注意力):
# - 它的 Attention 矩阵 **仅由位置决定**。
# - 直接省去 Q @ K^T 的点积计算（与 input 和 hidden states 无关），转而纯依靠相对位置的 Bias 进行 softmax 计算。
# - 是为了探索极限削弱版（无内容感知）的情况下单纯位置序列的推断能力。
# ---------------------------------------------------------
class GPT_Rel_Pos_Attention(GPT2Attention):
    def __init__(self, config, mode='rel', layer_idx=None):
        super().__init__(config, is_cross_attention=False, layer_idx=layer_idx)
        self.mode = mode # 'rel' or 'pos'
        self.relative_attention_bias = nn.Embedding(config.max_position_embeddings * 2, self.num_heads)

    def forward(
        self,
        hidden_states,
        past_key_values: Cache | None = None,
        cache_position: torch.LongTensor | None = None,
        attention_mask=None,
        **kwargs,
    ):
        if past_key_values is not None:
                curr_past_key_values = past_key_values

        query_length = hidden_states.size(1)
        if self.mode == 'rel':
            # GPT_rel：保持特征交互能力，执行 Q 与 K 的点积
            query_states, key_states, value_states = self.c_attn(hidden_states).split(self.split_size, dim=2)
            shape_kv = (*key_states.shape[:-1], -1, self.head_dim)
            query_states = query_states.view(shape_kv).transpose(1, 2) # qkv一样的 shape 变换
            key_states = key_states.view(shape_kv).transpose(1, 2)
            value_states = value_states.view(shape_kv).transpose(1, 2)


            if past_key_values is not None:
                key_states, value_states = curr_past_key_values.update(
                    key_states, value_states, self.layer_idx, {"cache_position": cache_position}
                )

            if cache_position is None:
                query_position = torch.arange(query_length, dtype=torch.long, device=hidden_states.device)
            else:
                query_position = cache_position.to(device=hidden_states.device, dtype=torch.long)

            key_length = key_states.size(-2)
            
            # 构建距离矩阵 (|j - i|) 映射为学习好的相对位置偏差(Bias)
            memory_position = torch.arange(key_length, dtype=torch.long, device=hidden_states.device)[None, :]
            relative_position = memory_position - query_position[:, None]
            relative_position = relative_position + self.config.max_position_embeddings
            relative_position = relative_position.clamp(0, self.config.max_position_embeddings * 2 - 1)
            rel_pos_bias = self.relative_attention_bias(relative_position).permute(2, 0, 1).unsqueeze(0).to(dtype=query_states.dtype)

            # 以下是正常的注意力机制

            combined_mask = rel_pos_bias
            if attention_mask is not None:
                combined_mask = combined_mask + attention_mask

            attention_interface: Callable = eager_attention_forward


            # 其余情况：走配置指定的 attention 后端（如 eager 默认路径/SDPA/flash 等）。
            attn_output, attn_weights = attention_interface(
                self,
                query_states,
                key_states,
                value_states,
                combined_mask,
                dropout=self.attn_dropout.p if self.training else 0.0,
                **kwargs,
            )

            attn_output = attn_output.reshape(*attn_output.shape[:-2], -1).contiguous()
            attn_output = self.c_proj(attn_output)
            attn_output = self.resid_dropout(attn_output)

            return attn_output, attn_weights

        else:
            # GPT_pos (纯位置): 去除了输入序列的Query和Key计算！
            # 注意力矩阵不再与隐藏状态发生交互内容。
            _, key_states, value_states = self.c_attn(hidden_states).split(self.split_size, dim=2)
            shape_kv = (*value_states.shape[:-1], -1, self.head_dim)
            key_states = key_states.view(shape_kv).transpose(1, 2)
            value_states = value_states.view(shape_kv).transpose(1, 2)


            if past_key_values is not None:
                _ , value_states = curr_past_key_values.update(
                    key_states, value_states, self.layer_idx, {"cache_position": cache_position}
                )

            key_length = value_states.size(-2)
            if cache_position is None:
                query_position = torch.arange(query_length, dtype=torch.long, device=hidden_states.device)
            else:
                query_position = cache_position.to(device=hidden_states.device, dtype=torch.long)

            # 初始化为空的全零注意力矩阵 (因为没有 Q和K 来计算特征内容相似度)
            attn_weights = torch.zeros(
                hidden_states.size(0),
                self.num_heads,
                query_length,
                key_length,
                device=hidden_states.device,
                dtype=value_states.dtype,
            )

            # 构建距离矩阵 (|j - i|) 映射为学习好的相对位置偏差(Bias)
            memory_position = torch.arange(key_length, dtype=torch.long, device=hidden_states.device)[None, :]
            relative_position = memory_position - query_position[:, None]
            relative_position = relative_position + self.config.max_position_embeddings
            relative_position = relative_position.clamp(0, self.config.max_position_embeddings * 2 - 1)
            rel_pos_bias = self.relative_attention_bias(relative_position).permute(2, 0, 1).unsqueeze(0)
            rel_pos_bias = rel_pos_bias.to(dtype=value_states.dtype)
            
            # 将结构/相对位置信息加上注意力矩阵，供最后 softmax
            attn_weights = attn_weights + rel_pos_bias

            if attention_mask is not None:
                attn_weights = attn_weights + attention_mask
                
            attn_weights = nn.functional.softmax(attn_weights, dim=-1)
            attn_output = torch.matmul(attn_weights, value_states).transpose(1, 2).reshape(*hidden_states.shape[:-1], -1)
            return self.c_proj(attn_output), attn_weights

# ---------------------------------------------------------
# GPT_uni (Uniform position-based attention)
# 对应论文中的 GPT_uni 架构。
#
# 最大改变和其代表的含义：
# 1. Attention 矩阵是被完全固定的（不可训练且与输入序列无关），只应用最简的位置注意力。
# 2. 这里的 h 头，不去做特征上的注意力投射 (无 Q 和 K 参数），只保留 Value 参数和输出投影。
# 3. 每 h 头对过去 (2^h - 1) 个 Tokens 执行一致均匀的权重平均 (Uniform Average)。
# ---------------------------------------------------------
class GPTUniAttention(nn.Module):
    def __init__(self, config, layer_idx=None):
        super().__init__()
        self.layer_idx = layer_idx
        self.num_heads = config.num_attention_heads # e.g. 论文中为 8 头
        self.embed_dim = config.hidden_size # 论文中指出：较小的保持参数一致时是 840维，较大设定是 1024维
        self.head_dim = self.embed_dim // self.num_heads
        
        # 极简设计：不需要提取特征的自注意力权重，只执行Value线性投射与输出整合。
        self.c_fc = nn.Linear(self.embed_dim, self.embed_dim) 
        self.c_proj = nn.Linear(self.embed_dim, self.embed_dim)

    def forward(
        self,
        hidden_states,
        past_key_values: Cache | None = None,
        cache_position: torch.LongTensor | None = None,
        attention_mask=None,
        **kwargs,
    ):
        if past_key_values is not None:
            curr_past_key_values = past_key_values
        bsz, query_len, _ = hidden_states.shape
        value_states = self.c_fc(hidden_states)
        value_states = value_states.view(bsz, query_len, self.num_heads, self.head_dim).transpose(1, 2)
        key_states = value_states
        if past_key_values is not None:
                _ , value_states = curr_past_key_values.update(
                    key_states, value_states, self.layer_idx, {"cache_position": cache_position}
                )

        key_len = value_states.size(-2)
        if cache_position is None:
            query_positions = torch.arange(query_len, device=hidden_states.device, dtype=torch.long)
        else:
            query_positions = cache_position.to(device=hidden_states.device, dtype=torch.long)
        
        # 建立固定的 Uniform 注意力矩阵（恒定不可变）
        attn_weights = torch.zeros(bsz, self.num_heads, query_len, key_len, device=hidden_states.device, dtype=value_states.dtype)
        for h in range(self.num_heads):
            # 取以前的 2^h - 1 个Token窗口（依据论文的指数增长型上下文衰减）
            window_size = (2 ** (h + 1)) - 1
            for i in range(query_len):
                abs_pos = int(query_positions[i].item())
                abs_pos = max(0, min(abs_pos, key_len - 1))
                start = max(0, abs_pos - window_size + 1)
                valid_len = abs_pos - start + 1
                # 执行均匀平均 (1.0 / window)
                attn_weights[:, h, i, start:abs_pos + 1] = 1.0 / valid_len
                
        if attention_mask is not None:
            # Mask out padding if any
            attn_weights = attn_weights.masked_fill(attention_mask < -1e4, 0.0)
            attn_weights = attn_weights / (attn_weights.sum(dim=-1, keepdim=True) + 1e-9)

        attn_output = torch.matmul(attn_weights, value_states).transpose(1, 2).reshape(bsz, query_len, -1)
        return self.c_proj(attn_output), attn_weights

# ---------------------------------------------------------
# Factory function for creating GPT blocks based on type
# ---------------------------------------------------------
def create_custom_gpt_block(config, layer_idx, gpt_type='standard'):
    block = GPT2Block(config, layer_idx=layer_idx)
    if gpt_type == 'rot':
        block.attn = GPTRotAttention(config, layer_idx=layer_idx)
    elif gpt_type == 'rel':
        block.attn = GPT_Rel_Pos_Attention(config, mode='rel', layer_idx=layer_idx)
    elif gpt_type == 'pos':
        block.attn = GPT_Rel_Pos_Attention(config, mode='pos', layer_idx=layer_idx)
    elif gpt_type == 'uni':
        block.attn = GPTUniAttention(config, layer_idx=layer_idx)
    return block




from modeling_gpt2 import GPT2LMHeadModel

class CustomGPT2LMHeadModel(GPT2LMHeadModel):
    """
    用于带有语言模型（LM Head）任务预训练的模型封装。
    """
    def __init__(self, config, gpt_type='standard'):
        super().__init__(config)
        self.gpt_type = gpt_type
        if gpt_type == 'standard':
            # 标准 GPT2 架构保持不变，使用原始的绝对位置编码和注意力机制
            return
        # 抛弃预设的 GPT2Model 实例，将其替换为包含变体注意力的定制骨干网络
        self.transformer = CustomGPT2Model(config, gpt_type=gpt_type)
        # 二次执行 post_init 将其权重重新随机初始化，并确保 Embedding 与新实例化的 transformer 权重重新绑定同步
        self.post_init()


def _build_test_config():
    return GPT2Config(
        vocab_size=128,
        n_positions=64,
        n_ctx=64,
        n_embd=64,
        n_layer=2,
        n_head=4,
        resid_pdrop=0.0,
        embd_pdrop=0.0,
        attn_pdrop=0.0,
        attn_implementation="eager", # eager flash_attention_2 # 目前rot支持flash
        gradient_checkpointing=True,
        dtype=torch.bfloat16,
    )


def test_all_variants_loss_not_nan():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    config = _build_test_config()
    input_ids = torch.tensor([[1, 2, 3, 4, 5, 6, 7, 8]], dtype=torch.long, device=device)
    attention_mask = torch.ones_like(input_ids)
    gpt_types = ["standard", "rot", "rel", "pos", "uni"]

    failures = []
    for gpt_type in gpt_types:
        try:
            torch.manual_seed(42)
            model = CustomGPT2LMHeadModel._from_config(config, gpt_type=gpt_type).to(device)
            # model = GPT2LMHeadModel.from_pretrained(
            #     "/ruilab/jxhe/CoE_Monitor/Physics_of_LM/Part1/checkpoints/GPT_2_Init/GPT_2_standard",
            #     attn_implementation="flash_attention_2",
            #     dtype=torch.bfloat16,
            #     device_map="auto"
		    # )
            model.eval()
            with torch.no_grad():
                outputs = model(
                    input_ids=input_ids.to(model.device),
                    attention_mask=attention_mask.to(model.device),
                    labels=input_ids.to(model.device),
                    use_cache=False,
                )

            loss = outputs.loss
            if loss is None:
                failures.append(f"{gpt_type}: loss is None")
                print(f"[{gpt_type}] loss FAIL | loss is None")
                continue

            loss_value = float(loss.detach().cpu().item())
            is_nan = torch.isnan(loss).any().item()
            is_finite = torch.isfinite(loss).all().item()

            if is_nan or (not is_finite):
                failures.append(f"{gpt_type}: loss invalid (nan={is_nan}, finite={is_finite}, value={loss_value})")
                print(f"[{gpt_type}] loss BAD | value={loss_value}, nan={is_nan}, finite={is_finite}")
            else:
                print(f"[{gpt_type}] loss OK | value={loss_value:.6f}")
        except Exception as exc:
            failures.append(f"{gpt_type}: {type(exc).__name__}: {exc}")
            print(f"[{gpt_type}] loss FAIL | {type(exc).__name__}: {exc}")
            raise

    if failures:
        print("Loss test failed:\n" + "\n".join(failures))

    print("All model variants passed loss validity test (non-NaN, finite).")


def test_all_variants_cache_consistency():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    config = _build_test_config()
    prefix_ids = torch.tensor([[1, 2, 3, 4, 5]], dtype=torch.long, device=device)
    next_ids = torch.tensor([[6, 7, 8]], dtype=torch.long, device=device)
    full_ids = torch.cat([prefix_ids, next_ids], dim=1)
    prefix_attention_mask = torch.ones_like(prefix_ids)
    full_attention_mask = torch.ones_like(full_ids)
    gpt_types = ["standard", "rot", "rel", "pos", "uni"]

    failures = []
    for gpt_type in gpt_types:
        try:
            torch.manual_seed(42)
            model = CustomGPT2LMHeadModel._from_config(config, gpt_type=gpt_type).to(device)
            # model = GPT2LMHeadModel.from_pretrained(
            #     "/ruilab/jxhe/CoE_Monitor/Physics_of_LM/Part1/checkpoints/GPT_2_Init/GPT_2_standard",
            #     attn_implementation="flash_attention_2",
            #     dtype=torch.bfloat16,
            #     device_map="auto"
		    # )
            model.eval()

            with torch.no_grad():
                first_outputs = model(
                    input_ids=prefix_ids,
                    attention_mask=prefix_attention_mask,
                    use_cache=True,
                )
                past_key_values = first_outputs.past_key_values
                if past_key_values is None:
                    failures.append(f"{gpt_type}: past_key_values is None when use_cache=True")
                    continue

                first_cache_len = past_key_values.get_seq_length()
                if first_cache_len != prefix_ids.size(1):
                    failures.append(
                        f"{gpt_type}: first cache length mismatch (got={first_cache_len}, expect={prefix_ids.size(1)})"
                    )
                    continue

                cached_outputs = model(
                    input_ids=next_ids,
                    attention_mask=full_attention_mask,
                    past_key_values=past_key_values,
                    use_cache=True,
                )
                cached_logits = cached_outputs.logits
                updated_cache = cached_outputs.past_key_values

                if updated_cache is None:
                    failures.append(f"{gpt_type}: updated past_key_values is None")
                    continue

                updated_cache_len = updated_cache.get_seq_length()
                if updated_cache_len != full_ids.size(1):
                    failures.append(
                        f"{gpt_type}: updated cache length mismatch (got={updated_cache_len}, expect={full_ids.size(1)})"
                    )
                    continue

                full_outputs = model(
                    input_ids=full_ids,
                    attention_mask=full_attention_mask,
                    use_cache=False,
                )
                full_tail_logits = full_outputs.logits[:, -next_ids.size(1):, :]

                max_abs_diff = (cached_logits - full_tail_logits).abs().max().item()
                if not torch.allclose(cached_logits, full_tail_logits, atol=1e-5, rtol=1e-4):
                    failures.append(f"{gpt_type}: cache logits mismatch (max_abs_diff={max_abs_diff:.6e})")
                else:
                    print(f"[{gpt_type}] cache OK | max_abs_diff={max_abs_diff:.6e}")

        except Exception as exc:
            failures.append(f"{gpt_type}: {type(exc).__name__}: {exc}")
            print(f"[{gpt_type}] cache FAIL | {type(exc).__name__}: {exc}")
            raise

    if failures:
        raise RuntimeError("Cache test failed:\n" + "\n".join(failures))

    print("All model variants passed cache consistency test.")


if __name__ == "__main__":
    
    test_all_variants_loss_not_nan()
    test_all_variants_cache_consistency()
    
