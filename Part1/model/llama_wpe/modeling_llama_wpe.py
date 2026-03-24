from collections.abc import Callable

import torch
from torch import nn

from transformers.cache_utils import Cache, DynamicCache
from transformers.generation import GenerationMixin
from transformers.modeling_layers import GradientCheckpointingLayer
from transformers.modeling_outputs import BaseModelOutputWithPast, CausalLMOutputWithPast
from transformers.modeling_utils import ALL_ATTENTION_FUNCTIONS
from transformers.processing_utils import Unpack
from transformers.utils import TransformersKwargs, auto_docstring, can_return_tuple
from transformers.masking_utils import create_causal_mask
from transformers.utils.output_capturing import capture_outputs
import sys
sys.path.append("/ruilab/jxhe/CoE_Monitor/Physics_of_LM/Part1/model/llama")
from configuration_llama import LlamaConfig
from modeling_llama import (
    LlamaForCausalLM,
    LlamaMLP,
    LlamaPreTrainedModel,
    LlamaRMSNorm,
    eager_attention_forward,
    LlamaAttention,
    LlamaDecoderLayer,
    LlamaModel
)



class LlamaAttentionNoRoPE(LlamaAttention):
    def __init__(self, config: LlamaConfig, layer_idx: int):
        super().__init__(config, layer_idx)

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
        past_key_values: Cache | None = None,
        cache_position: torch.LongTensor | None = None,
        **kwargs: Unpack[TransformersKwargs],
    ) -> tuple[torch.Tensor, torch.Tensor]:
        input_shape = hidden_states.shape[:-1]
        hidden_shape = (*input_shape, -1, self.head_dim)

        query_states = self.q_proj(hidden_states).view(hidden_shape).transpose(1, 2)
        key_states = self.k_proj(hidden_states).view(hidden_shape).transpose(1, 2)
        value_states = self.v_proj(hidden_states).view(hidden_shape).transpose(1, 2)

        if past_key_values is not None:
            cache_kwargs = {"cache_position": cache_position}
            key_states, value_states = past_key_values.update(key_states, value_states, self.layer_idx, cache_kwargs)

        attention_interface: Callable = ALL_ATTENTION_FUNCTIONS.get_interface(
            self.config._attn_implementation, eager_attention_forward
        )

        attn_output, attn_weights = attention_interface(
            self,
            query_states,
            key_states,
            value_states,
            attention_mask,
            dropout=0.0 if not self.training else self.attention_dropout,
            scaling=self.scaling,
            **kwargs,
        )

        attn_output = attn_output.reshape(*input_shape, -1).contiguous()
        attn_output = self.o_proj(attn_output)
        return attn_output, attn_weights

    def forward(
        self,
        hidden_states: torch.Tensor,
        position_embeddings: tuple[torch.Tensor, torch.Tensor] | None = None,
        attention_mask: torch.Tensor | None = None,
        past_key_values: Cache | None = None,
        cache_position: torch.LongTensor | None = None,
        **kwargs: Unpack[TransformersKwargs],
    ) -> tuple[torch.Tensor, torch.Tensor]:
        input_shape = hidden_states.shape[:-1]
        hidden_shape = (*input_shape, -1, self.head_dim)

        query_states = self.q_proj(hidden_states).view(hidden_shape).transpose(1, 2)
        key_states = self.k_proj(hidden_states).view(hidden_shape).transpose(1, 2)
        value_states = self.v_proj(hidden_states).view(hidden_shape).transpose(1, 2)

        if past_key_values is not None:
            # sin and cos are specific to RoPE models; cache_position needed for the static cache
            cache_kwargs = {"cache_position": cache_position}
            key_states, value_states = past_key_values.update(key_states, value_states, self.layer_idx, cache_kwargs)

        attention_interface: Callable = ALL_ATTENTION_FUNCTIONS.get_interface(
            self.config._attn_implementation, eager_attention_forward
        )

        attn_output, attn_weights = attention_interface(
            self,
            query_states,
            key_states,
            value_states,
            attention_mask,
            dropout=0.0 if not self.training else self.attention_dropout,
            scaling=self.scaling,
            **kwargs,
        )

        attn_output = attn_output.reshape(*input_shape, -1).contiguous()
        attn_output = self.o_proj(attn_output)
        return attn_output, attn_weights

class LlamaDecoderLayerWPE(LlamaDecoderLayer):
    def __init__(self, config: LlamaConfig, layer_idx: int):
        super().__init__(config, layer_idx)
        self.self_attn = LlamaAttentionNoRoPE(config=config, layer_idx=layer_idx)


class LlamaModelWPE(LlamaModel):
    def __init__(self, config: LlamaConfig):
        super().__init__(config)
        self.wpe = nn.Embedding(config.max_position_embeddings, config.hidden_size)
        self.layers = nn.ModuleList(
            [LlamaDecoderLayerWPE(config, layer_idx) for layer_idx in range(config.num_hidden_layers)]
        )
        self.rotary_emb = None
        self.post_init()
    @capture_outputs
    def forward(
        self,
        input_ids: torch.LongTensor | None = None,
        attention_mask: torch.Tensor | None = None,
        position_ids: torch.LongTensor | None = None,
        past_key_values: Cache | None = None,
        inputs_embeds: torch.FloatTensor | None = None,
        cache_position: torch.LongTensor | None = None,
        use_cache: bool | None = None,
        **kwargs: Unpack[TransformersKwargs],
    ) -> BaseModelOutputWithPast:
        if (input_ids is None) ^ (inputs_embeds is not None):
            raise ValueError("You must specify exactly one of input_ids or inputs_embeds")

        if inputs_embeds is None:
            inputs_embeds: torch.Tensor = self.embed_tokens(input_ids)


        if use_cache and past_key_values is None:
            past_key_values = DynamicCache(config=self.config)

        if cache_position is None:
            past_seen_tokens = past_key_values.get_seq_length() if past_key_values is not None else 0
            cache_position: torch.Tensor = (
                torch.arange(inputs_embeds.shape[1], device=inputs_embeds.device) + past_seen_tokens
            )

        if position_ids is None:
            position_ids = cache_position.unsqueeze(0)


        causal_mask = create_causal_mask(
            config=self.config,
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
            cache_position=cache_position,
            past_key_values=past_key_values,
            position_ids=position_ids,
        )

        position_embeds = self.wpe(position_ids)
        hidden_states = inputs_embeds + position_embeds.to(inputs_embeds.device)

        for decoder_layer in self.layers[: self.config.num_hidden_layers]:
            hidden_states = decoder_layer(
                hidden_states,
                attention_mask=causal_mask,
                position_ids=position_ids,
                past_key_values=past_key_values,
                use_cache=use_cache,
                cache_position=cache_position,
                **kwargs,
            )

        hidden_states = self.norm(hidden_states)
        return BaseModelOutputWithPast(
            last_hidden_state=hidden_states,
            past_key_values=past_key_values,
        )


class LlamaForCausalLMWPE(LlamaForCausalLM, GenerationMixin):

    def __init__(self, config):
        super().__init__(config)
        self.model = LlamaModelWPE(config)
        self.post_init()


def _build_test_config():
    return LlamaConfig(
        vocab_size=128,
        hidden_size=64,
        intermediate_size=128,
        num_hidden_layers=2,
        num_attention_heads=4,
        num_key_value_heads=4,
        max_position_embeddings=64,
        attention_dropout=0.0,
        attention_bias=False,
        mlp_bias=False,
        rope_parameters={"rope_type": "default", "rope_theta": 10000.0},
        pad_token_id=0,
        bos_token_id=1,
        eos_token_id=2,
    )

def test_all_variants_loss_not_nan():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    config = _build_test_config()
    input_ids = torch.tensor([[1, 2, 3, 4, 5, 6, 7, 8]], dtype=torch.long, device=device)
    attention_mask = torch.ones_like(input_ids)
    gpt_types = ["pos"]

    failures = []
    for gpt_type in gpt_types:
        try:
            torch.manual_seed(42)
            model = LlamaForCausalLMWPE._from_config(config).to(device)
            model.eval()
            with torch.no_grad():
                outputs = model(
                    input_ids=input_ids.to(model.device),
                    attention_mask=attention_mask.to(model.device),
                    labels=input_ids.to(model.device),
                    use_cache=False,
                    return_dict=True,
                    output_hidden_states=True,
                )

            hidden_state_stats = outputs.hidden_states[-1].float().mean().item()

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
    gpt_types = ["pos"]

    failures = []
    for gpt_type in gpt_types:
        try:
            torch.manual_seed(42)
            model = LlamaForCausalLMWPE._from_config(config, dtype=torch.bfloat16).to(device)
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
