# Copyright (c) Meta Platforms, Inc. and affiliates.

# This file is adapted from the original implementation in:
# https://github.com/facebookresearch/lingua/
# Released under the BSD 3-Clause License by:
# Mathurin Videau*, Badr Youbi Idrissi*, Daniel Haziza, Luca Wehrstedt, Jade Copet, Olivier Teytaud, and David Lopez-Paz.
#
# Modifications made by Zeyuan Allen-Zhu include:
# - greatly removed the original functionality, and adapted to use Huggingface models instead
#
# These modifications are licensed under the Apache 2.0 license, as stated in the root LICENSE file.
#

from dataclasses import dataclass
from typing import Optional, Tuple, Union

import torch
from torch import nn
from torch.nn.attention.flex_attention import create_block_mask, BlockMask

from torch.distributed._tensor import Replicate, Shard
from torch.distributed.tensor.parallel import (
    ColwiseParallel,
    RowwiseParallel,
    SequenceParallel,
    PrepareModuleInput,
    parallelize_module,
)

#from xformers.ops import fmha, AttentionBias
class AttentionBias: ...
from lingua.transformer import (
    BaseTransformer,
    BaseTransformerArgs,
    RMSNorm,
    TiedLinear,
    cross_entropy,
)

from huggingface.configuration_gla_canon import GLACanonConfig
from huggingface.modeling_gla_canon import GLACanonForCausalLM
from huggingface.configuration_gated_deltanet_canon import GatedDeltaNetCanonConfig
from huggingface.configuration_mamba2_canon import Mamba2CanonConfig
from huggingface.modeling_mamba2_canon import Mamba2CanonForCausalLM


def create_causal_mask(seqlen, attn_impl, sliding_window):
    assert False
    if sliding_window is not None and attn_impl == "fmha":
        return fmha.attn_bias.LocalAttentionFromBottomRightMask(
            window_left=sliding_window - 1, window_right=0
        )
    elif attn_impl == "fmha":
        return fmha.attn_bias.LowerTriangularMask()
    elif attn_impl == "sdpa":
        return "causal"
    elif attn_impl == "flex_attention":
        return create_block_mask(causal_mask, None, None, seqlen, seqlen)
    else:
        raise NotImplementedError(
            f"Attention {attn_impl} with {sliding_window} sliding window not implemented"
        )


def attention_flops_per_token(n_layers, seq_len, dim, causal):
    # Formula from https://github.com/Dao-AILab/flash-attention/blob/main/benchmarks/benchmark_flash_attention.py#L27-L30
    return 3.5 * (4 * n_layers * seq_len * dim // (2 if causal else 1))


def get_num_flop_per_token(
    num_non_embed_params: int, n_layers: int, dim: int, seq_len: int
) -> int:
    return 6 * num_non_embed_params + attention_flops_per_token(
        n_layers, seq_len, dim, True
    )


def causal_mask(b, h, q_idx, kv_idx):
    return q_idx >= kv_idx


@dataclass
class LMTransformerArgs(BaseTransformerArgs):

    seed: int = 42

    vocab_size: int = -1
    weight_tying: bool = False
    attn_impl: str = "sdpa"  

    z_loss: bool = False
    model_name: str = "GLA"
    mlp_type: Optional[str] = None  # Only used for Mamba2

    sliding_window: Optional[int] = None

    mamba_ssm: Optional[int] = None  # Only used for Mamba2, default is None, which means ssm_state=256

    zeyuan_linearT: Optional[bool] = False  # Only used for GLA5 so far
    zeyuan_alternate: Optional[bool] = False  # Only used for GLA6 so far, alternating Attn and MLP layers
    zeyuan_layernorm: Optional[bool] = False  # Do not fuse layernorm
    zeyuan_double: Optional[int] = None  # Only used for GLA5 and GLA6, merge every * number of layers, to save FSDP rounds


#class LMTransformer(BaseTransformer):
class LMTransformer(nn.Module):
    def __init__(self, args: LMTransformerArgs):
        super().__init__()
        if args.model_name.startswith("GLA"):
            config = GLACanonForCausalLM.build_config_from_yaml(args)
            self.trans = GLACanonForCausalLM(config)
        elif args.model_name.startswith("GDN"):
            from huggingface.modeling_gated_deltanet_canon import GatedDeltaNetCanonForCausalLM
            config = GatedDeltaNetCanonForCausalLM.build_config_from_yaml(args)
            self.trans = GatedDeltaNetCanonForCausalLM(config)
        elif args.model_name.startswith('Mamba2'):
            config = Mamba2CanonForCausalLM.build_config_from_yaml(args)
            self.trans = Mamba2CanonForCausalLM(config)
        else:
            assert False, f"Unknown model name {args.model_name}"

        
        self.z_loss = args.z_loss

        assert args.vocab_size > 0

    def forward(
        self,
        token_values: torch.Tensor,
        target: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        tok_idx: Optional[torch.Tensor] = None,
        mask: Optional[Union[BlockMask, AttentionBias, torch.Tensor, str]] = None,
        attn_impl: Optional[str] = None,
    ):
        assert tok_idx is None
        logits = self.trans.forward(token_values, attention_mask=attention_mask, use_cache=False, return_dict=True).logits
        if target is not None:
            return cross_entropy(logits, target, z_loss=self.z_loss)
        else:
            return logits

    def reset_parameters(self, init_std=None):
        assert False
        for n, m in self.trans.named_modules():
            if getattr(m, "_is_hf_initialized", False):
                m._is_hf_initialized = False
        self.trans.init_weights()

    def init_weights(self):
        print("<----zeyuan reached init_weights")
        for n, m in self.trans.named_modules():
            if getattr(m, "_is_hf_initialized", False):
                m._is_hf_initialized = False
        self.trans.init_weights()


# Optional policy for activation checkpointing. With None, we stick to the default (defined distributed.py: default_no_recompute_ops)
def get_no_recompute_ops():
    return None


# Optional and only used for fully shard options (fsdp) is choose. Highly recommanded for large models
def build_fsdp_grouping_plan(model_args: LMTransformerArgs):
    group_plan: Tuple[int, bool] = []

    if model_args.model_name=='Mamba2':
        group_plan.append(("trans.backbone.embeddings", False)) # verified for Mamba2
        for i in range(model_args.n_layers):
            group_plan.append((f"trans.backbone.layers.{i}", False))
        group_plan.append(("trans.lm_head", True)) # verified for Mamba2
    elif model_args.model_name in ['Llama']:
        group_plan.append(("trans.model.embed_tokens", False)) # verified for Llama
        for i in range(model_args.n_layers):
            group_plan.append((f"trans.model.layers.{i}", False))
        group_plan.append(("trans.lm_head", True)) # verified for Llama
    else:
        # Grouping and output seperately
        group_plan.append(("trans.model.embeddings", False)) # verified for both GLA and GDN

        # if model_args.model_name=='GDN':
        #     for i in range(model_args.n_layers):
        #         group_plan.append((f"trans.model.layers.{i}.attn.no_shard", "fake"))

        # Grouping by layers
        for i in range(model_args.n_layers):
            group_plan.append((f"trans.model.layers.{i}", False))

        group_plan.append(("trans.lm_head", True)) # verified for both GLA and GDN

    return group_plan


# Optional and only used for model/tensor parallelism when tp_size > 1
def tp_parallelize(model, tp_mesh, model_args: LMTransformerArgs, distributed_args):
    assert model_args.dim % distributed_args.tp_size == 0
    assert model_args.vocab_size % distributed_args.tp_size == 0
    assert model_args.n_heads % distributed_args.tp_size == 0
    assert (model_args.n_kv_heads or 0) % distributed_args.tp_size == 0
    assert model_args.n_heads % (model_args.n_kv_heads or 1) == 0

    # Embedding layer tp
    main_plan = {}
    main_plan["tok_embeddings"] = ColwiseParallel(
        input_layouts=Replicate(), output_layouts=Shard(1)
    )
    main_plan["norm"] = SequenceParallel()
    main_plan["output"] = ColwiseParallel(
        input_layouts=Shard(1), output_layouts=Replicate()
    )

    parallelize_module(
        model,
        tp_mesh,
        main_plan,
    )

    # Attention layers tp
    for layer in model.layers:
        layer_plan = {}

        layer_plan["attention"] = PrepareModuleInput(
            input_layouts=(Shard(1), None),
            desired_input_layouts=(Replicate(), None),
        )
        layer_plan["attention_norm"] = SequenceParallel()
        layer_plan["attention.wq"] = ColwiseParallel()
        layer_plan["attention.wk"] = ColwiseParallel()
        layer_plan["attention.wv"] = ColwiseParallel()
        layer_plan["attention.wo"] = RowwiseParallel(output_layouts=Shard(1))

        # Feedforward layers tp
        layer_plan["feed_forward"] = PrepareModuleInput(
            input_layouts=(Shard(1),),
            desired_input_layouts=(Replicate(),),
        )
        layer_plan["ffn_norm"] = SequenceParallel()
        layer_plan["feed_forward.w1"] = ColwiseParallel()
        layer_plan["feed_forward.w3"] = ColwiseParallel()
        layer_plan["feed_forward.w2"] = RowwiseParallel(output_layouts=Shard(1))

        parallelize_module(
            layer,
            tp_mesh,
            layer_plan,
        )

        # Adjusting the number of heads and kv heads according to the tp size
        attn_layer = layer.attention
        attn_layer.n_heads = attn_layer.n_heads // distributed_args.tp_size
        attn_layer.n_kv_heads = attn_layer.n_kv_heads // distributed_args.tp_size
