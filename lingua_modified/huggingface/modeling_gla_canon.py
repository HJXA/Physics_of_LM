# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This code is modified from https://github.com/fla-org/flash-linear-attention/blob/main/fla/models/gla/modeling_gla.py (Jan/3/2025 version)
# Original authors: Songlin Yang, Yu Zhang and others
# Under MIT License
#
# Zeyuan's edits include: 
# - canon layers
# - from_pretrained/build_config_from_yaml/load_from_lingua_state (lingua <-> HF toolkit)
# - zeyuan_split_init
# - zeyuan_layernorm
# - output_gate_rank
#
from __future__ import annotations

# Zeyuan's edit note: this code is tested using fla==0.3.0 and transformers==4.47.1
#                     this code uses `num_logits_to_keep` which is compatible with transformers==4.47.1 but not with transformers>=4.53. 
#                     fla-org's newest codebase (>=0.4.0) is compatible with transformers>=4.53, but I don't have the time to modify this code.
import importlib.metadata as m, sys
if m.version("flash-linear-attention") != "0.3.0":
    sys.exit(f"❌ fla==0.3.0 required, found {m.version('fla')}")
if m.version("transformers") != "4.47.1":
    sys.exit(f"❌ transformers==4.47.1 required, found {m.version('transformers')}")
print("✅ Dependencies verified.")

import math
import warnings
from typing import TYPE_CHECKING, Dict, List, Optional, Tuple, Union

import torch
import torch.nn as nn
import torch.utils.checkpoint
#from transformers.activations import ACT2FN
from transformers.generation import GenerationMixin
from transformers.modeling_outputs import (BaseModelOutputWithPast,
                                           CausalLMOutputWithPast)
from transformers.modeling_utils import PreTrainedModel
from transformers.utils import logging

from fla.layers.attn import Attention
from .configuration_gla_canon import GLACanonConfig
from fla.models.utils import Cache
from fla.modules import (FusedCrossEntropyLoss, FusedLinearCrossEntropyLoss,
                         RMSNorm)
#from fla.modules.activations import swiglu_linear
from fla.modules.activations import swiglu_linear as swiglu_linear_old
@torch._dynamo.disable
def swiglu_linear(*args, **kwargs):
    return swiglu_linear_old(*args, **kwargs)


if TYPE_CHECKING:
    from transformers.processing_utils import Unpack

logger = logging.get_logger(__name__)

class MyRMSNorm(nn.Module):
    """
    Initialize the RMSNorm normalization layer.

    Args:
        dim (int): The dimension of the input tensor.
        eps (float, optional): A small value added to the denominator for numerical stability. Default is 1e-6.

    Attributes:
        eps (float): A small value added to the denominator for numerical stability.
        weight (nn.Parameter): Learnable scaling parameter.

    """

    def __init__(self, hidden_size: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(hidden_size))

    def _norm(self, x: torch.Tensor):
        # Accumulate in fp32 WITHOUT upcasting the whole x
        var = torch.mean(torch.square(x), dim=-1, keepdim=True, dtype=torch.float32)
        rstd = torch.rsqrt(var + self.eps)          # fp32, shape [..., 1]
        rstd = rstd.to(dtype=x.dtype)               # keep elementwise mul in x dtype
        return x * rstd

    def forward(self, x: torch.Tensor):
        out = self._norm(x)
        return (out * self.weight.to(dtype=x.dtype))

    def reset_parameters(self):
        torch.nn.init.ones_(self.weight)  # type: ignore

from .canon_helper import create_canon, apply_canon, make_canon_layer


from typing import TYPE_CHECKING, Dict, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange, repeat

from fla.modules import FusedRMSNormSwishGate, RMSNorm, ShortConvolution
from fla.modules.activations import ACT2FN

import fla
from packaging import version
if version.parse(fla.__version__) >= version.parse("0.3.0"):
    from fla.ops.utils import prepare_position_ids, prepare_sequence_ids
else:
    from fla.ops.common.utils import prepare_position_ids, prepare_sequence_ids
    assert False, "code not tested with fla<0.3.0; model generation may be incorrect."

#from fla.ops.gla import chunk_gla, fused_chunk_gla, fused_recurrent_gla
from fla.ops.gla import chunk_gla as _chunk_gla, fused_chunk_gla as _fused_chunk_gla, fused_recurrent_gla as _fused_recurrent_gla

@torch._dynamo.disable
def chunk_gla(*args, **kwargs):
    return _chunk_gla(*args, **kwargs)
@torch._dynamo.disable
def fused_chunk_gla(*args, **kwargs):
    return _fused_chunk_gla(*args, **kwargs)
@torch._dynamo.disable
def fused_recurrent_gla(*args, **kwargs):
    return _fused_recurrent_gla(*args, **kwargs)

if TYPE_CHECKING:
    from transformers.processing_utils import Unpack

    from fla.models.utils import Cache



class GatedCanonLinearAttention(nn.Module):
    r"""
    The layer implementaion for [Gated Linear Attention Transformers with Hardware-Efficient Training](https://arxiv.org/abs/2312.06635).  # noqa

    Args:
        mode (str, Optional):
            Which GLA kernel to use.
            Currently available: `chunk`, `fused_recurrent`, and `fused_chunk`.
            Default: `chunk`.
        hidden_size (int, Optional):
            The hidden size of the input. Default: 1024.
        expand_k (float, Optional):
            The expansion ratio for the key dim. Default: 0.5.
        expand_v (float, Optional):
            The expansion ratio for the value dim. Default: 1.0.
        num_heads (int, Optional):
            The number of heads. Default: 4.
        num_kv_heads (int, Optional):
            The number of key/value heads, used for MQA. Default: None.
        feature_map (str, Optional):
            Feature map function applied to queries/keys. Default: None.
        use_short_conv (bool, Optional):
            Whether to use short convolutions. Default: `False`.
        conv_size (int, Optional):
            The kernel size of the short convolution, only used when `use_short_conv` is `True`. Default: 4.
        conv_bias (bool, Optional):
            Whether to use bias in the short convolution, only used when `use_short_conv` is `True`. Default: `False`.
        use_output_gate (bool, Optional):
            Whether to use output gate. Default: `True`.
        gate_fn (str, Optional):
            The activation function for the output gate. Default: `swish`.
        elementwise_affine (bool, Optional):
            If `True`, applies elementwise affine to LayerNorm with learnable parameters. Default: `True`.
        norm_eps (float, Optional):
            The epsilon value for the layernorm/rmsnorm layer. Default: 1e-5.
        gate_logit_normalizer (int, Optional):
            The normalizer for the gate logits, appied after `logsigmoid`. Default: 16.
        gate_low_rank_dim (int, Optional):
            The low rank dim for the gate projection. Default: 16.
        clamp_min (float, Optional):
            The minimum value for the gate logits. Default: None.
        fuse_norm (bool, Optional):
            Whether to fuse the norm and the output gate for better memory footprint. Default: `True`.
        layer_idx (int, Optional):
            The index of the layer. Default: None.
    """

    def __init__(
        self,
        mode: str = 'chunk',
        hidden_size: int = 1024,
        expand_k: float = 0.5,
        expand_v: float = 1.0,
        num_heads: int = 4,
        num_kv_heads: Optional[int] = None,
        feature_map: Optional[str] = None,
        use_short_conv: bool = False,
        conv_size: int = 4,
        conv_bias: bool = False,
        use_output_gate: bool = True,
        gate_fn: str = 'swish',
        elementwise_affine: Optional[bool] = True,
        norm_eps: float = 1e-5,
        gate_logit_normalizer: int = 16,
        gate_low_rank_dim: int = 16,
        clamp_min: Optional[float] = None,
        fuse_norm: bool = True,
        layer_idx: int = None,
        config: Optional[GLACanonConfig] = None,
    ) -> GatedCanonLinearAttention:
        super().__init__()

        self.mode = mode
        self.hidden_size = hidden_size
        self.expand_k = expand_k
        self.expand_v = expand_v
        self.num_heads = num_heads
        self.num_kv_heads = num_kv_heads if num_kv_heads is not None else num_heads
        self.num_kv_groups = self.num_heads // self.num_kv_heads
        self.feature_map_fn = ACT2FN[feature_map] if feature_map is not None else None
        self.silu = nn.SiLU()

        self.use_short_conv = use_short_conv
        self.conv_size = conv_size
        self.conv_bias = conv_bias
        self.use_output_gate = use_output_gate

        self.key_dim = int(hidden_size * expand_k)
        self.value_dim = int(hidden_size * expand_v)
        self.key_dim_per_group = self.key_dim // self.num_kv_groups
        self.value_dim_per_group = self.value_dim // self.num_kv_groups
        self.clamp_min = clamp_min
        self.layer_idx = layer_idx

        assert mode in ['chunk', 'fused_recurrent', 'fused_chunk'], f"Not suppoerted mode `{mode}`."
        assert self.key_dim % num_heads == 0, f"key dim must be divisible by num_heads of {num_heads}"
        assert self.value_dim % num_heads == 0, f"value dim must be divisible by num_heads of {num_heads}"

        self.head_k_dim = self.key_dim // num_heads
        self.head_v_dim = self.value_dim // num_heads

        self.q_proj = nn.Linear(hidden_size, self.key_dim, bias=False)
        self.k_proj = nn.Linear(hidden_size, self.key_dim_per_group, bias=False)
        self.v_proj = nn.Linear(hidden_size, self.value_dim_per_group, bias=False)
        if self.use_output_gate:
            if getattr(config, 'output_gate_rank', None) is not None:   # Zeyuan's edit note: I determined that replacing g_proj with a low-rank version helps
                self.g_proj = nn.Sequential(nn.Linear(hidden_size, config.output_gate_rank, bias=False),
                                            nn.Linear(config.output_gate_rank, self.value_dim, bias=True))
                self.g_proj[0].zeyuan_split_init = config.output_gate_rank
                self.g_proj[1].zeyuan_split_init = config.output_gate_rank
            else:
                self.g_proj = nn.Linear(hidden_size, self.value_dim, bias=False)

        if use_short_conv:
            self.conv_size = conv_size
            self.q_conv1d = make_canon_layer(self.key_dim, conv_size, activation='silu')
            self.k_conv1d = make_canon_layer(self.key_dim_per_group, conv_size, activation='silu')
            self.v_conv1d = make_canon_layer(self.value_dim_per_group, conv_size, activation='silu')
            # self.q_conv1d = ShortConvolution(self.key_dim, conv_size, activation='silu')
            # self.k_conv1d = ShortConvolution(self.key_dim_per_group, conv_size, activation='silu')
            # self.v_conv1d = ShortConvolution(self.value_dim_per_group, conv_size, activation='silu')
            # self.q_conv1d.forward = torch._dynamo.disable(self.q_conv1d.forward)
            # self.k_conv1d.forward = torch._dynamo.disable(self.k_conv1d.forward)
            # self.v_conv1d.forward = torch._dynamo.disable(self.v_conv1d.forward)
        if 'B' in config.canon_set:
            self.canonB = create_canon(self.key_dim+self.key_dim_per_group+self.value_dim_per_group, config)
        else:
            self.canonB = None

        self.gk_proj = nn.Sequential(nn.Linear(hidden_size, gate_low_rank_dim, bias=False),
                                    nn.Linear(gate_low_rank_dim, self.key_dim_per_group, bias=True))
        self.gk_proj[0].zeyuan_split_init = gate_low_rank_dim
        self.gk_proj[1].zeyuan_split_init = gate_low_rank_dim
        self.o_proj = nn.Linear(self.value_dim, hidden_size, bias=False)

        if gate_fn == 'swish' and fuse_norm and use_output_gate and not config.zeyuan_layernorm:
            self.g_norm_swish_gate = FusedRMSNormSwishGate(self.head_v_dim, elementwise_affine, norm_eps)
            self.g_norm_swish_gate.forward = torch._dynamo.disable(self.g_norm_swish_gate.forward)  
            self.fuse_norm_and_gate = True
        else:
            self.fuse_norm_and_gate = False
            if config.zeyuan_layernorm:
                assert elementwise_affine, "zeyuan_layernorm requires elementwise_affine to be True"
                self.g_norm = MyRMSNorm(hidden_size=self.head_v_dim, eps=norm_eps)
                self.gate_fn = nn.SiLU()
                assert gate_fn == 'swish', "zeyuan_layernorm only supports swish activation"
            else:
                self.g_norm = RMSNorm(hidden_size=self.head_v_dim, elementwise_affine=elementwise_affine, eps=norm_eps)
                self.g_norm.forward = torch._dynamo.disable(self.g_norm.forward)  
                self.gate_fn = ACT2FN[gate_fn]

        self.gate_logit_normalizer = gate_logit_normalizer

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        past_key_values: Optional[Cache] = None,
        use_cache: Optional[bool] = False,
        output_attentions: Optional[bool] = False,
        **kwargs: Unpack[Dict]
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[Cache]]:
        if attention_mask is not None:
            assert len(attention_mask.shape) == 2, (
                "Expected attention_mask as a 0-1 matrix with shape [batch_size, seq_len] "
                "for padding purposes (0 indicating padding). "
                "Arbitrary attention masks of shape [batch_size, seq_len, seq_len] are not allowed."
            )

        # launching the triton kernel for just one token will actually be slower
        mode = 'fused_recurrent' if hidden_states.shape[1] <= 64 else self.mode

        last_state = None
        if past_key_values is not None and len(past_key_values) > self.layer_idx:
            last_state = past_key_values[self.layer_idx]

        cu_seqlens, position_ids, seq_idx = kwargs.get('cu_seqlens', None), kwargs.get('position_ids', None), None
        if self.use_short_conv:
            conv_state_q, conv_state_k, conv_state_v = None, None, None
            if last_state is not None:
                conv_state_q, conv_state_k, conv_state_v = last_state['conv_state']
            conv_mask = attention_mask[:, -hidden_states.shape[1]:] if attention_mask is not None else None
            if cu_seqlens is not None:
                if position_ids is None:
                    position_ids = prepare_position_ids(cu_seqlens)
                seq_idx = prepare_sequence_ids(position_ids).to(torch.int32).unsqueeze(0)
            q = self.q_proj(hidden_states)
            k = self.k_proj(hidden_states)
            v = self.v_proj(hidden_states)
            q, conv_state_q = self.q_conv1d(x=q,
                                            mask=conv_mask,
                                            cache=conv_state_q,
                                            output_final_state=use_cache,
                                            seq_idx=seq_idx)
            k, conv_state_k = self.k_conv1d(x=k,
                                            mask=conv_mask,
                                            cache=conv_state_k,
                                            output_final_state=use_cache,
                                            seq_idx=seq_idx)
            v, conv_state_v = self.v_conv1d(x=v,
                                            mask=conv_mask,
                                            cache=conv_state_v,
                                            output_final_state=use_cache,
                                            seq_idx=seq_idx)
        else:
            q = self.silu(self.q_proj(hidden_states))  # Zeyuan's edit note: if using Canon-B as opposed to Canon-b, I recommend adding silu here (similarly done in GDN)
            k = self.silu(self.k_proj(hidden_states))
            v = self.silu(self.v_proj(hidden_states))

        if self.canonB is not None:
            qkv = torch.cat([q,k,v], dim=-1)
            qkv = apply_canon('canonB', self.canonB, hidden_states=qkv, cache=past_key_values, layer_idx = self.layer_idx if past_key_values else -1, attention_mask=attention_mask)
            q, k, v = qkv.split([self.key_dim, self.key_dim_per_group, self.value_dim_per_group], dim=-1)
            assert False, "if using canon-B, I recommend adding feature_map (e.g., silu) and apply it to q+k+v after; you need to modify the code below accordingly"

        gk = self.gk_proj(hidden_states)

        if self.feature_map_fn is not None:
            q, k = map(self.feature_map_fn, (q, k))
            
        # dealing with left-padding
        if attention_mask is not None:
            v = v.mul_(attention_mask[:, -v.shape[-2]:, None])
        q = rearrange(q, 'b t (h d) -> b t h d', d=self.head_k_dim)

        if self.num_kv_groups > 1:
            k, = (repeat(x, 'b t (h d) -> b t (h g) d', g=self.num_kv_groups, d=self.head_k_dim) for x in (k, ))
            v = repeat(v, 'b t (h d) -> b t (h g) d', g=self.num_kv_groups, d=self.head_v_dim)
        else:
            k, = (rearrange(x, 'b t (h d) -> b t h d', d=self.head_k_dim) for x in (k, ))
            v = rearrange(v, 'b t (h d) -> b t h d', d=self.head_v_dim)

        if self.num_kv_groups > 1:
            gk, = (repeat(x, 'b t (h d) -> b t (h g) d', g=self.num_kv_groups, d=self.head_k_dim) for x in (gk,))
        else:
            gk, = (rearrange(x, 'b t (h d) -> b t h d', d=self.head_k_dim) for x in (gk,))
        gk = F.logsigmoid(gk) / self.gate_logit_normalizer
        if self.clamp_min is not None:
            gk = torch.clamp_min(gk, self.clamp_min)

        recurrent_state = last_state['recurrent_state'] if last_state is not None else None
        if mode == 'fused_recurrent':
            o, recurrent_state = fused_recurrent_gla(
                q=q,
                k=k,
                v=v,
                gk=gk,
                initial_state=recurrent_state,
                output_final_state=use_cache,
                cu_seqlens=cu_seqlens,
                #head_first=False
            )
        elif mode == 'fused_chunk':
            o, recurrent_state = fused_chunk_gla(
                q=q,
                k=k,
                v=v,
                g=gk,
                initial_state=recurrent_state,
                output_final_state=use_cache,
                head_first=False
            )
        elif mode == 'chunk':
            o, recurrent_state = chunk_gla(
                q=q,
                k=k,
                v=v,
                g=gk,
                initial_state=recurrent_state,
                output_final_state=use_cache,
                cu_seqlens=cu_seqlens,
                #head_first=False   # this is disabled in newest fla-org package
            )
        else:
            raise NotImplementedError(f"Not supported mode `{mode}`.")

        if past_key_values is not None:
            past_key_values.update(
                recurrent_state=recurrent_state,
                conv_state=(conv_state_q, conv_state_k, conv_state_v) if self.use_short_conv else None,
                layer_idx=self.layer_idx,
                offset=q.shape[1]
            )

        if self.use_output_gate:
            g = self.g_proj(hidden_states)
            if self.fuse_norm_and_gate:
                g = rearrange(g, 'b t (h d) -> b t h d', d=self.head_v_dim)
                o = self.g_norm_swish_gate(o, g)
                o = rearrange(o, 'b t h d -> b t (h d)')
            else:
                o = rearrange(self.g_norm(o), 'b t h d -> b t (h d)')
                o = o * self.gate_fn(g)
        else:
            o = rearrange(self.g_norm(o), 'b t h d -> b t (h d)')
        o = self.o_proj(o)

        return o, None, past_key_values

    def state_size(self, **kwargs) -> int:
        state_size = self.key_dim * self.head_v_dim
        for module in self.children():
            if isinstance(module, ShortConvolution):
                state_size += module.state_size
        return state_size
            


class GLACanonMLP(nn.Module):

    def __init__(
        self,
        hidden_size: int,
        hidden_ratio: Optional[int] = None,
        intermediate_size: Optional[int] = None,
        hidden_act: str = 'swish',
        config: Optional[GLACanonConfig] = None,
        layer_idx: Optional[int] = None,
    ) -> GLACanonMLP:
        super().__init__()
        self.layer_idx = layer_idx
        self.hidden_size = hidden_size
        # the final number of params is `hidden_ratio * hidden_size^2`
        # `intermediate_size` is chosen to be a multiple of 256 closest to `2/3 * hidden_size * hidden_ratio`
        if hidden_ratio is None:
            hidden_ratio = 4
        if intermediate_size is None:
            intermediate_size = int(hidden_size * hidden_ratio * 2 / 3)
            intermediate_size = 256 * ((intermediate_size + 256 - 1) // 256)
        self.hidden_ratio = hidden_ratio
        self.intermediate_size = intermediate_size

        self.gate_proj = nn.Linear(self.hidden_size, self.intermediate_size * 2, bias=False)
        if 'D' in config.canon_set:
            self.canonD = create_canon(self.intermediate_size * 2, config)
        else:
            self.canonD = None
        self.down_proj = nn.Linear(self.intermediate_size, self.hidden_size, bias=False)
        #self.act_fn = ACT2FN[hidden_act]
        self.act_fn = nn.SiLU()
        assert hidden_act in ['swish', 'silu'], "GLA MLP only supports swish/silu activation"

    def forward(
        self,
        x: torch.Tensor,
        past_key_values: Optional[Cache] = None,
        attention_mask: Optional[torch.Tensor] = None,
        **kwargs: Unpack[Dict],
    ) -> torch.Tensor:
        y = self.gate_proj(x)
        if self.canonD is not None:
            y = apply_canon('canonD', self.canonD, hidden_states=y, cache=past_key_values, layer_idx = self.layer_idx if past_key_values else -1, attention_mask=attention_mask)
        gate, y = y.chunk(2, -1)
        #return swiglu_linear(gate, y, self.down_proj.weight, self.down_proj.bias)
        return self.down_proj(self.act_fn(gate) * y)


class GLABlock(nn.Module):
    def __init__(self, config: GLACanonConfig, layer_idx: int):
        super().__init__()
        self.layer_idx = layer_idx
        self.hidden_size = config.hidden_size

        if config.zeyuan_layernorm:
            self.attn_norm = MyRMSNorm(hidden_size=config.hidden_size, eps=config.norm_eps)
        else:
            self.attn_norm = RMSNorm(hidden_size=config.hidden_size, eps=config.norm_eps)
            self.attn_norm.forward = torch._dynamo.disable(self.attn_norm.forward)  
        if 'A' in config.canon_set:
            self.canonA = create_canon(self.hidden_size, config)
        else:
            self.canonA = None
        if config.attn is not None and layer_idx in config.attn['layers']:
            self.attn = Attention(
                hidden_size=config.hidden_size,
                num_heads=config.attn['num_heads'],
                num_kv_heads=config.attn['num_kv_heads'],
                window_size=config.attn['window_size'],
                max_position_embeddings=config.max_position_embeddings,
                layer_idx=layer_idx
            )
        else:
            self.attn = GatedCanonLinearAttention(
                mode=config.attn_mode,
                hidden_size=config.hidden_size,
                expand_k=config.expand_k,
                expand_v=config.expand_v,
                num_heads=config.num_heads,
                num_kv_heads=config.num_kv_heads,
                feature_map=config.feature_map,
                use_short_conv=config.use_short_conv,
                conv_size=config.conv_size,
                use_output_gate=config.use_output_gate,
                gate_fn=config.hidden_act,
                elementwise_affine=config.elementwise_affine,
                norm_eps=config.norm_eps,
                clamp_min=config.clamp_min,
                fuse_norm=config.fuse_norm,
                layer_idx=layer_idx,
                config=config,
            )

        if 'C' in config.canon_set:
            self.canonC = create_canon(self.hidden_size, config)
        else:
            self.canonC = None
        if config.zeyuan_layernorm:
            self.mlp_norm = MyRMSNorm(hidden_size=config.hidden_size, eps=config.norm_eps)
            self.mlp_norm_fused = False
        else:
            self.mlp_norm = RMSNorm(hidden_size=config.hidden_size, eps=config.norm_eps)
            self.mlp_norm.forward = torch._dynamo.disable(self.mlp_norm.forward)
            self.mlp_norm_fused = True
        self.mlp = GLACanonMLP(
            hidden_size=config.hidden_size,
            hidden_ratio=config.hidden_ratio,
            intermediate_size=config.intermediate_size,
            hidden_act=config.hidden_act,
            config=config,
            layer_idx=layer_idx,
        )

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        past_key_values: Optional[Union[Cache, List[torch.FloatTensor]]] = None,
        use_cache: Optional[bool] = False,
        output_attentions: Optional[bool] = False,
        **kwargs: Unpack[Dict]
    ) -> Tuple[torch.FloatTensor, Optional[Tuple[torch.FloatTensor, torch.FloatTensor]]]:
        residual = hidden_states
        hidden_states = self.attn_norm(hidden_states)
        if self.canonA is not None:
            hidden_states = apply_canon('canonA', self.canonA, hidden_states=hidden_states, cache=past_key_values, layer_idx = self.layer_idx if past_key_values else -1, attention_mask=attention_mask)
        hidden_states, attentions, past_key_values = self.attn(
            hidden_states=hidden_states,
            attention_mask=attention_mask,
            past_key_values=past_key_values,
            use_cache=use_cache,
            output_attentions=output_attentions,
            **kwargs
        )
        if self.mlp_norm_fused:
            hidden_states, residual = self.mlp_norm(hidden_states, residual, True)
        else:
            hidden_states = residual + hidden_states
            residual = hidden_states
            hidden_states = self.mlp_norm(hidden_states)    

        if self.canonC is not None:
            hidden_states = apply_canon('canonC', self.canonC, hidden_states=hidden_states, cache=past_key_values, layer_idx = self.layer_idx if past_key_values else -1, attention_mask=attention_mask)
        hidden_states = self.mlp(hidden_states, past_key_values=past_key_values, attention_mask=attention_mask, **kwargs)
        hidden_states = residual + hidden_states

        outputs = (hidden_states, attentions, past_key_values)

        return outputs


class GLACanonPreTrainedModel(PreTrainedModel):

    config_class = GLACanonConfig
    supports_gradient_checkpointing = True
    _no_split_modules = ['GLABlock']

    def __init__(self, *inputs, **kwargs):
        super().__init__(*inputs, **kwargs)

    def _init_weights(
        self,
        module: nn.Module,
        rescale_prenorm_residual: bool = False,  # Zeyuan's edit note: default to be False, different from fla-org's original code
        num_residuals_per_layer: int = 2,
    ):
        if hasattr(module, '_zeyuan_no_reinit') and module._zeyuan_no_reinit:  # For Canon layeres, use Kaiming init
            module.reset_parameters()
            return           
        if isinstance(module, (nn.Linear, nn.Conv1d)):
            # Slightly different from the TF version which uses truncated_normal for initialization
            # cf https://github.com/pytorch/pytorch/pull/5617
            std = self.config.initializer_range
            if hasattr(module, 'zeyuan_split_init'):    # Zeyuan's edit note: I believe this is a better init scheme for double linear layers
                # need to scale up the init since it's now two matrices multiplied, for obvious reasons
                std = math.sqrt(std) / math.pow(module.zeyuan_split_init, 0.25)
            nn.init.normal_(module.weight, mean=0.0, std=std)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0.0, std=self.config.initializer_range)
            if module.padding_idx is not None:
                module.weight.data[module.padding_idx].zero_()
        elif isinstance(module, FusedRMSNormSwishGate) or isinstance(module, RMSNorm) or isinstance(module, MyRMSNorm):  # <--- Zeyuan's edit note: I added this since otherwise the norm weights will be zeros after FSDP init (in Lingua)
            if not hasattr(module, 'reset_parameters'):
                print(f"Warning: {module} does not have reset_parameters function; you might be using an earlier version of fla")
                print(f"Warning: - If you're loading model weights from a checkpoint, you can ignore this warning")
                print(f"Warning: - If you're training from scratch without FSDP, you might be fine as well")
            else:
                module.reset_parameters()

        if rescale_prenorm_residual:
            assert False, "Zeyuan's edit note: I disabled this"
            # Reinitialize selected weights subject to the OpenAI GPT-2 Paper Scheme:
            #   > A modified initialization which accounts for the accumulation on the residual path with model depth. Scale
            #   > the weights of residual layers at initialization by a factor of 1/√N where N is the # of residual layers.
            #   >   -- GPT-2 :: https://openai.com/blog/better-language-models/
            #
            # Reference (Megatron-LM): https://github.com/NVIDIA/Megatron-LM/blob/main/megatron/model/gpt_model.py
            for name, p in module.named_parameters():
                if name in ["o_proj.weight", "down_proj.weight"]:
                    # Special Scaled Initialization --> There are 2 Layer Norms per Transformer Block
                    # Following Pytorch init, except scale by 1/sqrt(2 * n_layer)
                    # We need to reinit p since this code could be called multiple times
                    # Having just p *= scale would repeatedly scale it down
                    with torch.no_grad():
                        p /= math.sqrt(num_residuals_per_layer * self.config.num_hidden_layers)


class GLACanonModel(GLACanonPreTrainedModel):

    def __init__(self, config: GLACanonConfig):
        super().__init__(config)
        self.padding_idx = config.pad_token_id
        self.vocab_size = config.vocab_size

        self.embeddings = nn.Embedding(config.vocab_size, config.hidden_size, self.padding_idx)
        self.layers = nn.ModuleList([GLABlock(config, layer_idx) for layer_idx in range(config.num_hidden_layers)])
        if config.zeyuan_layernorm:
            self.norm = MyRMSNorm(config.hidden_size, eps=config.norm_eps)
        else:
            self.norm = RMSNorm(config.hidden_size, eps=config.norm_eps)
            self.norm.forward = torch._dynamo.disable(self.norm.forward)  

        self.gradient_checkpointing = False

        self.post_init()

    def get_input_embeddings(self):
        return self.embeddings

    def set_input_embeddings(self, value):
        self.embeddings = value

    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.Tensor] = None,  # noqa
        inputs_embeds: Optional[torch.FloatTensor] = None,
        past_key_values: Optional[Union[Cache, List[torch.FloatTensor]]] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        **kwargs: Unpack[Dict]
    ) -> Union[Tuple, BaseModelOutputWithPast]:
        if output_attentions:
            warnings.warn("`GLACanonModel` does not `output_attentions` now, setting it to `False`.")
            output_attentions = False
        if attention_mask is not None and torch.all(attention_mask == 1):
            attention_mask = None
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        use_cache = use_cache if use_cache is not None else (self.config.use_cache if not self.training else False)
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        # retrieve input_ids and inputs_embeds
        if input_ids is not None and inputs_embeds is not None:
            raise ValueError("You cannot specify both input_ids and inputs_embeds at the same time")
        if input_ids is None and inputs_embeds is None:
            raise ValueError("You have to specify either input_ids or inputs_embeds")

        if inputs_embeds is None:
            inputs_embeds = self.embeddings(input_ids)
        hidden_states = inputs_embeds

        if use_cache and not isinstance(past_key_values, Cache):
            past_key_values = Cache.from_legacy_cache(past_key_values)

        if self.gradient_checkpointing and self.training and use_cache:
            logger.warning_once("`use_cache=True` is incompatible with gradient checkpointing. Setting `use_cache=False`...")
            use_cache = False

        all_hidden_states = () if output_hidden_states else None
        all_attns = () if output_attentions else None
        for idx, layer in enumerate(self.layers):
            if output_hidden_states:
                all_hidden_states += (hidden_states,)

            if self.gradient_checkpointing and self.training:
                hidden_states, attentions, past_key_values = self._gradient_checkpointing_func(
                    layer.__call__,
                    hidden_states,
                    attention_mask,
                    past_key_values,
                    use_cache,
                    output_attentions,
                    **kwargs
                )
            else:
                hidden_states, attentions, past_key_values = layer(
                    hidden_states,
                    attention_mask=attention_mask,
                    past_key_values=past_key_values,
                    use_cache=use_cache,
                    output_attentions=output_attentions,
                    **kwargs
                )

            if output_attentions:
                all_attns += (attentions,)

        hidden_states = self.norm(hidden_states)

        # add hidden states from the last decoder layer
        if output_hidden_states:
            all_hidden_states += (hidden_states,)

        if not return_dict:
            return tuple(i for i in [hidden_states, past_key_values, all_hidden_states, all_attns] if i is not None)
        return BaseModelOutputWithPast(
            last_hidden_state=hidden_states,
            past_key_values=past_key_values,
            hidden_states=all_hidden_states,
            attentions=all_attns
        )


class GLACanonForCausalLM(GLACanonPreTrainedModel, GenerationMixin):

    _tied_weights_keys = ["lm_head.weight"]

    def __init__(self, config):
        super().__init__(config)
        self.model = GLACanonModel(config)
        self.vocab_size = config.vocab_size
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)

        # Initialize weights and apply final processing
        self.post_init()

    def get_input_embeddings(self):
        return self.model.embeddings

    def set_input_embeddings(self, value):
        self.model.embeddings = value

    def get_output_embeddings(self):
        return self.lm_head

    def set_output_embeddings(self, new_embeddings):
        self.lm_head = new_embeddings

    def set_decoder(self, decoder):
        self.model = decoder

    def get_decoder(self):
        return self.model

    def generate(self, *args, **kwargs):
        try:
            return super().generate(*args, **kwargs)
        except AttributeError as exception:
            if 'past_key_values' in str(exception):
                raise AttributeError(
                    f"You tried to call `generate` with a decoding strategy that manipulates `past_key_values`, "
                    f"which is not supported for {self.__class__.__name__}. "
                    f"Try another generation strategy instead. "
                    f"For the available generation strategies, check this doc: "
                    f"https://huggingface.co/docs/transformers/en/generation_strategies#decoding-strategies"
                )
            else:
                raise exception

    def prepare_inputs_for_generation(
        self,
        input_ids: torch.LongTensor = None,
        past_key_values: Optional[Union[Cache, List[torch.FloatTensor]]] = None,
        attention_mask: Optional[torch.Tensor] = None,
        inputs_embeds: Optional[torch.Tensor] = None,
        use_cache: bool = True,
        num_logits_to_keep: Optional[int] = None,
        **kwargs
    ):
        # only last token for `inputs_ids` if the `past_key_values` is passed along.
        if past_key_values is not None:
            input_ids = input_ids[:, -1:]
        # if `inputs_embeds` are passed, we only want to use them in the 1st generation step
        if inputs_embeds is not None and past_key_values is None:
            model_inputs = {'inputs_embeds': inputs_embeds}
        else:
            # The `contiguous()` here is necessary to have a static stride during decoding. torchdynamo otherwise
            # recompiles graphs as the stride of the inputs is a guard.
            # Ref: https://github.com/huggingface/transformers/pull/29114
            # TODO: use `next_tokens` directly instead.
            model_inputs = {'input_ids': input_ids.contiguous()}

        if num_logits_to_keep is not None:
            model_inputs['num_logits_to_keep'] = num_logits_to_keep

        model_inputs.update({
            'past_key_values': past_key_values,
            'use_cache': use_cache,
            'attention_mask': attention_mask,
            'num_logits_to_keep': num_logits_to_keep,
        })
        return model_inputs

    def forward(
        self,
        input_ids: torch.LongTensor = None,
        attention_mask: Optional[torch.Tensor] = None,
        inputs_embeds: Optional[torch.Tensor] = None,
        past_key_values: Optional[Union[Cache, List[torch.FloatTensor]]] = None,
        labels: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        num_logits_to_keep: Optional[int] = 0,
        **kwargs: Unpack[Dict]
    ) -> Union[Tuple, CausalLMOutputWithPast]:
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        outputs = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            inputs_embeds=inputs_embeds,
            past_key_values=past_key_values,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
            **kwargs
        )

        hidden_states = outputs[0]
        fuse_linear_and_cross_entropy = self.config.fuse_cross_entropy and self.training
        logits = None if fuse_linear_and_cross_entropy else self.lm_head(hidden_states[:, -num_logits_to_keep:])

        loss = None
        if labels is not None:
            if self.config.fuse_cross_entropy:
                if fuse_linear_and_cross_entropy:
                    loss_fct = FusedLinearCrossEntropyLoss()
                else:
                    loss_fct = FusedCrossEntropyLoss(inplace_backward=True)
            else:
                loss_fct = nn.CrossEntropyLoss()
            # Enable model parallelism
            labels = labels.to(hidden_states.device)
            labels = torch.cat((labels[..., 1:], torch.full_like(labels[:, :1], loss_fct.ignore_index)), 1)
            if fuse_linear_and_cross_entropy:
                loss = loss_fct(hidden_states.view(-1, self.config.hidden_size),
                                labels.view(-1),
                                self.lm_head.weight,
                                self.lm_head.bias)
            else:
                loss = loss_fct(logits.view(-1, self.config.vocab_size), labels.view(-1))

        if not return_dict:
            output = (logits,) + outputs[1:]
            return (loss,) + output if loss is not None else output

        return CausalLMOutputWithPast(
            loss=loss,
            logits=logits,
            past_key_values=outputs.past_key_values,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )

    def load_from_lingua_state(self, state_dict: dict, strict: bool = True):
        mapped = {}
        for k,v in state_dict.items():
            mapped[k.replace("trans.","")] = v
        self.load_state_dict(mapped, strict=strict)

    @staticmethod
    def build_config_from_yaml(args):
        if args.model_name.startswith('GLA5'):  # GLA5 is my version number to simulate GDN setting, 6d^2 + 6d^2 + zeyuan split init + output_gate_rank=16s
            config = GLACanonConfig(hidden_size=args.dim,
                                    num_hidden_layers=args.n_layers,
                                    num_heads=args.n_heads,
                                    vocab_size=args.vocab_size,
                                    norm_eps=args.norm_eps,
                                    tie_word_embeddings=args.weight_tying,
                                    use_short_conv='b' in getattr(args, 'canon_set', ''),
                                    fuse_cross_entropy=False,
                                    expand_k=1,
                                    expand_v=2,
                                    output_gate_rank = 16,
                                    zeyuan_split_init = True,
                                    zeyuan_layernorm = getattr(args, 'zeyuan_layernorm', False),
            )
            assert config.hidden_size % (config.num_heads) == 0, f"hidden_size {config.hidden_size} must be divisible by num_heads {config.num_heads}"
            config.intermediate_size = config.hidden_size * 2
            if args.hidden_dim is not None:
                config.intermediate_size = args.hidden_dim
            config.canon_set = getattr(args, 'canon_set', '')
            config.canon_bias = getattr(args, 'canon_bias', False)
            config.canon_residual = getattr(args, 'canon_residual', True)
            config.canon_activation = getattr(args, 'canon_activation', False)
            config.canon_kernel = getattr(args, 'canon_kernel', 4)
        else:
            assert False, f"Model name {args.model_name} not recognized; this is GLA model"
        return config




    @classmethod
    def from_pretrained(cls, pretrained_model_name_or_path, *model_args, variant="default", **kwargs):
        """
        Overrides HF default loader to use custom .pth and config from subfolder.
        """
        from omegaconf import OmegaConf
        
        def device_map_to_map_location(device_map):
            if device_map == "cpu":
                return "cpu"
            elif device_map == "auto":
                return None  # Let torch figure it out
            elif isinstance(device_map, dict):
                # Could be a more complex mapping, may need custom handling
                return lambda storage, loc: loc  # identity (as fallback)
            elif isinstance(device_map, str):
                return device_map  # e.g., "cuda:0"
            else:
                return None
        device_map = kwargs.pop("device_map", None)
        map_location = device_map_to_map_location(device_map)

        from huggingface_hub import hf_hub_download
        import os, json
        if os.path.isfile(os.path.join(pretrained_model_name_or_path, variant, "params.json")):
            config_path = os.path.join(pretrained_model_name_or_path, variant, "params.json")
        else:
            config_path = hf_hub_download(
                repo_id=pretrained_model_name_or_path,
                filename=f"{variant}/params.json",
            )
        with open(config_path, "r") as f:
            dd = json.load(f)
        cfg = GLACanonForCausalLM.build_config_from_yaml(OmegaConf.create(dd).model)

        model = GLACanonForCausalLM(cfg)

        if os.path.isfile(os.path.join(pretrained_model_name_or_path, variant, "consolidated.pth")):
            weights_path = os.path.join(pretrained_model_name_or_path, variant, "consolidated.pth")
        else:
            weights_path = hf_hub_download(
                repo_id=pretrained_model_name_or_path,
                filename=f"{variant}/consolidated.pth",
            )
        logger.info(f"Loading lingua model weights from {weights_path}")
        state = torch.load(weights_path, map_location=map_location, weights_only=True)
        model.load_from_lingua_state(state['model'])
        logger.info(f"Successfully converted lingua state to Huggingface state")

        return model