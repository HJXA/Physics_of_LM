# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This code is modified from https://github.com/fla-org/flash-linear-attention/blob/main/fla/models/gated_deltanet/configuration_gated_deltanet.py (Jan/3/2025 version)
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

from .configuration_gated_deltanet_canon import GatedDeltaNetCanonConfig
from .canon_helper import create_canon, apply_canon, make_canon_layer


import math
from typing import TYPE_CHECKING, Dict, Optional, Tuple

import torch
import torch.nn as nn
from einops import rearrange
from torch.nn import functional as F

from fla.modules import FusedRMSNormSwishGate, RMSNorm, ShortConvolution

import fla
from packaging import version
if version.parse(fla.__version__) >= version.parse("0.3.0"):
    from fla.ops.utils import prepare_position_ids, prepare_sequence_ids
else:
    from fla.ops.common.utils import prepare_position_ids, prepare_sequence_ids
    assert False, "code not tested with fla<0.3.0; model generation may be incorrect."

#from fla.ops.gated_delta_rule import chunk_gated_delta_rule, fused_recurrent_gated_delta_rule
from fla.ops.gated_delta_rule import chunk_gated_delta_rule as chunk_gated_delta_rule_old, fused_recurrent_gated_delta_rule as fused_recurrent_gated_delta_rule_old
@torch._dynamo.disable
def chunk_gated_delta_rule(*args, **kwargs):
    return chunk_gated_delta_rule_old(*args, **kwargs)
@torch._dynamo.disable
def fused_recurrent_gated_delta_rule(*args, **kwargs):
    return fused_recurrent_gated_delta_rule_old(*args, **kwargs)


if TYPE_CHECKING:
    from transformers.processing_utils import Unpack

    from fla.models.utils import Cache

@torch.compile
def elu_p1(x):
    return (F.elu(x, 1., False) + 1.).to(x)


@torch.compile
def sum_norm(x):
    return (x / x.sum(-1, keepdim=True)).to(x)


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


class GatedDeltaNetCanon(nn.Module):
    """
    The layer implementaion for [Gated Delta Networks: Improving Mamba2 with Delta Rule](https://arxiv.org/abs/2412.06464).  # noqa

    Similar to Mamba2, each layer contains around 6*hidden_size*hidden_size parameters.
    Parameter alloation when use_gate=True:
        - 0.75 * hidden_size * hidden_size for the q_proj and k_proj each
        - 1.5 * hidden_size * hidden_size for the v_proj, g_proj and o_proj each
        - Others are ignorably small.
        - In total = 0.75 * 2 + 1.5 * 3 = 6 * hidden_size * hidden_size
    NOTE: num_heads * head_dim = 0.75 * hidden_size, please make sure to set the correct num_heads and head_dim.

    Parameter allocation when use_gate=False:
        - 1 * hidden_size * hidden_size for the q_proj and k_proj each
        - 2 * hidden_size * hidden_size for the v_proj and o_proj each
        - Others are ignorably small.
        - In total = 1 * 2 + 2 * 2 = 6 * hidden_size * hidden_size

    Args:
        hidden_size (int, Optional):
            The hidden size of the input. Default: 2048.
        expand_v (float, Optional):
            The expansion ratio for the value dim. Default: 2.0.
        head_dim (int, Optional):
            The dimension of each head. Default: 256.
        num_heads (int, Optional):
            The number of heads. Default: 4.
        mode (str, Optional):
            Which Gated DeltaNet kernel to use.
            Currently available: `chunk` and `fused_recurrent`.
            Default: `chunk`.
        use_beta (bool, Optional):
            Whether to use beta. Default: `True`.
        use_gate (bool, Optional):
            Whether to use output gate. Default: `True`.
        use_short_conv (bool, Optional):
            Whether to use short convolutions. Default: `True`.
        conv_size (int, Optional):
            The kernel size of the short convolution, only used when `use_short_conv` is `True`. Default: 4.
        conv_bias (bool, Optional):
            Whether to use bias in the short convolution, only used when `use_short_conv` is `True`. Default: `False`.
        layer_idx (int, Optional):
            The index of the layer. Default: None.
        norm_eps (float, Optional):
            The epsilon value for the normalization layer. Default: 1e-5.
    """

    def __init__(
        self,
        hidden_size: int = 2048,
        expand_v: float = 2,
        head_dim: int = 256,
        num_heads: int = 6,
        mode: str = 'chunk',
        use_gate: bool = True,
        use_short_conv: bool = True,
        conv_size: int = 4,
        conv_bias: bool = False,
        layer_idx: int = None,
        norm_eps: float = 1e-5,
        config: Optional[GatedDeltaNetCanonConfig] = None,
        **kwargs
    ) -> GatedDeltaNetCanon:
        super().__init__()

        self.mode = mode

        self.hidden_size = hidden_size
        self.expand_v = expand_v

        self.use_gate = use_gate
        self.use_short_conv = use_short_conv
        self.conv_size = conv_size
        self.conv_bias = conv_bias

        self.head_dim = head_dim
        self.num_heads = num_heads

        self.key_dim = self.num_heads * self.head_dim
        self.value_dim = self.key_dim * self.expand_v
        self.head_k_dim = head_dim
        self.head_v_dim = head_dim * self.expand_v
        self.layer_idx = layer_idx
        self.silu = nn.SiLU()
        self.config = config

        assert mode in ['chunk', 'fused_recurrent'], f"Not suppoerted mode `{mode}`."

        self.q_proj = nn.Linear(hidden_size, self.key_dim, bias=False)
        self.k_proj = nn.Linear(hidden_size, self.key_dim, bias=False)
        self.v_proj = nn.Linear(hidden_size, self.value_dim, bias=False)
        self.b_proj = nn.Linear(hidden_size, self.num_heads, bias=False)
        self.a_proj = nn.Linear(hidden_size, self.num_heads, bias=False)
        A = torch.empty(self.num_heads, dtype=torch.float32).uniform_(0, 16)
        A_log = torch.log(A)
        self.A_log = nn.Parameter(A_log)
        self.A_log._no_weight_decay = True
        # hard coded for now
        dt_min = 0.001
        dt_max = 0.1
        dt_init_floor = 1e-4
        dt = torch.exp(
            torch.rand(self.num_heads) * (math.log(dt_max) - math.log(dt_min))
            + math.log(dt_min)
        )
        dt = torch.clamp(dt, min=dt_init_floor)
        # Inverse of softplus: https://github.com/pytorch/pytorch/issues/72759
        inv_dt = dt + torch.log(-torch.expm1(-dt))
        self.dt_bias = nn.Parameter(inv_dt)
        # Just to be explicit. Without this we already don't put wd on dt_bias because of the check
        # name.endswith("bias") in param_grouping.py
        self.dt_bias._no_weight_decay = True

        if 'B' in config.canon_set:
            self.canonB = create_canon(self.key_dim*2+self.value_dim, config)
        else:
            self.canonB = None
        if use_short_conv:
            self.conv_size = conv_size
            self.q_conv1d = make_canon_layer(self.key_dim, conv_size, activation='silu')
            self.k_conv1d = make_canon_layer(self.key_dim, conv_size, activation='silu')
            self.v_conv1d = make_canon_layer(self.value_dim, conv_size, activation='silu')
            # self.q_conv1d = ShortConvolution(
            #     hidden_size=self.key_dim,
            #     kernel_size=conv_size,
            #     activation='silu'
            # )
            # self.k_conv1d = ShortConvolution(
            #     hidden_size=self.key_dim,
            #     kernel_size=conv_size,
            #     activation='silu'
            # )
            # self.v_conv1d = ShortConvolution(
            #     hidden_size=self.value_dim,
            #     kernel_size=conv_size,
            #     activation='silu'
            # )
            # self.q_conv1d.forward = torch._dynamo.disable(self.q_conv1d.forward)
            # self.k_conv1d.forward = torch._dynamo.disable(self.k_conv1d.forward)
            # self.v_conv1d.forward = torch._dynamo.disable(self.v_conv1d.forward)
        # else:   # disabled
        #     raise UserWarning(
        #         "ShortConvolution is crucial to the performance. "
        #         "Do not turn it off, i.e., setting `use_short_conv=False` unless you know what you are doing."
        #     )
        if use_gate:
            if getattr(config, 'output_gate_rank', None) is not None:
                self.g_proj = nn.Sequential(nn.Linear(hidden_size, config.output_gate_rank, bias=False),
                                            nn.Linear(config.output_gate_rank, self.value_dim, bias=True))
                self.g_proj[0].zeyuan_split_init = config.output_gate_rank
                self.g_proj[1].zeyuan_split_init = config.output_gate_rank
            else:
                self.g_proj = nn.Linear(hidden_size, self.value_dim, bias=False)
            if config.zeyuan_layernorm:
                self.o_norm = MyRMSNorm(self.head_v_dim, eps=norm_eps)
                #self.o_norm_gate_fn = ACT2FN['swish']
                self.o_norm_gate_fn = nn.SiLU()
                self.fuse_norm_and_gate = False
            else:
                self.o_norm = FusedRMSNormSwishGate(self.head_v_dim, eps=norm_eps)
                self.o_norm.forward = torch._dynamo.disable(self.o_norm.forward)
                self.fuse_norm_and_gate = True
        else:
            if config.zeyuan_layernorm:
                self.o_norm = MyRMSNorm(self.head_v_dim, eps=norm_eps)
            else:
                self.o_norm = RMSNorm(self.head_v_dim, eps=norm_eps)
                self.o_norm.forward = torch._dynamo.disable(self.o_norm.forward)
        self.o_proj = nn.Linear(self.value_dim, hidden_size, bias=False)

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

        mode = 'fused_recurrent' if hidden_states.shape[1] <= 64 else self.mode
        if self.training:
            assert mode == 'chunk', "Only chunk mode is supported in training."

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
            q, conv_state_q = self.q_conv1d(
                x=self.q_proj(hidden_states),
                mask=conv_mask,
                cache=conv_state_q,
                output_final_state=use_cache,
                seq_idx=seq_idx
            )
            k, conv_state_k = self.k_conv1d(
                x=self.k_proj(hidden_states),
                mask=conv_mask,
                cache=conv_state_k,
                output_final_state=use_cache,
                seq_idx=seq_idx
            )
            v, conv_state_v = self.v_conv1d(
                x=self.v_proj(hidden_states),
                mask=conv_mask,
                cache=conv_state_v,
                output_final_state=use_cache,
                seq_idx=seq_idx
            )
        else:
            q = self.silu(self.q_proj(hidden_states))
            k = self.silu(self.k_proj(hidden_states))
            v = self.silu(self.v_proj(hidden_states))

        if self.canonB is not None:
            qkv = torch.cat([q,k,v], dim=-1)
            qkv = apply_canon('canonB', self.canonB, hidden_states=qkv, cache=past_key_values, layer_idx = self.layer_idx if past_key_values else -1, attention_mask=attention_mask)
            q, k, v = qkv.split([self.key_dim, self.key_dim, self.value_dim], dim=-1)

        q, k = map(lambda x: rearrange(x, 'b t (h d) -> b t h d', d=self.head_k_dim), (q, k))
        v = rearrange(v, 'b t (h d) -> b t h d', d=self.head_v_dim)
        beta = self.b_proj(hidden_states).sigmoid()
        g = -self.A_log.float().exp() * F.softplus(self.a_proj(hidden_states).float() + self.dt_bias)

        # dealing with padding
        if attention_mask is not None:
            beta = beta.mul(attention_mask[:, -beta.shape[-2]:, None])
            g = g.mul(attention_mask[:, -g.shape[-2]:, None])

        recurrent_state = last_state['recurrent_state'] if last_state is not None else None
        if mode == 'chunk':
            o, recurrent_state = chunk_gated_delta_rule(
                q=q,
                k=k,
                v=v,
                g=g,
                beta=beta,
                initial_state=recurrent_state,
                output_final_state=use_cache,
                cu_seqlens=cu_seqlens,
                #head_first=False,  # removed in latest fla-org codebase
                use_qk_l2norm_in_kernel=True
            )
        elif mode == 'fused_recurrent':
            o, recurrent_state = fused_recurrent_gated_delta_rule(
                q=q,
                k=k,
                v=v,
                g=g,
                beta=beta,
                initial_state=recurrent_state,
                output_final_state=use_cache,
                cu_seqlens=cu_seqlens,
                #head_first=False,  # removed in latest fla-org codebase
                use_qk_l2norm_in_kernel=True
            )
        if past_key_values is not None:
            past_key_values.update(
                recurrent_state=recurrent_state,
                conv_state=(conv_state_q, conv_state_k, conv_state_v) if self.use_short_conv else None,
                layer_idx=self.layer_idx,
                offset=q.shape[1]
            )

        # if self.use_gate:
        #     g = rearrange(self.g_proj(hidden_states), '... (h d) -> ... h d', d=self.head_v_dim)
        #     o = self.o_norm(o, g)
        # else:
        #     o = self.o_norm(o)
        # o = rearrange(o, 'b t h d -> b t (h d)')
        # o = self.o_proj(o)

        if self.use_gate:
            g = self.g_proj(hidden_states)
            if self.fuse_norm_and_gate:
                g = rearrange(g, 'b t (h d) -> b t h d', d=self.head_v_dim)
                o = self.o_norm(o, g)
                o = rearrange(o, 'b t h d -> b t (h d)')
            else:
                o = rearrange(self.o_norm(o), 'b t h d -> b t (h d)')
                o = o * self.o_norm_gate_fn(g)
        else:
            o = rearrange(self.o_norm(o), 'b t h d -> b t (h d)')
        o = self.o_proj(o)


        return o, None, past_key_values

    # Zeyuan's edit note: modified here to support FSDP and DTensor
    def reset_parameters(self):
        if hasattr(self.A_log, 'to_local'):
            # DTensor case
            A_log_local = self.A_log.to_local()
            num_heads_local = A_log_local.numel()
        else:
            # Regular tensor
            A_log_local = self.A_log
            num_heads_local = self.A_log.numel()

        A = torch.empty(num_heads_local, dtype=torch.float32, device=A_log_local.device).uniform_(0, 16)
        A_log = torch.log(A)
        with torch.no_grad():
            A_log_local.copy_(A_log)
        
        # --- Handle possible sharding / DTensor for dt_bias ---
        if hasattr(self.dt_bias, 'to_local'):
            dt_bias_local = self.dt_bias.to_local()
            num_heads_local = dt_bias_local.numel()
        else:
            dt_bias_local = self.dt_bias
            num_heads_local = dt_bias_local.numel()

        dt_min = 0.001
        dt_max = 0.1
        dt_init_floor = 1e-4
        dt = torch.exp(
            torch.rand(num_heads_local, device=dt_bias_local.device) * (math.log(dt_max) - math.log(dt_min))
            + math.log(dt_min)
        )
        dt = torch.clamp(dt, min=dt_init_floor)
        inv_dt = dt + torch.log(-torch.expm1(-dt))
        with torch.no_grad():
            dt_bias_local.copy_(inv_dt)
            
import math
import warnings
from typing import TYPE_CHECKING, Dict, List, Optional, Tuple, Union, Any

import torch
import torch.nn as nn
import torch.utils.checkpoint
from transformers.generation import GenerationMixin
from transformers.modeling_outputs import BaseModelOutputWithPast, CausalLMOutputWithPast
from transformers.modeling_utils import PreTrainedModel
from transformers.utils import logging
from transformers.utils.deprecation import deprecate_kwarg

from fla.layers.attn import Attention
from fla.models.utils import Cache
from fla.modules import FusedCrossEntropyLoss, FusedLinearCrossEntropyLoss
from fla.modules import RMSNorm

if TYPE_CHECKING:
    from transformers.processing_utils import Unpack

logger = logging.get_logger(__name__)

#from fla.modules.activations import swiglu, swiglu_linear
from fla.modules.activations import swiglu as swiglu_old, swiglu_linear as swiglu_linear_old
@torch._dynamo.disable
def swiglu(*args, **kwargs):
    return swiglu_old(*args, **kwargs)
@torch._dynamo.disable
def swiglu_linear(*args, **kwargs):
    return swiglu_linear_old(*args, **kwargs)

class SwiGLULinear(nn.Module):
    def forward(self, x, y, weight, bias):
        return swiglu_linear(x, y, weight, bias)

class GatedDeltaNetCanonMLP(nn.Module):

    def __init__(
        self,
        hidden_size: int,
        hidden_ratio: Optional[int] = None,
        intermediate_size: Optional[int] = None,
        hidden_act: str = 'swish',
        fuse_swiglu: bool = True,
        layer_idx = None,
        config: Optional[GatedDeltaNetCanonConfig] = None
    ) -> GatedDeltaNetCanonMLP:
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
        self.hidden_act = hidden_act
        self.fuse_swiglu = fuse_swiglu

        if hidden_act != 'swish':
            raise ValueError(f'Unsupported hidden_act: {hidden_act}')

        self.down_proj = nn.Linear(self.intermediate_size, self.hidden_size, bias=False)
        if 'D' in config.canon_set:
            self.canonD = create_canon(self.intermediate_size * 2, config)
            self.both_proj = nn.Linear(self.hidden_size, self.intermediate_size*2, bias=False)
        else:
            self.canonD = None
            self.gate_proj = nn.Linear(self.hidden_size, self.intermediate_size, bias=False)
            self.up_proj = nn.Linear(self.hidden_size, self.intermediate_size, bias=False)
        if self.fuse_swiglu:
            self.swiglu_linear = SwiGLULinear()

    def forward(
        self,
        x: torch.Tensor,
        attention_mask, 
        layer_past=None, 
        **kwargs: Unpack[Any]
    ) -> torch.Tensor:
        if self.canonD is not None:
            hidden_states = self.both_proj(x)
            hidden_states = apply_canon('canonD', self.canonD, hidden_states=hidden_states, cache=layer_past, layer_idx = self.layer_idx if layer_past else -1, attention_mask=attention_mask)
            gate, y = hidden_states.chunk(2, dim=-1)
        else:
            gate, y = self.gate_proj(x), self.up_proj(x)
        if self.fuse_swiglu:
            return self.swiglu_linear(gate, y, self.down_proj.weight, self.down_proj.bias)
        else:
            return self.down_proj(swiglu(gate, y))




class GatedDeltaNetCanonBlock(nn.Module):
    def __init__(self, config: GatedDeltaNetCanonConfig, layer_idx: int):
        super().__init__()

        self.config = config
        self.layer_idx = layer_idx

        if config.zeyuan_layernorm:
            self.attn_norm = MyRMSNorm(config.hidden_size, eps=config.norm_eps)
        else:
            self.attn_norm = (RMSNorm if config.fuse_norm else nn.RMSNorm)(config.hidden_size, eps=config.norm_eps)
            if isinstance(self.attn_norm, RMSNorm):
                self.attn_norm.forward = torch._dynamo.disable(self.attn_norm.forward)
        if 'A' in config.canon_set:
            self.canonA = create_canon(config.hidden_size, config)
        else:
            self.canonA = None
        if 'C' in config.canon_set:
            self.canonC = create_canon(config.hidden_size, config)
        else:
            self.canonC = None
        if config.attn is not None and layer_idx in config.attn['layers']:
            assert False, "disabled for hybrid model; need to add Canon if added"
            self.attn = Attention(
                hidden_size=config.hidden_size,
                num_heads=config.attn['num_heads'],
                num_kv_heads=config.attn['num_kv_heads'],
                qkv_bias=config.attn['qkv_bias'],
                window_size=config.attn['window_size'],
                rope_theta=config.attn['rope_theta'],
                max_position_embeddings=config.max_position_embeddings,
                layer_idx=layer_idx
            )
        else:
            self.attn = GatedDeltaNetCanon(
                mode=config.attn_mode,
                hidden_size=config.hidden_size,
                expand_v=config.expand_v,
                head_dim=config.head_dim,
                num_heads=config.num_heads,
                use_gate=config.use_gate,
                use_short_conv=config.use_short_conv,
                conv_size=config.conv_size,
                norm_eps=config.norm_eps,
                layer_idx=layer_idx,
                config=config,
            )
        if config.zeyuan_layernorm:
            self.mlp_norm = MyRMSNorm(config.hidden_size, eps=config.norm_eps)
        else:
            self.mlp_norm = (RMSNorm if config.fuse_norm else nn.RMSNorm)(config.hidden_size, eps=config.norm_eps)
            if isinstance(self.mlp_norm, RMSNorm):
                self.mlp_norm.forward = torch._dynamo.disable(self.mlp_norm.forward)
        if hasattr(config, 'mlp_type') and config.mlp_type == 'mlp10':
            config.intermediate_size = config.hidden_size*2
        self.mlp = GatedDeltaNetCanonMLP(
            hidden_size=config.hidden_size,
            hidden_ratio=config.hidden_ratio,
            intermediate_size=config.intermediate_size,
            hidden_act=config.hidden_act,
            fuse_swiglu=config.fuse_swiglu,
            layer_idx=layer_idx,
            config=config,
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
        if self.config.fuse_norm and not self.config.zeyuan_layernorm:
            hidden_states, residual = self.mlp_norm(hidden_states, residual, True)
        else:
            hidden_states = residual + hidden_states
            residual = hidden_states
            hidden_states = self.mlp_norm(hidden_states)
        if self.canonC is not None:
            hidden_states = apply_canon('canonC', self.canonC, hidden_states=hidden_states, cache=past_key_values, layer_idx = self.layer_idx if past_key_values else -1, attention_mask=attention_mask)
        hidden_states = self.mlp(hidden_states, layer_past = past_key_values, attention_mask=attention_mask, **kwargs)
        hidden_states = residual + hidden_states

        outputs = (hidden_states, attentions, past_key_values)

        return outputs


class GatedDeltaNetCanonPreTrainedModel(PreTrainedModel):

    config_class = GatedDeltaNetCanonConfig
    base_model_prefix = 'model'
    supports_gradient_checkpointing = True
    _no_split_modules = ['GatedDeltaNetCanonBlock']
    _supports_cache_class = True

    def __init__(self, *inputs, **kwargs):
        super().__init__(*inputs, **kwargs)

    def _init_weights(
        self,
        module: nn.Module,
        prenorm_residual_strategy: Optional[str] = None,  # Zeyuan's edit note: changed to None
        num_residuals_per_layer: int = 2,
    ):
        if hasattr(module, '_zeyuan_no_reinit') and module._zeyuan_no_reinit:  # For Canon layeres, use Kaiming init
            module.reset_parameters()
            return           
        if isinstance(module, (nn.Linear, nn.Conv1d)):
            # Slightly different from the TF version which uses truncated_normal for initialization
            # cf https://github.com/pytorch/pytorch/pull/5617
            std = self.config.initializer_range
            if hasattr(module, 'zeyuan_split_init'):
                # need to scale up the init since it's now two matrices multiplied, for obvious reasons
                std = math.sqrt(std) / math.pow(module.zeyuan_split_init, 0.25)
            nn.init.normal_(module.weight, mean=0.0, std=std)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0.0, std=self.config.initializer_range)
        elif hasattr(module, 'reset_parameters'):
            module.reset_parameters()

        if prenorm_residual_strategy is not None:
            assert False, "Zeyuan: I disabled this for a controlled experiment"
            # Reinitialize selected weights subject to the OpenAI GPT-2 Paper Scheme:
            #   > A modified initialization which accounts for the accumulation on the residual path with model depth. Scale
            #   > the weights of residual layers at initialization by a factor of 1/√N where N is the # of residual layers.
            #   >   -- GPT-2 :: https://openai.com/blog/better-language-models/
            #
            # Reference (Megatron-LM): https://github.com/NVIDIA/Megatron-LM/blob/main/megatron/model/gpt_model.py
            p = None
            if hasattr(module, 'o_proj'):
                p = module.o_proj.weight
            elif hasattr(module, 'down_proj'):
                p = module.down_proj.weight
            if p is not None:
                # Special Scaled Initialization --> There are 2 Layer Norms per Transformer Block
                # Following Pytorch init, except scale by 1/sqrt(2 * n_layer)
                # We need to reinit p since this code could be called multiple times
                # Having just p *= scale would repeatedly scale it down
                if prenorm_residual_strategy == 'rescale':
                    nn.init.kaiming_uniform_(p, a=math.sqrt(5))
                    with torch.no_grad():
                        p /= math.sqrt(num_residuals_per_layer * self.config.num_hidden_layers)
                elif prenorm_residual_strategy == 'zero':
                    nn.init.zeros_(p)
                else:
                    raise ValueError(f"Invalid prenorm_residual_strategy: {prenorm_residual_strategy}")


class GatedDeltaNetCanonModel(GatedDeltaNetCanonPreTrainedModel):

    def __init__(self, config: GatedDeltaNetCanonConfig):
        super().__init__(config)
        self.padding_idx = config.pad_token_id
        self.vocab_size = config.vocab_size

        self.embeddings = nn.Embedding(config.vocab_size, config.hidden_size, self.padding_idx)
        self.layers = nn.ModuleList([GatedDeltaNetCanonBlock(config, layer_idx) for layer_idx in range(config.num_hidden_layers)])
        if config.zeyuan_layernorm:
            self.norm = MyRMSNorm(config.hidden_size, eps=config.norm_eps)
        else:
            self.norm = (RMSNorm if config.fuse_norm else nn.RMSNorm)(config.hidden_size, eps=config.norm_eps)
            if isinstance(self.norm, RMSNorm):
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
            warnings.warn("`GatedDeltaNetCanonModel` does not `output_attentions` now, setting it to `False`.")
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
        for layer in self.layers:
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


class GatedDeltaNetCanonForCausalLM(GatedDeltaNetCanonPreTrainedModel, GenerationMixin):

    _tied_weights_keys = ["lm_head.weight"]

    def __init__(self, config):
        super().__init__(config)
        self.model = GatedDeltaNetCanonModel(config)
        self.vocab_size = config.vocab_size
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)
        self.criterion = None

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

    @deprecate_kwarg("num_logits_to_keep", version="4.50", new_name="logits_to_keep")
    def prepare_inputs_for_generation(
        self,
        input_ids: torch.LongTensor = None,
        past_key_values: Optional[Union[Cache, List[torch.FloatTensor]]] = None,
        attention_mask: Optional[torch.Tensor] = None,
        inputs_embeds: Optional[torch.Tensor] = None,
        use_cache: bool = True,
        logits_to_keep: Optional[int] = None,
        **kwargs
    ):
        # only last token for `inputs_ids` if the `past_key_values` is not empty.
        if past_key_values is not None and len(past_key_values) > 0:
            input_ids = input_ids[:, -1:]
        # if `inputs_embeds` are passed, we only want to use them in the 1st generation step
        if inputs_embeds is not None and len(past_key_values) == 0:
            model_inputs = {'inputs_embeds': inputs_embeds}
        else:
            # The `contiguous()` here is necessary to have a static stride during decoding. torchdynamo otherwise
            # recompiles graphs as the stride of the inputs is a guard.
            # Ref: https://github.com/huggingface/transformers/pull/29114
            # TODO: use `next_tokens` directly instead.
            model_inputs = {'input_ids': input_ids.contiguous()}

        if logits_to_keep is not None:
            model_inputs['logits_to_keep'] = logits_to_keep

        model_inputs.update({
            'past_key_values': past_key_values,
            'use_cache': use_cache,
            'attention_mask': attention_mask,
        })
        return model_inputs

    @deprecate_kwarg("num_logits_to_keep", version="4.50", new_name="logits_to_keep")
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
        logits_to_keep: Optional[int] = 0,
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

        loss, logits = None, None
        if not fuse_linear_and_cross_entropy or labels is None:
            logits = self.lm_head(hidden_states if logits_to_keep is None else hidden_states[:, -logits_to_keep:])
        if labels is not None:
            if getattr(self, 'criterion', None) is None:
                if fuse_linear_and_cross_entropy:
                    criterion = FusedLinearCrossEntropyLoss()
                elif self.config.fuse_cross_entropy:
                    criterion = FusedCrossEntropyLoss(inplace_backward=True)
                else:
                    criterion = nn.CrossEntropyLoss()
            else:
                criterion = self.criterion
            labels = labels.to(hidden_states.device)
            labels = torch.cat((labels[..., 1:], torch.full_like(labels[:, :1], criterion.ignore_index)), 1)
            if fuse_linear_and_cross_entropy:
                loss = criterion(hidden_states, labels, self.lm_head.weight, self.lm_head.bias)
            else:
                loss = criterion(logits.view(labels.numel(), -1), labels.view(-1))

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
        if args.model_name.startswith('GDN2'):
            config = GatedDeltaNetCanonConfig(hidden_size=args.dim,
                                              num_hidden_layers=args.n_layers,
                                              num_heads=args.n_heads,
                                              vocab_size=args.vocab_size,
                                              norm_eps=args.norm_eps,
                                              tie_word_embeddings=args.weight_tying,
                                              use_short_conv='b' in getattr(args, 'canon_set', ''),
                                              fuse_cross_entropy=False,
                                              output_gate_rank = 16,
                                              zeyuan_split_init = True,
                                              zeyuan_layernorm = getattr(args, 'zeyuan_layernorm', False),
            )
            assert config.hidden_size % (config.num_heads) == 0
            config.head_dim = config.hidden_size // config.num_heads
            config.intermediate_size = config.hidden_size * 2
            if args.hidden_dim is not None:
                config.intermediate_size = args.hidden_dim
            config.canon_set = getattr(args, 'canon_set', '')
            config.canon_bias = getattr(args, 'canon_bias', False)
            config.canon_residual = getattr(args, 'canon_residual', True)
            config.canon_activation = getattr(args, 'canon_activation', False)
            config.canon_kernel = getattr(args, 'canon_kernel', 4)
        else:
            assert False, f"Model name {args.model_name} not recognized; this is GDN"
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
        cfg = GatedDeltaNetCanonForCausalLM.build_config_from_yaml(OmegaConf.create(dd).model)

        model = GatedDeltaNetCanonForCausalLM(cfg)

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