# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This code is modified from https://github.com/fla-org/flash-linear-attention/blob/main/fla/models/gated_deltanet/configuration_gated_deltanet.py (Jan/3/2025 version)
# Original authors: Songlin Yang, Yu Zhang and others
# Under MIT License
#
# Zeyuan's edits include: 
# - canon layers
# - zeyuan_split_init
# - zeyuan_layernorm
# - output_gate_rank
#
from typing import Dict, Optional

from transformers.configuration_utils import PretrainedConfig

class GatedDeltaNetCanonConfig(PretrainedConfig):
    model_type = 'GatedDeltaNetCanon'
    keys_to_ignore_at_inference = ['past_key_values']

    def __init__(
        self,
        attn_mode: str = "chunk",
        hidden_size: int = 2048,
        expand_v: int = 2,
        use_gate: bool = True,
        output_gate_rank: Optional[int] = None,  # Zeyuan's edit note: none means not using low-rankness on g_proj
        use_short_conv: bool = True,
        conv_size: int = 4,
        head_dim: int = 256,
        num_heads: int = 6,
        max_position_embeddings: int = 2048,
        hidden_ratio: Optional[int] = 4,
        intermediate_size: Optional[int] = None,
        hidden_act: str = "swish",
        num_hidden_layers: int = 21,
        norm_eps: float = 1e-6,
        attn: Optional[Dict] = None,
        use_cache: bool = True,
        pad_token_id: int = None,
        bos_token_id: int = 1,
        eos_token_id: int = 2,
        tie_word_embeddings: bool = False,
        initializer_range: float = 0.02,  # Zeyuan's edit note: changed to 0.02 from 0.006
        zeyuan_split_init: bool = False,  # Zeyuan's edit note: whether to scale up the init for low-rank matrices (g_proj)
        zeyuan_layernorm: bool = False,   # Zeyuan's edit note: whether to use pytorch default Layernorm
        fuse_norm: bool = True,
        fuse_swiglu: bool = True,
        fuse_cross_entropy: bool = True,
        vocab_size: int = 32000,
        **kwargs
    ):
        self.attn_mode = attn_mode
        self.hidden_size = hidden_size
        self.expand_v = expand_v
        self.use_gate = use_gate
        self.use_short_conv = use_short_conv
        self.conv_size = conv_size
        self.head_dim = head_dim
        self.num_heads = num_heads
        self.max_position_embeddings = max_position_embeddings

        self.hidden_ratio = hidden_ratio
        self.intermediate_size = intermediate_size
        self.hidden_act = hidden_act
        self.num_hidden_layers = num_hidden_layers
        self.norm_eps = norm_eps
        self.attn = attn
        self.use_cache = use_cache
        self.initializer_range = initializer_range

        self.fuse_norm = fuse_norm
        self.fuse_swiglu = fuse_swiglu
        self.fuse_cross_entropy = fuse_cross_entropy
        self.vocab_size = vocab_size

        # Zeyuan's edit note: added support for canon layers, see "Part 4.1, Architecture Design and the Magic of Canon Layers" (https://ssrn.com/abstract=5240330)
        self.canon_set = kwargs.pop("canon_set", "")
        self.canon_bias = kwargs.pop("canon_bias", False)
        self.canon_activation = kwargs.pop("canon_activation", False)
        self.canon_kernel = kwargs.pop("canon_kernel", 4)
        self.canon_residual = kwargs.pop("canon_residual", True)

        # Zeyuan's modification to GDN
        self.zeyuan_split_init = zeyuan_split_init
        self.output_gate_rank = output_gate_rank
        self.zeyuan_layernorm = zeyuan_layernorm

        if attn is not None:
            if not isinstance(attn, Dict):
                raise ValueError("attn must be a dictionary")
            if 'layers' not in attn:
                raise ValueError("Layer indices must be provided to initialize hybrid attention layers")
            if 'num_heads' not in attn:
                raise ValueError("Number of heads must be provided to initialize hybrid attention layers")
            attn['num_kv_heads'] = attn.get('num_kv_heads', attn['num_heads'])
            attn['qkv_bias'] = attn.get('qkv_bias', False)
            attn['window_size'] = attn.get('window_size', None)
            attn['rope_theta'] = attn.get('rope_theta', 10000.)

        super().__init__(
            pad_token_id=pad_token_id,
            bos_token_id=bos_token_id,
            eos_token_id=eos_token_id,
            tie_word_embeddings=tie_word_embeddings,
            **kwargs,
        )
