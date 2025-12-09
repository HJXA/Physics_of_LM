# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This code creates a new GLA/GDN/Mamba2Canon model from scratch, so that you can train using it

import sys, os, json
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from omegaconf import OmegaConf

lingua_linear_recipes_folder = "../../canon_linear_recipes"

################### GLA Canon Example ######################
from huggingface.configuration_gla_canon import GLACanonConfig
from huggingface.modeling_gla_canon import GLACanonForCausalLM

# You can load a lingua model config and convert it to HF config like this:
cfg = GLACanonForCausalLM.build_config_from_yaml(OmegaConf.create(json.load(open(os.path.join(lingua_linear_recipes_folder, "GLACanon-1B-Nemo-1T-lr0.002_model_lingua.json"), "r"))))
# You can also load directly from a HF json config like this (should be the same as above):
cfg = GLACanonConfig.from_dict(json.load(open(os.path.join(lingua_linear_recipes_folder, "GLACanon-1B-Nemo-1T-lr0.002_model_HF.json"), "r")))
model = GLACanonForCausalLM(cfg)
print(model)

# GLACanonForCausalLM(
#   (model): GLACanonModel(
#     (embeddings): Embedding(32000, 2048)
#     (layers): ModuleList(
#       (0-23): 24 x GLABlock(
#         (attn_norm): RMSNorm(2048, eps=1e-05)
#         (canonA): CanonLayerCustom(hidden_size=2048)
#         (attn): GatedCanonLinearAttention(
#           (q_proj): Linear(in_features=2048, out_features=2048, bias=False)
#           (k_proj): Linear(in_features=2048, out_features=2048, bias=False)
#           (v_proj): Linear(in_features=2048, out_features=4096, bias=False)
#           (g_proj): Sequential(
#             (0): Linear(in_features=2048, out_features=16, bias=False)
#             (1): Linear(in_features=16, out_features=4096, bias=True)
#           )
#           (q_conv1d): CanonLayerCustom(hidden_size=2048)
#           (k_conv1d): CanonLayerCustom(hidden_size=2048)
#           (v_conv1d): CanonLayerCustom(hidden_size=4096)
#           (gk_proj): Sequential(
#             (0): Linear(in_features=2048, out_features=16, bias=False)
#             (1): Linear(in_features=16, out_features=2048, bias=True)
#           )
#           (o_proj): Linear(in_features=4096, out_features=2048, bias=False)
#           (g_norm_swish_gate): FusedRMSNormSwishGate(256, eps=1e-05, activation=swish)
#         )
#         (canonC): CanonLayerCustom(hidden_size=2048)
#         (mlp_norm): RMSNorm(2048, eps=1e-05)
#         (mlp): GLACanonMLP(
#           (gate_proj): Linear(in_features=2048, out_features=8192, bias=False)
#           (canonD): CanonLayerCustom(hidden_size=8192)
#           (down_proj): Linear(in_features=4096, out_features=2048, bias=False)
#           (act_fn): SiLU()
#         )
#       )
#     )
#     (norm): RMSNorm(2048, eps=1e-05)
#   )


################## GDN Canon Example ######################
from huggingface.configuration_gated_deltanet_canon import GatedDeltaNetCanonConfig
from huggingface.modeling_gated_deltanet_canon import GatedDeltaNetCanonForCausalLM

# You can load a lingua model config and convert it to HF config like this:
cfg = GatedDeltaNetCanonForCausalLM.build_config_from_yaml(OmegaConf.create(json.load(open(os.path.join(lingua_linear_recipes_folder, "GDNCanon-1B-Nemo-1T-lr0.002_model_lingua.json"), "r"))))
# You can also load directly from a HF json config like this (should be the same as above):
cfg = GatedDeltaNetCanonConfig.from_dict(json.load(open(os.path.join(lingua_linear_recipes_folder, "GDNCanon-1B-Nemo-1T-lr0.002_model_HF.json"), "r")))
model = GatedDeltaNetCanonForCausalLM(cfg)
print(model)

# GatedDeltaNetCanonForCausalLM(
#   (model): GatedDeltaNetCanonModel(
#     (embeddings): Embedding(32000, 2048)
#     (layers): ModuleList(
#       (0-23): 24 x GatedDeltaNetCanonBlock(
#         (attn_norm): RMSNorm(2048, eps=1e-05)
#         (canonA): CanonLayerCustom(hidden_size=2048)
#         (canonC): CanonLayerCustom(hidden_size=2048)
#         (attn): GatedDeltaNetCanon(
#           (silu): SiLU()
#           (q_proj): Linear(in_features=2048, out_features=2048, bias=False)
#           (k_proj): Linear(in_features=2048, out_features=2048, bias=False)
#           (v_proj): Linear(in_features=2048, out_features=4096, bias=False)
#           (b_proj): Linear(in_features=2048, out_features=16, bias=False)
#           (a_proj): Linear(in_features=2048, out_features=16, bias=False)
#           (q_conv1d): CanonLayerCustom(hidden_size=2048)
#           (k_conv1d): CanonLayerCustom(hidden_size=2048)
#           (v_conv1d): CanonLayerCustom(hidden_size=4096)
#           (g_proj): Sequential(
#             (0): Linear(in_features=2048, out_features=16, bias=False)
#             (1): Linear(in_features=16, out_features=4096, bias=True)
#           )
#           (o_norm): FusedRMSNormSwishGate(256, eps=1e-05, activation=swish)
#           (o_proj): Linear(in_features=4096, out_features=2048, bias=False)
#         )
#         (mlp_norm): RMSNorm(2048, eps=1e-05)
#         (mlp): GatedDeltaNetCanonMLP(
#           (down_proj): Linear(in_features=4096, out_features=2048, bias=False)
#           (canonD): CanonLayerCustom(hidden_size=8192)
#           (both_proj): Linear(in_features=2048, out_features=8192, bias=False)
#           (swiglu_linear): SwiGLULinear()
#         )
#       )
#     )
#     (norm): RMSNorm(2048, eps=1e-05)
#   )
#   (lm_head): Linear(in_features=2048, out_features=32000, bias=False)
# )


################## Mamba2 Canon Example ######################
from huggingface.configuration_mamba2_canon import Mamba2CanonConfig
from huggingface.modeling_mamba2_canon import Mamba2CanonForCausalLM

# You can load a lingua model config and convert it to HF config like this:
cfg = Mamba2CanonForCausalLM.build_config_from_yaml(OmegaConf.create(json.load(open(os.path.join(lingua_linear_recipes_folder, "Mamba2Canon-1B-Nemo-1T-lr0.002_model_lingua.json"), "r"))))
# You can also load directly from a HF json config like this (should be the same as above):
cfg = Mamba2CanonConfig.from_dict(json.load(open(os.path.join(lingua_linear_recipes_folder, "Mamba2Canon-1B-Nemo-1T-lr0.002_model_HF.json"), "r")))
model = Mamba2CanonForCausalLM(cfg)
print(model)

# Mamba2CanonForCausalLM(
#   (backbone): Mamba2CanonModel(
#     (embeddings): Embedding(32000, 2048)
#     (layers): ModuleList(
#       (0-23): 24 x Mamba2CanonBlock(
#         (norm): Mamba2CanonRMSNorm()
#         (canonA): CanonLayerCustom(hidden_size=2048)
#         (mixer): Mamba2CanonMixer(
#           (act): SiLU()
#           (conv1d): Conv1d(4352, 4352, kernel_size=(4,), stride=(1,), padding=(3,), groups=4352)
#           (in_proj): Linear(in_features=2048, out_features=8464, bias=False)
#           (norm): MambaRMSNormGated()
#           (out_proj): Linear(in_features=4096, out_features=2048, bias=False)
#         )
#         (pre_mlp_layernorm): MistralRMSNorm((2048,), eps=1e-05)
#         (canonC): CanonLayerCustom(hidden_size=2048)
#         (mlp): MistralCanonMLP(
#           (both_proj): Linear(in_features=2048, out_features=8192, bias=False)
#           (down_proj): Linear(in_features=4096, out_features=2048, bias=False)
#           (act_fn): SiLU()
#           (canonD): CanonLayerCustom(hidden_size=8192)
#         )
#       )
#     )
#     (norm_f): Mamba2CanonRMSNorm()
#   )
#   (lm_head): Linear(in_features=2048, out_features=32000, bias=False)
# )