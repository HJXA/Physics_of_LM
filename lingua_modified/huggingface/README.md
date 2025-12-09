# Huggingface Linear Model (GLA, GDN, Mamba2) Code with Canon Layers and Enhanced Features

**Author**: Zeyuan Allen-Zhu  

This folder provides modified implementations of **GLA**, **GDN**, and **Mamba2** with support for **Canon Layers** and additional architectural improvements.  
The models can be used:
- **Within the Lingua codebase** — see [`lingua_modified/apps/gla`](../lingua_modified/apps/gla), or  
- **Standalone** — as Hugging Face–style models for training/evaluation (see [`demo_newmodel.py`](demo_newmodel.py)).

> The Hugging Face version of **LlamaCanon** is released under [`../../huggingface`](../../huggingface/).

---

## 🚀 Demo

[`demo_newmodel.py`](demo_newmodel.py) initializes a **GLA**, **GDN**, or **Mamba2** Canon model from scratch for pretraining.

---

## 🔧 Key Features and Enhancements

### 1. Canon Layers

Reference: [*Physics of Language Models: Part 4.1 — Architecture Design and the Magic of Canon Layers*](https://ssrn.com/abstract=5240330)

All GLA/GDN/Mamba2 Canon models support configurable **Canon layers** at positions **A**, **b**, **C**, and **D**.  
In the paper’s notation, *Canon-b* corresponds to the original authors’ untouched Conv1D implementation, while *Canon-ACD* can be selectively enabled via configuration.

- **`config.canon_set`** — defines where Canon layers are applied.  
  Example: `"AbbCD"` or `"AbCD"` applies Canon layers at all valid positions (duplicates ignored), while `"bD"` applies them only at **b** and **D**.  
  *Note:* For linear models, Canon-b (Conv1D) is enabled by default. To test Canon-B, manually bypass the `assert False` guard in code (see inline comments).

- **`config.canon_residual`** *(default: `True`)* — adds residual connections (except Canon-b).  
  *Recommended* for Canon-ACD.

- **`config.canon_activation`** *(default: `False`)* — applies SiLU activation on Canon (except Canon-b).  
  *Not recommended* per our findings.

- **`config.canon_kernel`** *(default: `4`)* — kernel size for `causal_conv1d` (`2`, `3`, or `4` supported).  
  Efficient CUDA support except Canon-b.

- **`config.canon_bias`** *(default: `False`)* — enables Canon bias (except Canon-b).  
  *Not recommended* per our findings.


---

### 2. **Modified Gated DeltaNet** (*codename: GDN2*)

Key changes:
- The original `g_proj` gating matrix is now **low-rank**, controlled by `config.output_gate_rank`.  
  **Recommended rank:** `16`.  
- `head_dim = hidden_size / num_heads` replaces the original 0.75 scaling, ensuring each GDN layer has  
  **≈6d² + o(d²)** parameters — Q/K: d² each; V/O: 2d² each.  
- Gated MLP layers use `intermediate_size = hidden_size * 2`, adding another **6d²** parameters.  
  → All of **GLA5**, **GDN2**, and **Mamba2(mlp)** share the same **6d² + 6d²** parameter structure for controlled comparison.  
> For 8B models, total per-layer size is **6d² + 7d²**, matching Llama-8B.
- `prenorm_residual_strategy` disabled for ablation control.  
- Low-rank matrices initialized with `sqrt(0.02)` variance.  


---

### 3. **Modified Gated Linear Attention** (*codename: GLA5*)

Key architectural changes:
- Replaced full `g_proj` gating with a **low-rank version** (`config.output_gate_rank`, recommended rank: `16`).  
- Set `expand_k = 1` and `expand_v = 2` (instead of 0.5 and 1), yielding **6d² + o(d²)** parameters per layer—matching GDN2.  
- Gated MLP with `intermediate_size = hidden_size * 2` adds **6d²** more parameters (or 7d² for 8B models).
  → Ensures **GLA5**, **GDN2**, and **Mamba2(mlp)** remain architecturally aligned.  
- Disabled `prenorm_residual_strategy` for controlled experiments.  
- Low-rank matrices initialized with `sqrt(0.02)` variance.

---

### 4. **Modified Mamba2**
- The original Mamba2 lacks MLP layers; each base layer has ≈**6d²** parameters.  
- We add a **gated MLP** (`intermediate_size = hidden_size * 2`), contributing another **6d²** parameters (or 7d² for 8B models).  
- This yields **architectural parity** across GDN2, GLA5, and Mamba2+MLP for direct performance comparison.

---

### 5. **Loading Pretrained Models from Lingua**

Utilities are provided to load **Lingua-pretrained** models into the HF architecture:

- **`model.load_from_lingua_state`** — maps a Lingua-trained `state_dict` to the HF-style model.  
- **`BlaCanonForCausalLM.build_config_from_yaml`** *(static method)* — converts a Lingua YAML config into an HF-compatible one, automatically mapping:  
  - `model_type: GLA5` → GLA5 configuration  
  - `model_type: GDN2` → GDN2 configuration


## 📖Citation

Please cite the following if you use our models or findings in your research:
```bibtex
@inproceedings{Allen2025-canon,
  author = {{Allen-Zhu}, Zeyuan},
  title = {{Physics of Language Models: Part 4.1, Architecture Design and the Magic of Canon Layers}},
  year = {2025},
  booktitle = {Proceedings of the 39th Conference on Neural Information Processing Systems},
  series = {NeurIPS~'25},
  note = {Full version available at \url{https://ssrn.com/abstract=5240330}} 
}
@misc{Allen2025-resonate,
    title = {{Physics of Language Models: Part 4.2, Canon Layers at Scale where Synthetic Pretraining Resonates in Reality}},
    author = {{Allen-Zhu}, Zeyuan},
    year = {2025},
    url = {https://physics.allen-zhu.com/part-4-architecture-design/part-4-2},
    note = {Code released at \url{https://github.com/facebookresearch/PhysicsLM4}},
}
```