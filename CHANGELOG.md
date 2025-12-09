# CHANGELOG

### Second Major Commit (Dec 9, 2025)

- **`lingua_modified`**: 
  - Added support for linear model training and huggingface model compatibility.
  - Added GLA5/GDN2/Mamba2 with Canon layers.

- **`canon_linear_recipes`**:
  - Contains training parameters for our 48 linear models.

- **`canon_linear_results`**:
  - Provides evaluation comparison across linear and llama models.

- **`data-synthetic-pretrain`**:

- **`data-reallife-eval`**:

### Initial Commit (July 23, 2025)

- **`lingua_modified`**: 
  - A modified [Meta Lingua](https://github.com/facebookresearch/lingua) pretraining codebase.
  - Added support for:
    - **[Canon layers](https://www.ssrn.com/abstract=5240330)** on Transformers
    - **QK-norm**
    - **Z-loss**
    - **Partial RoPE (Rotary Positional Encoding)**
  - Slightly changed the evaluation code to match lm_eval's default.

- **`canon_llama_recipes`**:
  - Contains training parameters for our 16 Llama models, available on [Huggingface](https://huggingface.co/facebook/PhysicsLM4.2__LlamaCanon-8B-Nemo-1T-lr0.003).

- **`canon_llama_results`**:
  - Provides evaluation performance metrics for the 16 models.
  - Includes interactive training-time charts for better visualization and comparison.

- **`huggingface`** (Llama with Canon Layers):
  - A modified Huggingface Llama model that includes support for:
    - **Canon layers**
    - **QK-norm**
    - **Partial RoPE**
  - Compatible (after code fix!) for loading models pretrained with the Meta Lingua codebase.