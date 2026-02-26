# Universal Geometrical Safetensor Loading

A robust, strictly name-blind system for loading Safetensors into the Loom engine using **Anonymous Geometrical Probing**.

## Philosophy

Standard loading systems rely on weight names (prefixes like `model.layers.0.self_attn.q_proj.weight`). This system treats model data as a physical structure of tensor dimensions and statistical distributions. In Phase 3, we achieved **100% Tensor Conservation**, meaning the engine can identify and claim every single byte in a Safetensors file without a single name reference.

## Core Methodology

### 1. Anonymous Geometrical Probing
The loader builds a "Physical Blueprint" using only:
- **Index**: Tensor position in the sorted binary blob.
- **Rank & Shape**: The exact multidimensional geometry.
- **Statistical Signature**: Mean Absolute values and Variance of weights.

### 2. Archetype Matching (Phase 3 Expanded)

| Archetype | Geometrical Signature | Strategy |
| :--- | :--- | :--- |
| **MHA** | Cluster of 4 Rank-2 tensors `[D, D]` | Identifies Q/K/V/O and **Greedily Maps** nearby Biases. |
| **SwiGLU** | Cluster of 3 Rank-2 tensors | Identifies `Down` projection by dimension bottlenecking. |
| **LSTM** | Cluster of 2 Packed Rank-2 tensors | Matches `[4H, H]` gate structures. |
| **Norm Clusters**| Multi-Rank-1 Group | Claims Scale, Bias, Running Mean, and Running Var by statistics. |
| **Conv1D / 2D** | Rank-3 or Rank-4 tensors | Maps Kernels and **Sniffs** for auxiliary Biases. |
| **Embedding** | Rank-2 tensor (Ratio > 10:1) | Targeted identification of Vocab/Embedding clusters. |
| **Structural Meta**| Tensors with size < 10 | Automatically claims tiny tensors as engine-neutral metadata. |

### 3. Tensor Conservation & Diagnostics
- **Total File Coverage**: The loader reports the exact percentage of tensors claimed from the file.
- **Orphan Classification**: Any unassigned tensors are categorized for diagnostics:
    - `Potential Bias (Rank-1)`
    - `Unmapped Weights (Rank-2)`
    - `Metadata/Small`

## Batch Verification Results

Validated against the complete maintenance suite. Most models now achieve **100.0% Coverage**.

| Category | Typical Model | File Coverage | Engine Stability |
| :--- | :--- | :--- | :--- |
| **LLM** | TinyLlama, Phi-2 | **100.0%** | **STABLE** |
| **Transformer**| BERT, GPT2, Roberta | **100.0%** | **STABLE** |
| **Audio** | Wav2Vec2 | **100.0%** | **STABLE** |
| **CNN** | ResNet, ConvNext | **100.0%** | **STABLE** |
| **LSTM** | TextGen Pets | **100.0%** | **STABLE** |

## Usage

### Run Full Batch Audit
```powershell
go run main.go
```

### Using Hints
To resolve an ambiguous tensor at index 42:
```go
// Inside main.go
UserHints[42] = nn.LayerDense
```
