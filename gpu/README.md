# GPU Verification Tool

This tool validates the correctness and performance of GPU-accelerated neural network layers in LOOM against their CPU reference implementations.

## Usage

Run the tool from this directory:

```bash
go run main.go [flags]
```

## Flags

| Flag | Description | Default |
|------|-------------|---------|
| `-layer` | Specific layers to test (comma-separated). | All layers |
| `-depth` | Network workload size: `shallow`, `medium`, `deep`, `stress`. | `deep` |
| `-dtype` | Data types to test (e.g., `float32`, `int8`). | All types |
| `-adapter`| Substring to filter GPU adapter (e.g., "NVIDIA"). | "" |

## Examples

**Run a quick check on specific layers:**
```bash
go run main.go -dtype=float32 -depth=shallow -layer=Dense,LayerNorm,RMSNorm
```

**Test mostly everything:**
```bash
go run main.go
```

**Stress test a specific layer with all numerical types:**
```bash
go run main.go -layer=Conv2D -depth=stress
```

**Filter by GPU adapter:**
```bash
go run main.go -adapter=NVIDIA
```

## Supported Features

**Layers:**
- `Dense`
- `LayerNorm`, `RMSNorm`
- `Softmax`
- `Embedding`
- `Residual`
- `SwiGLU`
- `Conv1D`, `Conv2D`
- `MHA` (Multi-Head Attention)
- `RNN`, `LSTM`

**Depths (Workload Sizes):**
- `shallow`: ~256KB
- `medium`: ~4MB
- `deep`: ~16MB (Default)
- `stress`: ~64MB

**Numerical Types:**
- Floating Point: `float32`, `float64`, `float16`
- Integer: `int8`, `int16`, `int32`, `int64`
- Unsigned: `uint8`, `uint16`, `uint32`, `uint64`
