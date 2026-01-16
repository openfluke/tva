# Loom Demos

This directory contains standalone demonstrations of the Loom framework in action.

## demos

### [Conv2D MNIST Demo](./conv2d-mnist)

A comprehensive demonstration of training and validating a Convolutional Neural Network (CNN) on the MNIST handwritten digit dataset.

**Key Features:**
- **Architecture**: Uses `Conv2D` layers for spatial feature extraction, followed by `Dense` layers for classification.
- **Hardware Agnostic**: Automatically detects and utilizes GPU via WebGPU if available, falling back to CPU otherwise.
- **Numerical Type Validation**: Performs exhaustive testing of saving and loading weights across all 13 supported Safetensors `DType` formats (F64, F32, F16, BF16, F4, I64, I32, I16, I8, U64, U32, U16, U8).
- **Parity Verification**: Confirms bit-deterministic output between CPU and GPU execution paths.
- **Resource Monitoring**: Tracks RAM usage and file size efficiency for different quantization levels.

#### Numerical Type Comparison Summary

The following table summarizes the performance and efficiency of different numerical types when saving and loading the MNIST CNN model:

| DType | Quality Score | Avg Dev | File Size | RAM Usage |
|-------|---------------|---------|-----------|-----------|
| **F32** | 100.00% | 0.0000% | 2.92 MB | 5.86 MB |
| **F64** | 100.00% | 0.0000% | 5.84 MB | 8.77 MB |
| **F16** | 100.00% | 0.0006% | 1.46 MB | 4.40 MB |
| **BF16**| 100.00% | 0.0009% | 1.46 MB | 4.40 MB |
| **F4**  | 99.40%  | 0.6029% | 374.23 KB| 3.30 MB |
| **I64** | 100.00% | 0.0000% | 5.84 MB | 8.77 MB |
| **I32** | 100.00% | 0.0000% | 2.92 MB | 5.86 MB |
| **I16** | 99.94%  | 0.0554% | 1.46 MB | 4.40 MB |
| **I8**  | 99.61%  | 0.3855% | 747.70 KB| 3.67 MB |
| **U64** | 100.00% | 0.0000% | 5.84 MB | 8.77 MB |
| **U32** | 100.00% | 0.0000% | 2.92 MB | 5.86 MB |
| **U16** | 99.95%  | 0.0523% | 1.46 MB | 4.40 MB |
| **U8**  | 99.61%  | 0.3855% | 747.70 KB| 3.67 MB |

#### Usage

```bash
go run tva/demo/conv2d-mnist/main.go
```

---

### [Dense XOR Demo](./dense-xor)

Demonstrates fully-connected (Dense) layers solving the classic XOR problem.

**Key Features:**
- **Architecture**: 3-layer Dense network (2 → 8 → 8 → 1)
- **Task**: Binary classification (XOR function)
- **Multi-Precision**: Tests all 13 Safetensors dtypes
- **CPU/GPU Parity**: Verifies identical results across compute backends

**Usage:**
```bash
go run tva/demo/dense-xor/main.go
```

---

### [Conv1D Sequence Demo](./conv1d-sequence)

Demonstrates 1D Convolution for sequence pattern detection.

**Key Features:**
- **Architecture**: Conv1D layers for temporal feature extraction
- **Task**: Sequence classification (rising vs falling trends)
- **Input**: 1D sequences of length 32
- **Multi-Precision**: Full dtype testing

**Usage:**
```bash
go run tva/demo/conv1d-sequence/main.go
```

---

### [RNN Sequence Demo](./rnn-sequence)

Demonstrates Recurrent Neural Networks for sequence modeling.

**Key Features:**
- **Architecture**: RNN layer with hidden state
- **Task**: Temporal pattern classification
- **Sequence Length**: 16 timesteps
- **Multi-Precision**: All 13 dtypes tested

**Usage:**
```bash
go run tva/demo/rnn-sequence/main.go
```

---

### [LSTM Memory Demo](./lstm-memory)

Demonstrates Long Short-Term Memory networks for long-range dependencies.

**Key Features:**
- **Architecture**: LSTM with gated cells
- **Task**: Long-term memory pattern detection
- **Advantage**: Better gradient flow than vanilla RNN
- **Multi-Precision**: Full dtype support

**Usage:**
```bash
go run tva/demo/lstm-memory/main.go
```

---

### [Multi-Head Attention Demo](./mha-attention)

Demonstrates self-attention mechanisms powering transformer architectures.

**Key Features:**
- **Architecture**: Multi-Head Attention (4 heads, dModel=64)
- **Task**: Sequence-to-sequence attention patterns
- **Mechanism**: Query-Key-Value attention
- **Multi-Precision**: All 13 dtypes

**Usage:**
```bash
go run tva/demo/mha-attention/main.go
```

---

### [Normalization Demo](./norm-comparison)

Compares LayerNorm vs RMSNorm for training stability.

**Key Features:**
- **Architectures**: Two networks (LayerNorm and RMSNorm)
- **Task**: Multi-class classification (10 classes)
- **Comparison**: Convergence speed and final accuracy
- **Multi-Precision**: Both normalization types tested with all dtypes

**Usage:**
```bash
go run tva/demo/norm-comparison/main.go
```

---

### [SwiGLU MLP Demo](./swiglu-mlp)

Demonstrates modern gated activation blocks used in state-of-the-art LLMs.

**Key Features:**
- **Architecture**: SwiGLU gated MLP blocks
- **Task**: High-dimensional classification
- **Activation**: SiLU-based gating mechanism
- **Multi-Precision**: Full dtype testing

**Usage:**
```bash
go run tva/demo/swiglu-mlp/main.go
```

---

## Common Features Across All Demos

Every demo includes:
- ✅ **CPU Training**: Baseline training on CPU
- ✅ **GPU Training**: Accelerated training via WebGPU
- ✅ **Parity Verification**: CPU/GPU output consistency
- ✅ **Multi-Precision**: Testing all 13 Safetensors dtypes
- ✅ **Resource Monitoring**: File size and RAM usage tracking
- ✅ **Quality Metrics**: Deviation analysis and quality scores
- ✅ **Summary Tables**: Formatted comparison of all dtypes
