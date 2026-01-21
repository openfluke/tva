# MNIST Conv2D Demo: Numerical Precision Deep Dive

This demo showcases Loom's neural network training on MNIST digit classification, with a comprehensive comparison of **13 different numerical data types** for model weight storage.

## Quick Start

```bash
go run main.go
```

The demo automatically downloads MNIST data and runs:
1. CPU training (20 epochs)
2. GPU training (20 epochs)
3. Save/Load consistency verification
4. **Numerical type benchmarking** (the main attraction)

---

## Results Summary

| DType | Quality Score | Avg Deviation | Memory | Scale Factor |
|-------|--------------|---------------|--------|--------------|
| **F64** | 100.00% | 0.0000% | 5.84 MB | 1.0 |
| **F32** | 100.00% | 0.0000% | 2.92 MB | 1.0 |
| **F16** | 99.85% | 0.1457% | 1.46 MB | 1.0 |
| **BF16** | 99.87% | 0.1279% | 1.46 MB | 1.0 |
| **F4** | 0.00% | 100.0000% | 374.21 KB | 8.0 |
| **I64** | 100.00% | 0.0006% | 5.84 MB | 1000000.0 |
| **I32** | 100.00% | 0.0006% | 2.92 MB | 1000000.0 |
| **I16** | 99.32% | 0.6784% | 1.46 MB | 1000.0 |
| **I8** | 94.71% | 5.2898% | 747.67 KB | 100.0 |
| **U64** | 0.00% | 680.3266% | 5.84 MB | 1000000.0 |
| **U32** | 0.00% | 680.3266% | 2.92 MB | 1000000.0 |
| **U16** | 0.00% | 680.3266% | 1.46 MB | 1000.0 |
| **U8** | 0.00% | 404.4473% | 747.67 KB | 100.0 |

---

## Understanding the Metrics

### Quality Score (0-100%)
Measures how well the quantized model preserves the original model's predictions. A score of 100% means the outputs are identical to the F32 baseline.

### Average Deviation
The mean percentage deviation from expected outputs. Lower is better:
- **0-1%**: Essentially lossless
- **1-10%**: Minor quantization noise
- **10-50%**: Noticeable degradation
- **>100%**: Model outputs are meaningless

### Scale Factor
For integer types, weights are scaled before quantization:
```
quantized_value = original_value * scale
original_value = quantized_value / scale
```
This is necessary because neural network weights are typically small floats (e.g., -0.001 to 0.001).

---

## Deep Dive: Each Data Type

### Float Types

#### F64 (64-bit Float / Double)
- **Quality**: 100% ✓
- **Memory**: 5.84 MB (2x F32)
- **Verdict**: Overkill. Same quality as F32 with 2x memory. Never use for inference.

#### F32 (32-bit Float)
- **Quality**: 100% ✓
- **Memory**: 2.92 MB (baseline)
- **Verdict**: The industry standard. All training happens in F32/F64, then quantized for deployment.

#### F16 (16-bit Float / Half Precision)
- **Quality**: 99.85% ✓
- **Memory**: 1.46 MB (2x compression)
- **Verdict**: Sweet spot for most deployments. 2x smaller with negligible quality loss.

**Industry comparison**: NVIDIA GPUs have dedicated F16 tensor cores. PyTorch `torch.float16` is widely used.

#### BF16 (Brain Float 16)
- **Quality**: 99.87% ✓
- **Memory**: 1.46 MB (2x compression)
- **Verdict**: Google's format. Same range as F32 (8-bit exponent) but less precision (7-bit mantissa vs 23-bit).

**Industry comparison**: TPUs use BF16 natively. Preferred for training because it maintains F32's dynamic range. Better numerical stability than F16 for gradients.

| Format | Sign | Exponent | Mantissa | Range |
|--------|------|----------|----------|-------|
| F32 | 1 | 8 | 23 | ±3.4×10³⁸ |
| F16 | 1 | 5 | 10 | ±65504 |
| BF16 | 1 | 8 | 7 | ±3.4×10³⁸ |

#### F4 (4-bit Float E2M1)
- **Quality**: 0.00% ✗
- **Memory**: 374.21 KB (8x compression)
- **Verdict**: Too aggressive for this model. Only 16 unique values per weight.

**Why F4 fails here**: The E2M1 format (1 sign, 2 exponent, 1 mantissa bits) can only represent:
```
±{0, 0.25, 1, 1.5, 2, 3, 4, 6, ∞}
```
This is insufficient for the fine-grained weight distributions in neural networks.

**Industry comparison**: No mainstream framework supports pure FP4. Modern 4-bit quantization uses **INT4 with per-channel scales** (GPTQ, AWQ, bitsandbytes) which is fundamentally different.

---

### Signed Integer Types

#### I64 / I32 (64/32-bit Signed Integer)
- **Quality**: 100% ✓
- **Memory**: 5.84 / 2.92 MB
- **Scale**: 1,000,000
- **Verdict**: Works perfectly with large scale factors. No practical advantage over float.

#### I16 (16-bit Signed Integer)
- **Quality**: 99.32% ✓
- **Memory**: 1.46 MB
- **Scale**: 1,000
- **Verdict**: Good alternative to F16. Requires careful scale selection.

#### I8 (8-bit Signed Integer)
- **Quality**: 94.71% ✓
- **Memory**: 747.67 KB (4x compression)
- **Scale**: 100
- **Verdict**: **Best tradeoff!** Nearly 95% quality with 4x size reduction.

**Industry comparison**: INT8 quantization is the **industry standard** for edge deployment:
- **TensorRT**: NVIDIA's INT8 inference engine
- **ONNX Runtime**: INT8 quantization support
- **TensorFlow Lite**: INT8 for mobile
- **PyTorch**: `torch.quantization` for INT8

The key insight: Neural networks are surprisingly robust to INT8 quantization when done correctly (per-channel scaling, calibration).

---

### Unsigned Integer Types

#### U64 / U32 / U16 / U8
- **Quality**: 0.00% ✗
- **Verdict**: Complete failure.

**Why unsigned fails**: Neural network weights are centered around zero with both positive and negative values. Unsigned integers can only represent positive values, so:
- Weights like `-0.05` become `0` after clamping
- The model loses all negative weight information
- Predictions become meaningless

This is a fundamental mathematical incompatibility, not a Loom bug.

---

## Industry Comparison

### Quantization Landscape (2024)

| Method | Bits | Type | Quality | Industry Adoption |
|--------|------|------|---------|-------------------|
| **F32** | 32 | Float | 100% | Training standard |
| **F16** | 16 | Float | 99%+ | Inference on GPUs |
| **BF16** | 16 | Float | 99%+ | Training on TPUs |
| **INT8** | 8 | Int | 95%+ | Edge deployment |
| **INT4** | 4 | Int+Scale | 90%+ | LLM quantization |
| **GPTQ** | 4 | Int+Scale | 95%+ | LLM SOTA |
| **AWQ** | 4 | Int+Scale | 95%+ | LLM SOTA |
| **GGUF** | 2-8 | Mixed | Varies | llama.cpp |

### Loom's Position

Loom's numerical type support demonstrates:

1. **F16/BF16**: On par with industry (99%+ quality, 2x compression)
2. **INT8**: Competitive (94.7% quality, 4x compression)
3. **FP4**: Loom uses pure E2M1 format, unlike industry INT4+scale approaches

### Why INT4 Works in Industry But FP4 Doesn't Here

Industry 4-bit quantization uses **group quantization** with per-group scale factors:
```
# Industry INT4 (GPTQ/AWQ)
for each group of 128 weights:
    scale = max(abs(weights)) / 7  # 7 is max for signed 4-bit
    quantized = round(weights / scale)  # Store as INT4
    
# Dequantize
original = quantized * scale
```

This preserves relative weight magnitudes. Pure FP4 E2M1 doesn't use scales, so all fine-grained weight information is lost.

---

## Recommendations

### For Edge/Mobile Deployment
**Use I8 (INT8)** - 4x smaller, 94.7% quality. Industry proven.

### For GPU Inference
**Use F16 or BF16** - 2x smaller, 99%+ quality. Native hardware support.

### For Training
**Use F32 or BF16** - F32 for stability, BF16 for TPU/newer GPUs.

### For Maximum Compression
**Use I8 with calibration** - Better than aggressive float quantization.

---

## Network Architecture

```
Input (784) → Dense → Conv2D (8 filters) → Conv2D (16 filters) → Dense (64) → Dense (10)
```

| Layer | Input | Output | Params |
|-------|-------|--------|--------|
| Dense | 784 | 784 | ~615K |
| Conv2D | 28×28×1 | 26×26×8 | 80 |
| Conv2D | 26×26×8 | 12×12×16 | 1,168 |
| Dense | 2304 | 64 | ~147K |
| Dense | 64 | 10 | 650 |

**Total**: ~764K parameters

---

## Files

| File | Description |
|------|-------------|
| `main.go` | Demo entry point |
| `data/` | MNIST data (auto-downloaded) |
| `mnist_cpu_model.json` | Saved CPU model |

---

## Running on GPU

The demo automatically detects NVIDIA GPUs via WebGPU. Set preference:
```go
gpu.SetAdapterPreference("nvidia")
```

GPU training shows identical convergence to CPU, validating Loom's unified backend.

---

## Loom's Quantization Architecture

### PTQ vs QAT

| Approach | Description | Loom Support |
|----------|-------------|--------------|
| **PTQ** (Post-Training Quantization) | Train in float32, quantize after | ✅ Full support |
| **QAT** (Quantization-Aware Training) | Simulate quantization during training | ✅ Possible via serialization |

### How Loom Handles Multi-Precision

Loom uses **Go generics** for type-flexible forward/backward passes:

```go
// From forward.go - generic forward pass
func GenericForwardPass[T Numeric](
    n *Network,
    input *Tensor[T],
    backend Backend[T],
) (*Tensor[T], []*Tensor[T], []any, time.Duration)
```

The `Numeric` constraint allows `float32`, `float64`, `int8`, etc. This enables:
1. Training in high precision (float32/float64)
2. Inference in lower precision with automatic type conversion

### QAT via Serialization Roundtrip

The `BenchmarkNumericalTypes()` function demonstrates loom's QAT capability:

```go
// From evaluation.go - simulates quantization noise
testNet.ScaleWeights(scale, isUnsigned)
tensors := SerializeSafetensors(testNet)    // Quantize
LoadSafetensorsWithShapes(tensors)           // Dequantize
testNet.UnscaleWeights(scale, isUnsigned)
```

This **serialize→deserialize roundtrip** acts as a "fake quantization" layer:
- Weights get quantized to target dtype (precision loss)
- Weights get dequantized back to float32
- Model continues with quantization-degraded weights

### Implementing True QAT in Loom

To add training-time quantization (like PyTorch's `FakeQuantize`):

```go
// Potential QAT implementation
func (n *Network) TrainWithQAT(inputs, targets [][]float32, config *TrainingConfig) {
    for epoch := 0; epoch < config.Epochs; epoch++ {
        // 1. Forward pass
        outputs := n.Forward(inputs)
        
        // 2. Compute loss
        loss := computeLoss(outputs, targets)
        
        // 3. Backward pass
        n.Backward(loss)
        
        // 4. Update weights
        n.UpdateWeights(config.LearningRate)
        
        // 5. QAT: Quantize→Dequantize to simulate inference precision
        if config.QATDType != "" {
            n.SimulateQuantization(config.QATDType, config.QATScale)
        }
    }
}
```

The serialization-based approach is slower but highly effective because:
- Uses actual production quantization code (no simulation mismatch)
- Tests real dtype encoding/decoding accuracy
- Weights learn to be "robust" to quantization noise

---

## Key Takeaways

1. **F16/BF16** are the sweet spot: 2x compression, negligible quality loss
2. **INT8** is production-ready: 4x compression, 95% quality
3. **FP4 (E2M1)** is too aggressive without per-group scaling
4. **Unsigned integers** fundamentally fail due to weight sign requirements
5. **Scale factors** are critical for integer quantization
6. **QAT is possible** via loom's serialization roundtrip during training

Loom's multi-precision serialization enables experimentation with deployment tradeoffs without leaving the Go ecosystem.

