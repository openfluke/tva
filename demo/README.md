# Loom Demos

This directory contains standalone demonstrations of the Loom framework in action.

## demos

### [MNIST Demo](./mnist)

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
go run tva/demo/mnist/main.go
```
