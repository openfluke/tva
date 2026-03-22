# Experiment: CNN3 Multi-Core Tiled L1 Caching

This experiment explores the performance impact of combining **Loop-Blocked Tiling** with **Multi-Core Parallelization** for 3D Convolutional layers (CNN3).

## The Goal
The primary objective is to break up the volumetric tiling technique across all available CPU cores. By doing so, we aim to utilize the independent **L1/L2 CPU caches** of each core simultaneously, effectively bypassing the memory bandwidth bottleneck that traditionally strikes single-threaded volumetric operations.

## Multi-Platform Results

### 1. Apple Silicon (Mac Mini M4, 16GB)
On the M4, the native implementation is exceptionally fast, likely due to massive L2 caches and aggressive hardware prefetching.

```text
=== Performance Results ===
| Implementation       | Time         | Speedup (vs Normal) | Parity (vs Normal) |
|----------------------|--------------|-------------------|-------------------|
| Normal (Native)      | 1.300888583s | 1.00x             | BASE              |
| Single-Core Tiled    | 1.347335508s | 0.97x             | 0.000000e+00      |
| Multi-Core Tiled     | 430.649283ms | 3.02x             | 0.000000e+00      |
```
> **Observation**: Single-core tiling actually incurred a slight overhead on M4. This suggests the M4's hardware cache management is already highly optimized for the "Native" loop pattern.

### 2. Linux (Nitro 51 / Beast - x86_64)
The classic "Memory Wall" is much more evident here, and the tiling optimizations deliver massive gains.

```text
=== Performance Results ===
| Implementation       | Time         | Speedup (vs Normal) | Parity (vs Normal) |
|----------------------|--------------|-------------------|-------------------|
| Normal (Native)      | 2.811934892s | 1.00x             | BASE              |
| Single-Core Tiled    | 1.609353206s | 1.75x             | 0.000000e+00      |
| Multi-Core Tiled     | 475.451844ms | 5.91x             | 0.000000e+00      |
```
> **Observation**: A near **6x Speedup** confirms that for standard x86 architectures, multi-core tiling is a transformative optimization.

## Deep Breakdown

### Why the difference?
- **Apple M4 Architecture**: Apple's Unified Memory and oversized caches mean that "cache misses" are less penalizing than on x86. The overhead of managing tiles (calculating offsets, bounds checking) can sometimes eclipse the benefit when the hardware is already hiding the latency.
- **x86/Linux Optimization**: On traditional CPUs, the jump from 2.8s to 0.47s proves that the "Memory Wall" is the primary bottleneck. By parallelizing the tiles, we effectively created a **~5.9x wider pipe** to the L1 caches.

## Conclusion
The **Multi-Core Tiled Dispatcher** is a success. While Apple Silicon provides a high baseline, the optimizations provide critical performance boosts on both platforms and establish a scalable "Bedrock" for the next version of the engine.

✅ **Full Numerical Parity Verified across all platforms.**
