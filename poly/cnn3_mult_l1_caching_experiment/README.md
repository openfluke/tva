# Experiment: CNN3 Multi-Core Tiled L1 Caching

This experiment explores the performance impact of combining **Loop-Blocked Tiling** with **Multi-Core Parallelization** for 3D Convolutional layers (CNN3).

## The Goal
The primary objective is to break up the volumetric tiling technique across all available CPU cores. By doing so, we aim to utilize the independent **L1/L2 CPU caches** of each core simultaneously, effectively bypassing the memory bandwidth bottleneck that traditionally strikes single-threaded volumetric operations.

## Methodology
- **Implementation**: We added an `EnableMultiCoreTiling` flag to `VolumetricLayer` and implemented `cnn3ForwardTiledGenericParallel` in `poly/cnn3.go`.
- **Parallelization Strategy**: The workload is partitioned across `batch` and `filter tiles` using Go's `runtime.NumCPU()` and `sync.WaitGroup`.
- **Benchmarking**: A 32x32x32 volume with 32 channels and 32 filters was used for the comparison.

## Results
The Following output was captured on the test machine:

```text
=== CNN3 Multi-Core Tiling Experiment ===
Running Normal (Non-Tiled)...  5.75234406s
Running Single-Core Tiled...    3.6560858s
Running Multi-Core Tiled...     1.08672434s

=== Performance Results ===
| Implementation       | Time         | Speedup (vs Normal) | Parity (vs Normal) |
|----------------------|--------------|-------------------|-------------------|
| Normal (Native)      | 5.75234406s  | 1.00x             | BASE              |
| Single-Core Tiled    | 3.6560858s   | 1.57x             | 0.000000e+00      |
| Multi-Core Tiled     | 1.08672434s  | 5.29x             | 0.000000e+00      |

=== Numerical Sample (First 3 elements) ===
| Implementation       | Sample Values                            |
|----------------------|------------------------------------------|
| Normal (Native)      | 12.800031, 19.199993, 19.199993 |
| Single-Core Tiled    | 12.800031, 19.199993, 19.199993 |
| Multi-Core Tiled     | 12.800031, 19.199993, 19.199993 |

✅ All parity checks passed!
```

## Deep Breakdown

### 1. Normal (Native) - 5.75s
This is the baseline implementation using standard nested loops. It suffered from frequent cache misses as the 3D volume is too large to fit in the CPU's local cache, forcing constant trips to slower main RAM (DDR4/DDR5).

### 2. Single-Core Tiled - 3.66s (1.57x Speedup)
By introducing loop-blocking (tiling), we constrained the working set of the convolution to fit within the L1/L2 cache of a single core. This significantly reduced the "Memory Wall" effect, but remained bound by the frequency and throughput of a single core.

### 3. Multi-Core Tiled - 1.09s (5.29x Speedup)
This is the "Symphony" approach. By breaking the tiled workload across all cores, we not only multiplied the raw processing power but also the **Aggregate L1 Cache Capacity**. Each core processes its own independent tile in its own local cache, resulting in a near-linear speedup and a massive reduction in total execution time.

## Key Takeaway
The **5.29x Speedup** with **0.00e+00 Numerical Divergence** proves that multi-core tiling is a viable and highly effective strategy for bridging the gap between CPU and GPU performance in the `poly` engine. This technique is now scheduled for engine-wide integration in **v0.75.0**.
