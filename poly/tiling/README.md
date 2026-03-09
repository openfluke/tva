# Poly Tiling & VTD Optimization

This directory contains the benchmarking suite for the **M-POLY-VTD** (Morphing-Polymorphic Volumetric Tensor Dispatch) tiling architecture.

## 🚀 Multi-Type Monster Benchmark (135M+ Params)

Testing across the full numerical spectrum reveals a massive reduction in memory footprint while maintaining high performance via tiling.

| DType | Memory (MB) | NoTile (s) | Tiled (s) | Speedup |
| :--- | :--- | :--- | :--- | :--- |
| **float64** | 1080.0 | 4.59s | 1.93s | **2.38x** |
| **float32** | 540.0 | 4.10s | 1.73s | **2.37x** |
| **float16** | 270.0 | 4.83s | 1.72s | **2.81x** |
| **int8** | 135.0 | 4.23s | 1.80s | **2.35x** |
| **fp4** | 67.5 | 5.86s | 1.82s | **3.22x** |
| **int2** | 33.8 | 6.36s | 1.82s | **3.48x** |
| **binary** | 16.9 | 6.34s | 3.54s | **1.79x** |

### Key Takeaways
1. **Memory Reduction**: We've achieved a **64-fold reduction** in theoretical memory footprint from `float64` (1080MB) to `binary` (16.9MB).
2. **Native Tiled Fast-Paths**: We achieved a **~20% speedup** by implementing zero-allocation weight access. The engine no longer allocates temporary 540MB buffers for conversion; it reads quantized data (`int8`, etc.) directly from RAM.
3. **Loop Index Math (Lifting)**: By lifting coordinate-based address calculations out of the 8-nested loops, we significantly reduced the "mathematical friction" of the 3D dispatcher, dropping the `float32` baseline from 2.1s to 1.73s.
4. **The Binary Trade-off**: The `binary` type uses **True Bit-Packing** (1 bit per weight). While this shaves memory to a tiny 16.9MB, it is currently slower on CPU (3.5s) due to the overhead of manual bit-shifting and masking logic in high-level Go code. This serves as the foundation for future WebGPU/SIMD acceleration.

## 🛠️ Usage

To run the full suite:
```powershell
go run tiling.go
```

To run the Multi-Type Monster suite:
```powershell
go run numerical_monster.go
```

To run using the standard Go testing tool:
```powershell
go test -v -bench . -run ^$ .
```

## 🧠 Cache Optimization Details

Contrary to standard linear convolution, this implementation uses:
1. **64-Byte Cache Alignment**: All weight buffers are allocated on 64-byte boundaries (see `AlignedFloat32` in `poly.go`) to prevent cache-line splits.
2. **Spatial Blocking**: Instead of `for z... for y... for x...`, we process the grid in cubic "tiles".
3. **Systolic Parallelism**: In `SystolicForward`, we process tiles in parallel across all CPU cores to saturate the memory bus.

> [!NOTE]
> Currently, the `TileSize` is a hardcoded heuristic (8 for layers, 4 for grid blocks) designed to fit typical L1/L2 cache sizes on modern x86/ARM CPUs. While it doesn't "probe" the cache size dynamically yet, these values are the generic "sweet spot" for most consumer hardware.
