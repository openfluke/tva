# Poly Tiling & VTD Optimization

This directory contains the benchmarking suite for the **M-POLY-VTD** (Morphing-Polymorphic Volumetric Tensor Dispatch) tiling architecture.

## 🚀 Multi-Type Monster Benchmark (135M+ Params)

Testing across the full numerical spectrum reveals a massive reduction in memory footprint while maintaining high performance via tiling.

| DType | Memory (MB) | NoTile (s) | Tiled (s) | Speedup |
| :--- | :--- | :--- | :--- | :--- |
| **float64** | 1080.0 | 4.36s | 2.20s | **1.98x** |
| **float32** | 540.0 | 4.31s | 2.18s | **1.97x** |
| **float16** | 270.0 | 4.56s | 2.13s | **2.14x** |
| **int8** | 135.0 | 4.36s | 2.10s | **2.07x** |
| **fp4** | 67.5 | 5.54s | 2.03s | **2.73x** |
| **int2** | 33.8 | 6.05s | 2.05s | **2.95x** |
| **binary** | 16.9 | 6.37s | 2.08s | **3.06x** |

### Key Takeaways
1. **Memory Reduction**: We've achieved a **64-fold reduction** in theoretical memory footprint from `float64` (1080MB) to `binary` (16.9MB).
2. **The "Precision Penalty"**: In the "No Tiling" case, lower-bit types actually run *slower* due to the overhead of simulated arithmetic (masking/scaling).
3. **Tiling as a Leveler**: Our **3D Spatial Tiling** effectively mitigates this penalty, keeping the working set in cache and maintaining a consistent ~2.0s forward pass time regardless of numerical complexity. This makes low-bit inference practically "free" in terms of CPU overhead while yielding massive VRAM/Storage savings.

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
