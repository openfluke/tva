# Experiment: CNN3 Multi-Core Tiled L1 Caching — All Numerical Types

This experiment validates the performance and correctness of combining **Loop-Blocked Tiling**, **Multi-Core Parallelization**, and **Dynamic Hardware Detection** for 3D Convolutional layers (CNN3) across every numerical type used in the industry.

---

## The Goal

Break up the 3D convolution across all available CPU cores, using each core's **private L1 cache** simultaneously to bypass the memory bandwidth bottleneck. Then prove it works correctly for every numerical type — from 64-bit doubles down to 1-bit binary weights.

---

## How It Works

### 1. Dynamic Hardware Detection

`GetHardwareInfo()` queries the real L1 cache size at runtime:

| Platform | Detection Method |
|----------|-----------------|
| Windows | `wmic cpu get L2CacheSize,L3CacheSize` (L1 defaults to 32KB) |
| macOS / iOS | `sysctl -n hw.l1dcachesize` |
| Linux / Android | `/sys/devices/system/cpu/cpu0/cache/index*/` |

### 2. Dtype-Aware Tile Size

The tile size is computed per-dtype using the **actual storage bytes per weight** — not a fixed 4-byte assumption. Smaller types fit more into L1, allowing larger tiles:

```
T < cuberoot(L1 / (bytesPerWeight × inChannels))
```

| DType group | Bytes/weight stored in RAM | Tile formula (32KB L1, 32ch) | Tile on 32KB L1 | Tile on 128KB L1 (M-series) |
|---|---|---|---|---|
| Float64, Int64, Uint64 | 8 | cuberoot(128) = 5.0 | 8 (min) | 8 (min) |
| Float32, Int32, Uint32, Float16, BFloat16 | 4 | cuberoot(256) = 6.4 | 8 (min) | 8 (min) |
| Int16, Uint16 | 2 | cuberoot(512) = 8.0 | 8 | 8 |
| Int8, Uint8, FP8, Int4, Uint4, FP4, Int2, Uint2, Ternary, Binary | 1 | cuberoot(1024) = 10.1 | 8 | **16** |

On Apple M-series (128KB L1D), sub-byte types automatically get tile=16. On standard x86 (32KB L1), everything sits at 8 — the L1 is the constraint, not the formula.

> **Note on Float16/BFloat16:** These are stored as `[]float32` internally (simulated precision via Morph). Their storage footprint is 4 bytes/weight until a native 16-bit storage path is added.

### 3. Core Saturation via Semaphore Dispatch

Goroutines are dispatched **one per filter**, capped by a semaphore at `runtime.NumCPU()`:

```go
sem := make(chan struct{}, numCPUs)
for f := 0; f < filters; f++ {
    sem <- struct{}{}
    wg.Add(1)
    go func(b, f int) {
        defer func() { <-sem; wg.Done() }()
        // ... spatial tiling for this filter
    }(b, f)
}
```

This saturates every physical core regardless of tile size. Previously the dispatch was per filter-tile (`filters/tileSize = 4` goroutines), leaving most cores idle on many-core machines.

---

## Canonical Arithmetic — Bit-Determinism Across All Paths

All three paths (Normal, Single-Core Tiled, Multi-Core Tiled) use the **same canonical arithmetic** per dtype class, guaranteeing identical outputs:

| Storage type | Accumulator | Scale applied |
|---|---|---|
| `[]float32` (Float32, Float16, BFloat16) | `float32` | None |
| `[]float64` (Float64) | `float64` | None |
| `[]int64` (Int64, Uint64) | `float64` | At end: `sum *= scale` |
| `[]int32` (Int32, Uint32) | `float32` | At end: `sum *= scale` |
| `[]int16` (Int16, Uint16) | `float32` | At end: `sum *= scale` |
| `[]int8` (Int8, Uint8, FP8, Int4, Uint4, FP4, Int2, Uint2, Ternary, Binary) | `float32` | At end: `sum *= scale` |

Float64 gets its own dedicated tiled functions (`cnn3ForwardTiledF64`, `cnn3ForwardTiledF64Parallel`) that accumulate in `float64` to preserve precision. All other paths apply scale once after the full accumulation loop, not per-element — this is both faster and numerically identical between tiling strategies.

---

## Full Numerical Type Coverage — Windows Results

All 21 industry numerical types validated. Every type passes parity across all three paths (MaxDiff = 0.00e+00).

```text
=== CNN3 Multi-Core Tiling — All Numerical Types ===

| DType      | Tile  | Normal         | Single-Core    | Multi-Core     | 1C-Spd  | MC-Spd  | MaxDiff  | 1C-Par | MC-Par |
|------------|-------|----------------|----------------|----------------|---------|---------|----------|--------|--------|
| Float64    | 8     | 6.034232533s   | 3.999765433s   | 671.889666ms   | 1.51x   | 8.98x   | 0.00e+00 | PASS   | PASS   |
| Float32    | 8     | 5.808043800s   | 3.821229900s   | 594.818366ms   | 1.52x   | 9.76x   | 0.00e+00 | PASS   | PASS   |
| Float16    | 8     | 6.022105966s   | 3.907457966s   | 617.401666ms   | 1.54x   | 9.75x   | 0.00e+00 | PASS   | PASS   |
| BFloat16   | 8     | 5.947592833s   | 3.851163266s   | 600.530733ms   | 1.54x   | 9.90x   | 0.00e+00 | PASS   | PASS   |
| FP8-E4M3   | 8     | 5.879233033s   | 3.960553833s   | 622.865700ms   | 1.48x   | 9.44x   | 0.00e+00 | PASS   | PASS   |
| FP8-E5M2   | 8     | 5.849661166s   | 3.810731133s   | 639.642666ms   | 1.54x   | 9.15x   | 0.00e+00 | PASS   | PASS   |
| Int64      | 8     | 6.087717533s   | 3.960493233s   | 672.391866ms   | 1.54x   | 9.05x   | 0.00e+00 | PASS   | PASS   |
| Uint64     | 8     | 5.963018400s   | 3.840649300s   | 617.497300ms   | 1.55x   | 9.66x   | 0.00e+00 | PASS   | PASS   |
| Int32      | 8     | 5.836331766s   | 3.998992633s   | 626.164300ms   | 1.46x   | 9.32x   | 0.00e+00 | PASS   | PASS   |
| Uint32     | 8     | 5.654550800s   | 3.849519633s   | 612.622966ms   | 1.47x   | 9.23x   | 0.00e+00 | PASS   | PASS   |
| Int16      | 8     | 5.864517666s   | 3.914753166s   | 644.127733ms   | 1.50x   | 9.10x   | 0.00e+00 | PASS   | PASS   |
| Uint16     | 8     | 5.869446666s   | 3.901839900s   | 621.418200ms   | 1.50x   | 9.45x   | 0.00e+00 | PASS   | PASS   |
| Int8       | 8     | 5.812613066s   | 3.872499600s   | 617.053700ms   | 1.50x   | 9.42x   | 0.00e+00 | PASS   | PASS   |
| Uint8      | 8     | 5.900448500s   | 3.908468166s   | 634.776666ms   | 1.51x   | 9.30x   | 0.00e+00 | PASS   | PASS   |
| Int4       | 8     | 5.864039800s   | 3.870071933s   | 648.315800ms   | 1.52x   | 9.05x   | 0.00e+00 | PASS   | PASS   |
| Uint4      | 8     | 6.002881100s   | 3.924815766s   | 648.543666ms   | 1.53x   | 9.26x   | 0.00e+00 | PASS   | PASS   |
| FP4        | 8     | 5.959968366s   | 3.954932533s   | 617.204666ms   | 1.51x   | 9.66x   | 0.00e+00 | PASS   | PASS   |
| Int2       | 8     | 6.045082666s   | 3.892814166s   | 624.882266ms   | 1.55x   | 9.67x   | 0.00e+00 | PASS   | PASS   |
| Uint2      | 8     | 5.904033766s   | 3.890896166s   | 623.552866ms   | 1.52x   | 9.47x   | 0.00e+00 | PASS   | PASS   |
| Ternary    | 8     | 5.988010133s   | 3.903254300s   | 621.902900ms   | 1.53x   | 9.63x   | 0.00e+00 | PASS   | PASS   |
| Binary     | 8     | 5.967329766s   | 3.923752533s   | 620.740966ms   | 1.52x   | 9.61x   | 0.00e+00 | PASS   | PASS   |

✅ All parity checks passed across all numerical types!
```

---

## Multi-Platform Summary (Float32 reference)

| Platform | Normal | Single-Core | Multi-Core | MC Speedup |
|---|---|---|---|---|
| Windows (x86_64, ~12 cores) | 5.81s | 3.82s | 595ms | **9.76x** |
| Linux / Beast (x86_64) | 2.84s | 1.65s | 326ms | **8.73x** |
| Apple Silicon M-series (ARM64) | 1.31s | 1.33s | 279ms | **4.68x** |

---

## Why All Types Run at the Same Speed on CPU

Every path — regardless of dtype — ultimately executes:
```go
sum += float32(input) * float32(weight)
```

The weight may be stored as `int8`, `int16`, or `int64`, but it's cast to `float32` for the multiply. The CPU always executes **32-bit floating point MACs**. The dtype affects:
- Cache footprint (smaller = more fits per cache line)
- The tile size formula (now correctly tuned per dtype)
- Correctness of the output values (each type's quantization is preserved)

But it does NOT yet change the actual instruction type executed. True per-type acceleration requires:

| Technique | Potential speedup for INT8 | What's needed |
|---|---|---|
| AVX-512 VNNI (x86) | 4–16x over FP32 | CGo + C intrinsics or Go assembly |
| ARM NEON dot product | 4–8x | CGo or plan9 ASM |
| GPU tensor cores (INT8) | 16–64x | WebGPU / CUDA (next phase) |

The CPU implementation is the **correctness and scaling reference**. It is the verified ground truth that all GPU results will be validated against.

---

## Bugs Fixed During This Experiment

1. **`TileSize=0` guard bypass** — `CNN3ForwardPolymorphic` only entered the tiled path if `TileSize > 0`. When `SyncToCPU()` wasn't called, single-core tiled silently ran the non-tiled path — appearing as identical performance to Normal.

2. **`SyncToCPU()` on wrong receiver** — The experiment called `l.Network.SyncToCPU()`, which iterates `n.Layers`. Since `l` was a standalone layer not added to the network, `l.EnableMultiCoreTiling` was never set and multi-core silently ran as single-core.

3. **Dead core saturation** — The parallel dispatcher spawned `filters / tileSize = 4` goroutines regardless of core count. A 12-core machine had 8 cores idle. Fixed by dispatching one goroutine per filter behind a `numCPUs`-wide semaphore.

4. **Integer fast paths truncated float input** — Normal path for all integer types did `int32(float32_input)`, truncating 0.5 → 0. Tiled path did `float32(input) * float32(weight)`. Fixed by rewriting all integer normal paths to use float32 accumulation with scale applied at end.

5. **Generic fallthrough re-quantized stored weights** — The fallback path called `SimulatePrecision()` on weights already morphed by `Morph()`, causing int8 overflow for FP8/sub-byte types. Fixed by removing the per-element `SimulatePrecision` call and applying scale once to the full sum.

6. **Float64 precision lost in tiled paths** — The generic tiled function used `var sum float32` even for `[]float64` weights. Float64 now has dedicated tiled functions (`cnn3ForwardTiledF64`, `cnn3ForwardTiledF64Parallel`) that accumulate in `float64`.

7. **`default` branch in tiled switch hardcoded `scale=1.0`** — Sub-byte and non-standard types going through the `default` CastWeights path lost their scale factor. Fixed to pass `layer.WeightStore.Scale`.

8. **Tile size formula assumed 4 bytes per weight** — All dtypes got the same tile regardless of precision. Fixed by making `CalculateOptimalCNN3TileSize` dtype-aware, computing tile from actual storage bytes per weight.

---

## What This Establishes

This is the **complete CPU reference implementation** for 3D CNN forward passes across the full industry numerical type spectrum:

- Every type used in production ML (Float32, Float16, BFloat16, INT8, INT4, Binary, and more) is implemented, correct, and parallelised
- All three execution modes are bit-deterministic against each other for every type
- Hardware is auto-detected and tile sizes are tuned at runtime per dtype and per machine
- The implementation is pure Go — zero CGo, zero platform-specific assembly, single binary for Windows / Linux / macOS / Android / iOS

The next frontier is the **GPU path** — where INT8 and sub-byte types will show their true hardware acceleration via tensor cores, validated against this CPU reference.
