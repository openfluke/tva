# Experiment: CNN3 Backward GPU WebGPU — All Numerical Types

This experiment validates WebGPU-accelerated 3D convolution backward pass (DX + DW) across all 21 numerical types, comparing three GPU execution paths against a CPU Multi-Core Tiled reference.

---

## The Goal

Take the CPU CNN3 backward pass and push both gradient computations onto the GPU via WebGPU — proving that:

1. The GPU backward path is **dramatically faster** than the CPU reference for every dtype
2. All GPU paths produce **bit-identical results** (0.00e+00 diff) against the CPU reference
3. Two distinct backward passes (DX = input gradient, DW = weight gradient) execute correctly on GPU
4. Tiled shared-memory variants are validated against the normal global-memory path

---

## Paths Under Test

| # | Path | Dispatch | Notes |
|---|------|----------|-------|
| 1 | **CPU MC-Tiled** | `CNN3BackwardTiledParallel` — goroutines over (batch, inC) for DX, over filters for DW | Canonical reference |
| 2 | **GPU Normal** | `DispatchCNN3BackwardDX` + `DispatchCNN3BackwardDW` — global memory | One thread per output element |
| 3 | **GPU Tiled SC** | `DispatchCNN3TiledBackwardDX/DW(scTile=64)` — workgroup shared-mem cache | 2 NVIDIA warps per workgroup |
| 4 | **GPU Tiled MC** | `DispatchCNN3TiledBackwardDX/DW(mcTile=256)` — larger workgroup | 8 NVIDIA warps per workgroup |

---

## How It Works

### 1. Two Backward Passes Per Iteration

Each backward evaluation runs two GPU dispatches:

- **DX pass** — computes `gradInput[b, ic, id, ih, iw]` by summing over all filters and kernel positions.
  Each thread handles one input element; all threads in a workgroup cooperatively load one filter's full 864-weight kernel into `wCache` before iterating.

- **DW pass** — computes `gradWeights[f, ic, kd, kh, kw]` by summing over all batch/spatial positions.
  Each thread handles one weight gradient; all threads in a workgroup cooperatively cache `tileSize` pre-multiplied dy values (`gradOutput * actDeriv`) into `dyCache`, then all threads reuse that cache, reducing global gradOutput reads by `tileSize×`.

### 2. Raw Weights — No Scale in Backward

The CPU backward (`CNN3BackwardTiledParallel`) uses raw integer values cast to `float32` without multiplying the scale factor — the backward pass propagates gradients through the unscaled arithmetic. GPU backward matches exactly:

```go
// rawF32: raw int values as float32, NO scale multiplication
raw := rawF32(ws, cfg.dtype)
weightBuf = upload(raw)   // both DX and DW use the same raw weight buffer
```

This is why all 21 types achieve **0.00e+00 diff** — no floating-point scale multiplication to introduce rounding divergence.

### 3. Auto-Detected GPU Tile Sizes

`CNN3GPUTileSizes(ctx)` derives SC and MC workgroup sizes from the GPU's own reported limits:

```go
func CNN3GPUTileSizes(ctx *WGPUContext) (scTile, mcTile int) {
    scTile = ctx.GPUTileSize * 4
    if scTile < 64 { scTile = 64 }   // never below 2 NVIDIA warps

    mcTile = int(ctx.Limits.MaxComputeInvocationsPerWorkgroup)
    if mcTile > 256 { mcTile = 256 }
    mcTile = (mcTile / 64) * 64      // align to wavefront boundary
}
```

On the GTX 1650 Super (`MaxComputeInvocationsPerWorkgroup = 256`):
- **SC tile = 64** — 2 NVIDIA warps, moderate register pressure
- **MC tile = 256** — 8 NVIDIA warps, higher SM occupancy

### 4. Tiled DX Shader — Cooperative Filter-Kernel Cache

`ShaderTiledCNN3BackwardDX(tileSize, kernelVol int)` generates WGSL with the cache size baked in:

```wgsl
var<workgroup> wCache: array<f32, 864>;   // kernelVol = inC * kD * kH * kW

@compute @workgroup_size(64, 1, 1)
fn main(...) {
    // For each filter f: load filter kernel cooperatively into wCache
    for (var f = 0u; f < params.Filters; f++) {
        for (var t = local_id.x; t < 864u; t += 64u) {
            wCache[t] = weights[f * 864u + t];
        }
        workgroupBarrier();

        // Each thread accumulates its input gradient using cached kernel
        for (var kIdx = 0u; kIdx < 864u; kIdx++) {
            // reverse-map kernel index to (ic, kd, kh, kw)
            // check if corresponding output position is valid
            gradInput[inIdx] += wCache[kIdx] * gradOutput[outIdx] * actDeriv;
        }
        workgroupBarrier();
    }
}
```

### 5. Tiled DW Shader — Cooperative dy-Value Cache

`ShaderTiledCNN3BackwardDW(tileSize int)` caches the pre-multiplied activation derivative output values, not the weights (since weights are the *output* here):

```wgsl
var<workgroup> dyCache: array<f32, 64>;   // tileSize dy values per tile

@compute @workgroup_size(64, 1, 1)
fn main(...) {
    // Loop spatial positions in tiles of tileSize
    for (var tile = 0u; ...) {
        // Load tileSize dy values cooperatively: dy = gradOutput * actDeriv
        if (local_id.x < remaining) {
            dyCache[local_id.x] = gradOutput[pos] * actDeriv(preAct[pos]);
        }
        workgroupBarrier();

        // All threads accumulate weight gradient using cached dy values
        for (var t = 0u; t < tileSize; t++) {
            gradWeights[wIdx] += input[inputIdx] * dyCache[t];
        }
    }
}
```

This reduces `gradOutput` global reads from `O(weightSize × spatialPositions)` to `O(spatialPositions + weightSize × spatialPositions / tileSize)`.

### 6. Timing Methodology

Each GPU path is measured by polling after every individual iteration — this captures true GPU execution time, not just command submission overhead:

```go
var normTotal time.Duration
for i := 0; i < gpuIters; i++ {
    tStart := time.Now()
    ctx.DispatchCNN3BackwardDX(...)
    ctx.DispatchCNN3BackwardDW(...)
    ctx.Device.Poll(true, nil)   // blocks until GPU finishes this pair
    normTotal += time.Since(tStart)
}
tGPUNorm := normTotal / time.Duration(gpuIters)
```

Without the per-iteration Poll, `time.Since(start) / gpuIters` truncates to `0s` because GPU command submission is async and completes in nanoseconds from the CPU's perspective.

---

## Benchmark Results — GTX 1650 Super

**Config:** batch=1, inC=32, inD=32, inH=32, inW=32, filters=32, kernel=3×3×3, stride=1, padding=1
**GPU:** NVIDIA GTX 1650 Super (SC tile=64, MC tile=256, MaxComputeInvocations=256)
**Measurement:** 10 iterations each, per-iteration Poll synchronization

| DType    | Tile | CPU MC      | GPU Normal  | GPU Tiled SC | GPU Tiled MC | GN-Spd  | SC-Spd  | MC-Spd  | DX diff | DW diff | GN   | SC   | MC   |
|----------|------|-------------|-------------|--------------|--------------|---------|---------|---------|---------|---------|------|------|------|
| Float64  | 8    | 4.934s      | 65.89ms     | 98.22ms      | 100.12ms     | 74.9x   | 50.2x   | 49.3x   | 0.00e+00 | 0.00e+00 | PASS | PASS | PASS |
| Float32  | 8    | 4.936s      | 67.49ms     | 100.41ms     | 102.37ms     | 73.1x   | 49.2x   | 48.2x   | 0.00e+00 | 0.00e+00 | PASS | PASS | PASS |
| Float16  | 8    | 4.923s      | 69.17ms     | 97.49ms      | 100.57ms     | 71.2x   | 50.5x   | 48.9x   | 0.00e+00 | 0.00e+00 | PASS | PASS | PASS |
| BFloat16 | 8    | 5.153s      | 66.88ms     | 96.70ms      | 101.00ms     | 77.1x   | 53.3x   | 51.0x   | 0.00e+00 | 0.00e+00 | PASS | PASS | PASS |
| FP8-E4M3 | 8    | 5.052s      | 67.00ms     | 97.66ms      | 100.79ms     | 75.4x   | 51.7x   | 50.1x   | 0.00e+00 | 0.00e+00 | PASS | PASS | PASS |
| FP8-E5M2 | 8    | 4.838s      | 65.90ms     | 96.70ms      | 99.38ms      | 73.4x   | 50.0x   | 48.7x   | 0.00e+00 | 0.00e+00 | PASS | PASS | PASS |
| Int64    | 8    | 4.845s      | 66.96ms     | 98.46ms      | 99.31ms      | 72.3x   | 49.2x   | 48.8x   | 0.00e+00 | 0.00e+00 | PASS | PASS | PASS |
| Uint64   | 8    | 4.828s      | 67.68ms     | 98.01ms      | 101.00ms     | 71.3x   | 49.3x   | 47.8x   | 0.00e+00 | 0.00e+00 | PASS | PASS | PASS |
| Int32    | 8    | 4.855s      | 66.43ms     | 99.78ms      | 101.04ms     | 73.1x   | 48.7x   | 48.0x   | 0.00e+00 | 0.00e+00 | PASS | PASS | PASS |
| Uint32   | 8    | 4.851s      | 66.90ms     | 98.97ms      | 101.79ms     | 72.5x   | 49.0x   | 47.7x   | 0.00e+00 | 0.00e+00 | PASS | PASS | PASS |
| Int16    | 8    | 4.851s      | 66.47ms     | 98.03ms      | 100.67ms     | 73.0x   | 49.5x   | 48.2x   | 0.00e+00 | 0.00e+00 | PASS | PASS | PASS |
| Uint16   | 8    | 4.882s      | 67.21ms     | 96.81ms      | 101.71ms     | 72.6x   | 50.4x   | 48.0x   | 0.00e+00 | 0.00e+00 | PASS | PASS | PASS |
| Int8     | 8    | 4.884s      | 67.93ms     | 97.29ms      | 99.47ms      | 71.9x   | 50.2x   | 49.1x   | 0.00e+00 | 0.00e+00 | PASS | PASS | PASS |
| Uint8    | 8    | 4.881s      | 66.51ms     | 98.54ms      | 100.72ms     | 73.4x   | 49.5x   | 48.5x   | 0.00e+00 | 0.00e+00 | PASS | PASS | PASS |
| Int4     | 8    | 4.868s      | 66.45ms     | 98.22ms      | 100.80ms     | 73.2x   | 49.6x   | 48.3x   | 0.00e+00 | 0.00e+00 | PASS | PASS | PASS |
| Uint4    | 8    | 4.867s      | 65.73ms     | 98.58ms      | 102.40ms     | 74.0x   | 49.4x   | 47.5x   | 0.00e+00 | 0.00e+00 | PASS | PASS | PASS |
| FP4      | 8    | 4.895s      | 66.62ms     | 98.71ms      | 102.64ms     | 73.5x   | 49.6x   | 47.7x   | 0.00e+00 | 0.00e+00 | PASS | PASS | PASS |
| Int2     | 8    | 4.862s      | 66.49ms     | 98.67ms      | 102.31ms     | 73.1x   | 49.3x   | 47.5x   | 0.00e+00 | 0.00e+00 | PASS | PASS | PASS |
| Uint2    | 8    | 4.930s      | 66.04ms     | 97.80ms      | 101.46ms     | 74.6x   | 50.4x   | 48.6x   | 0.00e+00 | 0.00e+00 | PASS | PASS | PASS |
| Ternary  | 8    | 4.959s      | 68.01ms     | 98.41ms      | 100.03ms     | 72.9x   | 50.4x   | 49.6x   | 0.00e+00 | 0.00e+00 | PASS | PASS | PASS |
| Binary   | 8    | 4.904s      | 66.50ms     | 97.92ms      | 100.09ms     | 73.7x   | 50.1x   | 49.0x   | 0.00e+00 | 0.00e+00 | PASS | PASS | PASS |

**All 21 types × 3 GPU paths × 2 gradients (DX + DW) = 126 parity checks passed.**

---

## Key Observations

### GPU Normal Wins the Backward Pass

Unlike the forward pass where tiled workgroups outperformed global memory, the backward pass shows the opposite:

- **GPU Normal: ~66–69ms** (fastest)
- **GPU Tiled SC: ~96–100ms** (~47% slower than Normal)
- **GPU Tiled MC: ~99–104ms** (~50% slower than Normal)

Why? The backward tiled shaders have *two* cooperative loads per tile (filter kernel cache for DX, dy-value cache for DW) plus two `workgroupBarrier()` calls each. At this problem size the synchronization overhead outweighs the reduction in global memory reads. The forward tiled shader only needed one cooperative kernel load per output element with a single barrier — less coordination cost.

### Uniform Speedup Across All 21 Types

CPU time: ~4.8–5.2s across all types — the CPU processes integer types through the same float32 accumulation path.
GPU time: ~66–69ms across all types — the GPU shader operates entirely in f32 regardless of source dtype.

Speedup range: **71–77x** for GPU Normal across all 21 types.

### 0.00e+00 Diff for Every Type, Both Gradients

Both DX and DW achieve exact zero difference because:
- CPU backward: `float32(raw_int_weight)` — casts raw integer bits to f32, no scale
- GPU backward: uploads the same raw values as f32, no scale
- Both execute identical float32 arithmetic → identical results

This is the same principle as the forward experiment, but critically the backward pass does **not** apply scale even though the forward pass does — scale is a forward-only transformation.

---

## Implementation Summary

### New Files

| File | Purpose |
|------|---------|
| `poly/wgpu_cnn3_backward_tiled.go` | `WGPUCNN3Backward3DParams` struct, `ShaderTiledCNN3BackwardDX`, `ShaderTiledCNN3BackwardDW`, `DispatchCNN3TiledBackwardDX`, `DispatchCNN3TiledBackwardDW` |
| `tva/poly/cnn3_backward_gpu_experiment/main.go` | This experiment |

### Modified Files

| File | What Changed |
|------|-------------|
| `poly/cnn3.go` | Added `CNN3BackwardTiledParallel` — multi-core CPU reference using goroutines for DX (parallel over `(batch, inC)`) and DW (parallel over `filters`) |

### Uniform Struct

`WGPUCNN3Backward3DParams` (80 bytes, 20 × uint32, multiple of 16 for WebGPU alignment):

```go
type WGPUCNN3Backward3DParams struct {
    BatchSize             uint32
    InC, InD, InH, InW   uint32
    Filters               uint32
    OutD, OutH, OutW      uint32
    KD, KH, KW            uint32
    SD, SH, SW            uint32
    PD, PH, PW            uint32
    Activation            uint32
    Pad                   uint32   // padding to 80 bytes
}
```

### Dispatch Signatures

```go
// DX: tileSize × batchSize dispatch — each thread handles one input element
func (c *WGPUContext) DispatchCNN3TiledBackwardDX(
    tileSize, kernelVol int,
    batchSize, inC, inD, inH, inW, filters, outD, outH, outW,
    kD, kH, kW, sD, sH, sW, pD, pH, pW int,
    activation ActivationType,
    gradOutputBuf, weightBuf, preActBuf, gradInputBuf *wgpu.Buffer,
) error

// DW: kernelVol/tileSize × filters dispatch — each thread handles one weight gradient
func (c *WGPUContext) DispatchCNN3TiledBackwardDW(
    tileSize int,
    batchSize, inC, inD, inH, inW, filters, outD, outH, outW,
    kD, kH, kW, sD, sH, sW, pD, pH, pW int,
    activation ActivationType,
    gradOutputBuf, inputBuf, preActBuf, gradWeightsBuf *wgpu.Buffer,
) error
```

---

## Bugs Fixed During Development

1. **Nil output buffers in timing loops** — first draft passed `nil` as `gradInputBuf`/`gradWeightsBuf` inside timing iterations; fixed by allocating persistent `timingGI`/`timingGW` buffers reused across iterations.

2. **GPU timing showed `0s` / `+Inf` speedup** — batching all 10 GPU iterations before a single `Poll` meant `time.Since(start) / 10` measured only async submission overhead (~nanoseconds), truncating to `0s`. Fixed by moving `Poll(true, nil)` inside the iteration loop.

3. **`WGPUCNN3Backward3DParams` alignment** — initial struct was 76 bytes (not a multiple of 16); added `Pad uint32` to reach 80 bytes, satisfying WebGPU's minimum-binding-size alignment requirement.

4. **`wgslBwdActivateDeriv` constant** — both DX and DW shaders need the activation derivative inline function; extracted as a shared WGSL constant to avoid duplicating the switch block in both shader generators.

5. **DX dispatch dimensions** — initial dispatch used `(inElements/tileSize, 1, batchSize)` which over-allocated workgroups; corrected to `ceil(inElements/tileSize)` in X, 1 in Y, 1 in Z with bounds check inside the shader.

6. **DW dispatch workgroup Y** — DW dispatch Y dimension is `filters`, not `filters/tileSize`; each Y workgroup handles one complete filter's weight gradient reduction.
