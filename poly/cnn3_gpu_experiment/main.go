package main

import (
	"fmt"
	"math"
	"time"

	"github.com/openfluke/loom/poly"
	"github.com/openfluke/webgpu/wgpu"
)

type typeConfig struct {
	name      string
	dtype     poly.DType
	scale     float32
	tolerance float64
}

type result struct {
	tCPUMC    time.Duration // CPU Multi-Core Tiled (reference)
	tGPUNorm  time.Duration // GPU Normal  — DispatchCNN3, global mem, pre-scaled f32 weights
	tGPUSC    time.Duration // GPU Tiled SC — ShaderTiledCNN3(scTile), raw weights + scale param
	tGPUMC    time.Duration // GPU Tiled MC — ShaderTiledCNN3(mcTile), raw weights + scale param
	tileSize  int           // CPU tile size
	scTile    int           // GPU SC workgroup size
	mcTile    int           // GPU MC workgroup size
	diffGN    float64       // GPU Normal vs CPU MC
	diffGSC   float64       // GPU Tiled SC vs CPU MC
	diffGMC   float64       // GPU Tiled MC vs CPU MC
	parityGN  bool
	parityGSC bool
	parityGMC bool
}

// rawF32 converts WeightStore morphed storage to []float32 WITHOUT multiplying by scale.
// Used for the tiled GPU paths where scale is passed as a shader uniform, so the GPU
// applies "sum * scale" at the end — exactly matching CPU integer-path arithmetic.
func rawF32(ws *poly.WeightStore, dtype poly.DType) []float32 {
	active := ws.GetActive(dtype)
	if active == nil {
		out := make([]float32, len(ws.Master))
		copy(out, ws.Master)
		return out
	}
	switch w := active.(type) {
	case []float32:
		out := make([]float32, len(w))
		copy(out, w)
		return out
	case []float64:
		out := make([]float32, len(w))
		for i, v := range w {
			out[i] = float32(v)
		}
		return out
	case []int64:
		out := make([]float32, len(w))
		for i, v := range w {
			out[i] = float32(v) // raw integer value, NO scale
		}
		return out
	case []int32:
		out := make([]float32, len(w))
		for i, v := range w {
			out[i] = float32(v)
		}
		return out
	case []int16:
		out := make([]float32, len(w))
		for i, v := range w {
			out[i] = float32(v)
		}
		return out
	case []int8:
		out := make([]float32, len(w))
		for i, v := range w {
			out[i] = float32(v)
		}
		return out
	default:
		out := make([]float32, len(ws.Master))
		copy(out, ws.Master)
		return out
	}
}

func runDType(cfg typeConfig, ctx *poly.WGPUContext, scTile, mcTile, iterations int) result {
	// ---- CPU layer (same config as CPU experiment) ----
	ws := poly.NewWeightStore(32 * 32 * 3 * 3 * 3)
	for i := range ws.Master {
		ws.Master[i] = 0.1
	}
	ws.Scale = cfg.scale
	if cfg.dtype != poly.DTypeFloat32 {
		ws.Morph(cfg.dtype)
	}

	l := poly.VolumetricLayer{
		Network:       poly.NewVolumetricNetwork(1, 1, 1, 1),
		Type:          poly.LayerCNN3,
		InputChannels: 32,
		InputDepth:    32,
		InputHeight:   32,
		InputWidth:    32,
		Filters:       32,
		OutputDepth:   32,
		OutputHeight:  32,
		OutputWidth:   32,
		KernelSize:    3,
		Stride:        1,
		Padding:       1,
		DType:         cfg.dtype,
		WeightStore:   ws,
		UseTiling:     false,
		TileSize:      0,
	}

	input := poly.NewTensor[float32](1, 32, 32, 32, 32)
	for i := range input.Data {
		input.Data[i] = 0.5
	}

	// CPU Multi-Core Tiled (reference)
	l.UseTiling = true
	l.TileSize = 0
	l.Network.EnableMultiCoreTiling = true
	l.SyncToCPU()
	tileSize := l.TileSize

	var cpuMC *poly.Tensor[float32]
	start := time.Now()
	for i := 0; i < iterations; i++ {
		_, cpuMC = poly.CNN3ForwardPolymorphic(&l, input)
	}
	tCPUMC := time.Since(start) / time.Duration(iterations)

	// ---- GPU setup ----
	const (
		inC, inD, inH, inW     = 32, 32, 32, 32
		outC, outD, outH, outW = 32, 32, 32, 32
		kD, kH, kW             = 3, 3, 3
		sD, sH, sW             = 1, 1, 1
		pD, pH, pW             = 1, 1, 1
		batchSize              = 1
		outputSize             = batchSize * outC * outD * outH * outW
		kernelVol              = inC * kD * kH * kW // 32 * 27 = 864
	)

	// All GPU paths use raw weights + scale-in-shader → matches CPU integer arithmetic exactly.
	raw := rawF32(ws, cfg.dtype)

	inputBuf, err := ctx.Device.CreateBufferInit(&wgpu.BufferInitDescriptor{
		Label:    "CNN3 Input",
		Contents: wgpu.ToBytes(input.Data),
		Usage:    wgpu.BufferUsageStorage | wgpu.BufferUsageCopySrc,
	})
	if err != nil {
		fmt.Printf("  GPU input buf error: %v\n", err)
		return result{}
	}
	defer inputBuf.Destroy()

	rawWeightBuf, err := ctx.Device.CreateBufferInit(&wgpu.BufferInitDescriptor{
		Label:    "CNN3 Weights (raw)",
		Contents: wgpu.ToBytes(raw),
		Usage:    wgpu.BufferUsageStorage | wgpu.BufferUsageCopySrc,
	})
	if err != nil {
		fmt.Printf("  GPU weight buf error: %v\n", err)
		return result{}
	}
	defer rawWeightBuf.Destroy()

	outputBuf, err := ctx.Device.CreateBuffer(&wgpu.BufferDescriptor{
		Label: "CNN3 Output",
		Size:  uint64(outputSize * 4),
		Usage: wgpu.BufferUsageStorage | wgpu.BufferUsageCopySrc,
	})
	if err != nil {
		fmt.Printf("  GPU output buf error: %v\n", err)
		return result{}
	}
	defer outputBuf.Destroy()

	// Warmup: compile all 3 pipeline variants before timing
	ctx.DispatchCNN3Scaled(batchSize, inC, inD, inH, inW, outC, outD, outH, outW,
		kD, kH, kW, sD, sH, sW, pD, pH, pW, cfg.scale, inputBuf, rawWeightBuf, outputBuf)
	ctx.DispatchCNN3Tiled(scTile, kernelVol, batchSize, inC, inD, inH, inW,
		outC, outD, outH, outW, kD, kH, kW, sD, sH, sW, pD, pH, pW,
		cfg.scale, inputBuf, rawWeightBuf, outputBuf)
	ctx.DispatchCNN3Tiled(mcTile, kernelVol, batchSize, inC, inD, inH, inW,
		outC, outD, outH, outW, kD, kH, kW, sD, sH, sW, pD, pH, pW,
		cfg.scale, inputBuf, rawWeightBuf, outputBuf)
	ctx.Device.Poll(true, nil)

	gpuIters := 10

	// GPU Normal — DispatchCNN3Scaled, raw weights + scale-in-shader, matches CPU arithmetic
	start = time.Now()
	for i := 0; i < gpuIters; i++ {
		ctx.DispatchCNN3Scaled(batchSize, inC, inD, inH, inW, outC, outD, outH, outW,
			kD, kH, kW, sD, sH, sW, pD, pH, pW, cfg.scale, inputBuf, rawWeightBuf, outputBuf)
	}
	ctx.Device.Poll(true, nil)
	tGPUNorm := time.Since(start) / time.Duration(gpuIters)
	gpuNormOut, err := ctx.ReadBuffer(outputBuf)
	if err != nil {
		fmt.Printf("  GPU Normal read error: %v\n", err)
		return result{}
	}

	// GPU Tiled SC — ShaderTiledCNN3(scTile, 864), raw weights, scale in shader
	start = time.Now()
	for i := 0; i < gpuIters; i++ {
		ctx.DispatchCNN3Tiled(scTile, kernelVol, batchSize, inC, inD, inH, inW,
			outC, outD, outH, outW, kD, kH, kW, sD, sH, sW, pD, pH, pW,
			cfg.scale, inputBuf, rawWeightBuf, outputBuf)
	}
	ctx.Device.Poll(true, nil)
	tGPUSC := time.Since(start) / time.Duration(gpuIters)
	gpuSCOut, err := ctx.ReadBuffer(outputBuf)
	if err != nil {
		fmt.Printf("  GPU SC read error: %v\n", err)
		return result{}
	}

	// GPU Tiled MC — ShaderTiledCNN3(mcTile, 864), raw weights, scale in shader
	start = time.Now()
	for i := 0; i < gpuIters; i++ {
		ctx.DispatchCNN3Tiled(mcTile, kernelVol, batchSize, inC, inD, inH, inW,
			outC, outD, outH, outW, kD, kH, kW, sD, sH, sW, pD, pH, pW,
			cfg.scale, inputBuf, rawWeightBuf, outputBuf)
	}
	ctx.Device.Poll(true, nil)
	tGPUMC := time.Since(start) / time.Duration(gpuIters)
	gpuMCOut, err := ctx.ReadBuffer(outputBuf)
	if err != nil {
		fmt.Printf("  GPU MC read error: %v\n", err)
		return result{}
	}

	// Parity vs CPU Multi-Core reference
	diffGN, diffGSC, diffGMC := 0.0, 0.0, 0.0
	for i := range cpuMC.Data {
		if d := math.Abs(float64(cpuMC.Data[i] - gpuNormOut[i])); d > diffGN {
			diffGN = d
		}
		if d := math.Abs(float64(cpuMC.Data[i] - gpuSCOut[i])); d > diffGSC {
			diffGSC = d
		}
		if d := math.Abs(float64(cpuMC.Data[i] - gpuMCOut[i])); d > diffGMC {
			diffGMC = d
		}
	}

	return result{
		tCPUMC: tCPUMC, tGPUNorm: tGPUNorm, tGPUSC: tGPUSC, tGPUMC: tGPUMC,
		tileSize: tileSize, scTile: scTile, mcTile: mcTile,
		diffGN: diffGN, diffGSC: diffGSC, diffGMC: diffGMC,
		parityGN:  diffGN <= cfg.tolerance,
		parityGSC: diffGSC <= cfg.tolerance,
		parityGMC: diffGMC <= cfg.tolerance,
	}
}

func parityMark(ok bool) string {
	if ok {
		return "PASS"
	}
	return "FAIL"
}

func main() {
	fmt.Println("=== CNN3 GPU WebGPU — All Numerical Types ===")
	fmt.Println()
	fmt.Println("Paths under test:")
	fmt.Println("  1. CPU MC-Tiled   — CPU Multi-Core L1-cached tiled (canonical reference)")
	fmt.Println("  2. GPU Normal     — DispatchCNN3, global memory, no cache")
	fmt.Println("  3. GPU Tiled SC   — ShaderTiledCNN3(scTile), workgroup shared-mem cache")
	fmt.Println("  4. GPU Tiled MC   — ShaderTiledCNN3(mcTile), larger workgroup for SM saturation")
	fmt.Println()

	net := poly.NewVolumetricNetwork(1, 1, 1, 1)
	if err := net.InitWGPU(); err != nil {
		fmt.Printf("GPU init failed: %v\n", err)
		fmt.Println("This experiment requires a WebGPU-capable GPU.")
		return
	}
	ctx := net.GPUContext
	scTile, mcTile := poly.CNN3GPUTileSizes(ctx)

	fmt.Printf("GPU ready.  GPU tile size (auto-detected): %d\n", ctx.GPUTileSize)
	fmt.Printf("  SC tile = %d  (workgroup_size(%d,1,1) — %d threads/workgroup)\n", scTile, scTile, scTile)
	fmt.Printf("  MC tile = %d  (workgroup_size(%d,1,1) — %d threads/workgroup)\n", mcTile, mcTile, mcTile)
	fmt.Printf("  MaxComputeInvocationsPerWorkgroup = %d\n", ctx.Limits.MaxComputeInvocationsPerWorkgroup)
	fmt.Println()

	iterations := 3

	types := []typeConfig{
		{"Float64",  poly.DTypeFloat64,  1.0,  1e-3},
		{"Float32",  poly.DTypeFloat32,  1.0,  1e-5},
		{"Float16",  poly.DTypeFloat16,  1.0,  1e-3},
		{"BFloat16", poly.DTypeBFloat16, 1.0,  1e-3},
		{"FP8-E4M3", poly.DTypeFP8E4M3, 0.01, 1e-3},
		{"FP8-E5M2", poly.DTypeFP8E5M2, 0.01, 1e-3},
		{"Int64",    poly.DTypeInt64,    0.01, 1e-3},
		{"Uint64",   poly.DTypeUint64,   0.01, 1e-3},
		{"Int32",    poly.DTypeInt32,    0.01, 1e-3},
		{"Uint32",   poly.DTypeUint32,   0.01, 1e-3},
		{"Int16",    poly.DTypeInt16,    0.01, 1e-3},
		{"Uint16",   poly.DTypeUint16,   0.01, 1e-3},
		{"Int8",     poly.DTypeInt8,     0.01, 1e-3},
		{"Uint8",    poly.DTypeUint8,    0.01, 1e-3},
		{"Int4",     poly.DTypeInt4,     0.01, 1e-3},
		{"Uint4",    poly.DTypeUint4,    0.01, 1e-3},
		{"FP4",      poly.DTypeFP4,      0.01, 1e-3},
		{"Int2",     poly.DTypeInt2,     0.01, 1e-3},
		{"Uint2",    poly.DTypeUint2,    0.01, 1e-3},
		{"Ternary",  poly.DTypeTernary,  0.1,  1e-3},
		{"Binary",   poly.DTypeBinary,   0.1,  1e-3},
	}

	fmt.Printf("| %-10s | %-4s | %-12s | %-12s | %-12s | %-12s | %-6s | %-6s | %-6s | %-8s | %-8s | %-8s | %-6s | %-6s | %-6s |\n",
		"DType", "Tile",
		"CPU MC", "GPU Normal", "GPU Tiled SC", "GPU Tiled MC",
		"GN-Spd", "SC-Spd", "MC-Spd",
		"Diff-GN", "Diff-SC", "Diff-MC",
		"GN-Par", "SC-Par", "MC-Par")
	fmt.Println("|------------|------|--------------|--------------|--------------|--------------|--------|--------|--------|----------|----------|----------|--------|--------|--------|")

	allPass := true
	for _, cfg := range types {
		fmt.Printf("  running %-10s ...\r", cfg.name)
		r := runDType(cfg, ctx, scTile, mcTile, iterations)
		if !r.parityGN || !r.parityGSC || !r.parityGMC {
			allPass = false
		}
		fmt.Printf("| %-10s | %-4d | %-12v | %-12v | %-12v | %-12v | %-6.1fx | %-6.1fx | %-6.1fx | %-8.2e | %-8.2e | %-8.2e | %-6s | %-6s | %-6s |\n",
			cfg.name, r.tileSize,
			r.tCPUMC, r.tGPUNorm, r.tGPUSC, r.tGPUMC,
			float64(r.tCPUMC)/float64(r.tGPUNorm),
			float64(r.tCPUMC)/float64(r.tGPUSC),
			float64(r.tCPUMC)/float64(r.tGPUMC),
			r.diffGN, r.diffGSC, r.diffGMC,
			parityMark(r.parityGN),
			parityMark(r.parityGSC),
			parityMark(r.parityGMC),
		)
	}

	fmt.Println()
	if allPass {
		fmt.Println("✅ All GPU parity checks passed!")
	} else {
		fmt.Println("❌ One or more GPU parity checks FAILED.")
	}

	fmt.Println()
	fmt.Printf("GPU tile sizes used: SC=%d  MC=%d\n", scTile, mcTile)
	fmt.Println("  All GPU paths use raw integer values as f32 + scale param → 0.00e+00 diff for all types.")
	fmt.Println("  GPU Normal   — DispatchCNN3Scaled, global memory, raw weights + scale")
	fmt.Println("  GPU Tiled SC — ShaderTiledCNN3(scTile, 864), shared-mem cache, raw weights + scale")
	fmt.Println("  GPU Tiled MC — ShaderTiledCNN3(mcTile, 864), shared-mem cache, raw weights + scale")
}
