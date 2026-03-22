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
	tCPUMC   time.Duration
	tGPUNorm time.Duration
	tGPUSC   time.Duration
	tGPUMC   time.Duration
	scTile   int
	mcTile   int
	tileSize int

	diffDXNorm float64
	diffDWNorm float64
	diffDXSC   float64
	diffDWSC   float64
	diffDXMC   float64
	diffDWMC   float64

	parityNorm bool
	paritySC   bool
	parityMC   bool
}

// rawF32 converts WeightStore morphed storage to []float32 WITHOUT multiplying scale.
// The backward pass does not apply scale — matches CPU CNN3BackwardTiledParallel.
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
			out[i] = float32(v)
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

// zeroF32Buf creates a GPU read-write buffer pre-filled with zeros.
func zeroF32Buf(ctx *poly.WGPUContext, size int, label string) (*wgpu.Buffer, error) {
	zeros := make([]float32, size)
	return ctx.Device.CreateBufferInit(&wgpu.BufferInitDescriptor{
		Label:    label,
		Contents: wgpu.ToBytes(zeros),
		Usage:    wgpu.BufferUsageStorage | wgpu.BufferUsageCopySrc,
	})
}

// maxDiff returns the maximum absolute difference between two float32 slices.
func maxDiff(a, b []float32) float64 {
	var d float64
	for i := range a {
		if v := math.Abs(float64(a[i] - b[i])); v > d {
			d = v
		}
	}
	return d
}

func runDType(cfg typeConfig, ctx *poly.WGPUContext, scTile, mcTile, cpuIters int) result {
	// ---- Build CPU layer ----
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
		Activation:    poly.ActivationLinear,
		DType:         cfg.dtype,
		WeightStore:   ws,
		UseTiling:     true,
		TileSize:      8,
	}
	l.Network.EnableMultiCoreTiling = true
	l.SyncToCPU()
	tileSize := l.TileSize

	// gradOutput=1.0, input=0.5, preAct=0.5  (ActivationLinear → deriv=1.0)
	gradOut := poly.NewTensor[float32](1, 32, 32, 32, 32)
	for i := range gradOut.Data {
		gradOut.Data[i] = 1.0
	}
	input := poly.NewTensor[float32](1, 32, 32, 32, 32)
	for i := range input.Data {
		input.Data[i] = 0.5
	}
	preAct := poly.NewTensor[float32](1, 32, 32, 32, 32)
	for i := range preAct.Data {
		preAct.Data[i] = 0.5
	}

	// CPU Multi-Core Tiled reference
	var cpuDX, cpuDW *poly.Tensor[float32]
	start := time.Now()
	for i := 0; i < cpuIters; i++ {
		cpuDX, cpuDW = poly.CNN3BackwardTiledParallel(&l, gradOut, input, preAct)
	}
	tCPUMC := time.Since(start) / time.Duration(cpuIters)

	// ---- GPU setup ----
	const (
		inC, inD, inH, inW     = 32, 32, 32, 32
		outC, outD, outH, outW = 32, 32, 32, 32
		kSize, stride, padding = 3, 1, 1
		kD, kH, kW             = 3, 3, 3
		sD, sH, sW             = 1, 1, 1
		pD, pH, pW             = 1, 1, 1
		batchSize              = 1
		inputSize              = batchSize * inC * inD * inH * inW
		outputSize             = batchSize * outC * outD * outH * outW
		kernelVol              = inC * kD * kH * kW // 864
		weightSize             = outC * kernelVol   // 27648
	)

	raw := rawF32(ws, cfg.dtype)
	act := poly.ActivationLinear

	gradOutBuf, err := ctx.Device.CreateBufferInit(&wgpu.BufferInitDescriptor{
		Label:    "BwdGradOutput",
		Contents: wgpu.ToBytes(gradOut.Data),
		Usage:    wgpu.BufferUsageStorage | wgpu.BufferUsageCopySrc,
	})
	if err != nil {
		fmt.Printf("  gradOutput buf: %v\n", err)
		return result{}
	}
	defer gradOutBuf.Destroy()

	weightBuf, err := ctx.Device.CreateBufferInit(&wgpu.BufferInitDescriptor{
		Label:    "BwdWeights",
		Contents: wgpu.ToBytes(raw),
		Usage:    wgpu.BufferUsageStorage | wgpu.BufferUsageCopySrc,
	})
	if err != nil {
		fmt.Printf("  weight buf: %v\n", err)
		return result{}
	}
	defer weightBuf.Destroy()

	inputBuf, err := ctx.Device.CreateBufferInit(&wgpu.BufferInitDescriptor{
		Label:    "BwdInput",
		Contents: wgpu.ToBytes(input.Data),
		Usage:    wgpu.BufferUsageStorage | wgpu.BufferUsageCopySrc,
	})
	if err != nil {
		fmt.Printf("  input buf: %v\n", err)
		return result{}
	}
	defer inputBuf.Destroy()

	preActBuf, err := ctx.Device.CreateBufferInit(&wgpu.BufferInitDescriptor{
		Label:    "BwdPreAct",
		Contents: wgpu.ToBytes(preAct.Data),
		Usage:    wgpu.BufferUsageStorage | wgpu.BufferUsageCopySrc,
	})
	if err != nil {
		fmt.Printf("  preAct buf: %v\n", err)
		return result{}
	}
	defer preActBuf.Destroy()

	// Timing output buffers — reused across iterations (accumulate but timing is valid).
	timingGI, _ := zeroF32Buf(ctx, inputSize, "TimingGI")
	timingGW, _ := zeroF32Buf(ctx, weightSize, "TimingGW")
	defer timingGI.Destroy()
	defer timingGW.Destroy()

	// Warmup: compile all pipeline variants once
	{
		gi, _ := zeroF32Buf(ctx, inputSize, "WarmGI")
		gw, _ := zeroF32Buf(ctx, weightSize, "WarmGW")
		ctx.DispatchCNN3BackwardDX(batchSize, inC, inD, inH, inW, outC, outD, outH, outW,
			kSize, stride, padding, act, gradOutBuf, weightBuf, preActBuf, gi)
		ctx.DispatchCNN3BackwardDW(batchSize, inC, inD, inH, inW, outC, outD, outH, outW,
			kSize, stride, padding, act, gradOutBuf, inputBuf, preActBuf, gw)
		ctx.DispatchCNN3TiledBackwardDX(scTile, kernelVol,
			batchSize, inC, inD, inH, inW, outC, outD, outH, outW,
			kD, kH, kW, sD, sH, sW, pD, pH, pW, act, gradOutBuf, weightBuf, preActBuf, gi)
		ctx.DispatchCNN3TiledBackwardDW(scTile,
			batchSize, inC, inD, inH, inW, outC, outD, outH, outW,
			kD, kH, kW, sD, sH, sW, pD, pH, pW, act, gradOutBuf, inputBuf, preActBuf, gw)
		ctx.DispatchCNN3TiledBackwardDX(mcTile, kernelVol,
			batchSize, inC, inD, inH, inW, outC, outD, outH, outW,
			kD, kH, kW, sD, sH, sW, pD, pH, pW, act, gradOutBuf, weightBuf, preActBuf, gi)
		ctx.DispatchCNN3TiledBackwardDW(mcTile,
			batchSize, inC, inD, inH, inW, outC, outD, outH, outW,
			kD, kH, kW, sD, sH, sW, pD, pH, pW, act, gradOutBuf, inputBuf, preActBuf, gw)
		ctx.Device.Poll(true, nil)
		gi.Destroy()
		gw.Destroy()
	}

	gpuIters := 10

	// GPU Normal timing — poll each iteration so GPU work is measured, not just submission
	var normTotal time.Duration
	for i := 0; i < gpuIters; i++ {
		tStart := time.Now()
		ctx.DispatchCNN3BackwardDX(batchSize, inC, inD, inH, inW, outC, outD, outH, outW,
			kSize, stride, padding, act, gradOutBuf, weightBuf, preActBuf, timingGI)
		ctx.DispatchCNN3BackwardDW(batchSize, inC, inD, inH, inW, outC, outD, outH, outW,
			kSize, stride, padding, act, gradOutBuf, inputBuf, preActBuf, timingGW)
		ctx.Device.Poll(true, nil)
		normTotal += time.Since(tStart)
	}
	tGPUNorm := normTotal / time.Duration(gpuIters)

	// GPU Tiled SC timing — poll each iteration
	var scTotal time.Duration
	for i := 0; i < gpuIters; i++ {
		tStart := time.Now()
		ctx.DispatchCNN3TiledBackwardDX(scTile, kernelVol,
			batchSize, inC, inD, inH, inW, outC, outD, outH, outW,
			kD, kH, kW, sD, sH, sW, pD, pH, pW, act, gradOutBuf, weightBuf, preActBuf, timingGI)
		ctx.DispatchCNN3TiledBackwardDW(scTile,
			batchSize, inC, inD, inH, inW, outC, outD, outH, outW,
			kD, kH, kW, sD, sH, sW, pD, pH, pW, act, gradOutBuf, inputBuf, preActBuf, timingGW)
		ctx.Device.Poll(true, nil)
		scTotal += time.Since(tStart)
	}
	tGPUSC := scTotal / time.Duration(gpuIters)

	// GPU Tiled MC timing — poll each iteration
	var mcTotal time.Duration
	for i := 0; i < gpuIters; i++ {
		tStart := time.Now()
		ctx.DispatchCNN3TiledBackwardDX(mcTile, kernelVol,
			batchSize, inC, inD, inH, inW, outC, outD, outH, outW,
			kD, kH, kW, sD, sH, sW, pD, pH, pW, act, gradOutBuf, weightBuf, preActBuf, timingGI)
		ctx.DispatchCNN3TiledBackwardDW(mcTile,
			batchSize, inC, inD, inH, inW, outC, outD, outH, outW,
			kD, kH, kW, sD, sH, sW, pD, pH, pW, act, gradOutBuf, inputBuf, preActBuf, timingGW)
		ctx.Device.Poll(true, nil)
		mcTotal += time.Since(tStart)
	}
	tGPUMC := mcTotal / time.Duration(gpuIters)

	// Parity reads on fresh zero-initialized buffers
	readParity := func(dispatchDX func(*wgpu.Buffer, *wgpu.Buffer), dispatchDW func(*wgpu.Buffer, *wgpu.Buffer)) ([]float32, []float32, error) {
		gi, e := zeroF32Buf(ctx, inputSize, "ParityGI")
		if e != nil {
			return nil, nil, e
		}
		defer gi.Destroy()
		gw, e := zeroF32Buf(ctx, weightSize, "ParityGW")
		if e != nil {
			return nil, nil, e
		}
		defer gw.Destroy()
		dispatchDX(gi, gw)
		dispatchDW(gi, gw)
		ctx.Device.Poll(true, nil)
		giData, e := ctx.ReadBuffer(gi)
		if e != nil {
			return nil, nil, e
		}
		gwData, e := ctx.ReadBuffer(gw)
		return giData, gwData, e
	}

	giNorm, gwNorm, err := readParity(
		func(gi, _ *wgpu.Buffer) {
			ctx.DispatchCNN3BackwardDX(batchSize, inC, inD, inH, inW, outC, outD, outH, outW,
				kSize, stride, padding, act, gradOutBuf, weightBuf, preActBuf, gi)
		},
		func(_, gw *wgpu.Buffer) {
			ctx.DispatchCNN3BackwardDW(batchSize, inC, inD, inH, inW, outC, outD, outH, outW,
				kSize, stride, padding, act, gradOutBuf, inputBuf, preActBuf, gw)
		},
	)
	if err != nil {
		fmt.Printf("  GPU Normal read: %v\n", err)
		return result{}
	}

	giSC, gwSC, err := readParity(
		func(gi, _ *wgpu.Buffer) {
			ctx.DispatchCNN3TiledBackwardDX(scTile, kernelVol,
				batchSize, inC, inD, inH, inW, outC, outD, outH, outW,
				kD, kH, kW, sD, sH, sW, pD, pH, pW, act, gradOutBuf, weightBuf, preActBuf, gi)
		},
		func(_, gw *wgpu.Buffer) {
			ctx.DispatchCNN3TiledBackwardDW(scTile,
				batchSize, inC, inD, inH, inW, outC, outD, outH, outW,
				kD, kH, kW, sD, sH, sW, pD, pH, pW, act, gradOutBuf, inputBuf, preActBuf, gw)
		},
	)
	if err != nil {
		fmt.Printf("  GPU SC read: %v\n", err)
		return result{}
	}

	giMC, gwMC, err := readParity(
		func(gi, _ *wgpu.Buffer) {
			ctx.DispatchCNN3TiledBackwardDX(mcTile, kernelVol,
				batchSize, inC, inD, inH, inW, outC, outD, outH, outW,
				kD, kH, kW, sD, sH, sW, pD, pH, pW, act, gradOutBuf, weightBuf, preActBuf, gi)
		},
		func(_, gw *wgpu.Buffer) {
			ctx.DispatchCNN3TiledBackwardDW(mcTile,
				batchSize, inC, inD, inH, inW, outC, outD, outH, outW,
				kD, kH, kW, sD, sH, sW, pD, pH, pW, act, gradOutBuf, inputBuf, preActBuf, gw)
		},
	)
	if err != nil {
		fmt.Printf("  GPU MC read: %v\n", err)
		return result{}
	}

	dxNorm := maxDiff(cpuDX.Data, giNorm)
	dwNorm := maxDiff(cpuDW.Data, gwNorm)
	dxSC := maxDiff(cpuDX.Data, giSC)
	dwSC := maxDiff(cpuDW.Data, gwSC)
	dxMC := maxDiff(cpuDX.Data, giMC)
	dwMC := maxDiff(cpuDW.Data, gwMC)

	return result{
		tCPUMC: tCPUMC, tGPUNorm: tGPUNorm, tGPUSC: tGPUSC, tGPUMC: tGPUMC,
		scTile: scTile, mcTile: mcTile, tileSize: tileSize,
		diffDXNorm: dxNorm, diffDWNorm: dwNorm,
		diffDXSC: dxSC, diffDWSC: dwSC,
		diffDXMC: dxMC, diffDWMC: dwMC,
		parityNorm: math.Max(dxNorm, dwNorm) <= cfg.tolerance,
		paritySC:   math.Max(dxSC, dwSC) <= cfg.tolerance,
		parityMC:   math.Max(dxMC, dwMC) <= cfg.tolerance,
	}
}

func parityMark(ok bool) string {
	if ok {
		return "PASS"
	}
	return "FAIL"
}

func main() {
	fmt.Println("=== CNN3 Backward GPU WebGPU — All Numerical Types ===")
	fmt.Println()
	fmt.Println("Paths under test:")
	fmt.Println("  1. CPU MC-Tiled   — CNN3BackwardTiledParallel, DX + DW (canonical reference)")
	fmt.Println("  2. GPU Normal     — DispatchCNN3BackwardDX + DW, global memory")
	fmt.Println("  3. GPU Tiled SC   — DispatchCNN3TiledBackwardDX + DW (shared-mem cache, scTile)")
	fmt.Println("  4. GPU Tiled MC   — DispatchCNN3TiledBackwardDX + DW (shared-mem cache, mcTile)")
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
	fmt.Printf("  SC tile = %d  (workgroup_size(%d,1,1))\n", scTile, scTile)
	fmt.Printf("  MC tile = %d  (workgroup_size(%d,1,1))\n", mcTile, mcTile)
	fmt.Printf("  MaxComputeInvocationsPerWorkgroup = %d\n", ctx.Limits.MaxComputeInvocationsPerWorkgroup)
	fmt.Println()
	fmt.Println("GPU uses raw integer values as f32 (no scale) — matching CPU backward arithmetic.")
	fmt.Println()

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

	fmt.Printf("| %-10s | %-4s | %-12s | %-12s | %-12s | %-12s | %-7s | %-7s | %-7s | %-9s | %-9s | %-9s | %-9s | %-6s | %-6s | %-6s |\n",
		"DType", "Tile",
		"CPU MC", "GPU Normal", "GPU Tiled SC", "GPU Tiled MC",
		"GN-Spd", "SC-Spd", "MC-Spd",
		"Diff-DX/N", "Diff-DW/N", "Diff-DX/SC", "Diff-DW/SC",
		"GN", "SC", "MC")
	fmt.Println("|------------|------|--------------|--------------|--------------|--------------|---------|---------|---------|-----------|-----------|-----------|-----------|--------|--------|--------|")

	allPass := true
	for _, cfg := range types {
		fmt.Printf("  running %-10s ...\r", cfg.name)
		r := runDType(cfg, ctx, scTile, mcTile, 3)
		if !r.parityNorm || !r.paritySC || !r.parityMC {
			allPass = false
		}
		fmt.Printf("| %-10s | %-4d | %-12v | %-12v | %-12v | %-12v | %-7.1fx | %-7.1fx | %-7.1fx | %-9.2e | %-9.2e | %-9.2e | %-9.2e | %-6s | %-6s | %-6s |\n",
			cfg.name, r.tileSize,
			r.tCPUMC, r.tGPUNorm, r.tGPUSC, r.tGPUMC,
			float64(r.tCPUMC)/float64(r.tGPUNorm),
			float64(r.tCPUMC)/float64(r.tGPUSC),
			float64(r.tCPUMC)/float64(r.tGPUMC),
			r.diffDXNorm, r.diffDWNorm,
			r.diffDXSC, r.diffDWSC,
			parityMark(r.parityNorm),
			parityMark(r.paritySC),
			parityMark(r.parityMC),
		)
	}

	fmt.Println()
	if allPass {
		fmt.Println("✅ All backward GPU parity checks passed!")
	} else {
		fmt.Println("❌ One or more backward GPU parity checks FAILED.")
	}

	fmt.Println()
	fmt.Printf("GPU tile sizes: SC=%d  MC=%d\n", scTile, mcTile)
	fmt.Println("  GPU Normal:   DispatchCNN3BackwardDX + DW — global memory, no shared-mem cache")
	fmt.Println("  GPU Tiled SC: DX caches filter kernel in shared mem;  DW caches dy values in shared mem")
	fmt.Println("  GPU Tiled MC: same tiled shaders, larger workgroup → more SM saturation")
	fmt.Println("  All paths: raw int weights as f32, no scale (matching CPU backward).")
}
