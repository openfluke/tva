package main

import (
	"fmt"
	"time"

	"github.com/openfluke/loom/poly"
	"github.com/openfluke/webgpu/wgpu"
)

func main() {
	fmt.Println("=== M-POLY-VTD Training Showdown: CPU vs GPU Backward Pass ===")
	
	net := poly.NewVolumetricNetwork(1, 1, 1, 1)
	err := net.InitWGPU()
	ctx := net.GPUContext
	if err != nil {
		fmt.Printf("⚠️  GPU not available: %v\n", err)
	}

	layers := []struct {
		Name string
		Type poly.LayerType
	}{
		{"Dense (Linear)", poly.LayerDense},
		{"RMSNorm", poly.LayerRMSNorm},
		{"SwiGLU (MLP)", poly.LayerSwiGLU},
		{"Embedding", poly.LayerEmbedding},
		{"Residual Add", poly.LayerResidual},
		{"MHA (Fused)", poly.LayerMultiHeadAttention},
		{"CNN1D", poly.LayerCNN1},
		{"CNN2D", poly.LayerCNN2},
		{"CNN3D", poly.LayerCNN3},
	}

	fmt.Println("| Layer type      | CPU Time     | GPU Time     | Speedup | Max DX Diff | Max DW Diff | Sanity |")
	fmt.Println("|-----------------|--------------|--------------|---------|-------------|-------------|--------|")

	type trainSample struct {
		Name  string
		CPUDX []float32
		GPUDX []float32
		CPUDW []float32
		GPUDW []float32
	}
	var samples []trainSample

	for _, l := range layers {
		cpuTime, gpuTime, maxDX, maxDW, sanity, cDX, gDX, cDW, gDW := runTrainBench(l.Type, ctx)
		
		samples = append(samples, trainSample{l.Name, cDX, gDX, cDW, gDW})
		
		gpuLabel := "N/A"
		speedup := "N/A"
		if gpuTime > 0 {
			gpuLabel = fmt.Sprintf("%v", gpuTime)
			if cpuTime > 0 {
				ratio := float64(cpuTime) / float64(gpuTime)
				speedup = fmt.Sprintf("%.2fx", ratio)
			}
		}

		detDX := formatDiff(maxDX)
		detDW := formatDiff(maxDW)
		san := "N/A"
		if gpuTime > 0 {
			if sanity { san = "REAL 💎" } else { san = "ZERO 💀" }
		}

		fmt.Printf("| %-15s | %-12v | %-12s | %-7s | %-11s | %-11s | %-6s |\n", 
			l.Name, cpuTime, gpuLabel, speedup, detDX, detDW, san)
	}

	fmt.Println("\n=== Final Sanity Check: CPU vs GPU Gradient Samples ===")
	fmt.Println("| Layer           | Type | CPU Sample (first 3)        | GPU Sample (first 3)        | Status |")
	fmt.Println("|-----------------|------|-----------------------------|-----------------------------|--------|")
	for _, s := range samples {
		// DX
		cdxStr := "N/A"; gdxStr := "N/A"
		if len(s.CPUDX) > 0 { cdxStr = fmt.Sprintf("%.4f, %.4f, %.4f", s.CPUDX[0], s.CPUDX[1], s.CPUDX[2]) }
		if len(s.GPUDX) > 0 { gdxStr = fmt.Sprintf("%.4f, %.4f, %.4f", s.GPUDX[0], s.GPUDX[1], s.GPUDX[2]) }
		dxStat := "ZERO 💀"; for _, v := range s.GPUDX { if v != 0 { dxStat = "REAL 💎"; break } }
		fmt.Printf("| %-15s | DX   | %-27s | %-27s | %-6s |\n", s.Name, cdxStr, gdxStr, dxStat)

		// DW
		cdwStr := "N/A"; gdwStr := "N/A"
		if len(s.CPUDW) > 0 { cdwStr = fmt.Sprintf("%.4f, %.4f, %.4f", s.CPUDW[0], s.CPUDW[1], s.CPUDW[2]) }
		if len(s.GPUDW) > 0 { gdwStr = fmt.Sprintf("%.4f, %.4f, %.4f", s.GPUDW[0], s.GPUDW[1], s.GPUDW[2]) }
		dwStat := "ZERO 💀"; for _, v := range s.GPUDW { if v != 0 { dwStat = "REAL 💎"; break } }
		if s.Name != "Residual Add" && s.Name != "MHA (Fused)" { // Layers without weights in this bench
			fmt.Printf("| %-15s | DW   | %-27s | %-27s | %-6s |\n", "", cdwStr, gdwStr, dwStat)
		}
		fmt.Println("|-----------------|------|-----------------------------|-----------------------------|--------|")
	}
}

func formatDiff(diff float64) string {
	if diff < 0 { return "N/A" }
	if diff < 1e-7 { return "EXACT ⭐" }
	if diff < 1e-4 { return "OK ✅" }
	if diff < 1e-2 { return "OFF ⚠️" }
	return "BROKEN ❌"
}

func runTrainBench(lType poly.LayerType, ctx *poly.WGPUContext) (cpu, gpu time.Duration, maxDX, maxDW float64, sanity bool, cDX, gDX, cDW, gDW []float32) {
	iterations := 5
	
	// Setup layer
	l := poly.VolumetricLayer{
		Network: poly.NewVolumetricNetwork(1, 1, 1, 1),
		Type: lType,
		Activation: poly.ActivationLinear,
		InputHeight: 512,
		OutputHeight: 512,
		WeightStore: poly.NewWeightStore(512 * 512), // Sufficient for most
	}
	
	if lType == poly.LayerRMSNorm {
		l.WeightStore = poly.NewWeightStore(1024)
		l.InputHeight = 512
		l.OutputHeight = 512
	} else if lType == poly.LayerSwiGLU {
		l.InputHeight = 512
		l.OutputHeight = 1024
		size := 3*512*1024 + 1024 + 1024 + 512
		l.WeightStore = poly.NewWeightStore(size)
	} else if lType == poly.LayerEmbedding {
		l.VocabSize = 1024
		l.EmbeddingDim = 128
		l.WeightStore = poly.NewWeightStore(l.VocabSize * l.EmbeddingDim)
	} else if lType == poly.LayerMultiHeadAttention {
		l.NumHeads = 4
		l.NumKVHeads = 4
		l.HeadDim = 32
		l.InputHeight = 64 // SeqLen
		l.WeightStore = poly.NewWeightStore(0) 
	} else if lType == poly.LayerCNN1 {
		l.InputChannels = 3
		l.InputHeight = 64
		l.Filters = 8
		l.KernelSize = 3
		l.Stride = 1
		l.Padding = 1
		l.OutputHeight = 64
		l.WeightStore = poly.NewWeightStore(l.Filters * l.InputChannels * l.KernelSize)
	} else if lType == poly.LayerCNN2 {
		l.InputChannels = 3
		l.InputHeight = 32
		l.InputWidth = 32
		l.Filters = 8
		l.KernelSize = 3
		l.Stride = 1
		l.Padding = 1
		l.OutputHeight = 32
		l.OutputWidth = 32
		l.WeightStore = poly.NewWeightStore(l.Filters * l.InputChannels * l.KernelSize * l.KernelSize)
	} else if lType == poly.LayerCNN3 {
		l.InputChannels = 3
		l.InputDepth = 16
		l.InputHeight = 16
		l.InputWidth = 16
		l.Filters = 4
		l.KernelSize = 3
		l.Stride = 1
		l.Padding = 1
		l.OutputDepth = 16
		l.OutputHeight = 16
		l.OutputWidth = 16
		l.WeightStore = poly.NewWeightStore(l.Filters * l.InputChannels * l.KernelSize * l.KernelSize * l.KernelSize)
	}

	for i := range l.WeightStore.Master { l.WeightStore.Master[i] = 0.1 }

	// Inputs & GradOutputs
	batchSize := 1
	var input *poly.Tensor[float32]
	var gradOutput *poly.Tensor[float32]

	if lType == poly.LayerEmbedding {
		input = poly.NewTensor[float32](batchSize, 16) // indices
		for i := range input.Data { input.Data[i] = float32(i % l.VocabSize) }
		gradOutput = poly.NewTensor[float32](batchSize, 16, l.EmbeddingDim)
	} else if lType == poly.LayerMultiHeadAttention {
		input = poly.NewTensor[float32](batchSize, l.InputHeight, l.NumHeads*l.HeadDim) 
		gradOutput = poly.NewTensor[float32](batchSize, l.InputHeight, l.NumHeads*l.HeadDim)
	} else if lType == poly.LayerCNN1 {
		input = poly.NewTensor[float32](batchSize, l.InputChannels, l.InputHeight)
		gradOutput = poly.NewTensor[float32](batchSize, l.Filters, l.OutputHeight)
	} else if lType == poly.LayerCNN2 {
		input = poly.NewTensor[float32](batchSize, l.InputChannels, l.InputHeight, l.InputWidth)
		gradOutput = poly.NewTensor[float32](batchSize, l.Filters, l.OutputHeight, l.OutputWidth)
	} else if lType == poly.LayerCNN3 {
		input = poly.NewTensor[float32](batchSize, l.InputChannels, l.InputDepth, l.InputHeight, l.InputWidth)
		gradOutput = poly.NewTensor[float32](batchSize, l.Filters, l.OutputDepth, l.OutputHeight, l.OutputWidth)
	} else {
		input = poly.NewTensor[float32](batchSize, l.InputHeight)
		gradOutput = poly.NewTensor[float32](batchSize, l.OutputHeight)
	}
	for i := range gradOutput.Data { gradOutput.Data[i] = 0.05 }
	if lType != poly.LayerEmbedding {
		for i := range input.Data { input.Data[i] = 0.5 }
	}

	// 1. CPU Backward
	var cpuDX, cpuDW *poly.Tensor[float32]
	start := time.Now()
	for i := 0; i < iterations; i++ {
		// preAct dummy - set to exactly 1.0 for RMSNorm/Dense
		preAct := poly.NewTensor[float32](batchSize, 2) // typical for RMSNorm
		if lType != poly.LayerRMSNorm {
			preAct = poly.NewTensor[float32](gradOutput.Shape...)
		}
		for j := range preAct.Data { preAct.Data[j] = 1.0 }
		cpuDX, cpuDW = poly.DispatchLayerBackward(&l, gradOutput, input, nil, preAct)
	}
	cpu = time.Since(start) / time.Duration(iterations)

	// 2. GPU Backward
	if ctx != nil {
		l.Network.GPUContext = ctx
		l.SyncToGPU()
		
		inBuf, _ := ctx.CreatePersistentBuffer(input.Data, "input")
		goBuf, _ := ctx.CreatePersistentBuffer(gradOutput.Data, "gradOutput")
		
		dxBuf := ctx.GetActivationBuffer("dx", uint64(len(cpuDX.Data)*4), wgpu.BufferUsageStorage | wgpu.BufferUsageCopySrc)
		dwBuf := ctx.GetActivationBuffer("dw", uint64(len(cpuDW.Data)*4), wgpu.BufferUsageStorage | wgpu.BufferUsageCopySrc)
		
		start = time.Now()
		for i := 0; i < iterations; i++ {
			// Clear grad buffers each iteration for parity testing
			ctx.Queue.WriteBuffer(dxBuf, 0, make([]byte, uint64(len(cpuDX.Data)*4)))
			ctx.Queue.WriteBuffer(dwBuf, 0, make([]byte, uint64(len(cpuDW.Data)*4)))

			switch lType {
			case poly.LayerDense:
				wBuf := l.WeightStore.GPUWeights[poly.DTypeFloat32].(*wgpu.Buffer)
				ctx.DispatchDenseBackwardDX(batchSize, l.InputHeight, l.OutputHeight, goBuf, wBuf, dxBuf, 16)
				ctx.DispatchDenseBackwardDW(batchSize, l.InputHeight, l.OutputHeight, goBuf, inBuf, dwBuf, 16)
			case poly.LayerRMSNorm:
				wBuf := l.WeightStore.GPUWeights[poly.DTypeFloat32].(*wgpu.Buffer)
				rmsBuf := ctx.GetActivationBuffer("rms", uint64(batchSize*4), wgpu.BufferUsageStorage)
				ctx.Queue.WriteBuffer(rmsBuf, 0, wgpu.ToBytes([]float32{1.0})) // Match CPU preAct
				ctx.DispatchRMSNormBackward(batchSize, l.InputHeight, 1e-5, goBuf, inBuf, rmsBuf, wBuf, dxBuf, dwBuf)
			case poly.LayerSwiGLU:
				// SwiGLU backward needs gateIn, upIn
				gIn := ctx.GetActivationBuffer("gateIn", uint64(len(gradOutput.Data)*4), wgpu.BufferUsageStorage)
				uIn := ctx.GetActivationBuffer("upIn", uint64(len(gradOutput.Data)*4), wgpu.BufferUsageStorage)
				
				// Fill with non-zero dummy data for verification
				dummyData := make([]float32, len(gradOutput.Data))
				for i := range dummyData { dummyData[i] = 0.5 }
				ctx.Queue.WriteBuffer(gIn, 0, wgpu.ToBytes(dummyData))
				ctx.Queue.WriteBuffer(uIn, 0, wgpu.ToBytes(dummyData))
				
				ctx.DispatchSwiGLUBackward(batchSize, l.InputHeight, l.OutputHeight, goBuf, gIn, uIn, dxBuf, dwBuf)
			case poly.LayerEmbedding:
				idxBuf := ctx.GetActivationBuffer("indices", uint64(len(input.Data)*4), wgpu.BufferUsageStorage)
				indicesU32 := make([]uint32, len(input.Data))
				for i, v := range input.Data { indicesU32[i] = uint32(v) }
				ctx.Queue.WriteBuffer(idxBuf, 0, wgpu.ToBytes(indicesU32))
				ctx.DispatchEmbeddingBackward(l.VocabSize, l.EmbeddingDim, 16*batchSize, idxBuf, goBuf, dwBuf)
			case poly.LayerResidual:
				ctx.DispatchResidualBackward(len(gradOutput.Data), goBuf, dxBuf, dwBuf)
			case poly.LayerMultiHeadAttention:
				// MHA needs Q, K, V buffers
				qBuf, _ := ctx.CreatePersistentBuffer(input.Data, "Q")
				kBuf, _ := ctx.CreatePersistentBuffer(input.Data, "K")
				vBuf, _ := ctx.CreatePersistentBuffer(input.Data, "V")
				dqBuf := ctx.GetActivationBuffer("dQ", uint64(len(input.Data)*4), wgpu.BufferUsageStorage)
				dkBuf := ctx.GetActivationBuffer("dK", uint64(len(input.Data)*4), wgpu.BufferUsageStorage)
				dvBuf := ctx.GetActivationBuffer("dV", uint64(len(input.Data)*4), wgpu.BufferUsageStorage)
				ctx.DispatchMHABackward(batchSize, l.NumHeads, l.NumKVHeads, l.HeadDim, l.InputHeight, 1.0, goBuf, qBuf, kBuf, vBuf, dqBuf, dkBuf, dvBuf)
			case poly.LayerCNN1:
				wBuf := l.WeightStore.GPUWeights[poly.DTypeFloat32].(*wgpu.Buffer)
				preBuf, _ := ctx.CreatePersistentBuffer(gradOutput.Data, "preAct")
				ctx.DispatchCNN1BackwardDX(batchSize, l.InputChannels, l.InputHeight, l.Filters, l.OutputHeight, l.KernelSize, l.Stride, l.Padding, l.Activation, goBuf, wBuf, preBuf, dxBuf)
				ctx.DispatchCNN1BackwardDW(batchSize, l.InputChannels, l.InputHeight, l.Filters, l.OutputHeight, l.KernelSize, l.Stride, l.Padding, l.Activation, goBuf, inBuf, preBuf, dwBuf)
			case poly.LayerCNN2:
				wBuf := l.WeightStore.GPUWeights[poly.DTypeFloat32].(*wgpu.Buffer)
				preBuf, _ := ctx.CreatePersistentBuffer(gradOutput.Data, "preAct")
				ctx.DispatchCNN2BackwardDX(batchSize, l.InputChannels, l.InputHeight, l.InputWidth, l.Filters, l.OutputHeight, l.OutputWidth, l.KernelSize, l.Stride, l.Padding, l.Activation, goBuf, wBuf, preBuf, dxBuf)
				ctx.DispatchCNN2BackwardDW(batchSize, l.InputChannels, l.InputHeight, l.InputWidth, l.Filters, l.OutputHeight, l.OutputWidth, l.KernelSize, l.Stride, l.Padding, l.Activation, goBuf, inBuf, preBuf, dwBuf)
			case poly.LayerCNN3:
				wBuf := l.WeightStore.GPUWeights[poly.DTypeFloat32].(*wgpu.Buffer)
				preBuf, _ := ctx.CreatePersistentBuffer(gradOutput.Data, "preAct")
				ctx.DispatchCNN3BackwardDX(batchSize, l.InputChannels, l.InputDepth, l.InputHeight, l.InputWidth, l.Filters, l.OutputDepth, l.OutputHeight, l.OutputWidth, l.KernelSize, l.Stride, l.Padding, l.Activation, goBuf, wBuf, preBuf, dxBuf)
				ctx.DispatchCNN3BackwardDW(batchSize, l.InputChannels, l.InputDepth, l.InputHeight, l.InputWidth, l.Filters, l.OutputDepth, l.OutputHeight, l.OutputWidth, l.KernelSize, l.Stride, l.Padding, l.Activation, goBuf, inBuf, preBuf, dwBuf)
			}
		}
		gpu = time.Since(start) / time.Duration(iterations)

		// Parity Check
		resDX, _ := ctx.ReadBuffer(dxBuf)
		resDW, _ := ctx.ReadBuffer(dwBuf)

		if len(resDX) >= 3 && len(cpuDX.Data) >= 3 {
			gDX = resDX[:3]
			cDX = cpuDX.Data[:3]
		}
		if len(resDW) >= 3 && len(cpuDW.Data) >= 3 {
			gDW = resDW[:3]
			cDW = cpuDW.Data[:3]
		}
		
		maxDX = calcMaxDiff(resDX, cpuDX.Data)
		maxDW = calcMaxDiff(resDW, cpuDW.Data)
		
		for _, v := range resDX { if v != 0 { sanity = true; break } }
		if !sanity {
			for _, v := range resDW { if v != 0 { sanity = true; break } }
		}
	}

	return
}

func calcMaxDiff(a, b []float32) float64 {
	if len(a) == 0 || len(b) == 0 { return -1 }
	max := 0.0
	for i := 0; i < len(a) && i < len(b); i++ {
		d := float64(a[i] - b[i])
		if d < 0 { d = -d }
		if d > max { max = d }
	}
	return max
}
