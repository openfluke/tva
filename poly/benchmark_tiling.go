package main

import (
	"fmt"
	"time"

	"github.com/openfluke/loom/poly"
	"github.com/openfluke/webgpu/wgpu"
)

func main() {
	fmt.Println("=== M-POLY-VTD Performance Showdown: CPU Tiling vs GPU Acceleration ===")
	
	// Initialize GPU via a dummy network
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
		{"RNN Cell", poly.LayerRNN},
		{"LSTM Cell", poly.LayerLSTM},
		{"CNN 1D", poly.LayerCNN1},
		{"CNN 2D", poly.LayerCNN2},
		{"CNN 3D", poly.LayerCNN3},
		{"Embedding", poly.LayerEmbedding},
		{"RMSNorm", poly.LayerRMSNorm},
		{"MHA (Attn)", poly.LayerMultiHeadAttention},
		{"SwiGLU (MLP)", poly.LayerSwiGLU},
		{"Residual Add", poly.LayerResidual},
	}

	fmt.Println("| Layer type      | CPU (Simple) | CPU (Tiled)  | GPU (WebGPU) | Speedup (vs Tiled) | Deterministic | Sanity        |")
	fmt.Println("|-----------------|--------------|--------------|--------------|-------------------|---------------|---------------|")

	for _, l := range layers {
		cpuSimple, cpuTiled, gpuTime, maxDiff, sanity := runBench(l.Type, ctx)
		
		gpuLabel := "N/A"
		speedup := "N/A"
		if gpuTime > 0 {
			gpuLabel = fmt.Sprintf("%v", gpuTime)
			if cpuTiled > 0 {
				ratio := float64(cpuTiled) / float64(gpuTime)
				speedup = fmt.Sprintf("%.2fx", ratio)
			}
		}

		det := "Wait..."
		if gpuTime == 0 {
			det = "N/A"
		} else if maxDiff < 1e-7 {
			det = "EXACT ⭐"
		} else if maxDiff < 1e-4 {
			det = "INDUSTRY ✅"
		} else if maxDiff < 1e-2 {
			det = "SLIGHTLY OFF ⚠️"
		} else {
			det = "BROKEN ❌"
		}

		san := "Wait..."
		if sanity { san = "REAL 💎" } else if gpuTime > 0 { san = "ZERO 💀" } else { san = "N/A" }

		fmt.Printf("| %-15s | %-12v | %-12v | %-12s | %-17s | %-13s | %-13s |\n", 
			l.Name, cpuSimple, cpuTiled, gpuLabel, speedup, det, san)
	}
}

func runBench(lType poly.LayerType, ctx *poly.WGPUContext) (simple, tiled, gpu time.Duration, maxDiff float64, sanity bool) {
	iterations := 10
	
	// Setup layer
	l := poly.VolumetricLayer{
		Network: poly.NewVolumetricNetwork(1, 1, 1, 1),
		Type: lType,
		InputChannels: 128,
		InputHeight: 128,
		InputWidth: 64,
		InputDepth: 64,
		Filters: 128,
		OutputHeight: 128,
		OutputWidth: 64,
		OutputDepth: 64,
		KernelSize: 3,
		Stride: 1,
		Padding: 1,
		SeqLength: 1,
		NumHeads: 4,
		NumKVHeads: 4,
		HeadDim: 32,
		DModel: 128,
		MaxSeqLen: 512,
	}
	
	if lType == poly.LayerDense {
		l.InputHeight = 1024
		l.OutputHeight = 1024
		l.WeightStore = poly.NewWeightStore(1024 * 1024)
	} else if lType == poly.LayerRNN {
		l.InputHeight = 512
		l.OutputHeight = 512
		l.WeightStore = poly.NewWeightStore(512*512 + 512*512 + 512)
	} else if lType == poly.LayerLSTM {
		l.InputHeight = 512
		l.OutputHeight = 512
		l.WeightStore = poly.NewWeightStore(4 * (512*512 + 512*512 + 512))
	} else if lType == poly.LayerCNN1 {
		l.InputChannels = 64
		l.InputHeight = 512
		l.Filters = 64
		l.WeightStore = poly.NewWeightStore(64 * 64 * 3)
	} else if lType == poly.LayerCNN2 {
		l.InputChannels = 32
		l.InputHeight = 64
		l.InputWidth = 64
		l.Filters = 32
		l.WeightStore = poly.NewWeightStore(32 * 32 * 3 * 3)
	} else if lType == poly.LayerCNN3 {
		l.InputChannels = 16
		l.InputDepth = 16
		l.InputHeight = 16
		l.InputWidth = 16
		l.Filters = 16
		l.WeightStore = poly.NewWeightStore(16 * 16 * 3 * 3 * 3)
	} else if lType == poly.LayerEmbedding {
		l.VocabSize = 2048
		l.EmbeddingDim = 128
		l.WeightStore = poly.NewWeightStore(l.VocabSize * l.EmbeddingDim)
	} else if lType == poly.LayerRMSNorm {
		l.InputHeight = 1024
		l.OutputHeight = 1024
		l.WeightStore = poly.NewWeightStore(1024)
	} else if lType == poly.LayerMultiHeadAttention {
		l.DModel = 128
		l.NumHeads = 4
		l.NumKVHeads = 4
		l.HeadDim = 32
		kvDim := l.NumKVHeads * l.HeadDim
		// Weights (Q, K, V, O) + Biases (Q, K, V, O)
		size := 2*(l.DModel*l.DModel) + 2*(l.DModel*kvDim) + 2*l.DModel + 2*kvDim
		l.WeightStore = poly.NewWeightStore(size)
	} else if lType == poly.LayerSwiGLU {
		l.InputHeight = 512
		l.OutputHeight = 1024
		// Weights (Gate, Up, Down) + Biases (Gate, Up, Down)
		size := 3*512*1024 + 2*1024 + 512
		l.WeightStore = poly.NewWeightStore(size)
	} else if lType == poly.LayerResidual {
		l.InputHeight = 1024
		l.OutputHeight = 1024
		l.WeightStore = poly.NewWeightStore(0)
	}

	for i := range l.WeightStore.Master {
		l.WeightStore.Master[i] = 0.1
	}

	// Inputs
	var input *poly.Tensor[float32]
	if lType == poly.LayerCNN3 {
		input = poly.NewTensor[float32](1, l.InputChannels, l.InputDepth, l.InputHeight, l.InputWidth)
	} else if lType == poly.LayerCNN2 {
		input = poly.NewTensor[float32](1, l.InputChannels, l.InputHeight, l.InputWidth)
	} else if lType == poly.LayerCNN1 {
		input = poly.NewTensor[float32](1, l.InputChannels, l.InputHeight)
	} else if lType == poly.LayerEmbedding {
		input = poly.NewTensor[float32](1, 64) // seqLen=64
		for i := range input.Data { input.Data[i] = float32(i % l.VocabSize) }
	} else {
		input = poly.NewTensor[float32](1, l.InputHeight)
	}
	for i := range input.Data { if lType != poly.LayerEmbedding { input.Data[i] = 0.5 } }

	// 1. CPU Simple
	l.UseTiling = false
	start := time.Now()
	var cpuOut *poly.Tensor[float32]
	for i := 0; i < iterations; i++ {
		_, cpuOut = poly.DispatchLayer(&l, input, nil)
	}
	simple = time.Since(start) / time.Duration(iterations)

	// 2. CPU Tiled
	l.UseTiling = true
	l.TileSize = 32
	start = time.Now()
	for i := 0; i < iterations; i++ {
		poly.DispatchLayer(&l, input, nil)
	}
	tiled = time.Since(start) / time.Duration(iterations)

	// 3. GPU
	if ctx != nil {
		l.Network.GPUContext = ctx
		// Sync weights
		l.SyncToGPU()
		
		inBuf := ctx.GetActivationBuffer("input", uint64(len(input.Data) * 4), wgpu.BufferUsageStorage)
		ctx.Queue.WriteBuffer(inBuf, 0, wgpu.ToBytes(input.Data))
		
		outSize := len(cpuOut.Data)
		outBuf := ctx.GetActivationBuffer("output", uint64(outSize * 4), wgpu.BufferUsageStorage | wgpu.BufferUsageCopySrc)
		
		start = time.Now()
		for i := 0; i < iterations; i++ {
			switch lType {
			case poly.LayerDense:
				wBuf, _ := l.WeightStore.GPUWeights[poly.DTypeFloat32].(*wgpu.Buffer)
				ctx.DispatchDense(1, l.InputHeight, l.OutputHeight, inBuf, wBuf, outBuf, 32)
			case poly.LayerRNN:
				wIH, _ := l.WeightStore.GPUWeights[poly.DTypeFloat32].(*wgpu.Buffer)
				wHH := wIH // simplified for bench
				hPrev := ctx.GetActivationBuffer("hPrev", uint64(l.OutputHeight*4), wgpu.BufferUsageStorage)
				bias := ctx.GetActivationBuffer("bias", uint64(l.OutputHeight*4), wgpu.BufferUsageStorage)
				ctx.DispatchRNNStep(1, l.InputHeight, l.OutputHeight, inBuf, hPrev, wIH, wHH, bias, outBuf)
			case poly.LayerLSTM:
				weights, _ := l.WeightStore.GPUWeights[poly.DTypeFloat32].(*wgpu.Buffer)
				hPrev := ctx.GetActivationBuffer("hPrev", uint64(l.OutputHeight*4), wgpu.BufferUsageStorage)
				cPrev := ctx.GetActivationBuffer("cPrev", uint64(l.OutputHeight*4), wgpu.BufferUsageStorage)
				cCurr := ctx.GetActivationBuffer("cCurr", uint64(l.OutputHeight*4), wgpu.BufferUsageStorage)
				ctx.DispatchLSTMStep(1, l.InputHeight, l.OutputHeight, inBuf, hPrev, cPrev, weights, outBuf, cCurr)
			case poly.LayerCNN1:
				w, _ := l.WeightStore.GPUWeights[poly.DTypeFloat32].(*wgpu.Buffer)
				ctx.DispatchCNN1(1, l.InputChannels, l.InputHeight, l.Filters, l.OutputHeight, l.KernelSize, l.Stride, l.Padding, inBuf, w, outBuf)
			case poly.LayerCNN2:
				w, _ := l.WeightStore.GPUWeights[poly.DTypeFloat32].(*wgpu.Buffer)
				ctx.DispatchCNN2(1, l.InputChannels, l.InputHeight, l.InputWidth, l.Filters, l.OutputHeight, l.OutputWidth, l.KernelSize, l.KernelSize, l.Stride, l.Stride, l.Padding, l.Padding, inBuf, w, outBuf)
			case poly.LayerCNN3:
				w, _ := l.WeightStore.GPUWeights[poly.DTypeFloat32].(*wgpu.Buffer)
				ctx.DispatchCNN3(1, l.InputChannels, l.InputDepth, l.InputHeight, l.InputWidth, l.Filters, l.OutputDepth, l.OutputHeight, l.OutputWidth, l.KernelSize, l.KernelSize, l.KernelSize, l.Stride, l.Stride, l.Stride, l.Padding, l.Padding, l.Padding, inBuf, w, outBuf)
			case poly.LayerEmbedding:
				w, _ := l.WeightStore.GPUWeights[poly.DTypeFloat32].(*wgpu.Buffer)
				ctx.DispatchEmbedding(l.VocabSize, l.EmbeddingDim, 64, inBuf, w, outBuf)
			case poly.LayerRMSNorm:
				w, _ := l.WeightStore.GPUWeights[poly.DTypeFloat32].(*wgpu.Buffer)
				ctx.DispatchRMSNorm(1, l.InputHeight, 1e-5, inBuf, w, outBuf)
			case poly.LayerMultiHeadAttention:
				q, _ := l.WeightStore.GPUWeights[poly.DType(200)].(*wgpu.Buffer)
				k, _ := l.WeightStore.GPUWeights[poly.DType(201)].(*wgpu.Buffer)
				v, _ := l.WeightStore.GPUWeights[poly.DType(202)].(*wgpu.Buffer)
				oWeights, _ := l.WeightStore.GPUWeights[poly.DType(203)].(*wgpu.Buffer)
				
				// Attention output temp buffer
				attnOut := ctx.GetActivationBuffer("attn_out", uint64(64 * l.DModel * 4), wgpu.BufferUsageStorage)
				ctx.DispatchMHA(l.NumHeads, l.NumKVHeads, l.HeadDim, 64, 0, 512, q, k, v, attnOut, 32)
				// Final O-projection
				ctx.DispatchDense(1, l.DModel, l.DModel, attnOut, oWeights, outBuf, 32)
				
			case poly.LayerSwiGLU:
				g, _ := l.WeightStore.GPUWeights[poly.DType(100)].(*wgpu.Buffer)
				u, _ := l.WeightStore.GPUWeights[poly.DType(101)].(*wgpu.Buffer)
				wDown, _ := l.WeightStore.GPUWeights[poly.DType(102)].(*wgpu.Buffer)
				
				preOut := ctx.GetActivationBuffer("preOut", uint64(l.OutputHeight*4), wgpu.BufferUsageStorage)
				ctx.DispatchSwiGLU(1, l.InputHeight, l.OutputHeight, inBuf, g, u, preOut, 32)
				ctx.DispatchDense(1, l.OutputHeight, l.InputHeight, preOut, wDown, outBuf, 32)

			case poly.LayerResidual:
				// Residual Add needs outBuf to be initialized with 'input' (identity)
				// so we add inBuf to outBuf to get 2*input (or similar)
				ctx.Queue.WriteBuffer(outBuf, 0, wgpu.ToBytes(input.Data))
				ctx.DispatchResidual(l.InputHeight, outBuf, inBuf)
			}
		}
		gpu = time.Since(start) / time.Duration(iterations)
		
		maxDiff = -1.0
		// Parity & Sanity Check
		gpuRes, _ := ctx.ReadBuffer(outBuf)
		if len(gpuRes) >= len(cpuOut.Data) && len(cpuOut.Data) > 0 {
			maxDiff = 0
			for i := 0; i < 100 && i < len(cpuOut.Data); i++ {
				diff := float64(gpuRes[i] - cpuOut.Data[i])
				if diff < 0 { diff = -diff }
				if diff > maxDiff {
					maxDiff = diff
				}
				
				// Sanity: Is it non-zero?
				val := gpuRes[i]
				if val < 0 { val = -val }
				if val > 1e-6 {
					sanity = true
				}
			}
		}
	}

	return
}
