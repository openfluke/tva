package main

import (
	"fmt"
	"time"

	"github.com/openfluke/loom/poly"
)

type Result struct {
	Label        string
	Forward      time.Duration
	ForwardTile  time.Duration
	Backward     time.Duration
	Total        time.Duration
	SizeKb       float64
	PctSaved     float64
	SimTax       float64 // % of time spent in dispatch/overhead vs pure math
}

func main() {
	fmt.Println("=== M-POLY-VTD Truly Exhaustive Multi-Numerical Benchmark ===")

	allTypes := []struct {
		Label string
		Type  poly.DType
	}{
		{"Pure FLOAT64", poly.DTypeFloat64},
		{"Pure FLOAT32", poly.DTypeFloat32},
		{"Pure FLOAT16", poly.DTypeFloat16},
		{"Pure BFLOAT16", poly.DTypeBFloat16},
		{"Pure FP8 (E4M3)", poly.DTypeFP8E4M3},
		{"Pure FP8 (E5M2)", poly.DTypeFP8E5M2},
		{"Pure INT64", poly.DTypeInt64},
		{"Pure INT32", poly.DTypeInt32},
		{"Pure INT16", poly.DTypeInt16},
		{"Pure INT8", poly.DTypeInt8},
		{"Pure UINT64", poly.DTypeUint64},
		{"Pure UINT32", poly.DTypeUint32},
		{"Pure UINT16", poly.DTypeUint16},
		{"Pure UINT8", poly.DTypeUint8},
		{"Pure INT4", poly.DTypeInt4},
		{"Pure UINT4", poly.DTypeUint4},
		{"Pure FP4", poly.DTypeFP4},
		{"Pure INT2", poly.DTypeInt2},
		{"Pure UINT2", poly.DTypeUint2},
		{"Pure TERNARY", poly.DTypeTernary},
		{"Pure BINARY", poly.DTypeBinary},
	}

	// 1. DENSE BENCHMARKS
	runBenchSet("DENSE", poly.LayerDense, allTypes)

	// 2. CNN3 BENCHMARKS
	runBenchSet("CNN3", poly.LayerCNN3, allTypes)
}

func runBenchSet(title string, lType poly.LayerType, allTypes []struct {
	Label string
	Type  poly.DType
}) {
	fmt.Printf("\n=== %s Benchmarks ===\n", title)

	batchSize := 1
	iterations := 20
	depth := 4

	// Dimensions
	inputSize := 256
	outputSize := 256
	if lType == poly.LayerCNN3 {
		inputSize = 512
		outputSize = 512
	}

	resList := []Result{}

	// Individual Type Benchmarks
	for _, s := range allTypes {
		net := poly.NewVolumetricNetwork(1, 1, 1, depth)
		for i := 0; i < depth; i++ {
			l := net.GetLayer(0, 0, 0, i)
			l.Type = lType
			l.DType = s.Type
			l.Activation = poly.ActivationReLU

			if lType == poly.LayerCNN3 {
				l.InputChannels = 8
				l.InputDepth = 4
				l.InputHeight = 4
				l.InputWidth = 4
				l.Filters = 8
				l.KernelSize = 3
				l.Stride = 1
				l.Padding = 1
				l.OutputDepth = 4
				l.OutputHeight = 4
				l.OutputWidth = 4
				l.WeightStore = poly.NewWeightStore(l.Filters * l.InputChannels * l.KernelSize * l.KernelSize * l.KernelSize)
			} else {
				l.InputHeight = inputSize
				l.OutputHeight = outputSize
				l.WeightStore = poly.NewWeightStore(inputSize * outputSize)
			}
			l.WeightStore.Scale = 0.01
		}
		res := runBenchmark(s.Label, net, batchSize, lType, inputSize, outputSize, iterations)

		// Calculate PctSaved
		baselineSize := float64(depth*(inputSize*outputSize)*4) / 1024.0
		res.PctSaved = (1.0 - (res.SizeKb / baselineSize)) * 100.0
		resList = append(resList, res)
	}

	// Hybrid / Mixed
	mixedDepth := len(allTypes)
	net := poly.NewVolumetricNetwork(1, 1, 1, mixedDepth)
	for i := 0; i < mixedDepth; i++ {
		l := net.GetLayer(0, 0, 0, i)
		l.Type = lType
		l.DType = allTypes[i].Type
		l.Activation = poly.ActivationReLU

		if lType == poly.LayerCNN3 {
			l.InputChannels = 8
			l.InputDepth = 4
			l.InputHeight = 4
			l.InputWidth = 4
			l.Filters = 8
			l.KernelSize = 3
			l.Stride = 1
			l.Padding = 1
			l.OutputDepth = 4
			l.OutputHeight = 4
			l.OutputWidth = 4
			l.WeightStore = poly.NewWeightStore(l.Filters * l.InputChannels * l.KernelSize * l.KernelSize * l.KernelSize)
		} else {
			l.InputHeight = inputSize
			l.OutputHeight = outputSize
			l.WeightStore = poly.NewWeightStore(inputSize * outputSize)
		}
		l.WeightStore.Scale = 0.01
	}
	res := runBenchmark("Exhaustive Multi-Type", net, batchSize, lType, inputSize, outputSize, iterations)
	currentBaseline := (float64(len(allTypes)) * (float64(inputSize*outputSize) * 4)) / 1024.0
	res.PctSaved = (1.0 - (res.SizeKb / currentBaseline)) * 100.0
	resList = append(resList, res)

	fmt.Println("| Scenario              | Forward (avg) | Forward (Tile)| Training (total) | Size (KB) | % Saved | Sim Tax |")
	fmt.Println("|-----------------------|---------------|---------------|------------------|-----------|---------|---------|")
	for _, res := range resList {
		fmt.Printf("| %-21s | %-13v | %-13v | %-16v | %-9.1f | %-7.1f%% | %-7.1f%% |\n",
			res.Label, res.Forward, res.ForwardTile, res.Total, res.SizeKb, res.PctSaved, res.SimTax)
	}
}

func runBenchmark(label string, net *poly.VolumetricNetwork, batch int, lType poly.LayerType, in, out, iter int) Result {
	var input *poly.Tensor[float32]
	var gradOut *poly.Tensor[float32]

	if lType == poly.LayerCNN3 {
		input = poly.NewTensor[float32](batch, 8, 4, 4, 4)
		gradOut = poly.NewTensor[float32](batch, 8, 4, 4, 4)
	} else {
		input = poly.NewTensor[float32](batch, in)
		gradOut = poly.NewTensor[float32](batch, out)
	}

	sizeBytes := net.CalculateTotalMemory()
	sizeKb := float64(sizeBytes) / 1024.0

	// 1. Standard Forward
	net.UseTiling = false
	for i := range net.Layers {
		net.Layers[i].UseTiling = false
	}

	// Warmup
	poly.ForwardPolymorphic(net, input)

	var totalLayerForward time.Duration
	start := time.Now()
	for i := 0; i < iter; i++ {
		_, _, layerTimes := poly.ForwardPolymorphic(net, input)
		for _, d := range layerTimes {
			totalLayerForward += d
		}
	}
	fwd := time.Since(start) / time.Duration(iter)
	avgLayerForward := totalLayerForward / time.Duration(iter)

	// 2. Tiled Forward
	net.UseTiling = true
	optTile := poly.CalculateOptimalTileSize(in)
	for i := range net.Layers {
		net.Layers[i].UseTiling = true
		net.Layers[i].TileSize = optTile
	}

	// Warmup
	poly.ForwardPolymorphic(net, input)

	start = time.Now()
	for i := 0; i < iter; i++ {
		poly.ForwardPolymorphic(net, input)
	}
	fwdTile := time.Since(start) / time.Duration(iter)

	// Reset for training/backward
	net.UseTiling = false
	for i := range net.Layers {
		net.Layers[i].UseTiling = false
	}

	// 3. Backward / Training
	start = time.Now()
	for i := 0; i < iter; i++ {
		hist_in := make([]*poly.Tensor[float32], len(net.Layers))
		hist_pre := make([]*poly.Tensor[float32], len(net.Layers))
		curr := input
		for idx := range net.Layers {
			l := &net.Layers[idx]
			hist_in[idx] = curr
			pre, post := poly.DispatchLayer(l, curr, nil)
			hist_pre[idx] = pre
			curr = post
		}
		_, grads, _ := poly.BackwardPolymorphic(net, gradOut, hist_in, hist_pre)

		// REAL WEIGHT UPDATE
		lr := float32(0.001)
		for idx := range net.Layers {
			l := &net.Layers[idx]
			if l.WeightStore != nil && grads[idx][1] != nil {
				gW := poly.ConvertTensor[float32, float32](grads[idx][1])
				l.WeightStore.ApplyGradients(gW, lr)
			}
		}
	}
	bwd := time.Since(start) / time.Duration(iter)

	simTaxPct := 0.0
	if fwd > 0 {
		simTaxPct = (1.0 - (float64(avgLayerForward) / float64(fwd))) * 100.0
	}

	return Result{
		Label:       label,
		Forward:     fwd,
		ForwardTile: fwdTile,
		Backward:    bwd,
		Total:       fwd + bwd,
		SizeKb:      sizeKb,
		SimTax:      simTaxPct,
	}
}
