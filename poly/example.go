package main

import (
	"fmt"
	"sync"
	"time"

	"github.com/openfluke/loom/poly"
)

type Result struct {
	Label     string
	Forward   time.Duration
	Backward  time.Duration
	Total     time.Duration
	SizeKb    float64
	PctSaved  float64
	SimTax    float64 // % of time spent in dispatch/overhead vs pure math
}

func main() {
	fmt.Println("=== M-POLY-VTD Truly Exhaustive Multi-Numerical Benchmark ===")

	// Settings
	batchSize := 1
	inputSize := 1024
	outputSize := 1024
	iterations := 50
	depth := 4 // Standard depth for individual types

	allTypes := []struct {
		Label string
		Type  poly.DType
	}{
		// 64-bit
		{"Pure FLOAT64", poly.DTypeFloat64},
		{"Pure INT64", poly.DTypeInt64},
		{"Pure UINT64", poly.DTypeUint64},
		// 32-bit
		{"Pure FLOAT32", poly.DTypeFloat32},
		{"Pure INT32", poly.DTypeInt32},
		{"Pure UINT32", poly.DTypeUint32},
		// 16-bit
		{"Pure FLOAT16", poly.DTypeFloat16},
		{"Pure BFLOAT16", poly.DTypeBFloat16},
		{"Pure INT16", poly.DTypeInt16},
		{"Pure UINT16", poly.DTypeUint16},
		// 8-bit
		{"Pure FP8 (E4M3)", poly.DTypeFP8E4M3},
		{"Pure FP8 (E5M2)", poly.DTypeFP8E5M2},
		{"Pure INT8", poly.DTypeInt8},
		{"Pure UINT8", poly.DTypeUint8},
		// 4-bit
		{"Pure INT4", poly.DTypeInt4},
		{"Pure UINT4", poly.DTypeUint4},
		{"Pure FP4", poly.DTypeFP4},
		// 2-bit
		{"Pure INT2", poly.DTypeInt2},
		{"Pure UINT2", poly.DTypeUint2},
		{"Pure TERNARY", poly.DTypeTernary},
		// 1-bit
		{"Pure BINARY", poly.DTypeBinary},
	}

	results := make(chan Result, 30)
	var wg sync.WaitGroup

	// Baseline for savings calculation (depth layers of FP32)
	baselineSize := float64(depth*(inputSize*outputSize)*4) / 1024.0

	// 1. Individual Type Benchmarks
	for _, s := range allTypes {
		wg.Add(1)
		go func(label string, dtype poly.DType) {
			defer wg.Done()
			net := poly.NewVolumetricNetwork(1, 1, 1, depth)
			for i := 0; i < depth; i++ {
				l := net.GetLayer(0, 0, 0, i)
				l.Type = poly.LayerDense
				l.InputHeight = inputSize
				l.OutputHeight = outputSize
				l.WeightStore = poly.NewWeightStore(inputSize * outputSize)
				l.DType = dtype
				l.WeightStore.Scale = 0.01
			}
			res := runBenchmark(label, net, batchSize, inputSize, outputSize, iterations)
			results <- res
		}(s.Label, s.Type)
	}

	// 2. Hybrid Scenarios
	// Exhaustive Multi-Type (Mixed chain)
	wg.Add(1)
	go func() {
		defer wg.Done()
		mixedDepth := len(allTypes)
		net := poly.NewVolumetricNetwork(1, 1, 1, mixedDepth)
		for i := 0; i < mixedDepth; i++ {
			l := net.GetLayer(0, 0, 0, i)
			l.Type = poly.LayerDense
			l.InputHeight = inputSize
			l.OutputHeight = outputSize
			l.WeightStore = poly.NewWeightStore(inputSize * outputSize)
			l.DType = allTypes[i].Type
			l.WeightStore.Scale = 0.01
		}
		res := runBenchmark("Exhaustive Multi-Type", net, batchSize, inputSize, outputSize, iterations)
		results <- res
	}()

	wg.Wait()
	close(results)

	// Collect results
	resList := []Result{}
	for res := range results {
		// PctSaved calculation for individual types (based on standard 'depth')
		// For the exhaustive chain, the baseline would be mixedDepth * baselineSize/depth
		currentBaseline := baselineSize
		if res.Label == "Exhaustive Multi-Type" {
			currentBaseline = (float64(len(allTypes)) * (float64(inputSize*outputSize) * 4)) / 1024.0
		}
		res.PctSaved = (1.0 - (res.SizeKb / currentBaseline)) * 100.0
		resList = append(resList, res)
	}

	// Print Results Table
	fmt.Println("\n| Scenario              | Forward (avg) | Training (total) | Size (KB) | % Saved | Sim Tax |")
	fmt.Println("|-----------------------|---------------|------------------|-----------|---------|---------|")
	for _, res := range resList {
		fmt.Printf("| %-21s | %-13v | %-16v | %-9.1f | %-7.1f%% | %-7.1f%% |\n",
			res.Label, res.Forward, res.Total, res.SizeKb, res.PctSaved, res.SimTax)
	}
}

func runBenchmark(label string, net *poly.VolumetricNetwork, batch, in, out, iter int) Result {
	input := poly.NewTensor[float32](batch, in)
	gradOut := poly.NewTensor[float32](batch, out)

	sizeBytes := net.CalculateTotalMemory()
	sizeKb := float64(sizeBytes) / 1024.0

	// Warmup
	poly.ForwardPolymorphic(net, input)

	var totalLayerForward time.Duration
	var lastLayerTimes []time.Duration

	// Forward
	start := time.Now()
	for i := 0; i < iter; i++ {
		_, _, layerTimes := poly.ForwardPolymorphic(net, input)
		lastLayerTimes = layerTimes
		for _, d := range layerTimes {
			totalLayerForward += d
		}
	}
	fwd := time.Since(start) / time.Duration(iter)
	avgLayerForward := totalLayerForward / time.Duration(iter)

	// Backward / Training
	start = time.Now()
	var lastBackwardTimes []time.Duration
	for i := 0; i < iter; i++ {
		// Manual history capture since ForwardTraining was "stupid"
		hist_in := make([]*poly.Tensor[float32], len(net.Layers))
		hist_pre := make([]*poly.Tensor[float32], len(net.Layers))
		curr := input
		for idx := range net.Layers {
			l := &net.Layers[idx]
			hist_in[idx] = curr
			pre, post := poly.DispatchLayer(l, curr)
			hist_pre[idx] = pre
			curr = post
		}
		_, grads, bwdTimes := poly.BackwardPolymorphic(net, gradOut, hist_in, hist_pre)
		lastBackwardTimes = bwdTimes
		
		// REAL WEIGHT UPDATE (The "Learning" Step)
		lr := float32(0.001)
		for idx := range net.Layers {
			l := &net.Layers[idx]
			if l.WeightStore != nil && grads[idx][1] != nil {
				// Cast the gradient to float32 for the update step
				gW := poly.ConvertTensor[float32, float32](grads[idx][1])
				l.WeightStore.ApplyGradients(gW, lr)
			}
		}
	}
	bwd := time.Since(start) / time.Duration(iter)

	// If it's a multi-type or specifically interesting, print a small breakdown
	if label == "Exhaustive Multi-Type" {
		fmt.Printf("\n--- Per-Layer Detail (%s) ---\n", label)
		for i, d := range lastLayerTimes {
			fmt.Printf("  Layer %d: Forward=%v, Backward=%v\n", i, d, lastBackwardTimes[i])
		}
	}

	simTaxPct := 0.0
	if fwd > 0 {
		simTaxPct = (1.0 - (float64(avgLayerForward) / float64(fwd))) * 100.0
	}

	return Result{
		Label:    label,
		Forward:  fwd,
		Backward: bwd,
		Total:    fwd + bwd,
		SizeKb:   sizeKb,
		SimTax:   simTaxPct,
	}
}
