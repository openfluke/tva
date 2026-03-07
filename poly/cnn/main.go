package main

import (
	"fmt"
	"time"
	"sync"

	"github.com/openfluke/loom/poly"
)

type Result struct {
	Label     string
	Forward   time.Duration
	Backward  time.Duration
	Total     time.Duration
	SizeKb    float64
	PctSaved  float64
	SimTax    float64
}

var allTypes = []struct {
	Label string
	Type  poly.DType
}{
	{"Pure FLOAT64", poly.DTypeFloat64},
	{"Pure INT64", poly.DTypeInt64},
	{"Pure UINT64", poly.DTypeUint64},
	{"Pure FLOAT32", poly.DTypeFloat32},
	{"Pure INT32", poly.DTypeInt32},
	{"Pure UINT32", poly.DTypeUint32},
	{"Pure FLOAT16", poly.DTypeFloat16},
	{"Pure BFLOAT16", poly.DTypeBFloat16},
	{"Pure INT16", poly.DTypeInt16},
	{"Pure UINT16", poly.DTypeUint16},
	{"Pure FP8 (E4M3)", poly.DTypeFP8E4M3},
	{"Pure FP8 (E5M2)", poly.DTypeFP8E5M2},
	{"Pure INT8", poly.DTypeInt8},
	{"Pure UINT8", poly.DTypeUint8},
	{"Pure INT4", poly.DTypeInt4},
	{"Pure UINT4", poly.DTypeUint4},
	{"Pure FP4", poly.DTypeFP4},
	{"Pure INT2", poly.DTypeInt2},
	{"Pure UINT2", poly.DTypeUint2},
	{"Pure TERNARY", poly.DTypeTernary},
	{"Pure BINARY", poly.DTypeBinary},
}

func main() {
	depth := 4
	iterations := 50
	batchSize := 1

	fmt.Println("=== M-POLY-VTD CNN Truly Exhaustive Multi-Numerical Benchmark ===")

	runAllBenchmarks("CNN1 (1D)", depth, iterations, batchSize, setupCNN1Layer)
	runAllBenchmarks("CNN2 (2D)", depth, iterations, batchSize, setupCNN2Layer)
	runAllBenchmarks("CNN3 (3D)", depth, iterations, batchSize, setupCNN3Layer)
}

func runAllBenchmarks(title string, depth, iterations, batchSize int, setupFunc func(*poly.VolumetricLayer)) {
	fmt.Printf("\n--- Benchmark: %s ---\n", title)
	
	results := make(chan Result, 30)
	var wg sync.WaitGroup

	// Individual Benchmarks
	for _, s := range allTypes {
		wg.Add(1)
		go func(label string, dtype poly.DType) {
			defer wg.Done()
			net := poly.NewVolumetricNetwork(1, 1, 1, depth)
			for i := 0; i < depth; i++ {
				l := net.GetLayer(0, 0, 0, i)
				setupFunc(l)
				if i > 0 {
					l.InputChannels = l.Filters // Chain them
				}
				l.DType = dtype
				l.WeightStore.Scale = 0.01
				// Re-init weight store if size changed
				size := l.Filters * l.InputChannels * l.KernelSize
				if l.Type == poly.LayerCNN2 { size *= l.KernelSize }
				if l.Type == poly.LayerCNN3 { size *= l.KernelSize * l.KernelSize }
				l.WeightStore = poly.NewWeightStore(size)
				l.WeightStore.Randomize(42)
			}
			res := performBenchmark(label, net, iterations, batchSize)
			results <- res
		}(s.Label, s.Type)
	}

	// Hybrid Benchmark
	wg.Add(1)
	go func() {
		defer wg.Done()
		mixedDepth := len(allTypes)
		net := poly.NewVolumetricNetwork(1, 1, 1, mixedDepth)
		for i := 0; i < mixedDepth; i++ {
			l := net.GetLayer(0, 0, 0, i)
			setupFunc(l)
			if i > 0 {
				l.InputChannels = l.Filters // Chain them
			}
			l.DType = allTypes[i].Type
			l.WeightStore.Scale = 0.01
			// Re-init weight store
			size := l.Filters * l.InputChannels * l.KernelSize
			if l.Type == poly.LayerCNN2 { size *= l.KernelSize }
			if l.Type == poly.LayerCNN3 { size *= l.KernelSize * l.KernelSize }
			l.WeightStore = poly.NewWeightStore(size)
			l.WeightStore.Randomize(42)
		}
		res := performBenchmark("Exhaustive Multi-Type", net, iterations, batchSize)
		results <- res
	}()

	wg.Wait()
	close(results)

	resList := []Result{}
	for res := range results {
		resList = append(resList, res)
	}

	// Baseline size for depth layers in FP32
	baselineNet := poly.NewVolumetricNetwork(1, 1, 1, depth)
	for i := 0; i < depth; i++ {
		l := baselineNet.GetLayer(0, 0, 0, i)
		setupFunc(l)
		l.DType = poly.DTypeFloat32
	}
	baselineSizeKb := float64(baselineNet.CalculateTotalMemory()) / 1024.0

	// Print Table
	fmt.Println("\n| Scenario              | Forward (avg) | Backward (avg) | Training (total) | Size (KB) | % Saved | Sim Tax |")
	fmt.Println("|-----------------------|---------------|----------------|------------------|-----------|---------|---------|")
	for _, res := range resList {
		currentBaseline := baselineSizeKb
		if res.Label == "Exhaustive Multi-Type" {
			currentBaseline = baselineSizeKb * float64(len(allTypes)) / float64(depth)
		}
		res.PctSaved = (1.0 - (res.SizeKb / currentBaseline)) * 100.0
		
		fmt.Printf("| %-21s | %-13v | %-14v | %-16s | %-9.1f | %-7.1f%% | %-7.1f%% |\n",
			res.Label, res.Forward, res.Backward, res.Total, res.SizeKb, res.PctSaved, res.SimTax)
	}
}

func setupCNN1Layer(l *poly.VolumetricLayer) {
	l.Type = poly.LayerCNN1
	l.InputHeight = 1024
	l.InputChannels = 1
	l.OutputHeight = 1024
	l.Filters = 16
	l.KernelSize = 3
	l.Stride = 1
	l.Padding = 1
	l.WeightStore = poly.NewWeightStore(16 * 1 * 3)
	l.WeightStore.Randomize(42)
}

func setupCNN2Layer(l *poly.VolumetricLayer) {
	l.Type = poly.LayerCNN2
	l.InputHeight = 32
	l.InputWidth = 32
	l.InputChannels = 1
	l.OutputHeight = 32
	l.OutputWidth = 32
	l.Filters = 16
	l.KernelSize = 3
	l.Stride = 1
	l.Padding = 1
	l.WeightStore = poly.NewWeightStore(16 * 1 * 9)
	l.WeightStore.Randomize(42)
}

func setupCNN3Layer(l *poly.VolumetricLayer) {
	l.Type = poly.LayerCNN3
	l.InputDepth = 16
	l.InputHeight = 16
	l.InputWidth = 16
	l.InputChannels = 1
	l.OutputDepth = 16
	l.OutputHeight = 16
	l.OutputWidth = 16
	l.Filters = 4
	l.KernelSize = 3
	l.Stride = 1
	l.Padding = 1
	l.WeightStore = poly.NewWeightStore(4 * 1 * 27)
	l.WeightStore.Randomize(42)
}

func performBenchmark(label string, net *poly.VolumetricNetwork, iterations, batchSize int) Result {
	var input *poly.Tensor[float32]
	var target *poly.Tensor[float32]

	first := &net.Layers[0]
	switch first.Type {
	case poly.LayerCNN1:
		input = poly.NewTensor[float32](batchSize, first.InputChannels, first.InputHeight)
		target = poly.NewTensor[float32](batchSize, first.Filters, first.OutputHeight)
	case poly.LayerCNN2:
		input = poly.NewTensor[float32](batchSize, first.InputChannels, first.InputHeight, first.InputWidth)
		target = poly.NewTensor[float32](batchSize, first.Filters, first.OutputHeight, first.OutputWidth)
	case poly.LayerCNN3:
		input = poly.NewTensor[float32](batchSize, first.InputChannels, first.InputDepth, first.InputHeight, first.InputWidth)
		target = poly.NewTensor[float32](batchSize, first.Filters, first.OutputDepth, first.OutputHeight, first.OutputWidth)
	}

	for i := range input.Data { input.Data[i] = 1.0 }
	for i := range target.Data { target.Data[i] = 1.0 }

	sizeKb := float64(net.CalculateTotalMemory()) / 1024.0

	// Warmup
	poly.ForwardPolymorphic(net, input)

	var totalLayerForward time.Duration
	var lastLayerTimes []time.Duration

	// Forward Measurement Loop
	start := time.Now()
	for i := 0; i < iterations; i++ {
		_, _, layerTimes := poly.ForwardPolymorphic(net, input)
		lastLayerTimes = layerTimes
		for _, d := range layerTimes {
			totalLayerForward += d
		}
	}
	fwd := time.Since(start) / time.Duration(iterations)
	avgLayerForward := totalLayerForward / time.Duration(iterations)

	// Backward Measurement Loop (Matching example.go logic)
	gradOut := target // Use target as dummy gradOut
	start = time.Now()
	for i := 0; i < iterations; i++ {
		histIn := make([]*poly.Tensor[float32], len(net.Layers))
		histPre := make([]*poly.Tensor[float32], len(net.Layers))
		curr := input
		for idx := range net.Layers {
			l := &net.Layers[idx]
			histIn[idx] = curr
			pre, post := poly.DispatchLayer(l, curr)
			histPre[idx] = pre
			curr = post
		}
		poly.BackwardPolymorphic(net, gradOut, histIn, histPre)
	}
	bwd := time.Since(start) / time.Duration(iterations)

	// Training Measurement (Total process)
	config := poly.DefaultTrainingConfig()
	config.Epochs = 5
	config.Verbose = false
	if first.Type == poly.LayerCNN3 { config.LearningRate = 0.001 }
	batches := []poly.TrainingBatch[float32]{{Input: input, Target: target}}

	start = time.Now()
	poly.Train(net, batches, config)
	trainingTotal := time.Since(start)

	// Breakdown for Multi-Type
	if label == "Exhaustive Multi-Type" {
		fmt.Printf("\n--- Per-Layer Detail (%s) ---\n", label)
		for i, d := range lastLayerTimes {
			fmt.Printf("  Layer %d: Forward=%v\n", i, d)
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
		Total:    trainingTotal,
		SizeKb:   sizeKb,
		SimTax:   simTaxPct,
	}
}
