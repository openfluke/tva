package main

import (
	"fmt"
	"math/rand"
	"time"
	"github.com/openfluke/loom/poly"
)

func main() {
	fmt.Println("=== M-POLY-VTD Parallel Layer Benchmarking ===")
	
	inputSize := 128
	
	// Scenario 1: Additive Parallelism (Residual-like)
	branch1 := poly.VolumetricLayer{Type: poly.LayerDense, OutputHeight: inputSize, InputHeight: inputSize, DType: poly.DTypeFloat32}
	branch2 := poly.VolumetricLayer{Type: poly.LayerDense, OutputHeight: inputSize, InputHeight: inputSize, DType: poly.DTypeInt8}
	
	// Setup WeightStores (Minimal for benchmark)
	store1 := &poly.WeightStore{Master: make([]float32, inputSize*inputSize)}
	store2 := &poly.WeightStore{Master: make([]float32, inputSize*inputSize)}
	branch1.WeightStore = store1
	branch2.WeightStore = store2

	parallelLayer := &poly.VolumetricLayer{
		Type:             poly.LayerParallel,
		ParallelBranches: []poly.VolumetricLayer{branch1, branch2},
		CombineMode:      "add",
	}

	input := poly.NewTensor[float32](1, inputSize)
	for i := range input.Data { input.Data[i] = rand.Float32() }

	fmt.Println("\n--- Scenario 1: [Dense-F32 + Dense-I8] (Add Mode) ---")
	measure(parallelLayer, input)

	// Scenario 2: Mixed Concatenation (CNN + Dense + Softmax)
	cnnBranch := poly.VolumetricLayer{
		Type: poly.LayerCNN1, 
		InputHeight: inputSize, 
		Filters: 8, 
		KernelSize: 3, 
		Stride: 1, 
		Padding: 1,
		DType: poly.DTypeFloat16,
		WeightStore: &poly.WeightStore{Master: make([]float32, 8*3)},
	}
	denseBranch := poly.VolumetricLayer{
		Type: poly.LayerDense, 
		OutputHeight: 64, 
		InputHeight: inputSize, 
		DType: poly.DTypeInt4,
		WeightStore: &poly.WeightStore{Master: make([]float32, 64*inputSize)},
	}
	softmaxBranch := poly.VolumetricLayer{
		Type: poly.LayerSoftmax, 
		SoftmaxType: poly.SoftmaxStandard,
		DType: poly.DTypeFloat32,
	}

	mixedParallel := &poly.VolumetricLayer{
		Type: poly.LayerParallel,
		ParallelBranches: []poly.VolumetricLayer{cnnBranch, denseBranch, softmaxBranch},
		CombineMode:      "concat",
	}

	fmt.Println("\n--- Scenario 2: [CNN-F16 + Dense-I4 + Softmax-F32] (Concat Mode) ---")
	measure(mixedParallel, input)

	// Scenario 3: MoE-style Filtered Gating
	gateLayer := &poly.VolumetricLayer{
		Type: poly.LayerDense,
		OutputHeight: 2, // 2 branches
		InputHeight: inputSize,
		WeightStore: &poly.WeightStore{Master: make([]float32, 2*inputSize)},
	}
	
	filteredParallel := &poly.VolumetricLayer{
		Type: poly.LayerParallel,
		ParallelBranches: []poly.VolumetricLayer{branch1, branch2},
		CombineMode:      "filter",
		FilterGateConfig: gateLayer,
	}

	fmt.Println("\n--- Scenario 3: MoE Gated Parallel (Softmax Filter) ---")
	measure(filteredParallel, input)
}

func measure(l *poly.VolumetricLayer, input *poly.Tensor[float32]) {
	iterations := 100
	
	// Warmup
	poly.DispatchLayer(l, input)
	
	start := time.Now()
	for i := 0; i < iterations; i++ {
		poly.DispatchLayer(l, input)
	}
	elapsed := time.Since(start) / time.Duration(iterations)
	
	_, out := poly.DispatchLayer(l, input)
	fmt.Printf("Avg Latency: %v | Output Size: %d\n", elapsed, len(out.Data))
}
