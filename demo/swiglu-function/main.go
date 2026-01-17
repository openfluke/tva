package main

import (
	"fmt"
	"math"
	"math/rand"
	"time"

	"github.com/openfluke/loom/nn"
)

// SwiGLU Function Approximation Demo
// Demonstrates SwiGLU layers on complex function approximation

const (
	InputDim        = 2   // 2D functions (x, y)
	IntermediateDim = 128 // SwiGLU intermediate dimension
	NumFunctions    = 4   // Different function types
	SamplesPerFunc  = 250
	Epochs          = 100
	LearningRate    = 0.01
	BatchSize       = 20
)

const (
	FuncSinCos = iota
	FuncPolynomial
	FuncGaussian
	FuncTanhProd
)

func main() {
	rand.Seed(time.Now().UnixNano())

	fmt.Println("╔════════════════════════════════════════════════════════════════╗")
	fmt.Println("║   SwiGLU Demo: Complex Function Approximation                 ║")
	fmt.Println("╚════════════════════════════════════════════════════════════════╝")

	// Generate dataset
	fmt.Println("\n[1/5] Generating function data...")
	trainData, trainTargets := generateFunctionDataset()
	fmt.Printf("      Generated %d samples from 4 complex functions\n", len(trainData))

	// Build network config
	jsonConfig := buildNetworkConfig()

	// Train on CPU
	fmt.Println("\n[2/5] Training on CPU...")
	cpuNet, err := nn.BuildNetworkFromJSON(jsonConfig)
	if err != nil {
		panic(err)
	}
	cpuNet.InitializeWeights()

	startCPU := time.Now()
	lossCPU := trainNetwork(cpuNet, trainData, trainTargets, false)
	timeCPU := time.Since(startCPU)
	fmt.Printf("      CPU Training: Loss=%.6f, Time=%v\n", lossCPU, timeCPU)

	// Save weights for GPU
	initialWeights, _ := cpuNet.SaveModelToString("swiglu_init")

	// Train on GPU
	fmt.Println("\n[3/5] Training on GPU...")
	gpuNet, err := nn.LoadModelFromString(initialWeights, "swiglu_init")
	if err != nil {
		panic(err)
	}
	gpuNet.BatchSize = BatchSize

	startGPU := time.Now()
	lossGPU := trainNetwork(gpuNet, trainData, trainTargets, true)
	timeGPU := time.Since(startGPU)
	fmt.Printf("      GPU Training: Loss=%.6f, Time=%v\n", lossGPU, timeGPU)

	// Performance comparison
	fmt.Println("\n[4/5] Performance Comparison:")
	fmt.Println("╔════════════╦═══════════╦═══════════════╗")
	fmt.Println("║  Device    ║    Loss   ║  Time         ║")
	fmt.Println("╠════════════╬═══════════╬═══════════════╣")
	fmt.Printf("║  CPU       ║  %.6f ║  %12v ║\n", lossCPU, timeCPU)
	fmt.Printf("║  GPU       ║  %.6f ║  %12v ║\n", lossGPU, timeGPU)
	fmt.Println("╚════════════╩═══════════╩═══════════════╝")

	if timeGPU > 0 {
		speedup := float64(timeCPU) / float64(timeGPU)
		fmt.Printf("\n🚀 GPU Speedup: %.2fx\n", speedup)
	}

	// Show examples
	fmt.Println("\n[5/5] Example Approximations:")
	showExampleApproximations(cpuNet)

	// Clean up
	if gpuNet.GPU {
		gpuNet.ReleaseGPUWeights()
	}

	fmt.Println("\n✅ SwiGLU Demo Complete!")
}

func generateFunctionDataset() ([][]float32, []float32) {
	totalSamples := NumFunctions * SamplesPerFunc
	data := make([][]float32, totalSamples)
	targets := make([]float32, totalSamples)

	idx := 0
	for funcType := 0; funcType < NumFunctions; funcType++ {
		for sample := 0; sample < SamplesPerFunc; sample++ {
			x := rand.Float64()*4 - 2
			y := rand.Float64()*4 - 2

			input := []float32{float32(x), float32(y)}
			target := evaluateFunction(funcType, x, y)

			data[idx] = input
			targets[idx] = float32(target)
			idx++
		}
	}

	return data, targets
}

func evaluateFunction(funcType int, x, y float64) float64 {
	switch funcType {
	case FuncSinCos:
		return math.Sin(x) * math.Cos(y)
	case FuncPolynomial:
		return x*x + y*y - x*y
	case FuncGaussian:
		return math.Exp(-x*x-y*y) * math.Sin(x+y)
	case FuncTanhProd:
		return math.Tanh(x*y) + x/2
	}
	return 0
}

func buildNetworkConfig() string {
	return fmt.Sprintf(`{
		"grid_rows": 1,
		"grid_cols": 1,
		"layers_per_cell": 4,
		"batch_size": %d,
		"layers": [
			{
				"type": "dense",
				"input_size": %d,
				"output_size": 64,
				"activation": "tanh"
			},
			{
				"type": "swiglu",
				"input_size": 64,
				"intermediate_size": %d,
				"output_size": 64
			},
			{
				"type": "swiglu",
				"input_size": 64,
				"intermediate_size": %d,
				"output_size": 64
			},
			{
				"type": "dense",
				"input_size": 64,
				"output_size": 1,
				"activation": "tanh"
			}
		]
	}`, BatchSize, InputDim, IntermediateDim, IntermediateDim)
}

func trainNetwork(net *nn.Network, trainData [][]float32, trainTargets []float32, useGPU bool) float64 {
	batches := createBatches(trainData, trainTargets, BatchSize)

	config := &nn.TrainingConfig{
		Epochs:          Epochs,
		LearningRate:    LearningRate,
		UseGPU:          useGPU,
		LossType:        "mse",
		PrintEveryBatch: 0,
		Verbose:         false,
	}

	_, err := net.Train(batches, config)
	if err != nil {
		fmt.Printf("Warning: Training error: %v\n", err)
		return 999999
	}

	if useGPU {
		if err := net.WeightsToCPU(); err != nil {
			fmt.Printf("Warning: Failed to sync weights: %v\n", err)
		}
		net.GPU = false
	}

	return evaluateLoss(net, trainData, trainTargets)
}

func createBatches(data [][]float32, targets []float32, batchSize int) []nn.TrainingBatch {
	indices := rand.Perm(len(data))
	numBatches := len(data) / batchSize
	batches := make([]nn.TrainingBatch, numBatches)

	for b := 0; b < numBatches; b++ {
		input := make([]float32, batchSize*InputDim)
		target := make([]float32, batchSize)

		for i := 0; i < batchSize; i++ {
			idx := indices[b*batchSize+i]
			copy(input[i*InputDim:], data[idx])
			target[i] = targets[idx]
		}

		batches[b] = nn.TrainingBatch{Input: input, Target: target}
	}

	return batches
}

func evaluateLoss(net *nn.Network, data [][]float32, targets []float32) float64 {
	totalLoss := 0.0
	for i, input := range data {
		output, _ := net.ForwardCPU(input)
		diff := output[0] - targets[i]
		totalLoss += float64(diff * diff)
	}
	return totalLoss / float64(len(data))
}

func showExampleApproximations(net *nn.Network) {
	funcNames := []string{"sin(x)*cos(y)", "x²+y²-xy", "exp(-x²-y²)*sin(x+y)", "tanh(xy)+x/2"}

	for f := 0; f < NumFunctions; f++ {
		x, y := 1.0, 0.5
		input := []float32{float32(x), float32(y)}
		target := evaluateFunction(f, x, y)
		output, _ := net.ForwardCPU(input)

		error := math.Abs(float64(output[0]) - target)

		fmt.Printf("  %s at (%.1f, %.1f):\n", funcNames[f], x, y)
		fmt.Printf("    Target: %.6f, Predicted: %.6f, Error: %.6f\n",
			target, output[0], error)
	}
}
