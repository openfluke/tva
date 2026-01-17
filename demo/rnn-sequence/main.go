package main

import (
	"fmt"
	"math"
	"math/rand"
	"time"

	"github.com/openfluke/loom/nn"
)

// RNN Sequence Prediction Demo
// Demonstrates RNN layers on sequence pattern prediction

const (
	SequenceLength    = 10 // Look at 10 steps to predict
	NumPatterns       = 4  // Arithmetic, Geometric, Fibonacci, Alternating
	SamplesPerPattern = 250
	Epochs            = 100
	LearningRate      = 0.005
	HiddenSize        = 32
	BatchSize         = 20
)

const (
	PatternArithmetic = iota
	PatternGeometric
	PatternFibonacci
	PatternAlternating
)

func main() {
	rand.Seed(time.Now().UnixNano())

	fmt.Println("╔════════════════════════════════════════════════════════════════╗")
	fmt.Println("║   RNN Demo: Sequence Pattern Prediction                       ║")
	fmt.Println("╚════════════════════════════════════════════════════════════════╝")

	// Generate dataset
	fmt.Println("\n[1/5] Generating sequence patterns...")
	trainData, trainTargets := generateSequenceDataset()
	fmt.Printf("      Generated %d sequences (length %d each)\n", len(trainData), SequenceLength)

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
	initialWeights, _ := cpuNet.SaveModelToString("rnn_init")

	// Train on GPU
	fmt.Println("\n[3/5] Training on GPU...")
	gpuNet, err := nn.LoadModelFromString(initialWeights, "rnn_init")
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
	fmt.Println("\n[5/5] Example Predictions:")
	showExamplePredictions(cpuNet)

	// Clean up
	if gpuNet.GPU {
		gpuNet.ReleaseGPUWeights()
	}

	fmt.Println("\n✅ RNN Demo Complete!")
}

func generateSequenceDataset() ([][]float32, []float32) {
	totalSamples := NumPatterns * SamplesPerPattern
	data := make([][]float32, totalSamples)
	targets := make([]float32, totalSamples)

	idx := 0
	for pattern := 0; pattern < NumPatterns; pattern++ {
		for sample := 0; sample < SamplesPerPattern; sample++ {
			seq, target := generateSequence(pattern)
			data[idx] = seq
			targets[idx] = target
			idx++
		}
	}

	return data, targets
}

func generateSequence(patternType int) ([]float32, float32) {
	sequence := make([]float32, SequenceLength)

	switch patternType {
	case PatternArithmetic:
		start := rand.Float64()*10 - 5
		diff := rand.Float64()*4 - 2
		for i := 0; i < SequenceLength; i++ {
			sequence[i] = float32(start + float64(i)*diff)
		}
		nextValue := float32(start + float64(SequenceLength)*diff)
		return sequence, nextValue

	case PatternGeometric:
		start := rand.Float64()*2 + 0.5
		ratio := rand.Float64()*0.6 + 0.7
		for i := 0; i < SequenceLength; i++ {
			sequence[i] = float32(start * math.Pow(ratio, float64(i)))
		}
		nextValue := float32(start * math.Pow(ratio, float64(SequenceLength)))
		return sequence, nextValue

	case PatternFibonacci:
		a := rand.Float64()*2 - 1
		b := rand.Float64()*2 - 1
		sequence[0] = float32(a)
		if SequenceLength > 1 {
			sequence[1] = float32(b)
		}
		for i := 2; i < SequenceLength; i++ {
			sequence[i] = sequence[i-1] + sequence[i-2]
		}
		nextValue := sequence[SequenceLength-1] + sequence[SequenceLength-2]
		return sequence, nextValue

	case PatternAlternating:
		base := rand.Float64()*2 - 1
		trend := rand.Float64()*0.5 - 0.25
		amplitude := rand.Float64()*3 + 1
		for i := 0; i < SequenceLength; i++ {
			alt := amplitude * math.Pow(-1, float64(i))
			sequence[i] = float32(base + float64(i)*trend + alt)
		}
		alt := amplitude * math.Pow(-1, float64(SequenceLength))
		nextValue := float32(base + float64(SequenceLength)*trend + alt)
		return sequence, nextValue
	}

	return sequence, 0
}

func buildNetworkConfig() string {
	return fmt.Sprintf(`{
		"grid_rows": 1,
		"grid_cols": 1,
		"layers_per_cell": 2,
		"batch_size": %d,
		"layers": [
			{
				"type": "rnn",
				"input_size": 1,
				"hidden_size": %d,
				"seq_length": %d,
				"activation": "tanh"
			},
			{
				"type": "dense",
				"input_size": %d,
				"output_size": 1,
				"activation": "tanh"
			}
		]
	}`, BatchSize, HiddenSize, SequenceLength, SequenceLength*HiddenSize)
}

func trainNetwork(net *nn.Network, trainData [][]float32, trainTargets []float32, useGPU bool) float64 {
	// Create batches
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
		input := make([]float32, batchSize*SequenceLength)
		target := make([]float32, batchSize)

		for i := 0; i < batchSize; i++ {
			idx := indices[b*batchSize+i]
			copy(input[i*SequenceLength:], data[idx])
			target[i] = targets[idx]
		}

		batches[b] = nn.TrainingBatch{Input: input, Target: target}
	}

	return batches
}

func evaluateLoss(net *nn.Network, data [][]float32, targets []float32) float64 {
	originalBatchSize := net.BatchSize
	net.BatchSize = 1
	defer func() { net.BatchSize = originalBatchSize }()

	totalLoss := 0.0
	for i, input := range data {
		output, _ := net.ForwardCPU(input)
		diff := output[0] - targets[i]
		totalLoss += float64(diff * diff)
	}
	return totalLoss / float64(len(data))
}

func showExamplePredictions(net *nn.Network) {
	originalBatchSize := net.BatchSize
	net.BatchSize = 1
	defer func() { net.BatchSize = originalBatchSize }()

	patterns := []string{"Arithmetic", "Geometric", "Fibonacci", "Alternating"}

	for p := 0; p < min(NumPatterns, 4); p++ {
		seq, target := generateSequence(p)
		output, _ := net.ForwardCPU(seq)
		prediction := output[0]

		fmt.Printf("  %s: [", patterns[p])
		for i := 0; i < min(5, len(seq)); i++ {
			fmt.Printf("%.2f ", seq[i])
		}
		fmt.Printf("...] → Target: %.2f, Predicted: %.2f (Error: %.2f)\n",
			target, prediction, math.Abs(float64(target-prediction)))
	}
}

func min(a, b int) int {
	if a < b {
		return a
	}
	return b
}
