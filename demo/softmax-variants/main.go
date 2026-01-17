package main

import (
	"fmt"
	"math/rand"
	"time"

	"github.com/openfluke/loom/nn"
)

// Softmax Variants Demo
// Demonstrates different Softmax variants on multi-class classification

const (
	InputDim        = 2 // 2D for visualization
	NumClasses      = 4 // 4 classes
	SamplesPerClass = 250
	Epochs          = 50
	LearningRate    = 0.01
	BatchSize       = 20
)

func main() {
	rand.Seed(time.Now().UnixNano())

	fmt.Println("╔════════════════════════════════════════════════════════════════╗")
	fmt.Println("║   Softmax Demo: Multi-Class Classification Variants           ║")
	fmt.Println("╚════════════════════════════════════════════════════════════════╝")

	// Generate dataset (CPU only for simplicity)
	fmt.Println("\n[1/2] Generating synthetic classification data...")
	trainData, trainLabels := generateClassificationDataset()
	fmt.Printf("      Generated %d samples in %d classes\n", len(trainData), NumClasses)

	// Test Standard Softmax
	fmt.Println("\n[2/2] Testing Softmax Variants...")
	variants := []struct {
		name   string
		config string
	}{
		{"Standard", "\"soft max_variant\": \"standard\""},
		{"Temp=0.5 (Sharp)", "\"softmax_variant\": \"temperature\", \"temperature\": 0.5"},
		{"Temp=2.0 (Smooth)", "\"softmax_variant\": \"temperature\", \"temperature\": 2.0"},
	}

	results := make([]float64, len(variants))

	for i, variant := range variants {
		fmt.Printf("\n  Training with %s...\n", variant.name)
		jsonConfig := buildNetworkConfig(variant.config)
		net, err := nn.BuildNetworkFromJSON(jsonConfig)
		if err != nil {
			panic(err)
		}
		net.InitializeWeights()

		results[i] = trainNetwork(net, trainData, trainLabels)
	}

	// Comparison table
	fmt.Println("\n╔═══════════════════════╦════════════╗")
	fmt.Println("║  Softmax Variant      ║  Accuracy  ║")
	fmt.Println("╠═══════════════════════╬════════════╣")
	for i, variant := range variants {
		fmt.Printf("║  %-20s ║  %6.2f%%   ║\n", variant.name, results[i]*100)
	}
	fmt.Println("╚═══════════════════════╩════════════╝")

	fmt.Println("\n✅ Softmax Variants Demo Complete!")
	fmt.Println("\nKey Insights:")
	fmt.Println("  • T=0.5 (low temp): Sharper, more confident predictions")
	fmt.Println("  • T=2.0 (high temp): Smoother, less confident predictions")
	fmt.Println("  • Standard (T=1.0): Balanced behavior")
}

func generateClassificationDataset() ([][]float32, []int) {
	totalSamples := NumClasses * SamplesPerClass
	data := make([][]float32, totalSamples)
	labels := make([]int, totalSamples)

	idx := 0
	for class := 0; class < NumClasses; class++ {
		// Generate clusters
		centerX := rand.Float64()*4 - 2
		centerY := rand.Float64()*4 - 2

		for sample := 0; sample < SamplesPerClass; sample++ {
			noise := 0.5
			x := centerX + (rand.Float64()*2-1)*noise
			y := centerY + (rand.Float64()*2-1)*noise

			data[idx] = []float32{float32(x), float32(y)}
			labels[idx] = class
			idx++
		}
	}

	return data, labels
}

func buildNetworkConfig(softmaxConfig string) string {
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
				"type": "dense",
				"input_size": 64,
				"output_size": 64,
				"activation": "scaled_relu"
			},
			{
				"type": "dense",
				"input_size": 64,
				"output_size": %d,
				"activation": "tanh"
			},
			{
				"type": "softmax",
				%s
			}
		]
	}`, BatchSize, InputDim, NumClasses, softmaxConfig)
}

func trainNetwork(net *nn.Network, trainData [][]float32, trainLabels []int) float64 {
	batches := createBatches(trainData, trainLabels, BatchSize)

	config := &nn.TrainingConfig{
		Epochs:          Epochs,
		LearningRate:    LearningRate,
		UseGPU:          false, // CPU only for softmax demo
		LossType:        "cross_entropy",
		PrintEveryBatch: 0,
		Verbose:         false,
	}

	_, err := net.Train(batches, config)
	if err != nil {
		fmt.Printf("    Warning: Training error: %v\n", err)
		return 0
	}

	return evaluateAccuracy(net, trainData, trainLabels)
}

func createBatches(data [][]float32, labels []int, batchSize int) []nn.TrainingBatch {
	indices := rand.Perm(len(data))
	numBatches := len(data) / batchSize
	batches := make([]nn.TrainingBatch, numBatches)

	for b := 0; b < numBatches; b++ {
		input := make([]float32, batchSize*InputDim)
		target := make([]float32, batchSize*NumClasses)

		for i := 0; i < batchSize; i++ {
			idx := indices[b*batchSize+i]
			copy(input[i*InputDim:], data[idx])
			target[i*NumClasses+labels[idx]] = 1.0
		}

		batches[b] = nn.TrainingBatch{Input: input, Target: target}
	}

	return batches
}

func evaluateAccuracy(net *nn.Network, data [][]float32, labels []int) float64 {
	correct := 0
	for i, input := range data {
		output, _ := net.ForwardCPU(input)
		predicted := argmax(output)
		if predicted == labels[i] {
			correct++
		}
	}
	return float64(correct) / float64(len(data))
}

func argmax(vec []float32) int {
	maxIdx := 0
	maxVal := vec[0]
	for i, val := range vec {
		if val > maxVal {
			maxIdx = i
			maxVal = val
		}
	}
	return maxIdx
}
