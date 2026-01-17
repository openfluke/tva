package main

import (
	"fmt"
	"math"
	"math/rand"
	"time"

	"github.com/openfluke/loom/nn"
)

// Conv1D Audio Waveform Classification Demo
// Demonstrates Conv1D layers on audio classification task

const (
	SampleRate      = 128 // Samples per waveform
	NumClasses      = 4   // Sine, Square, Sawtooth, Triangle
	SamplesPerClass = 250 // Training samples per class
	Epochs          = 50
	LearningRate    = 0.01
	BatchSize       = 20
)

const (
	WaveformSine = iota
	WaveformSquare
	WaveformSawtooth
	WaveformTriangle
)

func main() {
	rand.Seed(time.Now().UnixNano())

	fmt.Println("╔════════════════════════════════════════════════════════════════╗")
	fmt.Println("║   Conv1D Demo: Audio Waveform Classification                   ║")
	fmt.Println("╚════════════════════════════════════════════════════════════════╝")

	// Generate dataset
	fmt.Println("\n[1/5] Generating synthetic audio waveforms...")
	trainData, trainLabels := generateWaveformDataset()
	fmt.Printf("      Generated %d waveforms (%d samples each)\n", len(trainData), SampleRate)

	// Build network from JSON
	fmt.Println("\n[2/5] Building network from JSON config...")
	jsonConfig := buildNetworkConfig()

	// Train on CPU
	fmt.Println("\n[3/5] Training on CPU...")
	cpuNet, err := nn.BuildNetworkFromJSON(jsonConfig)
	if err != nil {
		panic(err)
	}
	cpuNet.InitializeWeights()

	startCPU := time.Now()
	accCPU := trainNetwork(cpuNet, trainData, trainLabels, false)
	timeCPU := time.Since(startCPU)
	fmt.Printf("      CPU Training: Accuracy=%.2f%%, Time=%v\n", accCPU*100, timeCPU)

	// Save initial weights for GPU training
	initialWeights, _ := cpuNet.SaveModelToString("conv1d_init")

	// Train on GPU
	fmt.Println("\n[4/5] Training on GPU...")
	gpuNet, err := nn.LoadModelFromString(initialWeights, "conv1d_init")
	if err != nil {
		panic(err)
	}
	gpuNet.BatchSize = BatchSize

	startGPU := time.Now()
	accGPU := trainNetwork(gpuNet, trainData, trainLabels, true)
	timeGPU := time.Since(startGPU)
	fmt.Printf("      GPU Training: Accuracy=%.2f%%, Time=%v\n", accGPU*100, timeGPU)

	// Performance comparison
	fmt.Println("\n[5/5] Performance Comparison:")
	fmt.Println("╔════════════╦════════════╦═══════════════╗")
	fmt.Println("║  Device    ║  Accuracy  ║  Time         ║")
	fmt.Println("╠════════════╬════════════╬═══════════════╣")
	fmt.Printf("║  CPU       ║  %6.2f%%   ║  %12v ║\n", accCPU*100, timeCPU)
	fmt.Printf("║  GPU       ║  %6.2f%%   ║  %12v ║\n", accGPU*100, timeGPU)
	fmt.Println("╚════════════╩════════════╩═══════════════╝")

	if timeGPU > 0 {
		speedup := float64(timeCPU) / float64(timeGPU)
		fmt.Printf("\n🚀 GPU Speedup: %.2fx\n", speedup)
	}

	// Clean up
	if gpuNet.GPU {
		gpuNet.ReleaseGPUWeights()
	}

	// Verify save/reload
	fmt.Println("\n[Bonus] Verifying save/reload consistency...")
	verifySaveReload(cpuNet, trainData[:10], trainLabels[:10])

	fmt.Println("\n✅ Conv1D Demo Complete!")
}

func generateWaveformDataset() ([][]float32, []int) {
	totalSamples := NumClasses * SamplesPerClass
	data := make([][]float32, totalSamples)
	labels := make([]int, totalSamples)

	idx := 0
	for class := 0; class < NumClasses; class++ {
		for sample := 0; sample < SamplesPerClass; sample++ {
			waveform := generateWaveform(class)
			data[idx] = waveform
			labels[idx] = class
			idx++
		}
	}

	return data, labels
}

func generateWaveform(waveformType int) []float32 {
	waveform := make([]float32, SampleRate)

	// Random frequency between 1 and 5 Hz
	frequency := 1.0 + rand.Float64()*4.0

	// Add slight noise
	noiseLevel := 0.1

	for i := 0; i < SampleRate; i++ {
		t := float64(i) / float64(SampleRate)
		phase := 2.0 * math.Pi * frequency * t

		var value float64
		switch waveformType {
		case WaveformSine:
			value = math.Sin(phase)

		case WaveformSquare:
			if math.Sin(phase) >= 0 {
				value = 1.0
			} else {
				value = -1.0
			}

		case WaveformSawtooth:
			// Sawtooth from -1 to 1
			value = 2.0 * (frequency*t - math.Floor(frequency*t+0.5))

		case WaveformTriangle:
			// Triangle wave
			sawValue := 2.0 * (frequency*t - math.Floor(frequency*t+0.5))
			value = 2.0*math.Abs(sawValue) - 1.0
		}

		// Add noise
		noise := (rand.Float64()*2 - 1) * noiseLevel
		waveform[i] = float32(value + noise)
	}

	return waveform
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
				"output_size": %d,
				"activation": "tanh"
			},
			{
				"type": "conv1d",
				"input_channels": 1,
				"filters": 32,
				"kernel_size": 5,
				"stride": 2,
				"padding": 0,
				"input_length": %d,
				"activation": "scaled_relu"
			},
			{
				"type": "conv1d",
				"input_channels": 32,
				"filters": 64,
				"kernel_size": 3,
				"stride": 2,
				"padding": 0,
				"input_length": 62,
				"activation": "scaled_relu"
			},
			{
				"type": "dense",
				"input_size": 1920,
				"output_size": %d,
				"activation": "sigmoid"
			}
		]
	}`, BatchSize, SampleRate, SampleRate, SampleRate, NumClasses)
}

func trainNetwork(net *nn.Network, trainData [][]float32, trainLabels []int, useGPU bool) float64 {
	// Create batches
	batches := createBatches(trainData, trainLabels, BatchSize)

	config := &nn.TrainingConfig{
		Epochs:          Epochs,
		LearningRate:    LearningRate,
		UseGPU:          useGPU,
		LossType:        "cross_entropy",
		PrintEveryBatch: 0,
		Verbose:         false,
	}

	// Train
	_, err := net.Train(batches, config)
	if err != nil {
		fmt.Printf("Warning: Training error: %v\n", err)
		return 0
	}

	// Sync weights back if GPU
	if useGPU {
		if err := net.WeightsToCPU(); err != nil {
			fmt.Printf("Warning: Failed to sync weights: %v\n", err)
		}
		net.GPU = false
	}

	// Evaluate
	return evaluateAccuracy(net, trainData, trainLabels)
}

func createBatches(data [][]float32, labels []int, batchSize int) []nn.TrainingBatch {
	// Shuffle
	indices := rand.Perm(len(data))
	numBatches := len(data) / batchSize
	batches := make([]nn.TrainingBatch, numBatches)

	for b := 0; b < numBatches; b++ {
		input := make([]float32, batchSize*SampleRate)
		target := make([]float32, batchSize*NumClasses)

		for i := 0; i < batchSize; i++ {
			idx := indices[b*batchSize+i]
			copy(input[i*SampleRate:], data[idx])
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

func verifySaveReload(net *nn.Network, testData [][]float32, testLabels []int) {
	modelID := "conv1d_audio_model"

	// Save model
	jsonString, err := net.SaveModelToString(modelID)
	if err != nil {
		fmt.Printf("❌ Save failed: %v\n", err)
		return
	}

	// Reload model
	reloadedNet, err := nn.LoadModelFromString(jsonString, modelID)
	if err != nil {
		fmt.Printf("❌ Reload failed: %v\n", err)
		return
	}

	// Compare outputs
	maxDiff := 0.0
	for _, input := range testData {
		originalOut, _ := net.ForwardCPU(input)
		reloadedOut, _ := reloadedNet.ForwardCPU(input)

		for j := range originalOut {
			diff := math.Abs(float64(originalOut[j] - reloadedOut[j]))
			if diff > maxDiff {
				maxDiff = diff
			}
		}
	}

	if maxDiff < 1e-6 {
		fmt.Printf("✅ Save/Reload verified (max diff: %.2e)\n", maxDiff)
	} else {
		fmt.Printf("⚠️  Precision loss detected (max diff: %.2e)\n", maxDiff)
	}
}
