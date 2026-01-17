package main

import (
	"fmt"
	"math"
	"math/rand"
	"time"

	"github.com/openfluke/loom/nn"
)

// LSTM Time Series Forecasting Demo
// Demonstrates LSTM layers on time series prediction

const (
	WindowSize      = 20 // Look at 20 time steps
	ForecastHorizon = 5  // Predict next 5 steps
	NumTSTypes      = 4  // Seasonal, Trend, MultiFreq, Noisy
	SamplesPerType  = 200
	Epochs          = 80
	LearningRate    = 0.003
	HiddenSize      = 64
	BatchSize       = 20
)

const (
	TSTypeSeasonal = iota
	TSTypeTrend
	TSTypeMultiFreq
	TSTypeNoisy
)

func main() {
	rand.Seed(time.Now().UnixNano())

	fmt.Println("╔════════════════════════════════════════════════════════════════╗")
	fmt.Println("║   LSTM Demo: Time Series Forecasting                          ║")
	fmt.Println("╚════════════════════════════════════════════════════════════════╝")

	// Generate dataset
	fmt.Println("\n[1/5] Generating time series patterns...")
	trainData, trainTargets := generateTimeSeriesDataset()
	fmt.Printf("      Generated %d time series (window=%d, forecast=%d)\n",
		len(trainData), WindowSize, ForecastHorizon)

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
	initialWeights, _ := cpuNet.SaveModelToString("lstm_init")

	// Train on GPU
	fmt.Println("\n[3/5] Training on GPU...")
	gpuNet, err := nn.LoadModelFromString(initialWeights, "lstm_init")
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
	fmt.Println("\n[5/5] Example Forecasts:")
	showExampleForecasts(cpuNet)

	// Clean up
	if gpuNet.GPU {
		gpuNet.ReleaseGPUWeights()
	}

	fmt.Println("\n✅ LSTM Demo Complete!")
}

func generateTimeSeriesDataset() ([][]float32, [][]float32) {
	totalSamples := NumTSTypes * SamplesPerType
	data := make([][]float32, totalSamples)
	targets := make([][]float32, totalSamples)

	idx := 0
	for tsType := 0; tsType < NumTSTypes; tsType++ {
		for sample := 0; sample < SamplesPerType; sample++ {
			window, forecast := generateTS(tsType)
			data[idx] = window
			targets[idx] = forecast
			idx++
		}
	}

	return data, targets
}

func generateTS(tsType int) ([]float32, []float32) {
	window := make([]float32, WindowSize)
	forecast := make([]float32, ForecastHorizon)

	totalLength := WindowSize + ForecastHorizon
	fullSeries := make([]float64, totalLength)

	switch tsType {
	case TSTypeSeasonal:
		frequency := rand.Float64()*0.2 + 0.1
		amplitude := rand.Float64()*3 + 1
		phase := rand.Float64() * 2 * math.Pi

		for i := 0; i < totalLength; i++ {
			fullSeries[i] = amplitude * math.Sin(2*math.Pi*frequency*float64(i)+phase)
			fullSeries[i] += (rand.Float64()*2 - 1) * 0.1
		}

	case TSTypeTrend:
		start := rand.Float64()*10 - 5
		linear := rand.Float64()*0.5 - 0.25
		quadratic := rand.Float64()*0.02 - 0.01

		for i := 0; i < totalLength; i++ {
			t := float64(i)
			fullSeries[i] = start + linear*t + quadratic*t*t
			fullSeries[i] += (rand.Float64()*2 - 1) * 0.3
		}

	case TSTypeMultiFreq:
		f1 := rand.Float64()*0.15 + 0.05
		f2 := rand.Float64()*0.3 + 0.15
		a1 := rand.Float64()*2 + 0.5
		a2 := rand.Float64()*1.5 + 0.5

		for i := 0; i < totalLength; i++ {
			t := float64(i)
			fullSeries[i] = a1*math.Sin(2*math.Pi*f1*t) + a2*math.Sin(2*math.Pi*f2*t)
			fullSeries[i] += (rand.Float64()*2 - 1) * 0.15
		}

	case TSTypeNoisy:
		base := rand.Float64()*5 - 2.5
		drift := rand.Float64()*0.1 - 0.05

		for i := 0; i < totalLength; i++ {
			fullSeries[i] = base + float64(i)*drift
			fullSeries[i] += (rand.Float64()*2 - 1) * 2.0
		}
	}

	for i := 0; i < WindowSize; i++ {
		window[i] = float32(fullSeries[i])
	}
	for i := 0; i < ForecastHorizon; i++ {
		forecast[i] = float32(fullSeries[WindowSize+i])
	}

	return window, forecast
}

func buildNetworkConfig() string {
	return fmt.Sprintf(`{
		"grid_rows": 1,
		"grid_cols": 1,
		"layers_per_cell": 2,
		"batch_size": %d,
		"layers": [
			{
				"type": "lstm",
				"input_size": 1,
				"hidden_size": %d,
				"seq_length": %d,
				"activation": "tanh"
			},
			{
				"type": "dense",
				"input_size": %d,
				"output_size": %d,
				"activation": "tanh"
			}
		]
	}`, BatchSize, HiddenSize, WindowSize, WindowSize*HiddenSize, ForecastHorizon)
}

func trainNetwork(net *nn.Network, trainData [][]float32, trainTargets [][]float32, useGPU bool) float64 {
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

func createBatches(data [][]float32, targets [][]float32, batchSize int) []nn.TrainingBatch {
	indices := rand.Perm(len(data))
	numBatches := len(data) / batchSize
	batches := make([]nn.TrainingBatch, numBatches)

	for b := 0; b < numBatches; b++ {
		input := make([]float32, batchSize*WindowSize)
		target := make([]float32, batchSize*ForecastHorizon)

		for i := 0; i < batchSize; i++ {
			idx := indices[b*batchSize+i]
			copy(input[i*WindowSize:], data[idx])
			copy(target[i*ForecastHorizon:], targets[idx])
		}

		batches[b] = nn.TrainingBatch{Input: input, Target: target}
	}

	return batches
}

func evaluateLoss(net *nn.Network, data [][]float32, targets [][]float32) float64 {
	// Ideally processed in batches, but for simple eval loop:
	originalBatchSize := net.BatchSize
	net.BatchSize = 1
	defer func() { net.BatchSize = originalBatchSize }()

	totalLoss := 0.0
	for i, input := range data {
		output, _ := net.ForwardCPU(input)

		for j := 0; j < ForecastHorizon; j++ {
			diff := output[j] - targets[i][j]
			totalLoss += float64(diff * diff)
		}
	}
	return totalLoss / float64(len(data))
}

func showExampleForecasts(net *nn.Network) {
	// Ensure single item batch size for inference
	originalBatchSize := net.BatchSize
	net.BatchSize = 1
	defer func() { net.BatchSize = originalBatchSize }()

	tsTypes := []string{"Seasonal", "Trend", "MultiFreq", "Noisy"}

	for ts := 0; ts < NumTSTypes; ts++ {
		window, target := generateTS(ts)
		output, _ := net.ForwardCPU(window)

		totalError := 0.0
		for i := 0; i < ForecastHorizon; i++ {
			totalError += math.Abs(float64(output[i] - target[i]))
		}
		avgError := totalError / float64(ForecastHorizon)

		fmt.Printf("  %s:\n", tsTypes[ts])
		fmt.Printf("    Window: [%.2f %.2f %.2f ... %.2f %.2f]\n",
			window[0], window[1], window[2], window[WindowSize-2], window[WindowSize-1])
		fmt.Printf("    Target:  [%.2f %.2f %.2f %.2f %.2f]\n",
			target[0], target[1], target[2], target[3], target[4])
		fmt.Printf("    Predicted: [%.2f %.2f %.2f %.2f %.2f]\n",
			output[0], output[1], output[2], output[3], output[4])
		fmt.Printf("    Avg Error: %.4f\n", avgError)
	}
}
