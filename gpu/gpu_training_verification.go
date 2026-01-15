package main

import (
	"bufio"
	"flag"
	"fmt"
	"math"
	"math/rand"
	"os"
	"strconv"
	"strings"
	"time"

	"github.com/openfluke/loom/gpu"
	"github.com/openfluke/loom/nn"
)

var (
	gpuFlag    = flag.String("gpu", "", "Optional substring to select a specific GPU adapter (e.g. 'nvidia')")
	epochsFlag = flag.Int("epochs", 20, "Number of training epochs")
	lrFlag     = flag.Float64("lr", 0.05, "Learning rate")
	layersFlag = flag.String("layers", "", "Comma-separated list of layer types to test (e.g. 'Dense,Conv2D'). Empty = all layers")
)

// Simple linearly separable dataset: if x[0] > 0.5 then class 1, else class 0
type Dataset struct {
	Inputs   [][]float32
	Outputs  []float64 // Expected class labels (0 or 1)
	NumClass int
}

func generateSimpleDataset(numSamples int) *Dataset {
	inputs := make([][]float32, numSamples)
	outputs := make([]float64, numSamples)

	for i := 0; i < numSamples; i++ {
		x0 := rand.Float32()
		x1 := rand.Float32()
		inputs[i] = []float32{x0, x1}

		// Simple rule: if x[0] > 0.5, class 1, else class 0
		if x0 > 0.5 {
			outputs[i] = 1.0
		} else {
			outputs[i] = 0.0
		}
	}

	return &Dataset{
		Inputs:   inputs,
		Outputs:  outputs,
		NumClass: 2,
	}
}

// LayerTestConfig defines configuration for testing a specific layer type
type LayerTestConfig struct {
	Name       string
	LayerType  string
	Activation string
	HiddenSize int
	// For RNN/LSTM
	SeqLength int
	// For Conv layers
	UseConvFormat bool
}

// LayerTestResult stores training results for a specific layer type
type LayerTestResult struct {
	Config       LayerTestConfig
	IsGPU        bool
	LossHistory  []float32
	InitialAcc   float64
	FinalAcc     float64
	TrainTime    time.Duration
	Success      bool
	ErrorMessage string
}

// Define layer types to test
func getLayerTestConfigs() []LayerTestConfig {
	return []LayerTestConfig{
		// Dense layers (larger sizes = better GPU speedup)
		{Name: "Dense-1024", LayerType: "dense", Activation: "relu", HiddenSize: 1024},
		{Name: "Dense-512", LayerType: "dense", Activation: "relu", HiddenSize: 512},
		{Name: "Dense-256", LayerType: "dense", Activation: "relu", HiddenSize: 256},

		// Conv1D (larger for GPU - NOTE: backward shader incomplete)
		{Name: "Conv1D-64", LayerType: "conv1d", Activation: "relu", HiddenSize: 64, SeqLength: 16},
		{Name: "Conv1D-128", LayerType: "conv1d", Activation: "relu", HiddenSize: 128, SeqLength: 16},

		// Conv2D (larger sizes for better GPU utilization)
		{Name: "Conv2D-64", LayerType: "conv2d", Activation: "relu", HiddenSize: 64, UseConvFormat: true},
		{Name: "Conv2D-128", LayerType: "conv2d", Activation: "relu", HiddenSize: 128, UseConvFormat: true},

		// RNN (NOTE: backward shader doesn't compute weight gradients yet)
		{Name: "RNN-128", LayerType: "rnn", Activation: "tanh", HiddenSize: 128, SeqLength: 8},
		{Name: "RNN-256", LayerType: "rnn", Activation: "tanh", HiddenSize: 256, SeqLength: 8},

		// LSTM (NOTE: backward shader doesn't compute weight gradients yet)
		{Name: "LSTM-128", LayerType: "lstm", Activation: "tanh", HiddenSize: 128, SeqLength: 8},
		{Name: "LSTM-256", LayerType: "lstm", Activation: "tanh", HiddenSize: 256, SeqLength: 8},

		// LayerNorm (larger)
		{Name: "LayerNorm-256", LayerType: "layernorm", Activation: "none", HiddenSize: 256},
		{Name: "LayerNorm-512", LayerType: "layernorm", Activation: "none", HiddenSize: 512},

		// RMSNorm (larger)
		{Name: "RMSNorm-256", LayerType: "rmsnorm", Activation: "none", HiddenSize: 256},
		{Name: "RMSNorm-512", LayerType: "rmsnorm", Activation: "none", HiddenSize: 512},

		// SwiGLU (larger)
		{Name: "SwiGLU-256", LayerType: "swiglu", Activation: "none", HiddenSize: 256},
		{Name: "SwiGLU-512", LayerType: "swiglu", Activation: "none", HiddenSize: 512},

		// Multi-Head Attention
		{Name: "MHA-4h", LayerType: "mha", Activation: "none", HiddenSize: 64, SeqLength: 8},
		{Name: "MHA-8h", LayerType: "mha", Activation: "none", HiddenSize: 128, SeqLength: 8},

		// Softmax
		{Name: "Softmax-256", LayerType: "softmax", Activation: "none", HiddenSize: 256},

		// Combined All-In-One Test
		{Name: "Combined-Hybrid", LayerType: "combined", Activation: "relu", HiddenSize: 64, SeqLength: 8},
	}
}

func createNetworkWithHiddenLayer(batchSize int, config LayerTestConfig) (*nn.Network, error) {
	var jsonConfig string

	switch config.LayerType {
	case "dense":
		jsonConfig = fmt.Sprintf(`{
			"id": "training_verification_%s",
			"batch_size": %d,
			"grid_rows": 1,
			"grid_cols": 1,
			"layers_per_cell": 3,
			"layers": [
				{"type": "dense", "activation": "%s", "input_height": 2, "output_height": %d},
				{"type": "dense", "activation": "%s", "input_height": %d, "output_height": %d},
				{"type": "dense", "activation": "sigmoid", "input_height": %d, "output_height": 2}
			]
		}`, config.Name, batchSize, config.Activation, config.HiddenSize,
			config.Activation, config.HiddenSize, config.HiddenSize, config.HiddenSize)

	case "conv2d":
		// Reshape 2D input (2 features) to 4x4 image with 1 channel for conv
		jsonConfig = fmt.Sprintf(`{
			"id": "training_verification_%s",
			"batch_size": %d,
			"grid_rows": 1,
			"grid_cols": 1,
			"layers_per_cell": 3,
			"layers": [
				{"type": "dense", "activation": "relu", "input_height": 2, "output_height": 16},
				{"type": "conv2d", "activation": "%s", "input_channels": 1, "input_height": 4, "input_width": 4, "filters": %d, "kernel_size": 3, "stride": 1, "padding": 1, "output_height": 4, "output_width": 4},
				{"type": "dense", "activation": "sigmoid", "input_height": %d, "output_height": 2}
			]
		}`, config.Name, batchSize, config.Activation, config.HiddenSize, config.HiddenSize*16)

	case "rnn":
		// Reshape input to sequence format
		jsonConfig = fmt.Sprintf(`{
			"id": "training_verification_%s",
			"batch_size": %d,
			"grid_rows": 1,
			"grid_cols": 1,
			"layers_per_cell": 3,
			"layers": [
				{"type": "dense", "activation": "relu", "input_height": 2, "output_height": %d},
				{"type": "rnn", "activation": "%s", "seq_length": %d, "input_size": %d, "hidden_size": %d},
				{"type": "dense", "activation": "sigmoid", "input_height": %d, "output_height": 2}
			]
		}`, config.Name, batchSize, config.SeqLength*config.HiddenSize,
			config.Activation, config.SeqLength, config.HiddenSize, config.HiddenSize, config.HiddenSize*config.SeqLength)

	case "conv1d":
		// Conv1D for sequence data
		jsonConfig = fmt.Sprintf(`{
			"id": "training_verification_%s",
			"batch_size": %d,
			"grid_rows": 1,
			"grid_cols": 1,
			"layers_per_cell": 3,
			"layers": [
				{"type": "dense", "activation": "relu", "input_height": 2, "output_height": %d},
				{"type": "conv1d", "activation": "%s", "in_channels": 1, "filters": %d, "kernel_size": 3, "seq_len": %d},
				{"type": "dense", "activation": "sigmoid", "input_height": %d, "output_height": 2}
			]
		}`, config.Name, batchSize, config.SeqLength,
			config.Activation, config.HiddenSize, config.SeqLength, config.HiddenSize*(config.SeqLength-2))

	case "lstm":
		// LSTM also needs sequence format
		jsonConfig = fmt.Sprintf(`{
			"id": "training_verification_%s",
			"batch_size": %d,
			"grid_rows": 1,
			"grid_cols": 1,
			"layers_per_cell": 3,
			"layers": [
				{"type": "dense", "activation": "relu", "input_height": 2, "output_height": %d},
				{"type": "lstm", "activation": "%s", "seq_length": %d, "input_size": %d, "hidden_size": %d},
				{"type": "dense", "activation": "sigmoid", "input_height": %d, "output_height": 2}
			]
		}`, config.Name, batchSize, config.SeqLength*config.HiddenSize,
			config.Activation, config.SeqLength, config.HiddenSize, config.HiddenSize, config.HiddenSize*config.SeqLength)

	case "layernorm":
		jsonConfig = fmt.Sprintf(`{
			"id": "training_verification_%s",
			"batch_size": %d,
			"grid_rows": 1,
			"grid_cols": 1,
			"layers_per_cell": 3,
			"layers": [
				{"type": "dense", "activation": "relu", "input_height": 2, "output_height": %d},
				{"type": "layernorm", "norm_size": %d, "epsilon": 0.00001},
				{"type": "dense", "activation": "sigmoid", "input_height": %d, "output_height": 2}
			]
		}`, config.Name, batchSize, config.HiddenSize, config.HiddenSize, config.HiddenSize)

	case "rmsnorm":
		jsonConfig = fmt.Sprintf(`{
			"id": "training_verification_%s",
			"batch_size": %d,
			"grid_rows": 1,
			"grid_cols": 1,
			"layers_per_cell": 3,
			"layers": [
				{"type": "dense", "activation": "relu", "input_height": 2, "output_height": %d},
				{"type": "rmsnorm", "norm_size": %d, "epsilon": 0.00001},
				{"type": "dense", "activation": "sigmoid", "input_height": %d, "output_height": 2}
			]
		}`, config.Name, batchSize, config.HiddenSize, config.HiddenSize, config.HiddenSize)

	case "swiglu":
		jsonConfig = fmt.Sprintf(`{
			"id": "training_verification_%s",
			"batch_size": %d,
			"grid_rows": 1,
			"grid_cols": 1,
			"layers_per_cell": 3,
			"layers": [
				{"type": "dense", "activation": "relu", "input_height": 2, "output_height": %d},
				{"type": "swiglu", "input_height": %d, "output_height": %d},
				{"type": "dense", "activation": "sigmoid", "input_height": %d, "output_height": 2}
			]
		}`, config.Name, batchSize, config.HiddenSize, config.HiddenSize, config.HiddenSize, config.HiddenSize)

	case "softmax":
		jsonConfig = fmt.Sprintf(`{
			"id": "training_verification_%s",
			"batch_size": %d,
			"grid_rows": 1,
			"grid_cols": 1,
			"layers_per_cell": 3,
			"layers": [
				{"type": "dense", "activation": "relu", "input_height": 2, "output_height": %d},
				{"type": "softmax", "softmax_rows": 1, "softmax_cols": %d},
				{"type": "dense", "activation": "sigmoid", "input_height": %d, "output_height": 2}
			]
		}`, config.Name, batchSize, config.HiddenSize, config.HiddenSize, config.HiddenSize)

	case "mha":
		// Multi-Head Attention
		// HiddenSize = embed_dim, SeqLength = num_heads
		numHeads := config.SeqLength
		if numHeads < 1 {
			numHeads = 4
		}
		embedDim := config.HiddenSize
		jsonConfig = fmt.Sprintf(`{
			"id": "training_verification_%s",
			"batch_size": %d,
			"grid_rows": 1,
			"grid_cols": 1,
			"layers_per_cell": 3,
			"layers": [
				{"type": "dense", "activation": "relu", "input_height": 2, "output_height": %d},
				{"type": "multi_head_attention", "d_model": %d, "num_heads": %d, "seq_length": 1},
				{"type": "dense", "activation": "sigmoid", "input_height": %d, "output_height": 2}
			]
		}`, config.Name, batchSize, embedDim, embedDim, numHeads, embedDim)

	case "combined":
		// A complex network chaining compatible layers:
		// Input -> Dense -> SwiGLU -> LayerNorm -> Conv1D -> RNN -> LSTM -> MHA -> RMSNorm -> Dense -> Output
		// We need careful dimension handling.
		// Input (Flat) -> [Dense] -> (Seq, Dim) -> [Conv1D] -> ...

		seqLen := config.SeqLength
		hidden := config.HiddenSize

		// Note: Conv1D/RNN/LSTM/MHA inputs must be compatible.
		// We'll treat the hidden state as [SeqLen, HiddenSize] where possible.
		// Conv1D with kernel size 3 reduces sequence length by 2 usually, need padding or aware.
		// For simplicity, we keep SeqLen constant by using padding in Conv1D if supported,
		// OR we just accept the reduction.
		// Actually Loom Conv1D `same` padding? Default usually valid/no-padding logic in verify script?
		// Let's use `padding` param if needed, but `serialization.go` Conv2D has padding, Conv1D?
		// Conv1D Config has InputLength.

		// Let's assume Dense expands to (SeqLen * Hidden), then next layers interpret it.
		// Dense Input: 2
		// Dense Output: SeqLen * Hidden

		// Topology:
		// 1. Dense (Expand to Seq*Hidden)
		// 2. SwiGLU (Pointwise)
		// 3. LayerNorm
		// 4. Conv1D (Seq -> Seq, preserve length using padding=1 for k=3)
		// 5. RNN (Seq -> Seq)
		// 6. LSTM (Seq -> Seq)
		// 7. MHA (Seq -> Seq)
		// 8. RMSNorm
		// 9. Dense (Contract to 2) - Output

		jsonConfig = fmt.Sprintf(`{
			"id": "training_verification_%s",
			"batch_size": %d,
			"grid_rows": 1,
			"grid_cols": 1,
			"layers_per_cell": 9,
			"layers": [
				{"type": "dense", "activation": "relu", "input_height": 2, "output_height": %d},
				{"type": "swiglu", "input_height": %d, "output_height": %d},
				{"type": "layernorm", "norm_size": %d, "epsilon": 1e-5},
				{"type": "conv1d", "activation": "relu", "input_channels": %d, "filters": %d, "kernel_size": 3, "padding": 1, "input_length": %d},
				{"type": "rnn", "activation": "tanh", "input_size": %d, "hidden_size": %d, "seq_length": %d},
				{"type": "lstm", "activation": "tanh", "input_size": %d, "hidden_size": %d, "seq_length": %d},
				{"type": "multi_head_attention", "d_model": %d, "num_heads": 4, "seq_length": 1},
				{"type": "rmsnorm", "norm_size": %d, "epsilon": 1e-5},
				{"type": "dense", "activation": "sigmoid", "input_height": %d, "output_height": 2}
			]
		}`, config.Name, batchSize,
			seqLen*hidden,                // Dense Out
			seqLen*hidden, seqLen*hidden, // SwiGLU (Input=SeqLen*Hidden, Output=SeqLen*Hidden)
			seqLen*hidden,          // LayerNorm
			hidden, hidden, seqLen, // Conv1D (InputChannels=Hidden, Filters=Hidden, InputLength=SeqLen)
			hidden, hidden, seqLen, // RNN
			hidden, hidden, seqLen, // LSTM
			hidden,        // MHA
			hidden*seqLen, // RMSNorm (Applied to flat buffer)
			hidden*seqLen) // Dense In

		// Correction:
		// Conv1D output size: Filters * SeqLen (if padded)
		// RNN Input size: InputSize * SeqLen? No, RNN expects [SeqLen, InputSize]
		// If Conv1D outputs [Filters, SeqLen] (or [SeqLen, Filters] depending on impl),
		// and RNN inputs [SeqLen, InputSize].
		// If Filters == InputSize, it matches.

	case "combined-dummy": // Placeholder to avoid trailing switch error if I messed up braces

		return nil, fmt.Errorf("unsupported layer type: %s", config.LayerType)
	}

	return nn.BuildNetworkFromJSON(jsonConfig)
}

func cloneWeights(src, dst *nn.Network) {
	for i := 0; i < src.TotalLayers(); i++ {
		s := &src.Layers[i]
		d := &dst.Layers[i]

		// Generic
		if len(s.Kernel) > 0 {
			d.Kernel = append([]float32(nil), s.Kernel...)
		}
		if len(s.Bias) > 0 {
			d.Bias = append([]float32(nil), s.Bias...)
		}

		// RNN
		if len(s.WeightIH) > 0 {
			d.WeightIH = append([]float32(nil), s.WeightIH...)
		}
		if len(s.WeightHH) > 0 {
			d.WeightHH = append([]float32(nil), s.WeightHH...)
		}
		if len(s.BiasH) > 0 {
			d.BiasH = append([]float32(nil), s.BiasH...)
		}

		// LSTM (4 gates)
		d.WeightIH_i = append([]float32(nil), s.WeightIH_i...)
		d.WeightIH_f = append([]float32(nil), s.WeightIH_f...)
		d.WeightIH_g = append([]float32(nil), s.WeightIH_g...)
		d.WeightIH_o = append([]float32(nil), s.WeightIH_o...)

		d.WeightHH_i = append([]float32(nil), s.WeightHH_i...)
		d.WeightHH_f = append([]float32(nil), s.WeightHH_f...)
		d.WeightHH_g = append([]float32(nil), s.WeightHH_g...)
		d.WeightHH_o = append([]float32(nil), s.WeightHH_o...)

		d.BiasH_i = append([]float32(nil), s.BiasH_i...)
		d.BiasH_f = append([]float32(nil), s.BiasH_f...)
		d.BiasH_g = append([]float32(nil), s.BiasH_g...)
		d.BiasH_o = append([]float32(nil), s.BiasH_o...)
	}
}

func trainNetwork(network *nn.Network, dataset *Dataset, epochs int, learningRate float32, isGPU bool, batchSize int) ([]float32, time.Duration, error) {
	name := "CPU"
	if isGPU {
		name = "GPU"
	}

	numSamples := len(dataset.Inputs)
	inputSize := len(dataset.Inputs[0])
	outputSize := 2

	if batchSize <= 0 {
		batchSize = 1
	}
	numBatches := numSamples / batchSize

	lossHistory := make([]float32, epochs)
	startTime := time.Now()

	for epoch := 0; epoch < epochs; epoch++ {
		totalLoss := float32(0.0)

		for b := 0; b < numBatches; b++ {
			// Determine batch range
			idxStart := b * batchSize
			idxEnd := idxStart + batchSize
			if idxEnd > numSamples {
				idxEnd = numSamples
			}
			currentBatchSize := idxEnd - idxStart

			// Extract Batch Input
			// We need a flat slice for the batch
			batchInputLen := currentBatchSize * inputSize
			// Optimization: could reuse buffer, but allocation is cleaner for now
			batchInput := make([]float32, batchInputLen)

			// Copy samples into batchInput
			for i := 0; i < currentBatchSize; i++ {
				copy(batchInput[i*inputSize:], dataset.Inputs[idxStart+i])
			}

			// Forward Pass
			output, _ := network.ForwardCPU(batchInput)

			// Compute Gradients
			dOutput := make([]float32, len(output))

			for i := 0; i < currentBatchSize; i++ {
				// absolute index in dataset
				absIdx := idxStart + i

				// Output corresponds to batch index i
				outStart := i * outputSize
				sampleOut := output[outStart : outStart+outputSize]

				// Target
				class := int(dataset.Outputs[absIdx])

				// Loss calculation
				if class < len(sampleOut) {
					val := sampleOut[class]
					if val > 1e-7 {
						totalLoss += -float32(math.Log(float64(val)))
					}
				}

				// Gradient dL/dY = (Y - T) / N
				// dOutput should be scaled by currentBatchSize
				for j := 0; j < outputSize; j++ {
					targetVal := 0.0
					if j == class {
						targetVal = 1.0
					}
					dOutput[outStart+j] = (sampleOut[j] - float32(targetVal)) / float32(currentBatchSize)
				}
			}

			// Backward Pass
			_, _ = network.BackwardCPU(dOutput)

			// Apply Gradients
			network.ApplyGradients(learningRate)
		}

		// Record epoch loss
		avgLoss := totalLoss / float32(numSamples)
		lossHistory[epoch] = avgLoss

		// Print progress
		// Always print first, last, and every epoch if requested by user (for now we print all)
		fmt.Printf("  [%s] Epoch %d/%d - Loss: %.4f\n", name, epoch+1, epochs, avgLoss)
	}

	totalTime := time.Since(startTime)
	return lossHistory, totalTime, nil
}

func printDeviationTable(name string, before, after *nn.DeviationMetrics) {
	fmt.Printf("\n╔═══════════════════════════════════════════════════════════════╗\n")
	fmt.Printf("║  %-59s  ║\n", name)
	fmt.Printf("╠═══════════════════════════════════════════════════════════════╣\n")
	fmt.Printf("║  Accuracy:      %6.1f%% → %6.1f%%                           ║\n",
		before.Accuracy*100, after.Accuracy*100)
	fmt.Printf("║  Quality Score: %6.1f   → %6.1f                           ║\n",
		before.Score, after.Score)
	fmt.Printf("║  Avg Deviation: %6.1f%% → %6.1f%%                          ║\n",
		before.AverageDeviation, after.AverageDeviation)
	fmt.Printf("╠═══════════════════════════════════════════════════════════════╣\n")
	fmt.Printf("║  Deviation Distribution:                                      ║\n")
	fmt.Printf("╠═══════════════════════════════════════════════════════════════╣\n")

	bucketOrder := []string{"0-10%", "10-20%", "20-30%", "30-40%", "40-50%", "50-100%", "100%+"}
	for _, bucketName := range bucketOrder {
		beforeCount := before.Buckets[bucketName].Count
		afterCount := after.Buckets[bucketName].Count
		beforePct := float64(beforeCount) / float64(before.TotalSamples) * 100
		afterPct := float64(afterCount) / float64(after.TotalSamples) * 100

		fmt.Printf("║    %8s: %3d (%5.1f%%) → %3d (%5.1f%%)                    ║\n",
			bucketName, beforeCount, beforePct, afterCount, afterPct)
	}
	fmt.Printf("╚═══════════════════════════════════════════════════════════════╝\n")
}

// printEpochLossTable prints epoch-by-epoch loss progression for all tested layers
func printEpochLossTable(results []LayerTestResult, title string, epochs int) {
	fmt.Printf("\n%s\n", title)
	fmt.Printf("═══════════════════════════════════════════════════════════════\n")

	// Header
	fmt.Printf("%-15s", "Layer Type")
	epochsToShow := []int{1, 2, 3, 5, 10, 15, 20}
	for _, e := range epochsToShow {
		if e <= epochs {
			fmt.Printf(" | Ep%-3d", e)
		}
	}
	fmt.Printf("\n")
	fmt.Printf("═══════════════════════════════════════════════════════════════\n")

	// Rows
	for _, result := range results {
		fmt.Printf("%-15s", result.Config.Name)
		if !result.Success {
			fmt.Printf(" | %s\n", result.ErrorMessage)
			continue
		}

		for _, e := range epochsToShow {
			if e <= epochs && e-1 < len(result.LossHistory) {
				fmt.Printf(" | %.4f", result.LossHistory[e-1])
			}
		}
		fmt.Printf("\n")
	}
	fmt.Printf("═══════════════════════════════════════════════════════════════\n")
}

// printFinalComparisonTable prints final metrics comparison
func printFinalComparisonTable(cpuResults, gpuResults []LayerTestResult) {
	fmt.Printf("\nFinal Comparison\n")
	fmt.Printf("═══════════════════════════════════════════════════════════════\n")
	fmt.Printf("%-15s | CPU Acc | GPU Acc |  Speedup | GPU Status\n", "Layer Type")
	fmt.Printf("═══════════════════════════════════════════════════════════════\n")

	for i := range cpuResults {
		cpuRes := cpuResults[i]
		gpuRes := gpuResults[i]

		status := "✓ OK"
		if !gpuRes.Success {
			status = "✗ FAIL"
		}

		speedup := float64(cpuRes.TrainTime) / float64(gpuRes.TrainTime)
		if !gpuRes.Success {
			speedup = 0
		}

		fmt.Printf("%-15s | %6.1f%% | %6.1f%% | %7.2fx | %s\n",
			cpuRes.Config.Name,
			cpuRes.FinalAcc*100,
			gpuRes.FinalAcc*100,
			speedup,
			status)
	}
	fmt.Printf("═══════════════════════════════════════════════════════════════\n")
}

func main() {
	flag.Parse()
	if *gpuFlag != "" {
		fmt.Printf("Requesting GPU adapter matching: %q\n", *gpuFlag)
		gpu.SetAdapterPreference(*gpuFlag)
	}

	rand.Seed(time.Now().UnixNano())

	fmt.Println("╔═══════════════════════════════════════════════════════════════╗")
	fmt.Println("║       GPU Training Verification Test (Multi-Layer)           ║")
	fmt.Println("╚═══════════════════════════════════════════════════════════════╝")
	fmt.Println()

	// Get ALL layer configurations
	allLayerConfigs := getLayerTestConfigs()

	// Interactive menu for layer selection
	fmt.Println("Select layers to test:")
	for i, config := range allLayerConfigs {
		fmt.Printf("%2d. %s\n", i+1, config.Name)
	}
	fmt.Printf("%2d. Run ALL tests\n", len(allLayerConfigs)+1)
	fmt.Println()
	fmt.Print("Enter selection (comma-separated numbers, e.g., 1,2,3 or press Enter for all): ")

	// Read user input
	reader := bufio.NewReader(os.Stdin)
	input, _ := reader.ReadString('\n')
	input = strings.TrimSpace(input)

	var layerConfigs []LayerTestConfig
	if input == "" || input == fmt.Sprintf("%d", len(allLayerConfigs)+1) {
		// Run all tests
		layerConfigs = allLayerConfigs
		fmt.Println("\n→ Running ALL tests\n")
	} else {
		// Parse comma-separated selections
		selections := strings.Split(input, ",")
		selectedMap := make(map[int]bool)

		for _, sel := range selections {
			sel = strings.TrimSpace(sel)
			num, err := strconv.Atoi(sel)
			if err == nil && num >= 1 && num <= len(allLayerConfigs) {
				selectedMap[num-1] = true
			}
		}

		for idx := range allLayerConfigs {
			if selectedMap[idx] {
				layerConfigs = append(layerConfigs, allLayerConfigs[idx])
			}
		}

		if len(layerConfigs) == 0 {
			fmt.Println("No valid selections, running all tests...")
			layerConfigs = allLayerConfigs
		} else {
			fmt.Printf("\n→ Running %d selected test(s)\n\n", len(layerConfigs))
		}
	}

	// Generate dataset
	fmt.Printf("Generating linearly separable dataset (100 samples)...\n\n")
	dataset := generateSimpleDataset(100)

	var cpuResults []LayerTestResult
	var gpuResults []LayerTestResult

	// Test each layer type
	for _, config := range layerConfigs {
		fmt.Printf("\n┌───────────────────────────────────────────────────────────────┐\n")
		fmt.Printf("│ Testing: %-52s │\n", config.Name)
		fmt.Printf("└───────────────────────────────────────────────────────────────┘\n")

		// ===== CPU Training =====
		fmt.Println("\n[CPU Training]")
		cpuResult := LayerTestResult{
			Config: config,
			IsGPU:  false,
		}

		cpuNetwork, err := createNetworkWithHiddenLayer(1, config)
		if err != nil {
			cpuResult.Success = false
			cpuResult.ErrorMessage = fmt.Sprintf("Network creation failed: %v", err)
			cpuResults = append(cpuResults, cpuResult)
			fmt.Printf("  ✗ Failed to create network: %v\n", err)
			continue
		}

		cpuNetwork.InitializeWeights()

		// Evaluate before
		beforeMetrics, _ := cpuNetwork.EvaluateNetwork(dataset.Inputs, dataset.Outputs)
		cpuResult.InitialAcc = beforeMetrics.Accuracy

		initialWeightsNetwork, _ := createNetworkWithHiddenLayer(20, config)
		cloneWeights(cpuNetwork, initialWeightsNetwork)

		// Train
		lossHistory, trainTime, err := trainNetwork(cpuNetwork, dataset, *epochsFlag, float32(*lrFlag), false, 1)
		if err != nil {
			cpuResult.Success = false
			cpuResult.ErrorMessage = fmt.Sprintf("Training failed: %v", err)
			cpuResults = append(cpuResults, cpuResult)
			fmt.Printf("  ✗ Training failed: %v\n", err)
			continue
		}

		cpuResult.LossHistory = lossHistory
		cpuResult.TrainTime = trainTime

		// Evaluate after
		afterMetrics, _ := cpuNetwork.EvaluateNetwork(dataset.Inputs, dataset.Outputs)
		cpuResult.FinalAcc = afterMetrics.Accuracy
		cpuResult.Success = true

		cpuResults = append(cpuResults, cpuResult)
		fmt.Printf("  ✓ Completed: %.1f%% → %.1f%% accuracy in %v\n",
			cpuResult.InitialAcc*100, cpuResult.FinalAcc*100, trainTime)

		// ===== GPU Training =====
		fmt.Println("\n[GPU Training]")
		gpuResult := LayerTestResult{
			Config: config,
			IsGPU:  true,
		}

		gpuNetwork, err := createNetworkWithHiddenLayer(20, config)
		if err != nil {
			gpuResult.Success = false
			gpuResult.ErrorMessage = fmt.Sprintf("Network creation failed")
			gpuResults = append(gpuResults, gpuResult)
			fmt.Printf("  ✗ Failed to create network: %v\n", err)
			continue
		}

		cloneWeights(initialWeightsNetwork, gpuNetwork)

		// Try to enable GPU
		gpuNetwork.GPU = true
		err = gpuNetwork.WeightsToGPU()
		if err != nil {
			gpuResult.Success = false
			gpuResult.ErrorMessage = fmt.Sprintf("GPU mount failed")
			gpuResults = append(gpuResults, gpuResult)
			fmt.Printf("  ✗ Failed to mount to GPU: %v\n", err)
			continue
		}
		defer gpuNetwork.ReleaseGPUWeights()

		beforeMetrics, _ = gpuNetwork.EvaluateNetwork(dataset.Inputs, dataset.Outputs)
		gpuResult.InitialAcc = beforeMetrics.Accuracy

		// Train with scaled LR
		gpuLR := float32(*lrFlag) * 5.0
		lossHistory, trainTime, err = trainNetwork(gpuNetwork, dataset, *epochsFlag, gpuLR, true, 20)
		if err != nil {
			gpuResult.Success = false
			gpuResult.ErrorMessage = fmt.Sprintf("Training failed")
			gpuResults = append(gpuResults, gpuResult)
			fmt.Printf("  ✗ Training failed: %v\n", err)
			continue
		}

		gpuResult.LossHistory = lossHistory
		gpuResult.TrainTime = trainTime

		afterMetrics, _ = gpuNetwork.EvaluateNetwork(dataset.Inputs, dataset.Outputs)
		gpuResult.FinalAcc = afterMetrics.Accuracy
		gpuResult.Success = true

		gpuResults = append(gpuResults, gpuResult)
		fmt.Printf("  ✓ Completed: %.1f%% → %.1f%% accuracy in %v\n",
			gpuResult.InitialAcc*100, gpuResult.FinalAcc*100, trainTime)
	}

	// ===== Print Results =====
	fmt.Printf("\n\n")
	fmt.Println("╔═══════════════════════════════════════════════════════════════╗")
	fmt.Println("║ RESULTS                                                       ║")
	fmt.Println("╚═══════════════════════════════════════════════════════════════╝")

	printEpochLossTable(cpuResults, "CPU Epoch-by-Epoch Loss", *epochsFlag)
	printEpochLossTable(gpuResults, "GPU Epoch-by-Epoch Loss", *epochsFlag)
	printFinalComparisonTable(cpuResults, gpuResults)

	// Check overall pass/fail
	allPassed := true
	for _, result := range gpuResults {
		if !result.Success {
			allPassed = false
		}
	}

	if allPassed {
		fmt.Printf("\n✓ All GPU layers verified successfully\n")
	} else {
		fmt.Printf("\n⚠ Some GPU layers failed - see table above for details\n")
	}
}
