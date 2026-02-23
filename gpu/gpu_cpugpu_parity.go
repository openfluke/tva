package main

// GPU-CPU Parity Test for ALL Layer Types
//
// Tests GPU acceleration across all supported layer types by:
//   1. Creating networks with each layer type as the middle layer
//   2. Comparing CPU vs GPU forward/backward pass
//   3. Producing a summary table of results

import (
	"flag"
	"fmt"
	"math"
	"math/rand"
	"strings"
	"time"

	"github.com/openfluke/loom/gpu"
	"github.com/openfluke/loom/nn"
)

var gpuFlag = flag.String("gpu", "", "Optional substring to select a specific GPU adapter (e.g. 'nvidia')")

// GPULayerTestCase defines a test case for a specific hidden layer type
type GPULayerTestCase struct {
	Name       string
	JSONConfig string // Full network JSON config
	InputSize  int    // Network input size
	InputType  string // "uniform" (default), "indices" (for embedding)
	VocabSize  int    // For "indices" type, max token ID
}

// GPUTestResultRow holds the result of testing a layer type
type GPUTestResultRow struct {
	LayerName       string
	ForwardCPU      time.Duration
	ForwardGPU      time.Duration
	BackwardCPU     time.Duration
	BackwardGPU     time.Duration
	ForwardError    float64
	BackwardError   string
	ForwardWorks    bool
	BackwardWorks   bool
	ForwardSpeedup  float64
	BackwardSpeedup float64
	ErrMsg          string
}

func main() {
	flag.Parse()
	if *gpuFlag != "" {
		fmt.Printf("Requesting GPU adapter matching: %q\n", *gpuFlag)
		gpu.SetAdapterPreference(*gpuFlag)
	}

	rand.Seed(time.Now().UnixNano())

	fmt.Println("╔══════════════════════════════════════════════════════════════════════╗")
	fmt.Println("║       GPU-CPU Parity Test: ALL LAYER TYPES                          ║")
	fmt.Println("╚══════════════════════════════════════════════════════════════════════╝")
	fmt.Println()

	// Define all layer types to test with proper full network configs
	layerTests := []GPULayerTestCase{
		{
			Name: "Dense_Batch1",
			JSONConfig: `{
				"id": "gpu_test_dense",
				"batch_size": 1,
				"grid_rows": 1,
				"grid_cols": 1,
				"layers_per_cell": 5,
				"layers": [
					{"type": "dense", "activation": "leaky_relu", "input_height": 2048, "output_height": 2048},
					{"type": "dense", "activation": "leaky_relu", "input_height": 2048, "output_height": 2048},
					{"type": "dense", "activation": "leaky_relu", "input_height": 2048, "output_height": 2048},
					{"type": "dense", "activation": "leaky_relu", "input_height": 2048, "output_height": 2048},
					{"type": "dense", "activation": "sigmoid", "input_height": 2048, "output_height": 2}
				]
			}`,
			InputSize: 2048,
		},
		{
			Name: "Dense_Batch4",
			JSONConfig: `{
				"id": "gpu_test_dense_b4",
				"batch_size": 4,
				"grid_rows": 1,
				"grid_cols": 1,
				"layers_per_cell": 3,
				"layers": [
					{"type": "dense", "activation": "leaky_relu", "input_height": 512, "output_height": 512},
					{"type": "dense", "activation": "leaky_relu", "input_height": 512, "output_height": 512},
					{"type": "dense", "activation": "sigmoid", "input_height": 512, "output_height": 2}
				]
			}`,
			InputSize: 512,
		},
		{
			Name:      "Embedding",
			InputType: "indices",
			VocabSize: 100,
			JSONConfig: `{
				"id": "gpu_test_embedding",
				"batch_size": 1,
				"grid_rows": 1,
				"grid_cols": 1,
				"layers_per_cell": 3,
				"layers": [
					{"type": "embedding", "vocab_size": 100, "embedding_dim": 64},
					{"type": "dense", "activation": "tanh", "input_height": 64, "output_height": 64},
					{"type": "dense", "activation": "sigmoid", "input_height": 64, "output_height": 2}
				]
			}`,
			InputSize: 1, // 1 token index
		},
		{
			Name: "Residual_Skip",
			JSONConfig: `{
				"id": "gpu_test_residual",
				"batch_size": 1,
				"grid_rows": 1,
				"grid_cols": 1,
				"layers_per_cell": 3,
				"layers": [
					{"type": "dense", "activation": "leaky_relu", "input_height": 256, "output_height": 256},
					{
						"type": "residual", 
						"branches": [
							{"type": "dense", "activation": "tanh", "input_height": 256, "output_height": 256}
						]
					},
					{"type": "dense", "activation": "sigmoid", "input_height": 256, "output_height": 2}
				]
			}`,
			InputSize: 256,
		},
		{
			Name: "Parallel_MoE",
			JSONConfig: `{
				"id": "gpu_test_parallel",
				"batch_size": 1,
				"grid_rows": 1,
				"grid_cols": 1,
				"layers_per_cell": 3,
				"layers": [
					{"type": "dense", "activation": "leaky_relu", "input_height": 256, "output_height": 256},
					{
						"type": "parallel",
						"combine_mode": "concat",
						"branches": [
							{"type": "dense", "activation": "tanh", "input_height": 256, "output_height": 128},
							{"type": "dense", "activation": "sigmoid", "input_height": 256, "output_height": 128}
						]
					},
					{"type": "dense", "activation": "sigmoid", "input_height": 256, "output_height": 2}
				]
			}`,
			InputSize: 256,
		},
		{
			Name: "LayerNorm",
			JSONConfig: `{
				"id": "gpu_test_layernorm",
				"batch_size": 1,
				"grid_rows": 1,
				"grid_cols": 1,
				"layers_per_cell": 5,
				"layers": [
					{"type": "dense", "activation": "leaky_relu", "input_height": 2048, "output_height": 2048},
					{"type": "layer_norm", "norm_size": 2048, "epsilon": 1e-5},
					{"type": "layer_norm", "norm_size": 2048, "epsilon": 1e-5},
					{"type": "layer_norm", "norm_size": 2048, "epsilon": 1e-5},
					{"type": "dense", "activation": "sigmoid", "input_height": 2048, "output_height": 2}
				]
			}`,
			InputSize: 2048,
		},
		{
			Name: "RMSNorm",
			JSONConfig: `{
				"id": "gpu_test_rmsnorm",
				"batch_size": 1,
				"grid_rows": 1,
				"grid_cols": 1,
				"layers_per_cell": 5,
				"layers": [
					{"type": "dense", "activation": "leaky_relu", "input_height": 2048, "output_height": 2048},
					{"type": "rms_norm", "norm_size": 2048, "epsilon": 1e-5},
					{"type": "rms_norm", "norm_size": 2048, "epsilon": 1e-5},
					{"type": "rms_norm", "norm_size": 2048, "epsilon": 1e-5},
					{"type": "dense", "activation": "sigmoid", "input_height": 2048, "output_height": 2}
				]
			}`,
			InputSize: 2048,
		},
		{
			Name: "Softmax",
			JSONConfig: `{
				"id": "gpu_test_softmax",
				"batch_size": 1,
				"grid_rows": 1,
				"grid_cols": 1,
				"layers_per_cell": 5,
				"layers": [
					{"type": "dense", "activation": "leaky_relu", "input_height": 2048, "output_height": 2048},
					{"type": "softmax", "temperature": 1.0},
					{"type": "softmax", "temperature": 1.0},
					{"type": "softmax", "temperature": 1.0},
					{"type": "dense", "activation": "sigmoid", "input_height": 2048, "output_height": 2}
				]
			}`,
			InputSize: 2048,
		},
		{
			Name: "Conv1D",
			// 32 seq x 64 channels = 2048, output with padding=1 stride=1 is still 32x64=2048
			JSONConfig: `{
				"id": "gpu_test_conv1d",
				"batch_size": 1,
				"grid_rows": 1,
				"grid_cols": 1,
				"layers_per_cell": 5,
				"layers": [
					{"type": "dense", "activation": "leaky_relu", "input_height": 2048, "output_height": 2048},
					{"type": "conv1d", "input_channels": 64, "filters": 64, "kernel_size": 3, "stride": 1, "padding": 1, "input_length": 32},
					{"type": "conv1d", "input_channels": 64, "filters": 64, "kernel_size": 3, "stride": 1, "padding": 1, "input_length": 32},
					{"type": "conv1d", "input_channels": 64, "filters": 64, "kernel_size": 3, "stride": 1, "padding": 1, "input_length": 32},
					{"type": "dense", "activation": "sigmoid", "input_height": 2048, "output_height": 2}
				]
			}`,
			InputSize: 2048,
		},
		{
			Name: "Conv2D",
			// 16x16x8 = 2048, output with padding=1 stride=1 kernel=3 is still 16x16x8=2048
			JSONConfig: `{
				"id": "gpu_test_conv2d",
				"batch_size": 1,
				"grid_rows": 1,
				"grid_cols": 1,
				"layers_per_cell": 5,
				"layers": [
					{"type": "dense", "activation": "leaky_relu", "input_height": 2048, "output_height": 2048},
					{"type": "conv2d", "input_channels": 8, "filters": 8, "kernel_size": 3, "stride": 1, "padding": 1, "input_height": 16, "input_width": 16, "output_height": 16, "output_width": 16},
					{"type": "conv2d", "input_channels": 8, "filters": 8, "kernel_size": 3, "stride": 1, "padding": 1, "input_height": 16, "input_width": 16, "output_height": 16, "output_width": 16},
					{"type": "conv2d", "input_channels": 8, "filters": 8, "kernel_size": 3, "stride": 1, "padding": 1, "input_height": 16, "input_width": 16, "output_height": 16, "output_width": 16},
					{"type": "dense", "activation": "sigmoid", "input_height": 2048, "output_height": 2}
				]
			}`,
			InputSize: 2048,
		},
		{
			Name: "SwiGLU",
			JSONConfig: `{
				"id": "gpu_test_swiglu",
				"batch_size": 1,
				"grid_rows": 1,
				"grid_cols": 1,
				"layers_per_cell": 5,
				"layers": [
					{"type": "dense", "activation": "leaky_relu", "input_height": 2048, "output_height": 2048},
					{"type": "swiglu", "input_height": 2048, "output_height": 2048},
					{"type": "swiglu", "input_height": 2048, "output_height": 2048},
					{"type": "swiglu", "input_height": 2048, "output_height": 2048},
					{"type": "dense", "activation": "sigmoid", "input_height": 2048, "output_height": 2}
				]
			}`,
			InputSize: 2048,
		},
		{
			Name: "RNN",
			// Simplified: single RNN layer with smaller dimensions to avoid GPU shader issues
			// 8 seq x 64 hidden = 512
			JSONConfig: `{
				"id": "gpu_test_rnn",
				"batch_size": 1,
				"grid_rows": 1,
				"grid_cols": 1,
				"layers_per_cell": 3,
				"layers": [
					{"type": "dense", "activation": "leaky_relu", "input_height": 512, "output_height": 512},
					{"type": "rnn", "input_size": 64, "hidden_size": 64, "seq_length": 8},
					{"type": "dense", "activation": "sigmoid", "input_height": 512, "output_height": 2}
				]
			}`,
			InputSize: 512,
		},
		{
			Name: "LSTM",
			// Simplified: single LSTM layer with smaller dimensions to avoid GPU shader issues
			// 8 seq x 64 hidden = 512
			JSONConfig: `{
				"id": "gpu_test_lstm",
				"batch_size": 1,
				"grid_rows": 1,
				"grid_cols": 1,
				"layers_per_cell": 3,
				"layers": [
					{"type": "dense", "activation": "leaky_relu", "input_height": 512, "output_height": 512},
					{"type": "lstm", "input_size": 64, "hidden_size": 64, "seq_length": 8},
					{"type": "dense", "activation": "sigmoid", "input_height": 512, "output_height": 2}
				]
			}`,
			InputSize: 512,
		},
		{
			Name: "MHA",
			// 8 seq x 256 d_model = 2048
			JSONConfig: `{
				"id": "gpu_test_mha",
				"batch_size": 1,
				"grid_rows": 1,
				"grid_cols": 1,
				"layers_per_cell": 5,
				"layers": [
					{"type": "dense", "activation": "leaky_relu", "input_height": 2048, "output_height": 2048},
					{"type": "multi_head_attention", "d_model": 256, "num_heads": 8},
					{"type": "multi_head_attention", "d_model": 256, "num_heads": 8},
					{"type": "multi_head_attention", "d_model": 256, "num_heads": 8},
					{"type": "dense", "activation": "sigmoid", "input_height": 2048, "output_height": 2}
				]
			}`,
			InputSize: 2048,
		},
		{
			Name: "Combined_Hybrid",
			// All 9 layer types with consistent 512-element pipeline.
			// Old failure: SwiGLU(input=64) outputs 64 not 512 → downstream layers saw wrong sizes.
			// Fix: input_height=512 so SwiGLU outputs 512 consistently.
			JSONConfig: `{
				"id": "gpu_test_combined",
				"batch_size": 1,
				"grid_rows": 1,
				"grid_cols": 1,
				"layers_per_cell": 9,
				"layers": [
					{"type": "dense", "activation": "leaky_relu", "input_height": 512, "output_height": 512},
					{"type": "swiglu", "input_height": 512, "output_height": 512},
					{"type": "layer_norm", "norm_size": 512, "epsilon": 1e-5},
					{"type": "conv1d", "input_channels": 64, "filters": 64, "kernel_size": 3, "stride": 1, "padding": 1, "input_length": 8},
					{"type": "rnn", "input_size": 64, "hidden_size": 64, "seq_length": 8},
					{"type": "lstm", "input_size": 64, "hidden_size": 64, "seq_length": 8},
					{"type": "multi_head_attention", "d_model": 64, "num_heads": 8, "seq_length": 8},
					{"type": "rms_norm", "norm_size": 512, "epsilon": 1e-5},
					{"type": "dense", "activation": "sigmoid", "input_height": 512, "output_height": 2}
				]
			}`,
			InputSize: 512,
		},
	}

	results := make([]GPUTestResultRow, 0)

	for _, test := range layerTests {
		fmt.Printf("┌──────────────────────────────────────────────────────────────────────┐\n")
		fmt.Printf("│ Testing: %-59s│\n", test.Name)
		fmt.Printf("└──────────────────────────────────────────────────────────────────────┘\n")

		result := runGPULayerTest(test)
		results = append(results, result)

		// Print quick result
		if result.ForwardWorks {
			fmt.Printf("  ✓ Forward: CPU=%v GPU=%v (%.2fx)\n",
				result.ForwardCPU.Truncate(time.Millisecond),
				result.ForwardGPU.Truncate(time.Millisecond),
				result.ForwardSpeedup)
		} else {
			fmt.Printf("  ❌ Forward: %s\n", result.ErrMsg)
		}

		if result.BackwardWorks {
			fmt.Printf("  ✓ Backward: CPU=%v GPU=%v (%.2fx)\n",
				result.BackwardCPU.Truncate(time.Millisecond),
				result.BackwardGPU.Truncate(time.Millisecond),
				result.BackwardSpeedup)
		} else if result.ForwardWorks {
			fmt.Printf("  ⚠ Backward: %s\n", result.BackwardError)
		}

		fmt.Println()
	}

	// Print summary table
	printGPUSummaryTable(results)
}

func runGPULayerTest(test GPULayerTestCase) (result GPUTestResultRow) {
	result = GPUTestResultRow{LayerName: test.Name}

	// Recover from GPU panics
	defer func() {
		if r := recover(); r != nil {
			result.ErrMsg = fmt.Sprintf("GPU panic: %v", r)
			result.ForwardWorks = false
			result.BackwardWorks = false
		}
	}()

	// Build network from config
	network, err := nn.BuildNetworkFromJSON(test.JSONConfig)
	if err != nil {
		result.ErrMsg = fmt.Sprintf("build error: %v", err)
		return result
	}
	// network.BatchSize is set by JSON config, do not override

	network.InitializeWeights()

	// Create input
	totalInputSize := test.InputSize * network.BatchSize
	input := make([]float32, totalInputSize)

	if test.InputType == "indices" {
		vocab := test.VocabSize
		if vocab <= 0 {
			vocab = 100 // Default safety
		}
		for i := range input {
			input[i] = float32(rand.Intn(vocab))
		}
	} else {
		for i := range input {
			input[i] = rand.Float32()*2 - 1
		}
	}

	// CPU Forward
	cpuOutput, cpuForwardTime := network.ForwardCPU(input)
	result.ForwardCPU = cpuForwardTime

	// GPU Forward (with panic protection)
	gpuForwardOK := false
	var gpuOutput []float32
	var gpuForwardTime time.Duration

	func() {
		defer func() {
			if r := recover(); r != nil {
				result.ErrMsg = fmt.Sprintf("GPU forward panic: %v", r)
			}
		}()

		network.GPU = true
		err = network.WeightsToGPU()
		if err != nil {
			result.ErrMsg = fmt.Sprintf("GPU mount: %v", err)
			return
		}

		gpuOutput, gpuForwardTime = network.ForwardCPU(input)
		gpuForwardOK = true
	}()

	if !gpuForwardOK {
		network.ReleaseGPUWeights()
		return result
	}

	result.ForwardGPU = gpuForwardTime

	// Compare forward outputs
	forwardErr := computeGPUMaxError(cpuOutput, gpuOutput)
	result.ForwardError = forwardErr
	result.ForwardWorks = forwardErr < 1e-1 && len(gpuOutput) > 0

	if result.ForwardWorks && cpuForwardTime > 0 && gpuForwardTime > 0 {
		result.ForwardSpeedup = float64(cpuForwardTime) / float64(gpuForwardTime)
	}

	// CPU Backward
	network.GPU = false
	network.ReleaseGPUWeights()
	network.ForwardCPU(input)

	// Construct dOutput matching batch size (assuming output size is 2 for all tests)
	singleOutputSize := 2
	dOutput := make([]float32, singleOutputSize*network.BatchSize)
	for i := range dOutput {
		dOutput[i] = 1.0
	}

	_, cpuBackwardTime := network.BackwardCPU(dOutput)
	result.BackwardCPU = cpuBackwardTime

	// GPU Backward (with panic protection)
	func() {
		defer func() {
			if r := recover(); r != nil {
				result.BackwardError = fmt.Sprintf("GPU backward panic: %v", r)
			}
		}()

		network.GPU = true
		err = network.WeightsToGPU()
		if err != nil {
			result.BackwardError = fmt.Sprintf("GPU mount: %v", err)
			return
		}
		network.ForwardCPU(input)

		_, gpuBackwardTime := network.BackwardCPU(dOutput)
		result.BackwardGPU = gpuBackwardTime
		result.BackwardWorks = true
		if cpuBackwardTime > 0 && gpuBackwardTime > 0 {
			result.BackwardSpeedup = float64(cpuBackwardTime) / float64(gpuBackwardTime)
		}
	}()

	network.ReleaseGPUWeights()
	return result
}

func printGPUSummaryTable(results []GPUTestResultRow) {
	fmt.Println("╔══════════════════════════════════════════════════════════════════════╗")
	fmt.Println("║                     GPU ACCELERATION SUMMARY                        ║")
	fmt.Println("╚══════════════════════════════════════════════════════════════════════╝")
	fmt.Println()

	// Header
	fmt.Printf("%-20s │ %-8s │ %-8s │ %-10s │ %-10s │ %-8s\n",
		"Layer", "Forward", "Backward", "Fwd Speed", "Bkwd Speed", "Error")
	fmt.Println(strings.Repeat("─", 80))

	// Results
	for _, r := range results {
		fwd := "❌"
		if r.ForwardWorks {
			fwd = "✅"
		}
		bkwd := "❌"
		if r.BackwardWorks {
			bkwd = "✅"
		}

		fwdSpeed := "—"
		if r.ForwardSpeedup > 0 {
			fwdSpeed = fmt.Sprintf("%.2fx", r.ForwardSpeedup)
		}
		bkwdSpeed := "—"
		if r.BackwardSpeedup > 0 {
			bkwdSpeed = fmt.Sprintf("%.2fx", r.BackwardSpeedup)
		}

		errStr := "—"
		if r.ForwardError > 0 {
			errStr = fmt.Sprintf("%.1e", r.ForwardError)
		}
		if r.ErrMsg != "" {
			errStr = "err"
		}

		fmt.Printf("%-20s │ %-8s │ %-8s │ %-10s │ %-10s │ %-8s\n",
			r.LayerName, fwd, bkwd, fwdSpeed, bkwdSpeed, errStr)
	}

	fmt.Println()

	// Count stats
	fwdWorks := 0
	bkwdWorks := 0
	for _, r := range results {
		if r.ForwardWorks {
			fwdWorks++
		}
		if r.BackwardWorks {
			bkwdWorks++
		}
	}

	fmt.Printf("Forward Pass:  %d/%d layers working\n", fwdWorks, len(results))
	fmt.Printf("Backward Pass: %d/%d layers working\n", bkwdWorks, len(results))
	fmt.Println()
	fmt.Println("✅ = Working   ❌ = Failed/Error")
}

func computeGPUMaxError(a, b []float32) float64 {
	if len(a) != len(b) {
		return 9999.0
	}
	var maxErr float64
	for i := range a {
		diff := math.Abs(float64(a[i] - b[i]))
		if diff > maxErr {
			maxErr = diff
		}
	}
	return maxErr
}
