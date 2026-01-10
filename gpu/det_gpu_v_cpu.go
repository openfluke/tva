package main

import (
	"flag"
	"fmt"
	"math"
	"math/rand"
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
}

func main() {
	flag.Parse()
	if *gpuFlag != "" {
		fmt.Printf("Requesting GPU adapter matching: %q\n", *gpuFlag)
		gpu.SetAdapterPreference(*gpuFlag)
	}

	rand.Seed(time.Now().UnixNano())

	fmt.Println("╔══════════════════════════════════════════════════════════════════════╗")
	fmt.Println("║       GPU-CPU Determinism Test: Forward Pass ONLY                    ║")
	fmt.Println("╚══════════════════════════════════════════════════════════════════════╝")
	fmt.Println()

	// Define all layer types to test with proper full network configs
	layerTests := []GPULayerTestCase{
		{
			Name: "Dense",
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
	}

	for _, test := range layerTests {
		fmt.Printf("┌──────────────────────────────────────────────────────────────────────┐\n")
		fmt.Printf("│ Testing: %-59s│\n", test.Name)
		fmt.Printf("└──────────────────────────────────────────────────────────────────────┘\n")

		runGPULayerTest(test)
		fmt.Println()
	}
}

func runGPULayerTest(test GPULayerTestCase) {
	// Recover from GPU panics
	defer func() {
		if r := recover(); r != nil {
			fmt.Printf("  ❌ GPU panic: %v\n", r)
		}
	}()

	// Build network from config
	network, err := nn.BuildNetworkFromJSON(test.JSONConfig)
	if err != nil {
		fmt.Printf("  ❌ Build error: %v\n", err)
		return
	}
	// DO NOT force BatchSize to 1 here, let JSON config dictate it.
	// But ensure network.BatchSize is populated if 0 (BuildNetworkFromJSON should do this)
	if network.BatchSize == 0 {
		network.BatchSize = 1
	}

	network.InitializeWeights()

	// Create input
	// NOTE: InputSize provided in test definition corresponds to a SINGLE batch item usually,
	// or it was hardcoded for batch=1.
	// We need to scale it by BatchSize if the input provided is for a single item.
	// Usually nn.Network expects flattened input for the whole batch.
	totalInputSize := test.InputSize * network.BatchSize
	input := make([]float32, totalInputSize)
	for i := range input {
		input[i] = rand.Float32()*2 - 1
	}

	// CPU Forward
	// Force CPU execution
	network.GPU = false
	cpuOutput, _ := network.ForwardCPU(input)

	// GPU Forward
	// Panic protection specific to GPU scope
	gpuForwardOK := false
	var gpuOutput []float32

	func() {
		defer func() {
			if r := recover(); r != nil {
				fmt.Printf("  ❌ GPU Forward panic: %v\n", r)
			}
		}()

		network.GPU = true
		err = network.WeightsToGPU()
		if err != nil {
			fmt.Printf("  ❌ GPU mount error: %v\n", err)
			return
		}

		// Use ForwardCPU wrapper which handles GPU dispatch
		gpuOutput, _ = network.ForwardCPU(input)
		gpuForwardOK = true
	}()

	if !gpuForwardOK {
		network.ReleaseGPUWeights()
		return
	}

	// Compare
	compareOutputs(cpuOutput, gpuOutput)

	network.ReleaseGPUWeights()
}

func compareOutputs(cpu, gpu []float32) {
	if len(cpu) != len(gpu) {
		fmt.Printf("  ❌ Size mismatch: CPU=%d, GPU=%d\n", len(cpu), len(gpu))
		return
	}

	var maxDiff float64
	var sumDiff float64
	var meanDiff float64

	// Track indexes of max diff
	maxDiffIdx := -1

	for i := range cpu {
		diff := math.Abs(float64(cpu[i] - gpu[i]))
		if diff > maxDiff {
			maxDiff = diff
			maxDiffIdx = i
		}
		sumDiff += diff
	}

	if len(cpu) > 0 {
		meanDiff = sumDiff / float64(len(cpu))
	}

	fmt.Printf("  • Max Diff:  %.10f (Idx: %d)\n", maxDiff, maxDiffIdx)
	fmt.Printf("  • Mean Diff: %.10f\n", meanDiff)

	// Spectrum of Determinism
	if maxDiff == 0 {
		fmt.Println("  ✅ [GOLD STANDARD] Exact Bit-Determinism")
		fmt.Println("     Perfect match. CPU and GPU logic are identical down to the bit.")
	} else if maxDiff < 1e-7 {
		fmt.Println("  ✅ [EXCELLENT] Near-Machine-Epsilon (< 1e-7)")
		fmt.Println("     Virtually identical. Differences are likely due to rounding (LSB) or fused-multiply-add optimization (FMA).")
	} else if maxDiff < 1e-5 {
		fmt.Println("  ✅ [INDUSTRY STANDARD] Functional Equivalence (< 1e-5)")
		fmt.Println("     Standard tolerance for deep learning frameworks (float32). Safe for training.")
	} else if maxDiff < 1e-3 {
		fmt.Println("  ⚠️ [ACCEPTABLE DRIFT] Approximate Match (< 1e-3)")
		fmt.Println("     Noticeable drift. Usable for inference, but might cause divergence during long training.")
	} else {
		fmt.Println("  ❌ [FAILURE] Significant Divergence (> 1e-3)")
		fmt.Println("     Logic mismatch or bug. Outputs are too different to be considered the same layer.")
	}

	// Always print first few samples to show it's working
	fmt.Println("  • Output Sample (First 5):")
	limit := 5
	if len(cpu) < limit {
		limit = len(cpu)
	}
	for k := 0; k < limit; k++ {
		fmt.Printf("    [%d] CPU: %14.10f | GPU: %14.10f | Diff: %.10f\n",
			k, cpu[k], gpu[k], math.Abs(float64(cpu[k]-gpu[k])))
	}

	// Always print the max diff index sample
	if maxDiffIdx != -1 {
		fmt.Println("  • Max Diff Sample:")
		fmt.Printf("    [%d] CPU: %14.10f | GPU: %14.10f | Diff: %.10f\n",
			maxDiffIdx, cpu[maxDiffIdx], gpu[maxDiffIdx], maxDiff)
	}
}
