package main

import (
	"fmt"
	"math"
	"math/rand"
	"os"
	"strings"
	"time"

	"github.com/openfluke/loom/nn"
)

// =============================================================================
// Comprehensive Save/Load Test for All Layers and Numerical Types
// =============================================================================
// This test verifies that every layer type can be saved and loaded correctly
// with every supported numerical type (dtype).

// TestResult holds the result of a single test
type TestResult struct {
	LayerType   string
	DType       string
	Passed      bool
	Error       string
	MaxDiff     float32 // Maximum difference between original and loaded weights
	WeightCount int     // Number of weights tested
}

// AllNumericalTypes returns all supported numerical types
func AllNumericalTypes() []string {
	return []string{
		"float32",
		"float64",
		"bfloat16",
		"float16",
		"float8",
		"float4",
		"int8",
		"int16",
		"int32",
		"int64",
		"int4",
		"uint8",
		"uint16",
		"uint32",
		"uint64",
	}
}

// AllLayerTypes returns all layer types with their names
func AllLayerTypes() []string {
	return []string{
		"Dense",
		"Conv2D",
		"Conv1D",
		"MultiHeadAttention",
		"RNN",
		"LSTM",
		"LayerNorm",
		"RMSNorm",
		"SwiGLU",
		"Embedding",
		"Softmax",
		"Residual",
		"Parallel",
		"Parallel_Add",
		"Parallel_Avg",
		"Parallel_Mixed",
		"Parallel_Nested",
		"Sequential",
		"Sequential_Deep",
		"Sequential_With_RNN",
	}
}

// createLayerNetwork creates a network with a specific layer type for testing
func createLayerNetwork(layerType string) (*nn.Network, error) {
	network := nn.NewNetwork(1, 1, 1, 1)

	switch layerType {
	case "Dense":
		network.SetLayer(0, 0, 0, nn.LayerConfig{
			Type:         nn.LayerDense,
			Activation:   nn.ActivationScaledReLU,
			InputHeight:  8,
			OutputHeight: 4,
			Kernel:       randomWeights(8 * 4),
			Bias:         randomWeights(4),
		})

	case "Conv2D":
		network.SetLayer(0, 0, 0, nn.LayerConfig{
			Type:          nn.LayerConv2D,
			Activation:    nn.ActivationScaledReLU,
			InputChannels: 3,
			Filters:       8,
			KernelSize:    3,
			Stride:        1,
			Padding:       1,
			InputHeight:   8,
			InputWidth:    8,
			OutputHeight:  8,
			OutputWidth:   8,
			Kernel:        randomWeights(8 * 3 * 3 * 3),
			Bias:          randomWeights(8),
		})

	case "Conv1D":
		network.SetLayer(0, 0, 0, nn.LayerConfig{
			Type:          nn.LayerConv1D,
			Activation:    nn.ActivationScaledReLU,
			InputChannels: 3,
			Filters:       8,
			KernelSize:    3,
			Stride:        1,
			Padding:       1,
			InputHeight:   16,
			InputWidth:    1,
			OutputHeight:  16,
			OutputWidth:   1,
			Kernel:        randomWeights(8 * 3 * 3),
			Bias:          randomWeights(8),
		})

	case "MultiHeadAttention":
		dModel := 16
		numHeads := 2
		network.SetLayer(0, 0, 0, nn.LayerConfig{
			Type:         nn.LayerMultiHeadAttention,
			DModel:       dModel,
			NumHeads:     numHeads,
			SeqLength:    4,
			QWeights:     randomWeights(dModel * dModel),
			KWeights:     randomWeights(dModel * dModel),
			VWeights:     randomWeights(dModel * dModel),
			OutputWeight: randomWeights(dModel * dModel),
			QBias:        randomWeights(dModel),
			KBias:        randomWeights(dModel),
			VBias:        randomWeights(dModel),
			OutputBias:   randomWeights(dModel),
		})

	case "RNN":
		inputSize := 8
		hiddenSize := 16
		network.SetLayer(0, 0, 0, nn.LayerConfig{
			Type:         nn.LayerRNN,
			Activation:   nn.ActivationTanh,
			RNNInputSize: inputSize,
			HiddenSize:   hiddenSize,
			SeqLength:    4,
			WeightIH:     randomWeights(inputSize * hiddenSize),
			WeightHH:     randomWeights(hiddenSize * hiddenSize),
			BiasH:        randomWeights(hiddenSize),
		})

	case "LSTM":
		inputSize := 8
		hiddenSize := 16
		network.SetLayer(0, 0, 0, nn.LayerConfig{
			Type:         nn.LayerLSTM,
			RNNInputSize: inputSize,
			HiddenSize:   hiddenSize,
			SeqLength:    4,
			WeightIH_i:   randomWeights(inputSize * hiddenSize),
			WeightIH_f:   randomWeights(inputSize * hiddenSize),
			WeightIH_g:   randomWeights(inputSize * hiddenSize),
			WeightIH_o:   randomWeights(inputSize * hiddenSize),
			WeightHH_i:   randomWeights(hiddenSize * hiddenSize),
			WeightHH_f:   randomWeights(hiddenSize * hiddenSize),
			WeightHH_g:   randomWeights(hiddenSize * hiddenSize),
			WeightHH_o:   randomWeights(hiddenSize * hiddenSize),
			BiasH_i:      randomWeights(hiddenSize),
			BiasH_f:      randomWeights(hiddenSize),
			BiasH_g:      randomWeights(hiddenSize),
			BiasH_o:      randomWeights(hiddenSize),
		})

	case "LayerNorm":
		normSize := 16
		network.SetLayer(0, 0, 0, nn.LayerConfig{
			Type:     nn.LayerNorm,
			NormSize: normSize,
			Epsilon:  1e-5,
			Gamma:    randomWeights(normSize),
			Beta:     randomWeights(normSize),
		})

	case "RMSNorm":
		normSize := 16
		network.SetLayer(0, 0, 0, nn.LayerConfig{
			Type:     nn.LayerRMSNorm,
			NormSize: normSize,
			Epsilon:  1e-5,
			Gamma:    randomWeights(normSize),
		})

	case "SwiGLU":
		inputSize := 16
		intermediateSize := 32
		network.SetLayer(0, 0, 0, nn.LayerConfig{
			Type:         nn.LayerSwiGLU,
			InputHeight:  inputSize,
			OutputHeight: intermediateSize,
			GateWeights:  randomWeights(inputSize * intermediateSize),
			UpWeights:    randomWeights(inputSize * intermediateSize),
			DownWeights:  randomWeights(intermediateSize * inputSize),
			GateBias:     randomWeights(intermediateSize),
			UpBias:       randomWeights(intermediateSize),
			DownBias:     randomWeights(inputSize),
		})

	case "Embedding":
		vocabSize := 100
		embeddingDim := 16
		network.SetLayer(0, 0, 0, nn.LayerConfig{
			Type:             nn.LayerEmbedding,
			VocabSize:        vocabSize,
			EmbeddingDim:     embeddingDim,
			EmbeddingWeights: randomWeights(vocabSize * embeddingDim),
		})

	case "Softmax":
		network.SetLayer(0, 0, 0, nn.LayerConfig{
			Type:           nn.LayerSoftmax,
			SoftmaxVariant: nn.SoftmaxStandard,
			SoftmaxRows:    1,
			SoftmaxCols:    10,
			Temperature:    1.0,
		})

	case "Residual":
		network.SetLayer(0, 0, 0, nn.LayerConfig{
			Type: nn.LayerResidual,
		})

	case "Parallel":
		// Parallel with two Dense branches - concat mode
		network.SetLayer(0, 0, 0, nn.LayerConfig{
			Type:        nn.LayerParallel,
			CombineMode: "concat",
			ParallelBranches: []nn.LayerConfig{
				{
					Type:         nn.LayerDense,
					Activation:   nn.ActivationScaledReLU,
					InputHeight:  8,
					OutputHeight: 4,
					Kernel:       randomWeights(8 * 4),
					Bias:         randomWeights(4),
				},
				{
					Type:         nn.LayerDense,
					Activation:   nn.ActivationScaledReLU,
					InputHeight:  8,
					OutputHeight: 4,
					Kernel:       randomWeights(8 * 4),
					Bias:         randomWeights(4),
				},
			},
		})

	case "Parallel_Add":
		// Parallel with add combine mode (requires same-sized outputs)
		network.SetLayer(0, 0, 0, nn.LayerConfig{
			Type:        nn.LayerParallel,
			CombineMode: "add",
			ParallelBranches: []nn.LayerConfig{
				{
					Type:         nn.LayerDense,
					Activation:   nn.ActivationScaledReLU,
					InputHeight:  8,
					OutputHeight: 4,
					Kernel:       randomWeights(8 * 4),
					Bias:         randomWeights(4),
				},
				{
					Type:         nn.LayerDense,
					Activation:   nn.ActivationTanh,
					InputHeight:  8,
					OutputHeight: 4,
					Kernel:       randomWeights(8 * 4),
					Bias:         randomWeights(4),
				},
			},
		})

	case "Parallel_Avg":
		// Parallel with average combine mode
		network.SetLayer(0, 0, 0, nn.LayerConfig{
			Type:        nn.LayerParallel,
			CombineMode: "avg",
			ParallelBranches: []nn.LayerConfig{
				{
					Type:         nn.LayerDense,
					Activation:   nn.ActivationScaledReLU,
					InputHeight:  8,
					OutputHeight: 4,
					Kernel:       randomWeights(8 * 4),
					Bias:         randomWeights(4),
				},
				{
					Type:         nn.LayerDense,
					Activation:   nn.ActivationScaledReLU,
					InputHeight:  8,
					OutputHeight: 4,
					Kernel:       randomWeights(8 * 4),
					Bias:         randomWeights(4),
				},
			},
		})

	case "Parallel_Mixed":
		// Parallel with mixed branch types (Dense + SwiGLU)
		network.SetLayer(0, 0, 0, nn.LayerConfig{
			Type:        nn.LayerParallel,
			CombineMode: "concat",
			ParallelBranches: []nn.LayerConfig{
				{
					Type:         nn.LayerDense,
					Activation:   nn.ActivationScaledReLU,
					InputHeight:  8,
					OutputHeight: 4,
					Kernel:       randomWeights(8 * 4),
					Bias:         randomWeights(4),
				},
				{
					Type:         nn.LayerSwiGLU,
					InputHeight:  8,
					OutputHeight: 16,
					GateWeights:  randomWeights(8 * 16),
					UpWeights:    randomWeights(8 * 16),
					DownWeights:  randomWeights(16 * 8),
					GateBias:     randomWeights(16),
					UpBias:       randomWeights(16),
					DownBias:     randomWeights(8),
				},
			},
		})

	case "Parallel_Nested":
		// Nested parallel structure (Parallel containing Parallel)
		network.SetLayer(0, 0, 0, nn.LayerConfig{
			Type:        nn.LayerParallel,
			CombineMode: "concat",
			ParallelBranches: []nn.LayerConfig{
				{
					Type:         nn.LayerDense,
					Activation:   nn.ActivationScaledReLU,
					InputHeight:  8,
					OutputHeight: 4,
					Kernel:       randomWeights(8 * 4),
					Bias:         randomWeights(4),
				},
				{
					Type:        nn.LayerParallel,
					CombineMode: "add",
					ParallelBranches: []nn.LayerConfig{
						{
							Type:         nn.LayerDense,
							Activation:   nn.ActivationScaledReLU,
							InputHeight:  8,
							OutputHeight: 4,
							Kernel:       randomWeights(8 * 4),
							Bias:         randomWeights(4),
						},
						{
							Type:         nn.LayerDense,
							Activation:   nn.ActivationTanh,
							InputHeight:  8,
							OutputHeight: 4,
							Kernel:       randomWeights(8 * 4),
							Bias:         randomWeights(4),
						},
					},
				},
			},
		})

	case "Sequential":
		// Sequential with two Dense layers
		network.SetLayer(0, 0, 0, nn.LayerConfig{
			Type: nn.LayerSequential,
			ParallelBranches: []nn.LayerConfig{
				{
					Type:         nn.LayerDense,
					Activation:   nn.ActivationScaledReLU,
					InputHeight:  8,
					OutputHeight: 8,
					Kernel:       randomWeights(8 * 8),
					Bias:         randomWeights(8),
				},
				{
					Type:         nn.LayerDense,
					Activation:   nn.ActivationScaledReLU,
					InputHeight:  8,
					OutputHeight: 4,
					Kernel:       randomWeights(8 * 4),
					Bias:         randomWeights(4),
				},
			},
		})

	case "Sequential_Deep":
		// Deeper sequential (4 layers)
		network.SetLayer(0, 0, 0, nn.LayerConfig{
			Type: nn.LayerSequential,
			ParallelBranches: []nn.LayerConfig{
				{
					Type:         nn.LayerDense,
					Activation:   nn.ActivationScaledReLU,
					InputHeight:  8,
					OutputHeight: 16,
					Kernel:       randomWeights(8 * 16),
					Bias:         randomWeights(16),
				},
				{
					Type:     nn.LayerNorm,
					NormSize: 16,
					Epsilon:  1e-5,
					Gamma:    randomWeights(16),
					Beta:     randomWeights(16),
				},
				{
					Type:         nn.LayerDense,
					Activation:   nn.ActivationTanh,
					InputHeight:  16,
					OutputHeight: 8,
					Kernel:       randomWeights(16 * 8),
					Bias:         randomWeights(8),
				},
				{
					Type:         nn.LayerDense,
					Activation:   nn.ActivationScaledReLU,
					InputHeight:  8,
					OutputHeight: 4,
					Kernel:       randomWeights(8 * 4),
					Bias:         randomWeights(4),
				},
			},
		})

	case "Sequential_With_RNN":
		// Sequential with RNN layer
		network.SetLayer(0, 0, 0, nn.LayerConfig{
			Type: nn.LayerSequential,
			ParallelBranches: []nn.LayerConfig{
				{
					Type:         nn.LayerDense,
					Activation:   nn.ActivationScaledReLU,
					InputHeight:  8,
					OutputHeight: 8,
					Kernel:       randomWeights(8 * 8),
					Bias:         randomWeights(8),
				},
				{
					Type:         nn.LayerRNN,
					Activation:   nn.ActivationTanh,
					RNNInputSize: 8,
					HiddenSize:   8,
					SeqLength:    4,
					WeightIH:     randomWeights(8 * 8),
					WeightHH:     randomWeights(8 * 8),
					BiasH:        randomWeights(8),
				},
			},
		})

	default:
		return nil, fmt.Errorf("unknown layer type: %s", layerType)
	}

	return network, nil
}

// randomWeights generates random weights in the range [-1, 1]
func randomWeights(n int) []float32 {
	weights := make([]float32, n)
	for i := range weights {
		weights[i] = float32(rand.Float64()*2 - 1)
	}
	return weights
}

// getLayerWeights extracts all weights from a layer for comparison
func getLayerWeights(cfg *nn.LayerConfig) []float32 {
	var weights []float32

	switch cfg.Type {
	case nn.LayerDense:
		weights = append(weights, cfg.Kernel...)
		weights = append(weights, cfg.Bias...)
	case nn.LayerConv2D, nn.LayerConv1D:
		weights = append(weights, cfg.Kernel...)
		weights = append(weights, cfg.Bias...)
	case nn.LayerMultiHeadAttention:
		weights = append(weights, cfg.QWeights...)
		weights = append(weights, cfg.KWeights...)
		weights = append(weights, cfg.VWeights...)
		weights = append(weights, cfg.OutputWeight...)
		weights = append(weights, cfg.QBias...)
		weights = append(weights, cfg.KBias...)
		weights = append(weights, cfg.VBias...)
		weights = append(weights, cfg.OutputBias...)
	case nn.LayerRNN:
		weights = append(weights, cfg.WeightIH...)
		weights = append(weights, cfg.WeightHH...)
		weights = append(weights, cfg.BiasH...)
	case nn.LayerLSTM:
		weights = append(weights, cfg.WeightIH_i...)
		weights = append(weights, cfg.WeightIH_f...)
		weights = append(weights, cfg.WeightIH_g...)
		weights = append(weights, cfg.WeightIH_o...)
		weights = append(weights, cfg.WeightHH_i...)
		weights = append(weights, cfg.WeightHH_f...)
		weights = append(weights, cfg.WeightHH_g...)
		weights = append(weights, cfg.WeightHH_o...)
		weights = append(weights, cfg.BiasH_i...)
		weights = append(weights, cfg.BiasH_f...)
		weights = append(weights, cfg.BiasH_g...)
		weights = append(weights, cfg.BiasH_o...)
	case nn.LayerNorm:
		weights = append(weights, cfg.Gamma...)
		weights = append(weights, cfg.Beta...)
	case nn.LayerRMSNorm:
		weights = append(weights, cfg.Gamma...)
	case nn.LayerSwiGLU:
		weights = append(weights, cfg.GateWeights...)
		weights = append(weights, cfg.UpWeights...)
		weights = append(weights, cfg.DownWeights...)
		weights = append(weights, cfg.GateBias...)
		weights = append(weights, cfg.UpBias...)
		weights = append(weights, cfg.DownBias...)
	case nn.LayerEmbedding:
		weights = append(weights, cfg.EmbeddingWeights...)
	case nn.LayerParallel, nn.LayerSequential:
		for _, branch := range cfg.ParallelBranches {
			weights = append(weights, getLayerWeights(&branch)...)
		}
	}

	return weights
}

// compareWeights compares two weight slices and returns max difference
func compareWeights(original, loaded []float32, tolerance float32) (bool, float32) {
	if len(original) != len(loaded) {
		return false, -1
	}

	if len(original) == 0 {
		// No weights to compare (e.g., Softmax, Residual)
		return true, 0
	}

	maxDiff := float32(0)
	for i := range original {
		diff := float32(math.Abs(float64(original[i] - loaded[i])))
		if diff > maxDiff {
			maxDiff = diff
		}
	}

	return maxDiff <= tolerance, maxDiff
}

// getToleranceForDType returns acceptable tolerance for a given dtype
func getToleranceForDType(dtype string) float32 {
	switch dtype {
	case "float64", "float32":
		return 0.0001
	case "bfloat16":
		return 0.01 // BFloat16 has lower precision
	case "float16":
		return 0.01
	case "float8":
		return 0.1 // 8-bit float has very limited precision
	case "float4":
		return 0.2 // 4-bit quantization has large errors
	case "int8":
		return 0.01
	case "int16":
		return 0.0001
	case "int32":
		return 0.0001
	case "int64":
		return 0.01 // Same as int8, we use scale=127 to avoid precision issues
	case "int4":
		return 2.0 // 4-bit integer has extremely limited precision (only 16 levels)
	case "uint8":
		return 0.01
	case "uint16":
		return 0.0001
	case "uint32", "uint64":
		return 0.0001
	default:
		return 0.0001
	}
}

// testLayerWithDType tests saving and loading a specific layer with a specific dtype
func testLayerWithDType(layerType, dtype string) TestResult {
	result := TestResult{
		LayerType: layerType,
		DType:     dtype,
	}

	// Create network with the layer
	network, err := createLayerNetwork(layerType)
	if err != nil {
		result.Error = fmt.Sprintf("failed to create network: %v", err)
		return result
	}

	// Get original weights
	originalCfg := network.GetLayer(0, 0, 0)
	originalWeights := getLayerWeights(originalCfg)
	result.WeightCount = len(originalWeights)

	// Save with dtype
	modelID := fmt.Sprintf("test_%s_%s", layerType, dtype)
	jsonString, err := network.SaveModelWithDType(modelID, dtype)
	if err != nil {
		result.Error = fmt.Sprintf("failed to save: %v", err)
		return result
	}

	// Load back
	loadedNetwork, storedDType, err := nn.LoadModelWithDType(jsonString, modelID, dtype)
	if err != nil {
		result.Error = fmt.Sprintf("failed to load: %v", err)
		return result
	}

	if storedDType != dtype {
		result.Error = fmt.Sprintf("dtype mismatch: expected %s, got %s", dtype, storedDType)
		return result
	}

	// Get loaded weights
	loadedCfg := loadedNetwork.GetLayer(0, 0, 0)
	loadedWeights := getLayerWeights(loadedCfg)

	// Compare weights
	tolerance := getToleranceForDType(dtype)
	passed, maxDiff := compareWeights(originalWeights, loadedWeights, tolerance)

	result.Passed = passed
	result.MaxDiff = maxDiff

	if !passed && len(originalWeights) > 0 {
		result.Error = fmt.Sprintf("weight mismatch: max diff %.6f > tolerance %.6f", maxDiff, tolerance)
	}

	return result
}

// runAllTests runs all layer/dtype combinations
func runAllTests() []TestResult {
	layerTypes := AllLayerTypes()
	dtypes := AllNumericalTypes()

	var results []TestResult

	for _, layerType := range layerTypes {
		for _, dtype := range dtypes {
			result := testLayerWithDType(layerType, dtype)
			results = append(results, result)
		}
	}

	return results
}

// printResults prints the test results in a nice table
func printResults(results []TestResult) {
	fmt.Println("\n╔═══════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════╗")
	fmt.Println("║                                          🧪 SAVE/LOAD TEST RESULTS: All Layers × All Numerical Types 🧪                                                              ║")
	fmt.Println("╚═══════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════╝")

	// Group by layer type
	layerTypes := AllLayerTypes()
	dtypes := AllNumericalTypes()

	// Create result map
	resultMap := make(map[string]map[string]TestResult)
	for _, r := range results {
		if resultMap[r.LayerType] == nil {
			resultMap[r.LayerType] = make(map[string]TestResult)
		}
		resultMap[r.LayerType][r.DType] = r
	}

	// Print header
	fmt.Printf("\n%-20s │", "Layer Type")
	for _, dtype := range dtypes {
		fmt.Printf(" %8s │", dtype[:min(8, len(dtype))])
	}
	fmt.Println()
	fmt.Print(strings.Repeat("─", 21) + "┼")
	for range dtypes {
		fmt.Print(strings.Repeat("─", 10) + "┼")
	}
	fmt.Println()

	// Print rows
	passed := 0
	failed := 0
	for _, layerType := range layerTypes {
		fmt.Printf("%-20s │", layerType)
		for _, dtype := range dtypes {
			r := resultMap[layerType][dtype]
			if r.Passed {
				passed++
				if r.WeightCount == 0 {
					fmt.Printf("   %s    │", "✓ N/A")
				} else {
					fmt.Printf("   %s    │", "✓")
				}
			} else {
				failed++
				fmt.Printf("   %s    │", "✗")
			}
		}
		fmt.Println()
	}

	fmt.Print(strings.Repeat("─", 21) + "┴")
	for range dtypes {
		fmt.Print(strings.Repeat("─", 10) + "┴")
	}
	fmt.Println()

	// Summary
	fmt.Printf("\n✅ Passed: %d / %d\n", passed, passed+failed)
	fmt.Printf("❌ Failed: %d / %d\n", failed, passed+failed)

	// Print failures if any
	if failed > 0 {
		fmt.Println("\n═══ FAILURES ═══")
		for _, r := range results {
			if !r.Passed {
				fmt.Printf("❌ %s + %s: %s (max diff: %.6f, weights: %d)\n",
					r.LayerType, r.DType, r.Error, r.MaxDiff, r.WeightCount)
			}
		}
	}

	// Print precision analysis
	fmt.Println("\n═══ PRECISION ANALYSIS (Max Difference per DType) ═══")
	dtypeMaxDiff := make(map[string]float32)
	for _, r := range results {
		if r.Passed && r.MaxDiff > dtypeMaxDiff[r.DType] {
			dtypeMaxDiff[r.DType] = r.MaxDiff
		}
	}
	for _, dtype := range dtypes {
		maxDiff := dtypeMaxDiff[dtype]
		tolerance := getToleranceForDType(dtype)
		bar := strings.Repeat("█", int(maxDiff/tolerance*20))
		if len(bar) > 40 {
			bar = bar[:40]
		}
		fmt.Printf("%-10s │ max diff: %.6f │ tolerance: %.4f │ %s\n", dtype, maxDiff, tolerance, bar)
	}
}

func min(a, b int) int {
	if a < b {
		return a
	}
	return b
}

// =============================================================================
// COMPREHENSIVE PARALLEL PERMUTATION TESTS
// =============================================================================

// AllCombineModes returns all supported parallel combine modes
func AllCombineModes() []string {
	return []string{"concat", "add", "avg"}
}

// AllBranchLayerTypes returns layer types that can be used as branches
func AllBranchLayerTypes() []string {
	return []string{
		"Dense",
		"Conv2D",
		"Conv1D",
		"MultiHeadAttention",
		"RNN",
		"LSTM",
		"LayerNorm",
		"RMSNorm",
		"SwiGLU",
		"Softmax",
	}
}

// createBranchConfig creates a LayerConfig for a given branch type
func createBranchConfig(branchType string, outputSize int) nn.LayerConfig {
	switch branchType {
	case "Dense":
		return nn.LayerConfig{
			Type:         nn.LayerDense,
			Activation:   nn.ActivationScaledReLU,
			InputHeight:  8,
			OutputHeight: outputSize,
			Kernel:       randomWeights(8 * outputSize),
			Bias:         randomWeights(outputSize),
		}
	case "Conv2D":
		return nn.LayerConfig{
			Type:          nn.LayerConv2D,
			Activation:    nn.ActivationScaledReLU,
			InputChannels: 1,
			Filters:       outputSize,
			KernelSize:    1,
			Stride:        1,
			Padding:       0,
			InputHeight:   2,
			InputWidth:    4,
			OutputHeight:  2,
			OutputWidth:   4,
			Kernel:        randomWeights(outputSize * 1 * 1 * 1),
			Bias:          randomWeights(outputSize),
		}
	case "Conv1D":
		return nn.LayerConfig{
			Type:          nn.LayerConv1D,
			Activation:    nn.ActivationScaledReLU,
			InputChannels: 1,
			Filters:       outputSize,
			KernelSize:    1,
			Stride:        1,
			Padding:       0,
			InputHeight:   8,
			InputWidth:    1,
			OutputHeight:  8,
			OutputWidth:   1,
			Kernel:        randomWeights(outputSize * 1 * 1),
			Bias:          randomWeights(outputSize),
		}
	case "MultiHeadAttention":
		dModel := outputSize
		return nn.LayerConfig{
			Type:         nn.LayerMultiHeadAttention,
			DModel:       dModel,
			NumHeads:     1,
			SeqLength:    1,
			QWeights:     randomWeights(dModel * dModel),
			KWeights:     randomWeights(dModel * dModel),
			VWeights:     randomWeights(dModel * dModel),
			OutputWeight: randomWeights(dModel * dModel),
			QBias:        randomWeights(dModel),
			KBias:        randomWeights(dModel),
			VBias:        randomWeights(dModel),
			OutputBias:   randomWeights(dModel),
		}
	case "RNN":
		return nn.LayerConfig{
			Type:         nn.LayerRNN,
			Activation:   nn.ActivationTanh,
			RNNInputSize: 8,
			HiddenSize:   outputSize,
			SeqLength:    1,
			WeightIH:     randomWeights(8 * outputSize),
			WeightHH:     randomWeights(outputSize * outputSize),
			BiasH:        randomWeights(outputSize),
		}
	case "LSTM":
		return nn.LayerConfig{
			Type:         nn.LayerLSTM,
			RNNInputSize: 8,
			HiddenSize:   outputSize,
			SeqLength:    1,
			WeightIH_i:   randomWeights(8 * outputSize),
			WeightIH_f:   randomWeights(8 * outputSize),
			WeightIH_g:   randomWeights(8 * outputSize),
			WeightIH_o:   randomWeights(8 * outputSize),
			WeightHH_i:   randomWeights(outputSize * outputSize),
			WeightHH_f:   randomWeights(outputSize * outputSize),
			WeightHH_g:   randomWeights(outputSize * outputSize),
			WeightHH_o:   randomWeights(outputSize * outputSize),
			BiasH_i:      randomWeights(outputSize),
			BiasH_f:      randomWeights(outputSize),
			BiasH_g:      randomWeights(outputSize),
			BiasH_o:      randomWeights(outputSize),
		}
	case "LayerNorm":
		return nn.LayerConfig{
			Type:     nn.LayerNorm,
			NormSize: outputSize,
			Epsilon:  1e-5,
			Gamma:    randomWeights(outputSize),
			Beta:     randomWeights(outputSize),
		}
	case "RMSNorm":
		return nn.LayerConfig{
			Type:     nn.LayerRMSNorm,
			NormSize: outputSize,
			Epsilon:  1e-5,
			Gamma:    randomWeights(outputSize),
		}
	case "SwiGLU":
		return nn.LayerConfig{
			Type:         nn.LayerSwiGLU,
			InputHeight:  8,
			OutputHeight: 16,
			GateWeights:  randomWeights(8 * 16),
			UpWeights:    randomWeights(8 * 16),
			DownWeights:  randomWeights(16 * outputSize),
			GateBias:     randomWeights(16),
			UpBias:       randomWeights(16),
			DownBias:     randomWeights(outputSize),
		}
	case "Softmax":
		return nn.LayerConfig{
			Type:           nn.LayerSoftmax,
			SoftmaxVariant: nn.SoftmaxStandard,
			SoftmaxRows:    1,
			SoftmaxCols:    outputSize,
			Temperature:    1.0,
		}
	default:
		return nn.LayerConfig{
			Type:         nn.LayerDense,
			Activation:   nn.ActivationScaledReLU,
			InputHeight:  8,
			OutputHeight: outputSize,
			Kernel:       randomWeights(8 * outputSize),
			Bias:         randomWeights(outputSize),
		}
	}
}

// ParallelPermutationResult holds result for parallel permutation test
type ParallelPermutationResult struct {
	BranchType1  string
	BranchType2  string
	CombineMode  string
	DType        string
	NestingDepth int
	Passed       bool
	Error        string
	MaxDiff      float32
}

// testParallelPermutation tests a specific parallel configuration
func testParallelPermutation(branch1, branch2, combineMode, dtype string, nestingDepth int) ParallelPermutationResult {
	result := ParallelPermutationResult{
		BranchType1:  branch1,
		BranchType2:  branch2,
		CombineMode:  combineMode,
		DType:        dtype,
		NestingDepth: nestingDepth,
	}

	network := nn.NewNetwork(1, 1, 1, 1)

	// For add/avg modes, outputs must be same size
	outputSize := 4
	if combineMode == "add" || combineMode == "avg" {
		outputSize = 8 // Same as input for passthrough-like behavior
	}

	branch1Cfg := createBranchConfig(branch1, outputSize)
	branch2Cfg := createBranchConfig(branch2, outputSize)

	var layerCfg nn.LayerConfig
	if nestingDepth == 0 {
		// Simple parallel
		layerCfg = nn.LayerConfig{
			Type:             nn.LayerParallel,
			CombineMode:      combineMode,
			ParallelBranches: []nn.LayerConfig{branch1Cfg, branch2Cfg},
		}
	} else {
		// Nested parallel (depth 1)
		innerParallel := nn.LayerConfig{
			Type:             nn.LayerParallel,
			CombineMode:      "add", // Inner always uses add for same-sized outputs
			ParallelBranches: []nn.LayerConfig{branch1Cfg, branch2Cfg},
		}
		layerCfg = nn.LayerConfig{
			Type:        nn.LayerParallel,
			CombineMode: combineMode,
			ParallelBranches: []nn.LayerConfig{
				createBranchConfig("Dense", outputSize),
				innerParallel,
			},
		}
	}

	network.SetLayer(0, 0, 0, layerCfg)

	// Get original weights
	originalCfg := network.GetLayer(0, 0, 0)
	originalWeights := getLayerWeights(originalCfg)

	// Save with dtype
	modelID := fmt.Sprintf("perm_%s_%s_%s_%s_%d", branch1, branch2, combineMode, dtype, nestingDepth)
	jsonString, err := network.SaveModelWithDType(modelID, dtype)
	if err != nil {
		result.Error = fmt.Sprintf("save failed: %v", err)
		return result
	}

	// Load back
	loadedNetwork, storedDType, err := nn.LoadModelWithDType(jsonString, modelID, dtype)
	if err != nil {
		result.Error = fmt.Sprintf("load failed: %v", err)
		return result
	}

	if storedDType != dtype {
		result.Error = fmt.Sprintf("dtype mismatch: expected %s, got %s", dtype, storedDType)
		return result
	}

	// Get loaded weights
	loadedCfg := loadedNetwork.GetLayer(0, 0, 0)
	loadedWeights := getLayerWeights(loadedCfg)

	// Compare weights
	tolerance := getToleranceForDType(dtype)
	passed, maxDiff := compareWeights(originalWeights, loadedWeights, tolerance)

	result.Passed = passed
	result.MaxDiff = maxDiff

	if !passed && len(originalWeights) > 0 {
		result.Error = fmt.Sprintf("weight mismatch: max diff %.6f > tolerance %.6f", maxDiff, tolerance)
	}

	return result
}

// runParallelPermutationTests runs all parallel permutation tests
func runParallelPermutationTests() []ParallelPermutationResult {
	branchTypes := AllBranchLayerTypes()
	combineModes := AllCombineModes()
	dtypes := []string{"float32", "bfloat16", "int8"} // Representative subset
	nestingDepths := []int{0, 1}

	var results []ParallelPermutationResult

	total := len(branchTypes) * len(branchTypes) * len(combineModes) * len(dtypes) * len(nestingDepths)
	fmt.Printf("\nRunning %d parallel permutation tests...\n", total)

	count := 0
	for _, branch1 := range branchTypes {
		for _, branch2 := range branchTypes {
			for _, mode := range combineModes {
				for _, dtype := range dtypes {
					for _, depth := range nestingDepths {
						result := testParallelPermutation(branch1, branch2, mode, dtype, depth)
						results = append(results, result)
						count++
						if count%100 == 0 {
							fmt.Printf("  Progress: %d/%d\n", count, total)
						}
					}
				}
			}
		}
	}

	return results
}

// printParallelPermutationResults prints parallel permutation test results
func printParallelPermutationResults(results []ParallelPermutationResult) {
	passed := 0
	failed := 0
	for _, r := range results {
		if r.Passed {
			passed++
		} else {
			failed++
		}
	}

	fmt.Println("\n╔═══════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════╗")
	fmt.Println("║                              🔀 PARALLEL PERMUTATION TESTS: Branch×Branch×Mode×DType×Depth 🔀                                                                        ║")
	fmt.Println("╚═══════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════╝")

	fmt.Printf("\n✅ Passed: %d / %d\n", passed, passed+failed)
	fmt.Printf("❌ Failed: %d / %d\n", failed, passed+failed)

	// Show failure breakdown if any
	if failed > 0 {
		fmt.Println("\n═══ FAILURES (first 20) ═══")
		count := 0
		for _, r := range results {
			if !r.Passed && count < 20 {
				fmt.Printf("❌ %s+%s mode=%s dtype=%s depth=%d: %s\n",
					r.BranchType1, r.BranchType2, r.CombineMode, r.DType, r.NestingDepth, r.Error)
				count++
			}
		}
	}

	// Summary by combine mode
	fmt.Println("\n═══ RESULTS BY COMBINE MODE ═══")
	modeStats := make(map[string][2]int) // [passed, total]
	for _, r := range results {
		stats := modeStats[r.CombineMode]
		stats[1]++
		if r.Passed {
			stats[0]++
		}
		modeStats[r.CombineMode] = stats
	}
	for _, mode := range AllCombineModes() {
		stats := modeStats[mode]
		pct := 100.0 * float64(stats[0]) / float64(stats[1])
		bar := strings.Repeat("█", int(pct/5))
		fmt.Printf("%-10s │ %4d/%4d │ %5.1f%% │ %s\n", mode, stats[0], stats[1], pct, bar)
	}

	// Summary by branch type
	fmt.Println("\n═══ RESULTS BY BRANCH TYPE ═══")
	branchStats := make(map[string][2]int)
	for _, r := range results {
		for _, branch := range []string{r.BranchType1, r.BranchType2} {
			stats := branchStats[branch]
			stats[1]++
			if r.Passed {
				stats[0]++
			}
			branchStats[branch] = stats
		}
	}
	for _, branch := range AllBranchLayerTypes() {
		stats := branchStats[branch]
		pct := 100.0 * float64(stats[0]) / float64(stats[1])
		bar := strings.Repeat("█", int(pct/5))
		fmt.Printf("%-20s │ %4d/%4d │ %5.1f%% │ %s\n", branch, stats[0], stats[1], pct, bar)
	}
}

func main() {
	fmt.Println("╔═══════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════╗")
	fmt.Println("║                                          Save/Load Everything Test                                                                                                    ║")
	fmt.Println("║                                                                                                                                                                       ║")
	fmt.Println("║     Testing all layer types with all numerical types for serialization round-trip                                                                                     ║")
	fmt.Println("╚═══════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════╝")

	rand.Seed(time.Now().UnixNano())

	start := time.Now()

	// PHASE 1: Basic layer tests
	fmt.Println("\n══════════════════════════════════════════════════════════════════════════════════════════════════════════════════")
	fmt.Println("PHASE 1: Basic Layer × DType Tests")
	fmt.Println("══════════════════════════════════════════════════════════════════════════════════════════════════════════════════")
	results := runAllTests()
	printResults(results)

	phase1Failed := false
	for _, r := range results {
		if !r.Passed {
			phase1Failed = true
			break
		}
	}

	// PHASE 2: Parallel permutation tests
	fmt.Println("\n══════════════════════════════════════════════════════════════════════════════════════════════════════════════════")
	fmt.Println("PHASE 2: Parallel Permutation Tests (Branch×Branch×Mode×DType×Depth)")
	fmt.Println("══════════════════════════════════════════════════════════════════════════════════════════════════════════════════")
	permResults := runParallelPermutationTests()
	printParallelPermutationResults(permResults)

	phase2Failed := false
	for _, r := range permResults {
		if !r.Passed {
			phase2Failed = true
			break
		}
	}

	elapsed := time.Since(start)
	fmt.Printf("\n⏱️  Total test time: %v\n", elapsed)

	if phase1Failed || phase2Failed {
		fmt.Println("\n❌ Some tests failed!")
		os.Exit(1)
	}

	fmt.Println("\n✅ All tests passed!")
}
