package main

import (
	"fmt"
	"math"
	"math/rand"
	"os"
	"path/filepath"

	"github.com/openfluke/loom/nn"
)

// LayerTestResult holds result for a layer+dtype test
type LayerTestResult struct {
	LayerType string
	DType     string
	Passed    bool
	Error     string
	MaxDiff   float32
}

// AllLayerTypes returns all layer types to test
func AllLayerTypes() []string {
	return []string{
		"Dense",
		"Conv1D",
		"Conv2D",
		"LayerNorm",
		"RMSNorm",
		"Embedding",
		"MultiHeadAttention",
		"RNN",
		"LSTM",
		"SwiGLU",
		"Softmax",
		// Start with these, add more as they pass
	}
}

// createTestNetwork creates a network with a single layer of the specified type
func createTestNetwork(layerType string) (*nn.Network, error) {
	// Create a basic 1x1x1 network
	network := nn.NewNetwork(1, 1, 1, 1)

	var config nn.LayerConfig

	switch layerType {
	case "Dense":
		config = nn.LayerConfig{
			Type:         nn.LayerDense,
			Activation:   nn.ActivationScaledReLU,
			InputHeight:  4,
			OutputHeight: 3,
			Kernel:       randomWeights(4 * 3),
			Bias:         randomWeights(3),
		}
	case "Conv1D":
		config = nn.LayerConfig{
			Type:          nn.LayerConv1D,
			Activation:    nn.ActivationScaledReLU,
			InputChannels: 2,
			Filters:       3,
			KernelSize:    3,
			Stride:        1,
			Padding:       1,
			InputHeight:   8, // Signal length
			InputWidth:    1,
			Conv1DFilters: 3,                        // Set duplicate field just in case
			Kernel:        randomWeights(3 * 2 * 3), // [filters][inChannels][kernelSize]
			Bias:          randomWeights(3),
		}
	case "Conv2D":
		config = nn.LayerConfig{
			Type:          nn.LayerConv2D,
			Activation:    nn.ActivationScaledReLU,
			InputChannels: 2,
			Filters:       3,
			KernelSize:    3,
			Stride:        1,
			Padding:       1,
			InputHeight:   4,
			InputWidth:    4,
			Kernel:        randomWeights(3 * 2 * 3 * 3), // [filters][inChannels][H][W]
			Bias:          randomWeights(3),
		}
	case "LayerNorm":
		config = nn.LayerConfig{
			Type:     nn.LayerNorm,
			NormSize: 4,
			Epsilon:  1e-5,
			Gamma:    randomWeights(4),
			Beta:     randomWeights(4),
		}
	case "RMSNorm":
		config = nn.LayerConfig{
			Type:     nn.LayerRMSNorm,
			NormSize: 4,
			Epsilon:  1e-5,
			Gamma:    randomWeights(4),
		}
	case "Embedding":
		config = nn.LayerConfig{
			Type:             nn.LayerEmbedding,
			VocabSize:        10,
			EmbeddingDim:     4,
			EmbeddingWeights: randomWeights(10 * 4),
		}
	case "MultiHeadAttention":
		dModel := 8
		config = nn.LayerConfig{
			Type:         nn.LayerMultiHeadAttention,
			DModel:       dModel,
			NumHeads:     2,
			SeqLength:    4,
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
		config = nn.LayerConfig{
			Type:         nn.LayerRNN,
			Activation:   nn.ActivationTanh,
			RNNInputSize: 4,
			HiddenSize:   8,
			SeqLength:    4,
			WeightIH:     randomWeights(8 * 4),
			WeightHH:     randomWeights(8 * 8),
			BiasH:        randomWeights(8),
		}
	case "LSTM":
		inputSize := 4
		hiddenSize := 8
		config = nn.LayerConfig{
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
		}
	case "SwiGLU":
		inputSize := 4
		intermediateSize := 8
		config = nn.LayerConfig{
			Type:         nn.LayerSwiGLU,
			InputHeight:  inputSize,
			OutputHeight: intermediateSize,
			GateWeights:  randomWeights(inputSize * intermediateSize),
			UpWeights:    randomWeights(inputSize * intermediateSize),
			DownWeights:  randomWeights(intermediateSize * inputSize),
			GateBias:     randomWeights(intermediateSize),
			UpBias:       randomWeights(intermediateSize),
			DownBias:     randomWeights(inputSize),
		}
	case "Softmax":
		config = nn.LayerConfig{
			Type:           nn.LayerSoftmax,
			SoftmaxVariant: nn.SoftmaxStandard,
			// Softmax typically has no weights, so nothing to save/load here
			// but we test it to ensure it doesn't crash or save empty files improperly
		}
	default:
		return nil, fmt.Errorf("unsupported layer type: %s", layerType)
	}

	network.SetLayer(0, 0, 0, config)
	return network, nil
}

// randomWeights generates random weights
func randomWeights(n int) []float32 {
	w := make([]float32, n)
	for i := range w {
		w[i] = rand.Float32()*2 - 1
	}
	return w
}

// extractLayerWeights extracts all weights from a layer's config into a map
func extractLayerWeights(cfg *nn.LayerConfig) map[string][]float32 {
	weights := make(map[string][]float32)

	// Add weights based on what's available (non-nil/non-empty)
	// We use the same field names as definition

	switch cfg.Type {
	case nn.LayerDense, nn.LayerConv2D, nn.LayerConv1D:
		if len(cfg.Kernel) > 0 {
			weights["kernel"] = cfg.Kernel
		}
		if len(cfg.Bias) > 0 {
			weights["bias"] = cfg.Bias
		}

	case nn.LayerNorm:
		if len(cfg.Gamma) > 0 {
			weights["gamma"] = cfg.Gamma
		}
		if len(cfg.Beta) > 0 {
			weights["beta"] = cfg.Beta
		}

	case nn.LayerRMSNorm:
		if len(cfg.Gamma) > 0 {
			weights["gamma"] = cfg.Gamma
		}

	case nn.LayerEmbedding:
		if len(cfg.EmbeddingWeights) > 0 {
			weights["embedding_weights"] = cfg.EmbeddingWeights
		}

	case nn.LayerMultiHeadAttention:
		if len(cfg.QWeights) > 0 {
			weights["q_weights"] = cfg.QWeights
		}
		if len(cfg.KWeights) > 0 {
			weights["k_weights"] = cfg.KWeights
		}
		if len(cfg.VWeights) > 0 {
			weights["v_weights"] = cfg.VWeights
		}
		if len(cfg.OutputWeight) > 0 {
			weights["output_weight"] = cfg.OutputWeight
		}
		if len(cfg.QBias) > 0 {
			weights["q_bias"] = cfg.QBias
		}
		if len(cfg.KBias) > 0 {
			weights["k_bias"] = cfg.KBias
		}
		if len(cfg.VBias) > 0 {
			weights["v_bias"] = cfg.VBias
		}
		if len(cfg.OutputBias) > 0 {
			weights["output_bias"] = cfg.OutputBias
		}

	case nn.LayerRNN:
		if len(cfg.WeightIH) > 0 {
			weights["weight_ih"] = cfg.WeightIH
		}
		if len(cfg.WeightHH) > 0 {
			weights["weight_hh"] = cfg.WeightHH
		}
		if len(cfg.BiasH) > 0 {
			weights["bias_h"] = cfg.BiasH
		}

	case nn.LayerLSTM:
		if len(cfg.WeightIH_i) > 0 {
			weights["weight_ih_i"] = cfg.WeightIH_i
		}
		if len(cfg.WeightIH_f) > 0 {
			weights["weight_ih_f"] = cfg.WeightIH_f
		}
		if len(cfg.WeightIH_g) > 0 {
			weights["weight_ih_g"] = cfg.WeightIH_g
		}
		if len(cfg.WeightIH_o) > 0 {
			weights["weight_ih_o"] = cfg.WeightIH_o
		}

		if len(cfg.WeightHH_i) > 0 {
			weights["weight_hh_i"] = cfg.WeightHH_i
		}
		if len(cfg.WeightHH_f) > 0 {
			weights["weight_hh_f"] = cfg.WeightHH_f
		}
		if len(cfg.WeightHH_g) > 0 {
			weights["weight_hh_g"] = cfg.WeightHH_g
		}
		if len(cfg.WeightHH_o) > 0 {
			weights["weight_hh_o"] = cfg.WeightHH_o
		}

		if len(cfg.BiasH_i) > 0 {
			weights["bias_h_i"] = cfg.BiasH_i
		}
		if len(cfg.BiasH_f) > 0 {
			weights["bias_h_f"] = cfg.BiasH_f
		}
		if len(cfg.BiasH_g) > 0 {
			weights["bias_h_g"] = cfg.BiasH_g
		}
		if len(cfg.BiasH_o) > 0 {
			weights["bias_h_o"] = cfg.BiasH_o
		}

	case nn.LayerSwiGLU:
		if len(cfg.GateWeights) > 0 {
			weights["gate_weights"] = cfg.GateWeights
		}
		if len(cfg.UpWeights) > 0 {
			weights["up_weights"] = cfg.UpWeights
		}
		if len(cfg.DownWeights) > 0 {
			weights["down_weights"] = cfg.DownWeights
		}
		if len(cfg.GateBias) > 0 {
			weights["gate_bias"] = cfg.GateBias
		}
		if len(cfg.UpBias) > 0 {
			weights["up_bias"] = cfg.UpBias
		}
		if len(cfg.DownBias) > 0 {
			weights["down_bias"] = cfg.DownBias
		}
	}

	return weights
}

// testLayerWithType tests saving/loading a layer with a specific dtype
func testLayerWithType(layerType string, dtype nn.NumericType) LayerTestResult {
	result := LayerTestResult{
		LayerType: layerType,
		DType:     string(dtype),
		Passed:    false,
	}

	// Create network
	network, err := createTestNetwork(layerType)
	if err != nil {
		result.Error = fmt.Sprintf("Create network failed: %v", err)
		return result
	}

	// Get layer config
	layerCfg := network.GetLayer(0, 0, 0)

	// Extract original weights
	originalWeights := extractLayerWeights(layerCfg)

	// Skip weight check for parameter-less layers
	if len(originalWeights) == 0 {
		if layerType == "Softmax" {
			result.Passed = true
			return result
		}
		result.Error = "No weights found in layer (or extraction failed)"
		return result
	}

	// Convert all weights to target dtype and create SafeTensors
	tensors := make(map[string]nn.TensorWithShape)

	for name, weights := range originalWeights {
		// Verify conversion is possible (optional check)
		_, err := nn.ConvertSlice(weights, nn.TypeF32, dtype)
		if err != nil {
			result.Error = fmt.Sprintf("Conversion to %s failed: %v", dtype, err)
			return result
		}

		// Keep weights as F32 but specify target dtype, SaveSafetensors handles conversion
		tensors[name] = nn.TensorWithShape{
			Values: weights,
			Shape:  []int{len(weights)},
			DType:  string(dtype),
		}
	}

	// Save to temporary file
	tmpFile := filepath.Join(os.TempDir(), fmt.Sprintf("test_%s_%s.safetensors", layerType, dtype))
	defer os.Remove(tmpFile)

	err = nn.SaveSafetensors(tmpFile, tensors)
	if err != nil {
		result.Error = fmt.Sprintf("Save failed: %v", err)
		return result
	}

	// Load back
	data, err := os.ReadFile(tmpFile)
	if err != nil {
		result.Error = fmt.Sprintf("Read file failed: %v", err)
		return result
	}

	loadedTensors, err := nn.LoadSafetensorsWithShapes(data)
	if err != nil {
		result.Error = fmt.Sprintf("Load failed: %v", err)
		return result
	}

	// Verify all weights
	var maxDiff float32

	for name, original := range originalWeights {
		loaded, ok := loadedTensors[name]
		if !ok {
			result.Error = fmt.Sprintf("Weight %s not found in loaded file", name)
			return result
		}

		// Check dtype (metadata)
		if loaded.DType != string(dtype) {
			result.Error = fmt.Sprintf("DType mismatch for %s: expected %s, got %s", name, dtype, loaded.DType)
			return result
		}

		// Compare values
		if len(loaded.Values) != len(original) {
			result.Error = fmt.Sprintf("Length mismatch for %s: expected %d, got %d", name, len(original), len(loaded.Values))
			return result
		}

		for i := range original {
			// Calculate expected value by simulating conversion
			// We use ConvertValue to simulate F32 -> Target -> F32 round trip
			// This represents what happened during save/load

			// 1. F32 -> Target
			valAsTarget, _ := nn.ConvertValue(original[i], nn.TypeF32, dtype)

			// 2. Target -> F32
			valReconstructedInterface, _ := nn.ConvertValue(valAsTarget, dtype, nn.TypeF32)
			valReconstructed := valReconstructedInterface.(float32)

			// The loaded value should match the reconstructed value VERY closely
			// (basically exact, except for potential float intermediate jitter, but since we use same logic...).
			// Actually, let's compare against the *reconstructed* value, because that's what we expect to find on disk.
			// Comparing against *original* verifies that our quantization/tolerance logic is correct vs reality.
			// Let's stick to comparing against original and using appropriate tolerances,
			// OR compare against reconstructed with tight tolerance to verify IO integrity perfectly.
			//
			// Creating a "Perfect IO" test vs "Precision Loss" test.
			// Let's do IO Integrity test: Loaded == Reconstructed

			diff := float32(math.Abs(float64(loaded.Values[i] - valReconstructed)))

			// Use a very tight tolerance for IO integrity (should be basically zero/epsilon)
			ioTolerance := float32(1e-6)

			if diff > maxDiff {
				maxDiff = diff
			}

			if diff > ioTolerance {
				result.Error = fmt.Sprintf("IO Mismatch %s[%d]: Original=%.4f, Reconstructed(Expected)=%.4f, Loaded=%.4f, Diff=%.6f",
					name, i, original[i], valReconstructed, loaded.Values[i], diff)
				result.MaxDiff = maxDiff
				return result
			}
		}
	}

	result.MaxDiff = maxDiff
	result.Passed = true
	return result
}

// getToleranceForDType returns acceptable tolerance for each dtype (used if we compare to original)
// Kept for reference, but currently using IO integrity check against simulated conversion
func getToleranceForDType(dtype string) float32 {
	switch dtype {
	case "F32", "F64":
		return 1e-6
	case "F16":
		return 1e-3
	case "BF16":
		return 1e-2
	case "F4":
		return 0.5
	default:
		return 1.0 // Integers
	}
}

// runAllTests runs all layer+type combinations
func runAllTests() []LayerTestResult {
	layers := AllLayerTypes()
	types := []nn.NumericType{
		nn.TypeF32, nn.TypeF64, nn.TypeF16, nn.TypeBF16, nn.TypeF4,
		nn.TypeI8, nn.TypeI16, nn.TypeI32, nn.TypeI64,
		nn.TypeU8, nn.TypeU16, nn.TypeU32, nn.TypeU64,
	}

	var results []LayerTestResult
	total := len(layers) * len(types)
	current := 0

	fmt.Printf("Running %d tests (%d layers × %d types)...\n\n", total, len(layers), len(types))

	for _, layer := range layers {
		fmt.Printf("Testing %s layer:\n", layer)
		for _, dtype := range types {
			current++

			// Skip F4 for now if it's causing generic issues (F4 save needs fix from previous task!)
			// But we'll try running it to see failures.

			result := testLayerWithType(layer, dtype)
			results = append(results, result)

			status := "✅ PASS"
			if !result.Passed {
				status = "❌ FAIL"
			}

			// Short output
			fmt.Printf("  %-4s: %s (diff: %.6f) %s\n", dtype, status, result.MaxDiff, truncate(result.Error, 40))
		}
		fmt.Println()
	}

	return results
}

func truncate(s string, maxLen int) string {
	if len(s) == 0 {
		return ""
	}
	if len(s) <= maxLen {
		return s
	}
	return s[:maxLen-3] + "..."
}

// printSummary prints test summary
func printSummary(results []LayerTestResult) {
	fmt.Println("\n╔══════════════════════════════════════════════════════════════╗")
	fmt.Println("║     SafeTensors All Layers + All Types Test Summary         ║")
	fmt.Println("╚══════════════════════════════════════════════════════════════╝")
	fmt.Println()

	passed := 0
	failed := 0

	// Count by layer
	layerStats := make(map[string]struct{ pass, fail int })
	for _, r := range results {
		if r.Passed {
			passed++
			stats := layerStats[r.LayerType]
			stats.pass++
			layerStats[r.LayerType] = stats
		} else {
			failed++
			stats := layerStats[r.LayerType]
			stats.fail++
			layerStats[r.LayerType] = stats

		}
	}

	fmt.Println("By Layer Type:")
	fmt.Println("─────────────────────────────────────────")
	for _, layer := range AllLayerTypes() {
		stats := layerStats[layer]
		total := stats.pass + stats.fail
		if total > 0 {
			fmt.Printf("%-12s: %2d/%2d passed (%.0f%%)\n",
				layer, stats.pass, total, float64(stats.pass)/float64(total)*100)
		}
	}

	fmt.Println()
	fmt.Printf("TOTAL: %d tests | ✅ %d passed | ❌ %d failed (%.1f%% pass rate)\n\n",
		len(results), passed, failed, float64(passed)/float64(len(results))*100)

	if failed == 0 {
		fmt.Println("🎉 All tests passed!")
	} else {
		fmt.Printf("⚠️  %d test(s) failed\n", failed)
	}
}

func main() {
	fmt.Println("╔══════════════════════════════════════════════════════════════╗")
	fmt.Println("║   SafeTensors: All Layers × All Numerical Types Test        ║")
	fmt.Println("╚══════════════════════════════════════════════════════════════╝")
	fmt.Println()

	results := runAllTests()
	printSummary(results)
}
