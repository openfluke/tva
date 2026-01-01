package main

import (
	"encoding/json"
	"fmt"
	"math"
	"math/rand"
	"strings"
	"time"

	"github.com/openfluke/loom/nn"
)

// =============================================================================
// LOOM v0.0.7 Complete Test Suite
// Tests all v0.0.7 features + multi-precision save/load for all layer types
// =============================================================================

func main() {
	fmt.Println("╔══════════════════════════════════════════════════════════════════════╗")
	fmt.Println("║               LOOM v0.0.7 Complete Feature Test Suite               ║")
	fmt.Println("╚══════════════════════════════════════════════════════════════════════╝")
	fmt.Println()

	passed := 0
	failed := 0

	// =========================================================================
	// PART 1: Core v0.0.7 Feature Tests
	// =========================================================================
	fmt.Println("═══════════════════════════════════════════════════════════════════════")
	fmt.Println("                     PART 1: CORE FEATURE TESTS                        ")
	fmt.Println("═══════════════════════════════════════════════════════════════════════")

	coreTests := []func() bool{
		testArchitectureGeneration,
		testFilterCombineMode,
		testSequentialLayers,
		testKMeansClustering,
		testCorrelationAnalysis,
	}

	for _, test := range coreTests {
		if test() {
			passed++
		} else {
			failed++
		}
	}

	// =========================================================================
	// PART 2: Multi-Precision Serialization Tests
	// =========================================================================
	fmt.Println("\n═══════════════════════════════════════════════════════════════════════")
	fmt.Println("           PART 2: MULTI-PRECISION SAVE/LOAD FOR ALL LAYERS           ")
	fmt.Println("═══════════════════════════════════════════════════════════════════════")

	dtypes := []string{"float32", "float64", "int32", "int16", "int8"}

	layerTests := []struct {
		name      string
		inputSize int
	}{
		{"Dense", 8},
		{"MHA", 64},
		{"RNN", 16},
		{"LSTM", 16},
		{"LayerNorm", 16},
		{"RMSNorm", 16},
		{"SwiGLU", 32},
		{"Conv2D", 16},
		{"Parallel", 8},
		{"Sequential", 8},
		{"Softmax", 8},
	}

	for _, lt := range layerTests {
		for _, dtype := range dtypes {
			p, f := testLayerWithDType(lt.name, lt.inputSize, dtype)
			passed += p
			failed += f
		}
	}

	// Final Summary
	fmt.Println()
	fmt.Println("╔══════════════════════════════════════════════════════════════════════╗")
	fmt.Printf("║              FINAL RESULTS: %d/%d TESTS PASSED                       ║\n", passed, passed+failed)
	fmt.Println("╚══════════════════════════════════════════════════════════════════════╝")

	if failed > 0 {
		fmt.Printf("\n❌ %d test(s) failed!\n", failed)
	} else {
		fmt.Println("\n🎉 All tests passed! Ready for 0.0.7 release!")
	}
}

// =============================================================================
// PART 1: Core Feature Tests
// =============================================================================

func testArchitectureGeneration() bool {
	fmt.Println("\n┌──────────────────────────────────────────────────────────────────────┐")
	fmt.Println("│ Architecture Generation with DType (architecture.go)               │")
	fmt.Println("└──────────────────────────────────────────────────────────────────────┘")

	// Test default options
	opts := nn.DefaultArchGenOptions()
	if len(opts.GridShapes) < 1 || len(opts.DModels) < 1 {
		fmt.Println("  ❌ DefaultArchGenOptions returned empty options")
		return false
	}
	fmt.Printf("  ✓ DefaultArchGenOptions: %d grid shapes, %d DModel sizes\n", len(opts.GridShapes), len(opts.DModels))

	// Default should have float32
	if len(opts.DTypes) == 0 || opts.DTypes[0] != "float32" {
		fmt.Println("  ❌ Default DType should be float32")
		return false
	}
	fmt.Printf("  ✓ Default DTypes: %v\n", opts.DTypes)

	// Generate with default (float32)
	configs := nn.GenerateDiverseConfigs(3, opts)
	if len(configs) != 3 {
		fmt.Printf("  ❌ Expected 3 configs, got %d\n", len(configs))
		return false
	}
	for i, cfg := range configs {
		if cfg.DType == "" {
			fmt.Printf("  ❌ Config %d has empty DType\n", i)
			return false
		}
	}
	fmt.Printf("  ✓ Generated %d configs with DType=%s\n", len(configs), configs[0].DType)

	// Test multi-precision generation
	multiPrecOpts := nn.DefaultArchGenOptions()
	multiPrecOpts.DTypes = []string{"float32", "float64", "int16", "int8"}
	multiPrecOpts.DTypeDistribution = []float64{0.25, 0.25, 0.25, 0.25} // Equal distribution

	multiConfigs := nn.GenerateDiverseConfigs(20, multiPrecOpts)
	dtypeCounts := make(map[string]int)
	for _, cfg := range multiConfigs {
		dtypeCounts[cfg.DType]++
	}
	fmt.Printf("  ✓ Multi-precision generation: %v\n", dtypeCounts)

	// Test JSON serialization with DType
	jsonData, err := json.Marshal(configs[0])
	if err != nil {
		fmt.Printf("  ❌ JSON serialization failed: %v\n", err)
		return false
	}
	if !strings.Contains(string(jsonData), `"dtype"`) {
		fmt.Println("  ❌ JSON does not contain dtype field")
		return false
	}
	fmt.Printf("  ✓ Config serializable to JSON with dtype (%d bytes)\n", len(jsonData))

	fmt.Println("  ✅ PASSED: Architecture Generation with DType")
	return true
}

func testFilterCombineMode() bool {
	fmt.Println("\n┌──────────────────────────────────────────────────────────────────────┐")
	fmt.Println("│ Parallel Filter Combine Mode (parallel.go - MoE)                    │")
	fmt.Println("└──────────────────────────────────────────────────────────────────────┘")

	net := nn.NewNetwork(4, 1, 1, 2)
	net.BatchSize = 1

	gateLayer := nn.InitDenseLayer(4, 2, nn.ActivationSigmoid)
	parallel := nn.LayerConfig{
		Type:        nn.LayerParallel,
		CombineMode: "filter",
		ParallelBranches: []nn.LayerConfig{
			nn.InitDenseLayer(4, 2, nn.ActivationTanh),
			nn.InitDenseLayer(4, 2, nn.ActivationSigmoid),
		},
		FilterGateConfig:  &gateLayer,
		FilterTemperature: 1.0,
	}
	net.SetLayer(0, 0, 0, parallel)
	net.SetLayer(0, 0, 1, nn.InitDenseLayer(2, 2, nn.ActivationSigmoid))

	input := []float32{0.1, 0.2, 0.3, 0.4}
	output, _ := net.ForwardCPU(input)

	if len(output) != 2 {
		fmt.Printf("  ❌ Expected 2 outputs, got %d\n", len(output))
		return false
	}
	fmt.Printf("  ✓ Forward pass: output=[%.3f, %.3f]\n", output[0], output[1])
	fmt.Println("  ✅ PASSED: Filter Combine Mode (MoE)")
	return true
}

func testSequentialLayers() bool {
	fmt.Println("\n┌──────────────────────────────────────────────────────────────────────┐")
	fmt.Println("│ Sequential Layer Composition (sequential.go)                        │")
	fmt.Println("└──────────────────────────────────────────────────────────────────────┘")

	seq := nn.InitSequentialLayer(
		nn.InitDenseLayer(4, 8, nn.ActivationLeakyReLU),
		nn.InitDenseLayer(8, 2, nn.ActivationSigmoid),
	)

	if seq.Type != nn.LayerSequential {
		fmt.Println("  ❌ InitSequentialLayer did not create LayerSequential type")
		return false
	}
	fmt.Printf("  ✓ Sequential layer with %d sub-layers\n", len(seq.ParallelBranches))

	net := nn.NewNetwork(4, 1, 1, 1)
	net.BatchSize = 1
	net.SetLayer(0, 0, 0, seq)

	input := []float32{0.1, 0.2, 0.3, 0.4}
	output, _ := net.ForwardCPU(input)

	if len(output) != 2 {
		fmt.Printf("  ❌ Expected 2 outputs, got %d\n", len(output))
		return false
	}
	fmt.Printf("  ✓ Forward pass: output=[%.3f, %.3f]\n", output[0], output[1])
	fmt.Println("  ✅ PASSED: Sequential Layer Composition")
	return true
}

func testKMeansClustering() bool {
	fmt.Println("\n┌──────────────────────────────────────────────────────────────────────┐")
	fmt.Println("│ K-Means Clustering (clustering.go)                                  │")
	fmt.Println("└──────────────────────────────────────────────────────────────────────┘")

	rand.Seed(time.Now().UnixNano())
	data := make([][]float32, 30)
	for i := 0; i < 10; i++ {
		data[i] = []float32{rand.Float32()*2 - 1, rand.Float32()*2 - 1}
		data[10+i] = []float32{rand.Float32()*2 + 4, rand.Float32()*2 + 4}
		data[20+i] = []float32{rand.Float32()*2 - 1, rand.Float32()*2 + 4}
	}

	centroids, assignments := nn.KMeansCluster(data, 3, 100, false)

	if len(centroids) != 3 {
		fmt.Printf("  ❌ Expected 3 centroids, got %d\n", len(centroids))
		return false
	}
	fmt.Printf("  ✓ K-Means: %d centroids, %d assignments\n", len(centroids), len(assignments))

	silhouette := nn.ComputeSilhouetteScore(data, assignments)
	fmt.Printf("  ✓ Silhouette Score: %.3f\n", silhouette)

	fmt.Println("  ✅ PASSED: K-Means Clustering")
	return true
}

func testCorrelationAnalysis() bool {
	fmt.Println("\n┌──────────────────────────────────────────────────────────────────────┐")
	fmt.Println("│ Correlation Analysis (correlation.go)                               │")
	fmt.Println("└──────────────────────────────────────────────────────────────────────┘")

	n := 100
	data := make([][]float32, n)
	for i := 0; i < n; i++ {
		x := float32(i) / float32(n)
		y := x + rand.Float32()*0.1
		z := rand.Float32()
		data[i] = []float32{x, y, z}
	}

	result := nn.ComputeCorrelationMatrix(data, nil)
	if result == nil || len(result.Correlation.Matrix) != 3 {
		fmt.Println("  ❌ Failed to compute correlation matrix")
		return false
	}
	fmt.Printf("  ✓ Correlation matrix: %dx%d\n", len(result.Correlation.Matrix), len(result.Correlation.Matrix[0]))

	xyCorr := result.Correlation.Matrix[0][1]
	xzCorr := result.Correlation.Matrix[0][2]
	fmt.Printf("  ✓ X-Y corr: %.3f (high), X-Z corr: %.3f (low)\n", xyCorr, xzCorr)

	strong := result.GetStrongCorrelations(0.5)
	fmt.Printf("  ✓ Found %d strong correlations (>0.5)\n", len(strong))

	fmt.Println("  ✅ PASSED: Correlation Analysis")
	return true
}

// =============================================================================
// JSON Network Templates (dtype will be injected)
// =============================================================================

func getJSONConfig(layerType, dtype string) string {
	template := ""
	switch layerType {
	case "Dense":
		template = `{
			"id": "dense_test",
			"dtype": "%s",
			"batch_size": 1,
			"grid_rows": 1,
			"grid_cols": 1,
			"layers_per_cell": 3,
			"layers": [
				{"type": "dense", "activation": "leaky_relu", "input_height": 8, "output_height": 64},
				{"type": "dense", "activation": "tanh", "input_height": 64, "output_height": 32},
				{"type": "dense", "activation": "sigmoid", "input_height": 32, "output_height": 4}
			]
		}`
	case "MHA":
		template = `{
			"id": "mha_test",
			"dtype": "%s",
			"batch_size": 1,
			"grid_rows": 1,
			"grid_cols": 1,
			"layers_per_cell": 2,
			"layers": [
				{"type": "multi_head_attention", "d_model": 64, "num_heads": 8, "seq_length": 1},
				{"type": "dense", "activation": "sigmoid", "input_height": 64, "output_height": 4}
			]
		}`
	case "RNN":
		template = `{
			"id": "rnn_test",
			"dtype": "%s",
			"batch_size": 1,
			"grid_rows": 1,
			"grid_cols": 1,
			"layers_per_cell": 2,
			"layers": [
				{"type": "rnn", "input_size": 16, "hidden_size": 32, "activation": "tanh"},
				{"type": "dense", "activation": "sigmoid", "input_height": 32, "output_height": 4}
			]
		}`
	case "LSTM":
		template = `{
			"id": "lstm_test",
			"dtype": "%s",
			"batch_size": 1,
			"grid_rows": 1,
			"grid_cols": 1,
			"layers_per_cell": 2,
			"layers": [
				{"type": "lstm", "input_size": 16, "hidden_size": 32},
				{"type": "dense", "activation": "sigmoid", "input_height": 32, "output_height": 4}
			]
		}`
	case "LayerNorm":
		template = `{
			"id": "layernorm_test",
			"dtype": "%s",
			"batch_size": 1,
			"grid_rows": 1,
			"grid_cols": 1,
			"layers_per_cell": 3,
			"layers": [
				{"type": "dense", "activation": "leaky_relu", "input_height": 16, "output_height": 32},
				{"type": "layer_norm", "norm_size": 32, "epsilon": 1e-5},
				{"type": "dense", "activation": "sigmoid", "input_height": 32, "output_height": 4}
			]
		}`
	case "RMSNorm":
		template = `{
			"id": "rmsnorm_test",
			"dtype": "%s",
			"batch_size": 1,
			"grid_rows": 1,
			"grid_cols": 1,
			"layers_per_cell": 3,
			"layers": [
				{"type": "dense", "activation": "leaky_relu", "input_height": 16, "output_height": 32},
				{"type": "rms_norm", "norm_size": 32, "epsilon": 1e-5},
				{"type": "dense", "activation": "sigmoid", "input_height": 32, "output_height": 4}
			]
		}`
	case "SwiGLU":
		template = `{
			"id": "swiglu_test",
			"dtype": "%s",
			"batch_size": 1,
			"grid_rows": 1,
			"grid_cols": 1,
			"layers_per_cell": 3,
			"layers": [
				{"type": "dense", "activation": "leaky_relu", "input_height": 32, "output_height": 64},
				{"type": "swiglu", "input_height": 64, "output_height": 128},
				{"type": "dense", "activation": "sigmoid", "input_height": 64, "output_height": 4}
			]
		}`
	case "Conv2D":
		template = `{
			"id": "conv2d_test",
			"dtype": "%s",
			"batch_size": 1,
			"grid_rows": 1,
			"grid_cols": 1,
			"layers_per_cell": 2,
			"layers": [
				{"type": "conv2d", "input_channels": 1, "filters": 2, "kernel_size": 3, "stride": 1, "padding": 1, "input_height": 4, "input_width": 4, "activation": "leaky_relu"},
				{"type": "dense", "activation": "sigmoid", "input_height": 32, "output_height": 4}
			]
		}`
	case "Parallel":
		template = `{
			"id": "parallel_test",
			"dtype": "%s",
			"batch_size": 1,
			"grid_rows": 1,
			"grid_cols": 1,
			"layers_per_cell": 3,
			"layers": [
				{"type": "dense", "activation": "leaky_relu", "input_height": 8, "output_height": 16},
				{
					"type": "parallel",
					"combine_mode": "concat",
					"branches": [
						{"type": "dense", "activation": "tanh", "input_height": 16, "output_height": 8},
						{"type": "dense", "activation": "sigmoid", "input_height": 16, "output_height": 8}
					]
				},
				{"type": "dense", "activation": "sigmoid", "input_height": 16, "output_height": 4}
			]
		}`
	case "Sequential":
		template = `{
			"id": "sequential_test",
			"dtype": "%s",
			"batch_size": 1,
			"grid_rows": 1,
			"grid_cols": 1,
			"layers_per_cell": 2,
			"layers": [
				{
					"type": "sequential",
					"branches": [
						{"type": "dense", "activation": "leaky_relu", "input_height": 8, "output_height": 16},
						{"type": "dense", "activation": "tanh", "input_height": 16, "output_height": 8}
					]
				},
				{"type": "dense", "activation": "sigmoid", "input_height": 8, "output_height": 4}
			]
		}`
	case "Softmax":
		template = `{
			"id": "softmax_test",
			"dtype": "%s",
			"batch_size": 1,
			"grid_rows": 1,
			"grid_cols": 1,
			"layers_per_cell": 3,
			"layers": [
				{"type": "dense", "activation": "leaky_relu", "input_height": 8, "output_height": 16},
				{"type": "dense", "activation": "leaky_relu", "input_height": 16, "output_height": 4},
				{"type": "softmax", "softmax_variant": "standard", "temperature": 1.0}
			]
		}`
	}
	return fmt.Sprintf(template, dtype)
}

// =============================================================================
// Test Runner with DType from JSON
// =============================================================================

func testLayerWithDType(layerName string, inputSize int, dtype string) (int, int) {
	// Build network from JSON with dtype
	jsonConfig := getJSONConfig(layerName, dtype)
	net, configDType, err := nn.BuildNetworkFromJSONWithDType(jsonConfig)
	if err != nil {
		fmt.Printf("  ❌ %s/%s: Build failed: %v\n", layerName, dtype, err)
		return 0, 1
	}

	// Verify dtype was parsed correctly
	if configDType != dtype {
		fmt.Printf("  ❌ %s/%s: DType mismatch (got %s)\n", layerName, dtype, configDType)
		return 0, 1
	}

	// Create input
	input := make([]float32, inputSize)
	for i := range input {
		input[i] = float32(i+1) * 0.1
	}

	// Reference output (float32 inference)
	refOutput, _ := net.ForwardCPU(input)
	if len(refOutput) == 0 {
		fmt.Printf("  ❌ %s/%s: Forward pass failed\n", layerName, dtype)
		return 0, 1
	}

	// Save with dtype from config
	start := time.Now()
	jsonData, err := net.SaveModelWithDType(layerName+"_test", configDType)
	saveTime := time.Since(start).Seconds() * 1000
	if err != nil {
		fmt.Printf("  ❌ %s/%s: Save failed: %v\n", layerName, dtype, err)
		return 0, 1
	}

	// Load back
	start = time.Now()
	loaded, loadedDType, err := nn.LoadModelWithDType(jsonData, layerName+"_test", configDType)
	loadTime := time.Since(start).Seconds() * 1000
	if err != nil {
		fmt.Printf("  ❌ %s/%s: Load failed: %v\n", layerName, dtype, err)
		return 0, 1
	}

	// Verify loaded dtype matches
	if loadedDType != dtype {
		fmt.Printf("  ❌ %s/%s: Loaded dtype mismatch (got %s)\n", layerName, dtype, loadedDType)
		return 0, 1
	}

	// Verify output
	loadedOutput, _ := loaded.ForwardCPU(input)
	maxErr := computeMaxError(refOutput, loadedOutput)

	threshold := getThreshold(dtype)
	if maxErr > threshold {
		fmt.Printf("  ❌ %s/%-8s: error=%.2e (>%.2e) size=%d\n", layerName, dtype, maxErr, threshold, len(jsonData))
		return 0, 1
	}

	fmt.Printf("  ✓ %-10s/%-8s: save=%.1fms load=%.1fms size=%s err=%.1e\n",
		layerName, dtype, saveTime, loadTime, formatSize(len(jsonData)), maxErr)
	return 1, 0
}

// =============================================================================
// Helper Functions
// =============================================================================

func computeMaxError(a, b []float32) float64 {
	maxErr := float64(0)
	minLen := len(a)
	if len(b) < minLen {
		minLen = len(b)
	}
	for i := 0; i < minLen; i++ {
		diff := math.Abs(float64(a[i]) - float64(b[i]))
		if diff > maxErr {
			maxErr = diff
		}
	}
	return maxErr
}

func getThreshold(dtype string) float64 {
	switch dtype {
	case "float64":
		return 1e-6
	case "float32":
		return 1e-6
	case "int32":
		return 0.1
	case "int16":
		return 0.15
	case "int8":
		return 0.25
	default:
		return 0.1
	}
}

func formatSize(bytes int) string {
	if bytes < 1024 {
		return fmt.Sprintf("%dB", bytes)
	}
	return fmt.Sprintf("%.1fKB", float64(bytes)/1024)
}

// Print summary table header
func init() {
	// Print after startup
	go func() {
		time.Sleep(100 * time.Millisecond)
		fmt.Println("Testing all layer types with all numerical types...")
		fmt.Println(strings.Repeat("─", 70))
	}()
}
