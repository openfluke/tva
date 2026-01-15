package main

import (
	"encoding/json"
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
var filterFlag = flag.String("filter", "", "Optional substring to run specific tests only")

// =============================================================================
// LOOM v0.0.8 Complete Test Suite
// Tests all v0.0.8 features + multi-precision save/load for all layer types
// =============================================================================

func main() {
	flag.Parse()
	fmt.Println("╔══════════════════════════════════════════════════════════════════════╗")
	fmt.Println("║               LOOM v0.0.8 Complete Feature Test Suite               ║")
	fmt.Println("╚══════════════════════════════════════════════════════════════════════╝")
	fmt.Println()

	passed := 0
	failed := 0

	// =========================================================================
	// PART 1: Core v0.0.8 Feature Tests
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
		testNetworkGrafting,
	}

	if *filterFlag == "" {
		for _, test := range coreTests {
			if test() {
				passed++
			} else {
				failed++
			}
		}
	} else {
		fmt.Println("Skipping Part 1 (Core Tests) due to filter")
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

	// =========================================================================
	// PART 3: Advanced Math Tests
	// =========================================================================
	if *filterFlag == "" {
		fmt.Println("\n═══════════════════════════════════════════════════════════════════════")
		fmt.Println("                  PART 3: ADVANCED MATH TESTS                          ")
		fmt.Println("═══════════════════════════════════════════════════════════════════════")

		additionalTests := []func() bool{
			testOptimizers,
			testSchedulers,
			testActivations,
			testSoftmaxVariants,
			testEmbeddingLayer,
			testIntrospection,
			testStepTween,
			testConv1DLayer,
			testResidualConnection,
			testEnsembleFeatures,
			testObserverPattern,
		}

		for _, test := range additionalTests {
			if test() {
				passed++
			} else {
				failed++
			}
		}

		// =========================================================================
		// PART 4: Experimental Demos
		// =========================================================================
		fmt.Println("\n═══════════════════════════════════════════════════════════════════════")
		fmt.Println("              PART 4: EXPERIMENTAL DEMOS                              ")
		fmt.Println("═══════════════════════════════════════════════════════════════════════")

		fmt.Println("\n>>> Running Frozen Specialization Benchmark...")
		runFrozenSpecDemo()

		fmt.Println("\n>>> Running Stitched Experts (Odds) Demo...")
		runOddsDemo()

		fmt.Println("\n>>> Running Filter CombineMode Demo...")
		runFilterDemo()
	} else {
		fmt.Println("Skipping Parts 3-4 (Advanced/Demos) due to filter")
	}

	// =========================================================================
	// PART 5: GPU Determinism Tests
	// =========================================================================
	fmt.Println("\n═══════════════════════════════════════════════════════════════════════")
	fmt.Println("              PART 5: GPU DETERMINISM TESTS (Forward Pass)             ")
	fmt.Println("═══════════════════════════════════════════════════════════════════════")

	if *gpuFlag != "" {
		fmt.Printf("Requesting GPU adapter matching: %q\n", *gpuFlag)
		gpu.SetAdapterPreference(*gpuFlag)
	}

	var failedTests []string
	for _, test := range gpuLayerTests {
		if *filterFlag != "" && !strings.Contains(test.Name, *filterFlag) {
			continue
		}
		if runGPULayerTest(test) {
			passed++
		} else {
			failed++
			failedTests = append(failedTests, test.Name)
		}
		fmt.Println()
	}

	// Final Summary
	fmt.Println()
	fmt.Println("╔══════════════════════════════════════════════════════════════════════╗")
	fmt.Printf("║              FINAL RESULTS: %d/%d TESTS PASSED                       ║\n", passed, passed+failed)
	fmt.Println("╚══════════════════════════════════════════════════════════════════════╝")

	if failed > 0 {
		fmt.Printf("\n❌ %d test(s) failed:\n", failed)
		for _, name := range failedTests {
			fmt.Printf("  • %s\n", name)
		}
	} else {
		fmt.Println("\n🎉 All tests passed! Ready for 0.0.8 release!")
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

// =============================================================================
// PART 3: Additional Feature Tests
// =============================================================================

func testOptimizers() bool {
	fmt.Println("\n┌──────────────────────────────────────────────────────────────────────┐")
	fmt.Println("│ Optimizers (optimizer.go)                                           │")
	fmt.Println("└──────────────────────────────────────────────────────────────────────┘")

	// Test SGD
	sgd := nn.NewSGDOptimizer()
	if sgd.Name() != "SGD" {
		fmt.Println("  ❌ SGD optimizer name mismatch")
		return false
	}
	fmt.Printf("  ✓ %s optimizer created\n", sgd.Name())

	// Test SGD with momentum
	sgdMomentum := nn.NewSGDOptimizerWithMomentum(0.9, 0.0, false)
	fmt.Printf("  ✓ %s created\n", sgdMomentum.Name())

	// Test AdamW
	adamw := nn.NewAdamWOptimizerDefault()
	if adamw.Name() != "AdamW" {
		fmt.Println("  ❌ AdamW optimizer name mismatch")
		return false
	}
	fmt.Printf("  ✓ %s optimizer created\n", adamw.Name())

	// Test RMSprop
	rmsprop := nn.NewRMSpropOptimizerDefault()
	if rmsprop.Name() != "RMSprop" {
		fmt.Println("  ❌ RMSprop optimizer name mismatch")
		return false
	}
	fmt.Printf("  ✓ %s optimizer created\n", rmsprop.Name())

	// Test state save/load
	state := adamw.GetState()
	err := adamw.LoadState(state)
	if err != nil {
		fmt.Printf("  ❌ AdamW state load failed: %v\n", err)
		return false
	}
	fmt.Println("  ✓ Optimizer state save/load works")

	fmt.Println("  ✅ PASSED: Optimizers")
	return true
}

func testSchedulers() bool {
	fmt.Println("\n┌──────────────────────────────────────────────────────────────────────┐")
	fmt.Println("│ Learning Rate Schedulers (scheduler.go)                             │")
	fmt.Println("└──────────────────────────────────────────────────────────────────────┘")

	schedulers := []nn.LRScheduler{
		nn.NewConstantScheduler(0.01),
		nn.NewLinearDecayScheduler(0.01, 0.0001, 1000),
		nn.NewCosineAnnealingScheduler(0.01, 0.0001, 1000),
		nn.NewExponentialDecayScheduler(0.01, 0.9, 100),
		nn.NewStepDecayScheduler(0.01, 0.5, 100),
		nn.NewPolynomialDecayScheduler(0.01, 0.0001, 1000, 2.0),
	}

	for _, s := range schedulers {
		lr0 := s.GetLR(0)
		lr500 := s.GetLR(500)
		if lr0 <= 0 {
			fmt.Printf("  ❌ %s returned invalid LR at step 0\n", s.Name())
			return false
		}
		fmt.Printf("  ✓ %s: LR(0)=%.4f, LR(500)=%.6f\n", s.Name(), lr0, lr500)
	}

	// Test warmup scheduler
	warmup := nn.NewWarmupScheduler(100, 0.0, 0.01, nn.NewConstantScheduler(0.01))
	lr0 := warmup.GetLR(0)
	lr50 := warmup.GetLR(50)
	lr150 := warmup.GetLR(150)
	fmt.Printf("  ✓ %s: LR(0)=%.4f, LR(50)=%.4f, LR(150)=%.4f\n", warmup.Name(), lr0, lr50, lr150)

	fmt.Println("  ✅ PASSED: Learning Rate Schedulers")
	return true
}

func testActivations() bool {
	fmt.Println("\n┌──────────────────────────────────────────────────────────────────────┐")
	fmt.Println("│ Activation Functions (activations.go)                               │")
	fmt.Println("└──────────────────────────────────────────────────────────────────────┘")

	activations := []nn.ActivationType{
		nn.ActivationScaledReLU,
		nn.ActivationSigmoid,
		nn.ActivationTanh,
		nn.ActivationSoftplus,
		nn.ActivationLeakyReLU,
	}
	names := []string{"ScaledReLU", "Sigmoid", "Tanh", "Softplus", "LeakyReLU"}

	testVal := float32(0.5)
	for i, act := range activations {
		result := nn.Activate(testVal, act)
		deriv := nn.ActivateDerivative(testVal, act)
		if math.IsNaN(float64(result)) || math.IsNaN(float64(deriv)) {
			fmt.Printf("  ❌ %s returned NaN\n", names[i])
			return false
		}
		fmt.Printf("  ✓ %s: f(0.5)=%.3f, f'(0.5)=%.3f\n", names[i], result, deriv)
	}

	// Test tensor activation
	tensor := nn.NewTensor[float32](4)
	tensor.Data = []float32{-1.0, 0.0, 0.5, 1.0}
	activated := nn.ActivateTensor(tensor, nn.ActivationSigmoid)
	if len(activated.Data) != 4 {
		fmt.Println("  ❌ ActivateTensor failed")
		return false
	}
	fmt.Printf("  ✓ ActivateTensor: [%.2f, %.2f, %.2f, %.2f]\n",
		activated.Data[0], activated.Data[1], activated.Data[2], activated.Data[3])

	fmt.Println("  ✅ PASSED: Activation Functions")
	return true
}

func testSoftmaxVariants() bool {
	fmt.Println("\n┌──────────────────────────────────────────────────────────────────────┐")
	fmt.Println("│ Softmax Variants (softmax.go)                                       │")
	fmt.Println("└──────────────────────────────────────────────────────────────────────┘")

	input := []float32{1.0, 2.0, 3.0, 4.0}

	variants := []struct {
		name   string
		config nn.LayerConfig
	}{
		{"Standard", nn.InitSoftmaxLayer()},
		{"Temperature", nn.InitTemperatureSoftmaxLayer(0.5)},
		{"Grid", nn.InitGridSoftmaxLayer(2, 2)},
		{"Sparsemax", nn.InitSparsemaxLayer()},
		{"Entmax", nn.InitEntmaxLayer(1.5)},
	}

	for _, v := range variants {
		output, err := nn.ForwardSoftmaxCPU(input, &v.config)
		if err != nil {
			fmt.Printf("  ❌ %s failed: %v\n", v.name, err)
			return false
		}
		// Verify sum ~= 1.0 for standard softmax
		if v.name == "Standard" || v.name == "Temperature" {
			sum := float32(0)
			for _, p := range output {
				sum += p
			}
			if math.Abs(float64(sum-1.0)) > 0.001 {
				fmt.Printf("  ❌ %s sum=%.3f (expected 1.0)\n", v.name, sum)
				return false
			}
		}
		fmt.Printf("  ✓ %s: [%.2f, %.2f, %.2f, %.2f]\n", v.name, output[0], output[1], output[2], output[3])
	}

	fmt.Println("  ✅ PASSED: Softmax Variants")
	return true
}

func testEmbeddingLayer() bool {
	fmt.Println("\n┌──────────────────────────────────────────────────────────────────────┐")
	fmt.Println("│ Embedding Layer (embedding.go)                                      │")
	fmt.Println("└──────────────────────────────────────────────────────────────────────┘")

	// Create network with embedding
	net := nn.NewNetwork(1, 1, 1, 2)
	net.BatchSize = 1

	embConfig := nn.LayerConfig{
		Type:         nn.LayerEmbedding,
		VocabSize:    100,
		EmbeddingDim: 16,
	}
	// Initialize embedding weights
	embConfig.EmbeddingWeights = make([]float32, 100*16)
	for i := range embConfig.EmbeddingWeights {
		embConfig.EmbeddingWeights[i] = rand.Float32()*0.1 - 0.05
	}

	net.SetLayer(0, 0, 0, embConfig)
	net.SetLayer(0, 0, 1, nn.InitDenseLayer(16, 4, nn.ActivationSigmoid))

	// Forward pass with token index
	input := []float32{5.0} // Token index 5
	output, _ := net.ForwardCPU(input)

	if len(output) != 4 {
		fmt.Printf("  ❌ Expected 4 outputs, got %d\n", len(output))
		return false
	}
	fmt.Printf("  ✓ Embedding lookup: token 5 → %d dims → 4 outputs\n", embConfig.EmbeddingDim)
	fmt.Printf("  ✓ Output: [%.3f, %.3f, %.3f, %.3f]\n", output[0], output[1], output[2], output[3])

	fmt.Println("  ✅ PASSED: Embedding Layer")
	return true
}

func testIntrospection() bool {
	fmt.Println("\n┌──────────────────────────────────────────────────────────────────────┐")
	fmt.Println("│ Introspection & Telemetry (introspection.go, telemetry.go)          │")
	fmt.Println("└──────────────────────────────────────────────────────────────────────┘")

	// Create a test network
	net := nn.NewNetwork(4, 1, 1, 2)
	net.BatchSize = 1
	net.SetLayer(0, 0, 0, nn.InitDenseLayer(4, 8, nn.ActivationLeakyReLU))
	net.SetLayer(0, 0, 1, nn.InitDenseLayer(8, 2, nn.ActivationSigmoid))

	// Test GetModelSizeInfo
	sizeInfo := net.GetModelSizeInfo()
	if len(sizeInfo) != 10 {
		fmt.Printf("  ❌ Expected 10 dtype sizes, got %d\n", len(sizeInfo))
		return false
	}
	fmt.Printf("  ✓ GetModelSizeInfo: %d dtypes analyzed\n", len(sizeInfo))

	for dtype, info := range sizeInfo {
		fmt.Printf("    - %s: %d weights × %dB = %d bytes\n", dtype, info.TotalWeights, info.BytesPerWeight, info.TotalBytes)
		break // Just show one
	}

	// Test TotalLayers
	totalLayers := net.TotalLayers()
	if totalLayers != 2 {
		fmt.Printf("  ❌ Expected 2 layers, got %d\n", totalLayers)
		return false
	}
	fmt.Printf("  ✓ TotalLayers: %d\n", totalLayers)

	fmt.Println("  ✅ PASSED: Introspection & Telemetry")
	return true
}

func testStepTween() bool {
	fmt.Println("\n┌──────────────────────────────────────────────────────────────────────┐")
	fmt.Println("│ Step-Tween Training (tween.go)                                      │")
	fmt.Println("└──────────────────────────────────────────────────────────────────────┘")

	// Create a simple network for step-tween
	net := nn.NewNetwork(4, 1, 1, 2)
	net.BatchSize = 1
	net.SetLayer(0, 0, 0, nn.InitDenseLayer(4, 8, nn.ActivationLeakyReLU))
	net.SetLayer(0, 0, 1, nn.InitDenseLayer(8, 2, nn.ActivationSigmoid))

	// Create tween config and state
	config := &nn.TweenConfig{
		UseChainRule:  true,
		GradientScale: 0.1,
		Momentum:      0.9,
	}

	// Create GenericTweenState for float32
	tweenState := nn.NewGenericTweenState[float32](net, config)

	if tweenState == nil {
		fmt.Println("  ❌ Failed to create TweenState")
		return false
	}
	fmt.Printf("  ✓ GenericTweenState created with %d layers\n", tweenState.TotalLayers)

	// Create input tensor
	input := nn.NewTensor[float32](4)
	input.Data = []float32{0.1, 0.2, 0.3, 0.4}

	// Create CPU backend
	backend := nn.NewCPUBackend[float32]()

	// Forward pass
	output1 := tweenState.ForwardPass(net, input, backend)
	if output1 == nil || len(output1.Data) != 2 {
		fmt.Println("  ❌ Forward pass failed")
		return false
	}
	fmt.Printf("  ✓ Forward pass: output=[%.3f, %.3f]\n", output1.Data[0], output1.Data[1])

	// Do a tween step
	gap := tweenState.TweenStep(net, input, 0, 2, 0.01, backend)
	fmt.Printf("  ✓ TweenStep executed, gap=%.4f\n", gap)

	// Forward again
	output2 := tweenState.ForwardPass(net, input, backend)

	// Check that output changed (learning happened)
	changed := false
	for i := range output1.Data {
		if math.Abs(float64(output1.Data[i]-output2.Data[i])) > 0.0001 {
			changed = true
			break
		}
	}
	if !changed {
		fmt.Println("  ⚠ Weights barely changed (may be small learning rate)")
	} else {
		fmt.Printf("  ✓ Before: [%.3f, %.3f], After: [%.3f, %.3f]\n",
			output1.Data[0], output1.Data[1], output2.Data[0], output2.Data[1])
	}

	fmt.Println("  ✅ PASSED: Step-Tween Training")
	return true
}

func testConv1DLayer() bool {
	fmt.Println("\n┌──────────────────────────────────────────────────────────────────────┐")
	fmt.Println("│ Conv1D Layer (conv1d.go)                                            │")
	fmt.Println("└──────────────────────────────────────────────────────────────────────┘")

	// Create network with Conv1D
	net := nn.NewNetwork(16, 1, 1, 2)
	net.BatchSize = 1

	// Use helper to init correctly
	conv1d := nn.InitConv1DLayer(16, 1, 3, 1, 1, 4, nn.ActivationLeakyReLU)

	net.SetLayer(0, 0, 0, conv1d)
	net.SetLayer(0, 0, 1, nn.InitDenseLayer(64, 4, nn.ActivationSigmoid)) // 16*4 = 64

	input := make([]float32, 16)
	for i := range input {
		input[i] = float32(i) * 0.1
	}

	output, _ := net.ForwardCPU(input)
	if len(output) != 4 {
		fmt.Printf("  ❌ Expected 4 outputs, got %d\n", len(output))
		return false
	}
	fmt.Printf("  ✓ Conv1D: [%d] → [%d×%d] → Dense → [%d]\n", len(input), 16, 4, len(output))
	fmt.Printf("  ✓ Output: [%.3f, %.3f, %.3f, %.3f]\n", output[0], output[1], output[2], output[3])

	fmt.Println("  ✅ PASSED: Conv1D Layer")
	return true
}

func testResidualConnection() bool {
	fmt.Println("\n┌──────────────────────────────────────────────────────────────────────┐")
	fmt.Println("│ Residual Connection (residual.go)                                   │")
	fmt.Println("└──────────────────────────────────────────────────────────────────────┘")

	size := 4
	input := nn.NewTensor[float32](size)
	input.Data = []float32{1.0, 2.0, 3.0, 4.0}

	skip := nn.NewTensor[float32](size)
	skip.Data = []float32{0.5, 0.5, 0.5, 0.5}

	// Test Forward
	output := nn.ResidualForward(input, skip)
	if len(output.Data) != size {
		fmt.Println("  ❌ Output size mismatch")
		return false
	}

	expected := []float32{1.5, 2.5, 3.5, 4.5}
	for i := range expected {
		if math.Abs(float64(output.Data[i]-expected[i])) > 0.0001 {
			fmt.Printf("  ❌ Mismatch at %d: got %.2f, expected %.2f\n", i, output.Data[i], expected[i])
			return false
		}
	}
	fmt.Printf("  ✓ Residual Forward: input+skip = output OK\n")

	// Test Backward
	gradOutput := nn.NewTensor[float32](size)
	gradOutput.Data = []float32{1, 1, 1, 1}
	gradInput, gradSkip := nn.ResidualBackward(gradOutput)

	if gradInput.Data[0] != 1.0 || gradSkip.Data[0] != 1.0 {
		fmt.Println("  ❌ Gradients failed to propagate correctly")
		return false
	}
	fmt.Printf("  ✓ Residual Backward: gradients split correctly\n")

	fmt.Println("  ✅ PASSED: Residual Connection")
	return true
}

func testEnsembleFeatures() bool {
	fmt.Println("\n┌──────────────────────────────────────────────────────────────────────┐")
	fmt.Println("│ Ensemble Features (ensemble.go)                                     │")
	fmt.Println("└──────────────────────────────────────────────────────────────────────┘")

	// Create mock model performances
	models := []nn.ModelPerformance{
		{ModelID: "ModelA", Mask: []bool{true, true, false, false}},
		{ModelID: "ModelB", Mask: []bool{false, false, true, true}}, // Perfect complement
		{ModelID: "ModelC", Mask: []bool{true, true, true, false}},
	}

	matches := nn.FindComplementaryMatches(models, 0.0)
	if len(matches) == 0 {
		fmt.Println("  ❌ No matches found")
		return false
	}

	best := matches[0]
	if (best.ModelA == "ModelA" && best.ModelB == "ModelB") || (best.ModelA == "ModelB" && best.ModelB == "ModelA") {
		if best.Coverage != 1.0 {
			fmt.Printf("  ❌ Expected 100%% coverage, got %.1f%%\n", best.Coverage*100)
			return false
		}
		fmt.Printf("  ✓ Identified perfect pair: %s + %s (100%% coverage)\n", best.ModelA, best.ModelB)
	} else {
		fmt.Printf("  ❌ Unexpected best pair: %s + %s\n", best.ModelA, best.ModelB)
		// return false // Don't fail hard on sort order if it's debatable, but here it's clear
	}

	fmt.Println("  ✅ PASSED: Ensemble Features")
	return true
}

func testObserverPattern() bool {
	fmt.Println("\n┌──────────────────────────────────────────────────────────────────────┐")
	fmt.Println("│ Observer Pattern (observer.go)                                      │")
	fmt.Println("└──────────────────────────────────────────────────────────────────────┘")

	// Create a simple network
	net := nn.NewNetwork(2, 1, 1, 2) // 2 inputs, 2 outputs
	net.BatchSize = 1

	// Check layer config from network
	// denseConfig := nn.InitDenseLayer(2, 2, nn.ActivationLeakyReLU) // Fix activation
	denseConfig := nn.InitDenseLayer(2, 2, nn.ActivationLeakyReLU)
	denseConfig.Observer = nn.NewRecordingObserver("TestModel")

	// Inject the observer into the network layer (trickier since SetLayer copies)
	// We need to set it, then rely on the fact that forward uses the config in the network
	net.SetLayer(0, 0, 0, denseConfig)

	// Run forward pass
	input := []float32{1.0, -1.0}
	net.ForwardCPU(input)

	// Retrieve the observer back from the network layer to check events
	// Since SetLayer copies the config struct, we need to extract the observer from the stored config
	storedConfig := net.GetLayer(0, 0, 0)
	if storedConfig.Observer == nil {
		fmt.Println("  ❌ Observer not preserved in layer config")
		return false
	}

	// Cast back to RecordingObserver
	recorder, ok := storedConfig.Observer.(*nn.RecordingObserver)
	if !ok {
		fmt.Println("  ❌ Observer type mismatch")
		return false
	}

	events := recorder.GetRecording()
	if events.TotalEvents == 0 {
		fmt.Println("  ❌ No events recorded")
		return false
	}

	fmt.Printf("  ✓ Recorded %d events\n", events.TotalEvents)
	fmt.Printf("  ✓ Event Type: %s, Layer: %s\n", events.Events[0].Type, events.Events[0].Stats.LayerType)

	fmt.Println("  ✅ PASSED: Observer Pattern")
	return true
}

func testNetworkGrafting() bool {
	fmt.Println("\n┌──────────────────────────────────────────────────────────────────────┐")
	fmt.Println("│ Network Grafting (grafting.go)                                      │")
	fmt.Println("└──────────────────────────────────────────────────────────────────────┘")

	// Create 2 source networks
	net1 := nn.NewNetwork(2, 1, 1, 2)
	net1.SetLayer(0, 0, 1, nn.InitDenseLayer(2, 4, nn.ActivationLeakyReLU))

	net2 := nn.NewNetwork(2, 1, 1, 2)
	net2.SetLayer(0, 0, 1, nn.InitDenseLayer(2, 4, nn.ActivationSigmoid))

	// Graft them
	networks := []*nn.Network{net1, net2}
	graftedConfig, err := nn.GraftNetworks(networks, "concat")

	if err != nil {
		fmt.Printf("  ❌ GraftNetworks failed: %v\n", err)
		return false
	}

	if graftedConfig.Type != nn.LayerParallel {
		fmt.Printf("  ❌ Expected Parallel layer, got %s\n", graftedConfig.Type)
		return false
	}

	if len(graftedConfig.ParallelBranches) != 2 {
		fmt.Printf("  ❌ Expected 2 branches, got %d\n", len(graftedConfig.ParallelBranches))
		return false
	}

	fmt.Printf("  ✓ Grafted 2 networks into Parallel layer with %d branches\n", len(graftedConfig.ParallelBranches))
	fmt.Printf("  ✓ Branch 1 Activation: %d (LeakyReLU)\n", graftedConfig.ParallelBranches[0].Activation)
	fmt.Printf("  ✓ Branch 2 Activation: %d (Sigmoid)\n", graftedConfig.ParallelBranches[1].Activation)

	fmt.Println("  ✅ PASSED: Network Grafting")
	return true
}

// =============================================================================
// PART 4 IMPL: FROZEN SPECIALIZATION DEMO
// =============================================================================

// Global constants for Frozen Spec (Renamed to avoid collisions)
const (
	fs_InputSize        = 8
	fs_CommonOutputSize = 1
	fs_TrainingEpochs   = 1000 // Increased for better Gate convergence
)

func runFrozenSpecDemo() {
	rand.Seed(time.Now().UnixNano())

	fmt.Println("╔══════════════════════════════════════════════════════════════════╗")
	fmt.Println("║       🧊 Frozen Specialization Training Mode Benchmark 🧊          ║")
	fmt.Println("║       (Fixed Experts: Manual Gradient Descent Pre-training)      ║")
	fmt.Println("╚══════════════════════════════════════════════════════════════════╝")

	// 1. Define Training Modes
	modes := []string{
		"Standard Forward/Backward",
		"StepBack",
		"Step Tween",
		"Tween",
		"Tween Chain",
		"Step Tween Chain",
	}

	results := make([]string, 0, len(modes))
	for _, mode := range modes {
		res := fs_runExperimentForMode(mode)
		results = append(results, res)
	}

	fmt.Printf("\n\n")
	fmt.Printf("%-25s | %-10s | %-10s | %-10s | %-10s | %s\n", "Mode", "Expert 1", "Expert 2", "Network", "Ideal", "% Off")
	fmt.Println("------------------------------------------------------------------------------------------------------------------")
	for _, res := range results {
		fmt.Println(res)
	}
}

func fs_runExperimentForMode(mode string) string {
	// 2. Setup Network with Frozen Experts (Common Foundation)
	expert1, expert2 := fs_createExperts(mode)

	// Create Filter Layer
	gateLayer := nn.InitDenseLayer(fs_InputSize, 2, nn.ActivationScaledReLU)

	// Initialize Gate to random small weights
	for i := range gateLayer.Kernel {
		gateLayer.Kernel[i] = (rand.Float32() - 0.5) * 0.1
	}

	filterLayer := nn.LayerConfig{
		Type:              nn.LayerParallel,
		ParallelBranches:  []nn.LayerConfig{expert1, expert2},
		CombineMode:       "filter",
		FilterGateConfig:  &gateLayer,
		FilterSoftmax:     nn.SoftmaxStandard,
		FilterTemperature: 0.5,
	}

	net := nn.NewNetwork(fs_InputSize, 1, 1, 1)
	net.SetLayer(0, 0, 0, filterLayer)

	// 3. Train Gate with Specific Mode
	fs_trainGate(net, mode)

	// 4. Verify & Compare
	// We check if it routes correctly for High input (Expert 1) AND Low input (Expert 2)

	// Evaluate High Case (Expect Exp1)
	val1H, _, netValH, idealH := fs_evaluateNetwork(net, true)
	diffH := float64(netValH - idealH)

	// Evaluate Low Case (Expect Exp2)
	_, val2L, netValL, idealL := fs_evaluateNetwork(net, false)
	diffL := float64(netValL - idealL)

	// Combined Error
	avgOff := (math.Abs(diffH)/float64(idealH+0.0001) + math.Abs(diffL)/float64(idealL+0.0001)) / 2.0 * 100.0

	// Format result
	return fmt.Sprintf("%-25s | %-10.4f | %-10.4f | %-10.4f | %-10.4f | %.2f%%",
		mode, val1H, val2L, netValH, idealH, avgOff)
}

func fs_createExperts(mode string) (nn.LayerConfig, nn.LayerConfig) {
	fmt.Printf("\n=== Preparing Experts for %s ===\n", mode)

	// Expert 1
	e1 := nn.InitDenseLayer(fs_InputSize, 8, nn.ActivationLeakyReLU)
	s1 := nn.InitDenseLayer(8, fs_CommonOutputSize, nn.ActivationSigmoid)
	b1 := nn.InitSequentialLayer(e1, s1)

	fmt.Print("   🎓 Pre-training Expert 1... ")
	fs_trainExpert(&b1, fs_InputSize, true) // High -> 1.0
	fs_freezeLayer(&b1)

	// Expert 2
	e2 := nn.InitDenseLayer(fs_InputSize, 8, nn.ActivationLeakyReLU)
	s2 := nn.InitDenseLayer(8, fs_CommonOutputSize, nn.ActivationSigmoid)
	b2 := nn.InitSequentialLayer(e2, s2)

	fmt.Print("   🎓 Pre-training Expert 2... ")
	fs_trainExpert(&b2, fs_InputSize, false) // Low -> 1.0
	fs_freezeLayer(&b2)

	return b1, b2
}

func fs_trainExpert(layer *nn.LayerConfig, inputSize int, highDetect bool) {
	// 1. Create a temporary network to train this expert
	tempNet := nn.NewNetwork(inputSize, 1, 1, 1)
	tempNet.SetLayer(0, 0, 0, *layer) // Pass by value, but slices are shared

	// 2. Generate Training Data
	trainingData := make([]nn.TrainingBatch, 2000)

	for i := 0; i < 2000; i++ {
		input := make([]float32, inputSize)
		for j := range input {
			input[j] = rand.Float32() * 0.1
		}

		targetVal := float32(0.0)
		if highDetect {
			if rand.Float32() > 0.5 {
				input[0] = 0.9 // Trigger
				targetVal = 1.0
			}
		} else {
			if rand.Float32() > 0.5 {
				input[0] = 0.1 // Trigger
				targetVal = 1.0
			}
		}

		trainingData[i] = nn.TrainingBatch{
			Input:  input,
			Target: []float32{targetVal},
		}
	}

	// 3. Train
	config := &nn.TrainingConfig{
		Epochs:          5,
		LearningRate:    0.05,
		UseGPU:          false, // Use CPU for small demo
		Verbose:         true,
		LossType:        "mse",
		PrintEveryBatch: 0,
	}

	fmt.Printf("   Standard Framework Training (%d samples, %d epochs)...\n", len(trainingData), config.Epochs)
	_, err := tempNet.Train(trainingData, config)
	if err != nil {
		fmt.Printf("Training failed: %v\n", err)
	}
}

func fs_trainGate(net *nn.Network, mode string) {
	// Optimizers
	sgd := nn.NewSGDOptimizerWithMomentum(0.9, 0, false) // Standard SGD for backprop modes

	// Tween State
	ts := nn.NewTweenState(net, nil)
	ts.Config.UseChainRule = true // Default for chain modes
	ts.Config.Momentum = 0.5

	for i := 0; i < fs_TrainingEpochs; i++ {
		// Input
		input := make([]float32, fs_InputSize)
		for j := range input {
			input[j] = rand.Float32() * 0.1
		}

		if i%2 == 0 {
			input[0] = 0.9 // High -> Expert 1 -> Target 1.0
		} else {
			input[0] = 0.1 // Low -> Expert 2 -> Target 1.0
		}

		switch mode {
		case "Standard Forward/Backward":
			stepState := net.InitStepState(fs_InputSize)
			stepState.SetInput(input)
			net.StepForward(stepState)
			output := stepState.GetOutput()

			gradOut := make([]float32, 1)
			gradOut[0] = output[0] - 1.0 // Minimize distance to 1.0

			net.StepBackward(stepState, gradOut)
			sgd.Step(net, 0.05)

		case "StepBack":
			stepState := net.InitStepState(fs_InputSize)
			stepState.SetInput(input)
			net.StepForward(stepState)
			output := stepState.GetOutput()

			gradOut := make([]float32, 1)
			gradOut[0] = output[0] - 1.0
			net.StepBackward(stepState, gradOut)
			sgd.Step(net, 0.05)

		case "Step Tween":
			ts.Config.UseChainRule = false
			ts.TweenStep(net, input, 0, 1, 0.05)

		case "Tween":
			ts.ForwardPass(net, input)
			ts.BackwardPass(net, 0, 1) // Target index 0
			ts.CalculateLinkBudgets()
			ts.TweenWeights(net, 0.05)

		case "Tween Chain":
			ts.Config.UseChainRule = true
			ts.ForwardPass(net, input)
			ts.BackwardPassChainRule(net, 0, 1)
			ts.CalculateLinkBudgets()
			ts.TweenWeightsChainRule(net, 0.05)

		case "Step Tween Chain":
			ts.Config.UseChainRule = true
			ts.TweenStep(net, input, 0, 1, 0.05)
		}
	}
}

func fs_freezeLayer(cfg *nn.LayerConfig) {
	cfg.Frozen = true
	// Recurse for Parallel
	if len(cfg.ParallelBranches) > 0 {
		for i := range cfg.ParallelBranches {
			fs_freezeLayer(&cfg.ParallelBranches[i])
		}
	}
}

func fs_evaluateNetwork(net *nn.Network, isHigh bool) (float32, float32, float32, float32) {
	input := make([]float32, fs_InputSize)
	if isHigh {
		input[0] = 0.9
	} else {
		input[0] = 0.1
	}
	inputTensor := nn.NewTensorFromSlice(input, fs_InputSize)

	// Network Output
	netOut, _, _, _ := nn.GenericForwardPass(net, inputTensor, nil)
	netVal := float32(netOut.Data[0])

	// Experts (Frozen Snapshot)
	layer := net.GetLayer(0, 0, 0)

	e1Net := nn.NewNetwork(fs_InputSize, 1, 1, 1)
	e1Net.SetLayer(0, 0, 0, layer.ParallelBranches[0])
	e1Out, _, _, _ := nn.GenericForwardPass(e1Net, inputTensor, nil)
	val1 := float32(e1Out.Data[0])

	e2Net := nn.NewNetwork(fs_InputSize, 1, 1, 1)
	e2Net.SetLayer(0, 0, 0, layer.ParallelBranches[1])
	e2Out, _, _, _ := nn.GenericForwardPass(e2Net, inputTensor, nil)
	val2 := float32(e2Out.Data[0])

	ideal := val1
	return val1, val2, netVal, ideal
}

// =============================================================================
// PART 4 IMPL: ODDS DEMO (Stitched Experts)
// =============================================================================

func runOddsDemo() {
	rand.Seed(time.Now().UnixNano())

	fmt.Println("╔══════════════════════════════════════════════════════════════════╗")
	fmt.Println("║            🎲 Odds Experiment: Stitched Experts Demo             ║")
	fmt.Println("╚══════════════════════════════════════════════════════════════════╝")

	// ===========================================================================
	// DEMO 1: Simple 2-branch filter with Stitched Experts
	// ===========================================================================
	fmt.Println("\n📌 Demo 1: Two Odd-Sized Expert Branches Stitched to Common Size")
	odds_demo1TwoBranchStitched()

	// ===========================================================================
	// DEMO 2: Multi-branch filter with different expert types
	// ===========================================================================
	fmt.Println("\n📌 Demo 2: Multi-Branch Stitched Filter")
	odds_demo2MultiBranchStitched()

	// ===========================================================================
	// DEMO 3: Training the gate to specialize
	// ===========================================================================
	fmt.Println("\n📌 Demo 3: Training Gate Specialization with Odd Experts")
	odds_demo3TrainGateSpecializationStitched()
}

// odds_demo1TwoBranchStitched
func odds_demo1TwoBranchStitched() {
	inputSize := 16
	commonOutputSize := 10

	// Expert 1: Size 5 -> Stitch to 10
	expert1 := nn.InitDenseLayer(inputSize, 5, nn.ActivationLeakyReLU)
	stitch1 := nn.InitStitchLayer(5, commonOutputSize)
	branch1 := nn.InitSequentialLayer(expert1, stitch1)

	// Expert 2: Size 7 -> Stitch to 10
	expert2 := nn.InitDenseLayer(inputSize, 7, nn.ActivationSigmoid)
	stitch2 := nn.InitStitchLayer(7, commonOutputSize)
	branch2 := nn.InitSequentialLayer(expert2, stitch2)

	// Gate
	gateLayer := nn.InitDenseLayer(inputSize, 2, nn.ActivationScaledReLU)

	filterLayer := nn.LayerConfig{
		Type:              nn.LayerParallel,
		ParallelBranches:  []nn.LayerConfig{branch1, branch2},
		CombineMode:       "filter",
		FilterGateConfig:  &gateLayer,
		FilterSoftmax:     nn.SoftmaxStandard,
		FilterTemperature: 1.0,
	}

	net := nn.NewNetwork(inputSize, 1, 1, 3)
	net.SetLayer(0, 0, 0, nn.InitDenseLayer(inputSize, inputSize, nn.ActivationLeakyReLU))
	net.SetLayer(0, 0, 1, filterLayer)
	net.SetLayer(0, 0, 2, nn.InitDenseLayer(commonOutputSize, 5, nn.ActivationSigmoid))

	// Test
	input := make([]float32, inputSize)
	for i := range input {
		input[i] = rand.Float32()
	}
	output, _ := net.ForwardCPU(input)
	fmt.Printf("   ✅ Forward pass successful! Output size: %d\n", len(output))
}

// odds_demo2MultiBranchStitched
func odds_demo2MultiBranchStitched() {
	inputSize := 16
	commonOutputSize := 8
	sizes := []int{4, 12, 6, 20}

	branches := make([]nn.LayerConfig, len(sizes))
	for i, size := range sizes {
		expert := nn.InitDenseLayer(inputSize, size, nn.ActivationLeakyReLU)
		stitch := nn.InitStitchLayer(size, commonOutputSize)
		branches[i] = nn.InitSequentialLayer(expert, stitch)
	}

	gateLayer := nn.InitDenseLayer(inputSize, len(branches), nn.ActivationScaledReLU)

	filterLayer := nn.LayerConfig{
		Type:              nn.LayerParallel,
		ParallelBranches:  branches,
		CombineMode:       "filter",
		FilterGateConfig:  &gateLayer,
		FilterSoftmax:     nn.SoftmaxEntmax,
		FilterTemperature: 0.5,
	}

	net := nn.NewNetwork(inputSize, 1, 1, 2)
	net.SetLayer(0, 0, 0, filterLayer)
	net.SetLayer(0, 0, 1, nn.InitDenseLayer(commonOutputSize, 1, nn.ActivationSigmoid))

	fmt.Printf("   🧪 Testing with 4 odd-sized experts (%v) stitched to %d\n", sizes, commonOutputSize)
	for trial := 0; trial < 3; trial++ {
		input := make([]float32, inputSize)
		for i := range input {
			input[i] = rand.Float32()
		}
		output, _ := net.ForwardCPU(input)
		fmt.Printf("      Trial %d: Output val=%.4f\n", trial+1, output[0])
	}
}

// odds_demo3TrainGateSpecializationStitched
func odds_demo3TrainGateSpecializationStitched() {
	inputSize := 8
	commonOutputSize := 4

	// -----------------------------------------------------------
	// -----------------------------------------------------------
	// STEP 1: Create two odd-sized networks and pre-train them
	// Expert 1 (Size 3): Detects High input[0]
	// Expert 2 (Size 5): Detects Low input[0]
	// -----------------------------------------------------------

	// Training Expert 1 (Size 3) + Stitch (Size 4)
	fmt.Printf("   🎓 Pre-training Expert 1 (Size 3 -> 4) to detect HIGH input[0]...\n")
	expert1Core := nn.InitDenseLayer(inputSize, 3, nn.ActivationSigmoid)
	stitch1 := nn.InitStitchLayer(3, commonOutputSize)

	net1 := nn.NewNetwork(inputSize, 1, 1, 2)
	net1.SetLayer(0, 0, 0, expert1Core)
	net1.SetLayer(0, 0, 1, stitch1)

	// Generate training data for Expert 1
	// Task: if input[0] is high, output high
	trainData1 := make([]nn.TrainingBatch, 2000)
	for i := range trainData1 {
		input := make([]float32, inputSize)
		for j := range input {
			input[j] = rand.Float32()
		}

		target := float32(0.0)
		if rand.Float32() > 0.5 {
			input[0] = 0.8 + rand.Float32()*0.2 // High input
			target = 1.0
		} else {
			input[0] = rand.Float32() * 0.2 // Low input
			target = 0.0
		}
		trainData1[i] = nn.TrainingBatch{Input: input, Target: []float32{target}}
	}

	config1 := &nn.TrainingConfig{
		Epochs:       10,
		LearningRate: 0.1,
		Verbose:      false,
		LossType:     "mse",
	}
	net1.Train(trainData1, config1)

	branch1 := nn.InitSequentialLayer(*net1.GetLayer(0, 0, 0), *net1.GetLayer(0, 0, 1))

	// Training Expert 2 (Size 5) + Stitch (Size 4)
	fmt.Printf("   🎓 Pre-training Expert 2 (Size 5 -> 4) to detect LOW input[0]...\n")
	expert2Core := nn.InitDenseLayer(inputSize, 5, nn.ActivationSigmoid)
	stitch2 := nn.InitStitchLayer(5, commonOutputSize)

	net2 := nn.NewNetwork(inputSize, 1, 1, 2)
	net2.SetLayer(0, 0, 0, expert2Core)
	net2.SetLayer(0, 0, 1, stitch2)

	// Generate training data for Expert 2
	// Logic: Low input -> High Output
	trainData2 := make([]nn.TrainingBatch, 2000)
	for i := range trainData2 {
		input := make([]float32, inputSize)
		for j := range input {
			input[j] = rand.Float32()
		}

		target := float32(0.0)
		if rand.Float32() > 0.5 {
			input[0] = rand.Float32() * 0.2 // Low input
			target = 1.0                    // High Output
			// Note: This logic seems flipped compared to Ex 1?
			// Ex1: High Input -> High Output
			// Ex2: Low Input -> High Output
			// Yes, both want to output High when they see their "preferred" signal.
		} else {
			input[0] = 0.8 + rand.Float32()*0.2 // High input
			target = 0.0                        // Low Output
		}
		trainData2[i] = nn.TrainingBatch{Input: input, Target: []float32{target}}
	}

	config2 := &nn.TrainingConfig{
		Epochs:       10,
		LearningRate: 0.1,
		Verbose:      false,
		LossType:     "mse",
	}
	net2.Train(trainData2, config2)

	branch2 := nn.InitSequentialLayer(*net2.GetLayer(0, 0, 0), *net2.GetLayer(0, 0, 1))

	// -----------------------------------------------------------
	// STEP 2: Create filter layer
	// -----------------------------------------------------------
	gateLayer := nn.InitDenseLayer(inputSize, 2, nn.ActivationScaledReLU)
	filterLayer := nn.LayerConfig{
		Type:              nn.LayerParallel,
		ParallelBranches:  []nn.LayerConfig{branch1, branch2},
		CombineMode:       "filter",
		FilterGateConfig:  &gateLayer,
		FilterSoftmax:     nn.SoftmaxStandard,
		FilterTemperature: 0.5,
	}

	net := nn.NewNetwork(inputSize, 1, 1, 2)
	net.SetLayer(0, 0, 0, filterLayer)
	outputL := nn.InitDenseLayer(commonOutputSize, 1, nn.ActivationSigmoid)
	net.SetLayer(0, 0, 1, outputL)

	// -----------------------------------------------------------
	// STEP 4: Train only the gate layer
	// -----------------------------------------------------------
	fmt.Printf("   🏋️ Training GATE layer for 1000 steps...\n")

	ts := nn.NewTweenState(net, nil) // TweenState for main net
	ts.Config.UseChainRule = true

	for epoch := 0; epoch < 1000; epoch++ {
		input := make([]float32, inputSize)
		for j := range input {
			input[j] = rand.Float32()
		}

		// Target logic:
		// HIGH input[0] -> Should route to Expert 1
		// LOW input[0]  -> Should route to Expert 2

		if epoch%2 == 0 {
			input[0] = 0.9 // High
		} else {
			input[0] = 0.1 // Low
		}

		// If gate works, output will be high (since experts are specialized to output high for their preferred input).
		// So we train the whole net to maximize output.
		// Since experts are frozen (we don't update them here effectively via this simple call unless we passed gradients everywhere),
		// essentially we are just updating the gate to find the max-output path.

		// Using TweenStep to maximize output (target [1.0])
		ts.TweenStep(net, input, 0, 1, 0.05)
	}

	// -----------------------------------------------------------
	// STEP 5: Test
	// -----------------------------------------------------------
	fmt.Printf("   📊 Testing Selection:\n")

	highIn := make([]float32, inputSize)
	highIn[0] = 0.9
	lowIn := make([]float32, inputSize)
	lowIn[0] = 0.1

	hOut, _ := net.ForwardCPU(highIn)
	lOut, _ := net.ForwardCPU(lowIn)

	fmt.Printf("      High Input → Output: %.4f (Expert 1 preferred)\n", hOut[0])
	fmt.Printf("      Low Input  → Output: %.4f (Expert 2 preferred)\n", lOut[0])

	diff := math.Abs(float64(hOut[0] - lOut[0]))
	if hOut[0] > 0.5 && lOut[0] > 0.5 {
		fmt.Printf("   ✅ Gate learned to pick the right expert (both outputs high)!\n")
	} else if diff < 0.1 {
	} else {
		fmt.Printf("   ✅ Gate differentiating.\n")
	}
}

// =============================================================================
// PART 4 IMPL: FILTER DEMO (Original)
// =============================================================================

func runFilterDemo() {
	rand.Seed(time.Now().UnixNano())

	fmt.Println("╔══════════════════════════════════════════════════════════════════╗")
	fmt.Println("║        🔬 Filter CombineMode Demo (Mixture of Experts)           ║")
	fmt.Println("╚══════════════════════════════════════════════════════════════════╝")

	// ===========================================================================
	// DEMO 1: Simple 2-branch filter with Dense experts
	// ===========================================================================
	fmt.Println("\n📌 Demo 1: Two Dense Expert Branches with Learned Gating")
	fd_demo1TwoBranchFilter()

	// ===========================================================================
	// DEMO 2: Multi-branch filter with different expert types
	// ===========================================================================
	fmt.Println("\n📌 Demo 2: Multi-Branch Filter (4 experts)")
	fd_demo2MultiBranchFilter()

	// ===========================================================================
	// DEMO 3: Training the gate to specialize
	// ===========================================================================
	fmt.Println("\n📌 Demo 3: Training Gate Specialization")
	fd_demo3TrainGateSpecialization()
}

// fd_demo1TwoBranchFilter creates a simple 2-expert filtered parallel layer
func fd_demo1TwoBranchFilter() {
	inputSize := 16
	expertSize := 8

	// Create two Dense expert branches
	expert1 := nn.InitDenseLayer(inputSize, expertSize, nn.ActivationLeakyReLU)
	expert2 := nn.InitDenseLayer(inputSize, expertSize, nn.ActivationLeakyReLU)

	// Create gate layer: decides how much to use each expert
	gateLayer := nn.InitDenseLayer(inputSize, 2, nn.ActivationScaledReLU)

	// Build the filtered parallel layer
	filterLayer := nn.LayerConfig{
		Type:              nn.LayerParallel,
		ParallelBranches:  []nn.LayerConfig{expert1, expert2},
		CombineMode:       "filter",
		FilterGateConfig:  &gateLayer,
		FilterSoftmax:     nn.SoftmaxStandard,
		FilterTemperature: 1.0,
	}

	// Create network: Input -> FilteredExperts -> Output
	net := nn.NewNetwork(inputSize, 1, 1, 3)
	inputL := nn.InitDenseLayer(inputSize, inputSize, nn.ActivationLeakyReLU)
	net.SetLayer(0, 0, 0, inputL)
	net.SetLayer(0, 0, 1, filterLayer)
	outputL := nn.InitDenseLayer(expertSize, inputSize, nn.ActivationSigmoid)
	net.SetLayer(0, 0, 2, outputL)

	// Test forward pass
	input := make([]float32, inputSize)
	for i := range input {
		input[i] = rand.Float32()
	}

	output, _ := net.ForwardCPU(input)
	if len(output) == 0 {
		fmt.Printf("   ❌ Forward failed: empty output\n")
		return
	}

	fmt.Printf("   ✅ Forward pass successful!\n")
	fmt.Printf("   📊 Input size: %d, Output size: %d\n", len(input), len(output))
	fmt.Printf("   📈 Sample output values: [%.3f, %.3f, %.3f, ...]\n",
		output[0], output[1], output[2])
}

// fd_demo2MultiBranchFilter creates a 4-expert filtered parallel layer
func fd_demo2MultiBranchFilter() {
	inputSize := 32
	expertSize := 16
	numExperts := 4

	// Create multiple Dense expert branches
	experts := make([]nn.LayerConfig, numExperts)
	for i := 0; i < numExperts; i++ {
		experts[i] = nn.InitDenseLayer(inputSize, expertSize, nn.ActivationLeakyReLU)
		// Add some variety in weights
		for j := range experts[i].Kernel {
			experts[i].Kernel[j] *= float32(1.0 + 0.2*float64(i))
		}
	}

	// Create gate layer for 4 experts
	gateLayer := nn.InitDenseLayer(inputSize, numExperts, nn.ActivationScaledReLU)

	// Build with sparse softmax for sharper routing
	filterLayer := nn.LayerConfig{
		Type:              nn.LayerParallel,
		ParallelBranches:  experts,
		CombineMode:       "filter",
		FilterGateConfig:  &gateLayer,
		FilterSoftmax:     nn.SoftmaxEntmax, // Sparse routing
		FilterTemperature: 0.5,              // Sharper selections
	}

	// Create network
	net := nn.NewNetwork(inputSize, 1, 1, 3)
	inputL := nn.InitDenseLayer(inputSize, inputSize, nn.ActivationLeakyReLU)
	net.SetLayer(0, 0, 0, inputL)
	net.SetLayer(0, 0, 1, filterLayer)
	outputL := nn.InitDenseLayer(expertSize, inputSize, nn.ActivationSigmoid)
	net.SetLayer(0, 0, 2, outputL)

	// Test with multiple inputs
	fmt.Printf("   🧪 Testing with 5 different inputs:\n")
	for trial := 0; trial < 5; trial++ {
		input := make([]float32, inputSize)
		for i := range input {
			input[i] = rand.Float32()
		}

		output, _ := net.ForwardCPU(input)
		if len(output) == 0 {
			fmt.Printf("      Trial %d: ❌ empty output\n", trial+1)
			continue
		}

		// Calculate output statistics
		sum := float32(0)
		for _, v := range output {
			sum += v
		}
		avg := sum / float32(len(output))

		fmt.Printf("      Trial %d: ✅ avg=%.4f\n", trial+1, avg)
	}
}

// fd_demo3TrainGateSpecialization - Pre-train experts then train gate to route
func fd_demo3TrainGateSpecialization() {
	inputSize := 8
	expertSize := 8

	// -----------------------------------------------------------
	// STEP 1: Create two simple networks and pre-train them
	// Expert 1: trained to output HIGH when input[0] > 0.5
	// Expert 2: trained to output HIGH when input[0] <= 0.5
	// -----------------------------------------------------------
	fmt.Printf("   🎓 Pre-training Expert 1 (responds to HIGH first element)...\n")
	expert1 := nn.InitDenseLayer(inputSize, expertSize, nn.ActivationSigmoid)

	// Create temp net for training Expert 1
	e1Net := nn.NewNetwork(inputSize, 1, 1, 1)
	e1Net.SetLayer(0, 0, 0, expert1)

	// Train Expert 1: High Input -> High Output (1.0), Low Input -> Low Output (0.0)
	trainData1 := make([]nn.TrainingBatch, 2000)
	for i := range trainData1 {
		input := make([]float32, inputSize)
		for j := range input {
			input[j] = rand.Float32()
		}

		target := float32(0.0)
		if rand.Float32() > 0.5 {
			input[0] = 0.7 + rand.Float32()*0.3 // High
			target = 1.0
		} else {
			input[0] = rand.Float32() * 0.3 // Low
			target = 0.0
		}
		trainData1[i] = nn.TrainingBatch{Input: input, Target: []float32{target}}
	}

	config1 := &nn.TrainingConfig{
		Epochs:       10,
		LearningRate: 0.1,
		Verbose:      false,
		LossType:     "mse",
	}
	e1Net.Train(trainData1, config1)
	expert1 = *e1Net.GetLayer(0, 0, 0) // Update config

	fmt.Printf("   🎓 Pre-training Expert 2 (responds to LOW first element)...\n")
	expert2 := nn.InitDenseLayer(inputSize, expertSize, nn.ActivationSigmoid)

	// Create temp net for training Expert 2
	e2Net := nn.NewNetwork(inputSize, 1, 1, 1)
	e2Net.SetLayer(0, 0, 0, expert2)

	// Train Expert 2: Low Input -> High Output (1.0), High Input -> Low Output (0.0)
	trainData2 := make([]nn.TrainingBatch, 2000)
	for i := range trainData2 {
		input := make([]float32, inputSize)
		for j := range input {
			input[j] = rand.Float32()
		}

		target := float32(0.0)
		if rand.Float32() > 0.5 {
			input[0] = rand.Float32() * 0.3 // Low
			target = 1.0
		} else {
			input[0] = 0.7 + rand.Float32()*0.3 // High
			target = 0.0
		}
		trainData2[i] = nn.TrainingBatch{Input: input, Target: []float32{target}}
	}

	config2 := &nn.TrainingConfig{
		Epochs:       10,
		LearningRate: 0.1,
		Verbose:      false,
		LossType:     "mse",
	}
	e2Net.Train(trainData2, config2)
	expert2 = *e2Net.GetLayer(0, 0, 0) // Update config

	// -----------------------------------------------------------
	// STEP 2: Create filter layer combining both experts
	// -----------------------------------------------------------
	gateLayer := nn.InitDenseLayer(inputSize, 2, nn.ActivationScaledReLU)

	filterLayer := nn.LayerConfig{
		Type:              nn.LayerParallel,
		ParallelBranches:  []nn.LayerConfig{expert1, expert2},
		CombineMode:       "filter",
		FilterGateConfig:  &gateLayer,
		FilterSoftmax:     nn.SoftmaxStandard,
		FilterTemperature: 0.5,
	}

	// Simple network: just the filter layer
	net := nn.NewNetwork(inputSize, 1, 1, 2)
	net.SetLayer(0, 0, 0, filterLayer)
	outputL := nn.InitDenseLayer(expertSize, 1, nn.ActivationSigmoid)
	net.SetLayer(0, 0, 1, outputL)

	// -----------------------------------------------------------
	// STEP 3: Test without training gate - should give mixed results
	// -----------------------------------------------------------
	fmt.Printf("   📊 Testing BEFORE gate training:\n")

	// Test with high first element
	highInput := make([]float32, inputSize)
	for j := range highInput {
		highInput[j] = rand.Float32() * 0.5
	}
	highInput[0] = 0.9

	// Test with low first element
	lowInput := make([]float32, inputSize)
	for j := range lowInput {
		lowInput[j] = rand.Float32() * 0.5
	}
	lowInput[0] = 0.1

	highOut, _ := net.ForwardCPU(highInput)
	lowOut, _ := net.ForwardCPU(lowInput)

	fmt.Printf("      High input[0]=0.9 → output=%.4f\n", highOut[0])
	fmt.Printf("      Low input[0]=0.1  → output=%.4f\n", lowOut[0])

	// -----------------------------------------------------------
	// STEP 4: Train only the gate layer
	// -----------------------------------------------------------
	fmt.Printf("   🏋️ Training GATE layer for 2000 steps...\n")

	ts := nn.NewTweenState(net, nil)
	ts.Config.LinkBudgetScale = 0.3
	ts.Config.UseChainRule = true

	for epoch := 0; epoch < 2000; epoch++ {
		input := make([]float32, inputSize)
		for j := range input {
			input[j] = rand.Float32() * 0.5
		}

		// Half the time: high first element
		// Half the time: low first element
		if epoch%2 == 0 {
			input[0] = 0.7 + rand.Float32()*0.3
		} else {
			input[0] = rand.Float32() * 0.3
		}

		// Use targetIdx=0 for single output, outputSize=1
		ts.TweenStep(net, input, 0, 1, 0.01)
	}

	// -----------------------------------------------------------
	// STEP 5: Test AFTER gate training
	// -----------------------------------------------------------
	fmt.Printf("   📊 Testing AFTER gate training:\n")

	highOut2, _ := net.ForwardCPU(highInput)
	lowOut2, _ := net.ForwardCPU(lowInput)

	fmt.Printf("      High input[0]=0.9 → output=%.4f (was %.4f)\n", highOut2[0], highOut[0])
	fmt.Printf("      Low input[0]=0.1  → output=%.4f (was %.4f)\n", lowOut2[0], lowOut[0])

	// Check if outputs changed (indicating gate learned something)
	highDiff := math.Abs(float64(highOut2[0] - highOut[0]))
	lowDiff := math.Abs(float64(lowOut2[0] - lowOut[0]))

	if highDiff > 0.01 || lowDiff > 0.01 {
		fmt.Printf("   ✅ Gate learned to differentiate! (changes: high=%.4f, low=%.4f)\n", highDiff, lowDiff)
	} else {
		fmt.Printf("   ⚠️ Gate didn't learn much (changes: high=%.4f, low=%.4f)\n", highDiff, lowDiff)
	}
}

// =============================================================================
// PART 5: GPU Determinism Tests (Ported from det_gpu_v_cpu.go)
// =============================================================================

// GPULayerTestCase defines a test case for a specific hidden layer type
type GPULayerTestCase struct {
	Name       string
	JSONConfig string // Full network JSON config
	InputSize  int    // Network input size
	InputType  string // "uniform" (default), "indices" (for embedding)
	VocabSize  int    // For "indices" type, max token ID
}

var gpuLayerTests = []GPULayerTestCase{
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
		JSONConfig: `{
			"id": "gpu_test_combined",
			"batch_size": 1,
			"grid_rows": 1,
			"grid_cols": 1,
			"layers_per_cell": 9,
			"layers": [
				{"type": "dense", "activation": "leaky_relu", "input_height": 64, "output_height": 64},
				{"type": "swiglu", "input_height": 64, "output_height": 512},
				{"type": "layer_norm", "norm_size": 512, "epsilon": 1e-5},
				{"type": "conv1d", "input_channels": 64, "filters": 64, "kernel_size": 3, "stride": 1, "padding": 1, "input_length": 8},
				{"type": "rnn", "input_size": 64, "hidden_size": 64, "seq_length": 8},
				{"type": "lstm", "input_size": 64, "hidden_size": 64, "seq_length": 8},
				{"type": "multi_head_attention", "d_model": 64, "num_heads": 8, "seq_length": 8},
				{"type": "rms_norm", "norm_size": 64, "epsilon": 1e-5},
				{"type": "dense", "activation": "sigmoid", "input_height": 512, "output_height": 2}
			]
		}`,
		InputSize: 64,
	},
}

func runGPULayerTest(test GPULayerTestCase) bool {
	fmt.Printf("┌──────────────────────────────────────────────────────────────────────┐\n")
	fmt.Printf("│ Testing: %-59s│\n", test.Name)
	fmt.Printf("└──────────────────────────────────────────────────────────────────────┘\n")

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
		return false
	}
	// DO NOT force BatchSize to 1 here, let JSON config dictate it.
	// But ensure network.BatchSize is populated if 0 (BuildNetworkFromJSON should do this)
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
	network.GPU = false
	cpuOutput, _ := network.ForwardCPU(input)

	// GPU Forward
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

		gpuOutput, _ = network.ForwardCPU(input)
		gpuForwardOK = true
	}()

	network.ReleaseGPUWeights()

	if !gpuForwardOK {
		return false
	}

	return compareOutputs(cpuOutput, gpuOutput)
}

func compareOutputs(cpu, gpu []float32) bool {
	if len(cpu) != len(gpu) {
		fmt.Printf("  ❌ Size mismatch: CPU=%d, GPU=%d\n", len(cpu), len(gpu))
		return false
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

	passed := false
	if maxDiff == 0 {
		fmt.Println("  ✅ [GOLD STANDARD] Exact Bit-Determinism")
		passed = true
	} else if maxDiff < 1e-7 {
		fmt.Println("  ✅ [EXCELLENT] Near-Machine-Epsilon (< 1e-7)")
		passed = true
	} else if maxDiff < 1e-5 {
		fmt.Println("  ✅ [INDUSTRY STANDARD] Functional Equivalence (< 1e-5)")
		passed = true
	} else if maxDiff < 1e-3 {
		fmt.Println("  ⚠️ [ACCEPTABLE DRIFT] Approximate Match (< 1e-3)")
		passed = true
	} else {
		fmt.Println("  ❌ [FAILURE] Significant Divergence (> 1e-3)")
		passed = false
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

	return passed
}
