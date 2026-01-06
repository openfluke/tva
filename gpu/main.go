package main

import (
	"flag"
	"fmt"
	"log"
	"math"
	"os"
	"strings"
	"time"

	"github.com/openfluke/loom/nn"
)

// Global flags
var (
	flagLayer = flag.String("layer", "", "Specific layer type to test (e.g. 'Dense', 'Conv2D'). Comma-separated for multiple.")
	flagDepth = flag.String("depth", "", "Network depth: 'shallow', 'medium', 'deep'. Comma-separated or empty for all.")
	flagDType = flag.String("dtype", "", "Specific dtype to test (e.g. 'float32'). Comma-separated for multiple.")
	flagAll   = flag.Bool("all", false, "Run all combos")
)

// All supported DTypes (from nn/types.go)
var allDTypes = []string{
	"float32", "float64", "float16",
	"int8", "int16", "int32", "int64",
	"uint8", "uint16", "uint32", "uint64",
}

// All supported layer types
var allLayers = []string{
	"Dense", "MHA", "RNN", "LSTM", "LayerNorm", "RMSNorm",
	"SwiGLU", "Conv2D", "Parallel", "Sequential", "Softmax",
}

// All depths
var allDepths = []string{"shallow", "medium", "deep"}

func main() {
	flag.Parse()

	fmt.Println("╔══════════════════════════════════════════════════════════════════════╗")
	fmt.Println("║               LOOM GPU Verification Tool                            ║")
	fmt.Println("╚══════════════════════════════════════════════════════════════════════╝")

	// Determine layers to test
	layers := allLayers
	if *flagLayer != "" {
		layers = parseCSV(*flagLayer, allLayers)
		if len(layers) == 0 {
			log.Fatalf("No valid layers found in: %s", *flagLayer)
		}
	}

	// Determine dtypes to test
	dtypes := allDTypes
	if *flagDType != "" {
		dtypes = parseCSV(*flagDType, allDTypes)
		if len(dtypes) == 0 {
			log.Fatalf("No valid dtypes found in: %s", *flagDType)
		}
	}

	// Determine depths to test
	depths := allDepths
	if *flagDepth != "" {
		depths = parseCSV(*flagDepth, allDepths)
		if len(depths) == 0 {
			log.Fatalf("No valid depths found in: %s", *flagDepth)
		}
	}

	fmt.Printf("Layers: %v\n", layers)
	fmt.Printf("DTypes: %v\n", dtypes)
	fmt.Printf("Depths: %v\n\n", depths)

	total := 0
	passed := 0
	failed := 0
	skipped := 0

	for _, l := range layers {
		for _, depth := range depths {
			for _, d := range dtypes {
				res := verifyLayer(l, depth, d)
				total++
				switch res {
				case "PASS":
					passed++
				case "FAIL":
					failed++
				case "SKIP":
					skipped++
				}
			}
		}
	}

	fmt.Println("\n═══════════════════════════════════════════════════════════════════════")
	fmt.Printf("RESULTS: %d Passed, %d Failed, %d Skipped (Total %d)\n", passed, failed, skipped, total)
	if failed > 0 {
		os.Exit(1)
	}
}

// parseCSV parses a comma-separated string and filters against valid options
func parseCSV(input string, valid []string) []string {
	parts := strings.Split(input, ",")
	result := []string{}
	for _, p := range parts {
		p = strings.TrimSpace(p)
		for _, v := range valid {
			if strings.EqualFold(p, v) {
				result = append(result, v)
				break
			}
		}
	}
	return result
}

func verifyLayer(layerType, depth, dtype string) string {
	configStr := getJSONConfig(layerType, depth, dtype)
	if configStr == "" {
		fmt.Printf("  ⚠️  %s/%s: No config available (Skipping)\n", layerType, dtype)
		return "SKIP"
	}

	// Build network
	net, _, err := nn.BuildNetworkFromJSONWithDType(configStr)
	if err != nil {
		fmt.Printf("  ❌ %s/%s: Build failed: %v\n", layerType, dtype, err)
		return "FAIL"
	}

	// Prepare Input
	inputSize := getInputSize(layerType, depth)
	input := make([]float32, inputSize)
	for i := range input {
		input[i] = float32(i+1) * 0.01
	}

	// 1. CPU Reference
	startCPU := time.Now()
	cpuOut, _ := net.ForwardCPU(input)
	timeCPU := time.Since(startCPU)
	if len(cpuOut) == 0 {
		fmt.Printf("  ❌ %s/%s: CPU Forward failed (empty output)\n", layerType, dtype)
		return "FAIL"
	}

	// 2. GPU Candidate (Placeholder for now)
	startGPU := time.Now()
	gpuOut, err := forwardGPU_Placeholder(net, input)
	timeGPU := time.Since(startGPU)

	if err != nil {
		if strings.Contains(err.Error(), "not implemented") {
			fmt.Printf("  ⚠️  %s/%s [%s]: GPU Not Implemented Yet\n", layerType, dtype, depth)
			return "SKIP"
		}
		fmt.Printf("  ❌ %s/%s: GPU Forward failed: %v\n", layerType, dtype, err)
		return "FAIL"
	}

	// 3. Compare
	maxErr := computeMaxError(cpuOut, gpuOut)
	threshold := getThreshold(dtype)

	status := "✅"
	result := "PASS"
	if maxErr > threshold {
		status = "❌"
		result = "FAIL"
	}

	fmt.Printf("  %s %-10s %-8s [%-7s]: CPU=%.2fms GPU=%.2fms Err=%.2e (Lim %.2e)\n",
		status, layerType, dtype, depth,
		float64(timeCPU.Microseconds())/1000.0,
		float64(timeGPU.Microseconds())/1000.0,
		maxErr, threshold)

	return result
}

// forwardGPU_Placeholder mocks the GPU call.
// TODO: Replace with actual GPU implementation
func forwardGPU_Placeholder(net *nn.Network, input []float32) ([]float32, error) {
	return nil, fmt.Errorf("GPU execution not implemented")
}

func computeMaxError(a, b []float32) float64 {
	if len(a) != len(b) {
		return 999999.0
	}
	maxErr := 0.0
	for i := range a {
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
		return 1e-12
	case "float32":
		return 1e-5
	case "float16":
		return 1e-3 // Half precision has lower accuracy
	case "int32", "int64", "uint32", "uint64":
		return 0.1
	case "int16", "uint16":
		return 0.2
	case "int8", "uint8":
		return 0.5
	default:
		return 0.1
	}
}

func getInputSize(layerType, depth string) int {
	base := 8
	switch layerType {
	case "MHA":
		base = 64
	case "SwiGLU":
		base = 32
	case "Conv2D":
		base = 16
	case "LayerNorm", "RMSNorm", "RNN", "LSTM":
		base = 16
	}
	return base
}

// getJSONConfig generates network config with proper grid structure
func getJSONConfig(layerType, depth, dtype string) string {
	// Depth determines number of layers
	numLayers := 2
	switch depth {
	case "medium":
		numLayers = 4
	case "deep":
		numLayers = 8
	}

	switch layerType {
	case "Dense":
		return getDenseConfig(dtype, numLayers)
	default:
		// Other layers not yet implemented for this tool
		return ""
	}
}

// getDenseConfig creates a Dense-only network with specified depth
func getDenseConfig(dtype string, numLayers int) string {
	// Build layer array dynamically
	// Input: 8 -> expand to 64 -> ... -> contract to 4
	inputSize := 8
	hiddenSize := 64
	outputSize := 4

	layers := "[\n"

	for i := 0; i < numLayers; i++ {
		var inH, outH int
		var activation string

		if i == 0 {
			// First layer: input -> hidden
			inH = inputSize
			outH = hiddenSize
			activation = "leaky_relu"
		} else if i == numLayers-1 {
			// Last layer: hidden -> output
			inH = hiddenSize
			outH = outputSize
			activation = "sigmoid"
		} else {
			// Hidden layers
			inH = hiddenSize
			outH = hiddenSize
			activation = "tanh"
		}

		comma := ","
		if i == numLayers-1 {
			comma = ""
		}

		layers += fmt.Sprintf(`				{"type": "dense", "activation": "%s", "input_height": %d, "output_height": %d}%s
`, activation, inH, outH, comma)
	}

	layers += "			]"

	return fmt.Sprintf(`{
		"id": "dense_gpu_test",
		"dtype": "%s",
		"batch_size": 1,
		"grid_rows": 1,
		"grid_cols": 1,
		"layers_per_cell": %d,
		"layers": %s
	}`, dtype, numLayers, layers)
}
