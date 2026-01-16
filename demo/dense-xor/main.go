package main

import (
	"fmt"
	"os"
	"runtime"
	"text/tabwriter"

	"github.com/openfluke/loom/nn"
)

// XOR Dataset
type XORSample struct {
	Input  []float32
	Target []float32
	Label  int
}

type VerificationResult struct {
	DType    string
	Quality  float64
	AvgDev   float64
	FileSize int64
	RAM      uint64
}

func generateXORData() []XORSample {
	return []XORSample{
		{Input: []float32{0, 0}, Target: []float32{0}, Label: 0},
		{Input: []float32{0, 1}, Target: []float32{1}, Label: 1},
		{Input: []float32{1, 0}, Target: []float32{1}, Label: 1},
		{Input: []float32{1, 1}, Target: []float32{0}, Label: 0},
	}
}

func formatBytes(b int64) string {
	const unit = 1024
	if b < unit {
		return fmt.Sprintf("%d B", b)
	}
	div, exp := int64(unit), 0
	for n := b / unit; n >= unit; n /= unit {
		div *= unit
		exp++
	}
	return fmt.Sprintf("%.2f %cB", float64(b)/float64(div), "KMGTPE"[exp])
}

func verifyReload(name string, original, reloaded *nn.Network, samples []XORSample) {
	fmt.Printf("\nVerifying %s consistency:\n", name)
	maxDiff := float32(0)
	for i, s := range samples {
		o1, _ := original.Forward(s.Input)
		o2, _ := reloaded.Forward(s.Input)

		diff := float32(0)
		for j := 0; j < len(o1); j++ {
			d := o1[j] - o2[j]
			if d < 0 {
				d = -d
			}
			if d > maxDiff {
				maxDiff = d
			}
			diff += d
		}
		if i == 0 {
			fmt.Printf("  Sample 0: Orig %.4f vs Load %.4f\n", o1[0], o2[0])
		}
	}
	fmt.Printf("  Max Difference over %d samples: %.9f\n", len(samples), maxDiff)
	if maxDiff < 1e-6 {
		fmt.Println("  ✓ Consistency Confirmed")
	} else {
		fmt.Println("  ✗ Consistency FAILED")
	}
}

func unscaleWeights(n *nn.Network, scale float32, isUint bool) {
	invScale := 1.0 / scale
	for i := range n.Layers {
		l := &n.Layers[i]
		s := func(data []float32) {
			for j := range data {
				data[j] *= invScale
				if isUint {
					data[j] -= 1.0
				}
			}
		}
		s(l.Kernel)
		s(l.Bias)
	}
}

func main() {
	fmt.Println("=== Dense Layer Demo: XOR Problem ===\n")

	// Generate XOR data
	data := generateXORData()
	fmt.Printf("Generated %d XOR training samples\n\n", len(data))

	// Create network: Dense layers for XOR
	config := `{
		"id": "xor_dense",
		"batch_size": 1,
		"grid_rows": 1,
		"grid_cols": 1,
		"layers_per_cell": 3,
		"layers": [
			{"type": "dense", "activation": "relu", "input_height": 2, "output_height": 8},
			{"type": "dense", "activation": "relu", "input_height": 8, "output_height": 8},
			{"type": "dense", "activation": "sigmoid", "input_height": 8, "output_height": 1}
		]
	}`

	// Prepare batches
	batches := make([]nn.TrainingBatch, len(data))
	for i, sample := range data {
		batches[i] = nn.TrainingBatch{Input: sample.Input, Target: sample.Target}
	}

	// 1. CPU Training
	fmt.Println("=== Training on CPU ===")
	cpuNet, err := nn.BuildNetworkFromJSON(config)
	if err != nil {
		panic(err)
	}
	cpuNet.InitializeWeights()

	initialWeights, _ := cpuNet.SaveModelToString("xor_init")

	cpuConfig := &nn.TrainingConfig{
		Epochs:          2000,
		LearningRate:    0.1,
		UseGPU:          false,
		LossType:        "mse",
		PrintEveryBatch: 0,
		Verbose:         true,
	}

	cpuNet.Train(batches, cpuConfig)

	// Save CPU model
	fmt.Println("\nSaving CPU model...")
	if err := cpuNet.SaveModel("xor_cpu_model.json", "xor_cpu"); err != nil {
		panic(err)
	}

	// 2. GPU Training
	fmt.Println("\n=== Training on GPU ===")
	gpuNet, err := nn.LoadModelFromString(initialWeights, "xor_init")
	if err != nil {
		panic(err)
	}

	gpuConfig := &nn.TrainingConfig{
		Epochs:          2000,
		LearningRate:    0.1,
		UseGPU:          true,
		LossType:        "mse",
		PrintEveryBatch: 0,
		Verbose:         true,
	}

	if _, err := gpuNet.Train(batches, gpuConfig); err != nil {
		panic(err)
	}

	// Sync weights back
	fmt.Println("Syncing GPU weights to CPU...")
	if err := gpuNet.WeightsToCPU(); err != nil {
		panic(err)
	}
	gpuNet.GPU = false

	// Save GPU model
	fmt.Println("Saving GPU model...")
	if err := gpuNet.SaveModel("xor_gpu_model.json", "xor_gpu"); err != nil {
		panic(err)
	}

	// 3. Verify Save/Load
	fmt.Println("\n=== Verifying Save/Load Consistency ===")
	loadedCPU, err := nn.LoadModel("xor_cpu_model.json", "xor_cpu")
	if err != nil {
		panic(err)
	}
	loadedGPU, err := nn.LoadModel("xor_gpu_model.json", "xor_gpu")
	if err != nil {
		panic(err)
	}

	verifyReload("CPU JSON", cpuNet, loadedCPU, data)
	verifyReload("GPU JSON", gpuNet, loadedGPU, data)

	// 4. Safetensors
	fmt.Println("\n=== Verifying Safetensors ===")
	safetensorsPath := "xor_gpu.safetensors"
	if err := gpuNet.SaveWeightsToSafetensors(safetensorsPath); err != nil {
		panic(err)
	}
	fmt.Println("Saved " + safetensorsPath)

	// Load into fresh network
	safetensorsNet, err := nn.BuildNetworkFromJSON(config)
	if err != nil {
		panic(err)
	}
	if err := safetensorsNet.LoadWeightsFromSafetensors(safetensorsPath); err != nil {
		panic(err)
	}

	verifyReload("GPU Safetensors", gpuNet, safetensorsNet, data)

	// 5. Multi-Precision Testing
	fmt.Println("\n=== Testing All Numerical Types ===")

	// Load base tensors
	baseBytes, err := os.ReadFile(safetensorsPath)
	if err != nil {
		panic(err)
	}
	baseTensors, err := nn.LoadSafetensorsWithShapes(baseBytes)
	if err != nil {
		panic(err)
	}

	dtypes := []struct {
		Name  string
		Scale float32
	}{
		{"F32", 1.0},
		{"F64", 1.0},
		{"F16", 1.0},
		{"BF16", 1.0},
		{"F4", 8.0},
		{"I64", 1000.0},
		{"I32", 1000.0},
		{"I16", 100.0},
		{"I8", 10.0},
		{"U64", 1000.0},
		{"U32", 1000.0},
		{"U16", 100.0},
		{"U8", 10.0},
	}

	results := []VerificationResult{}

	for _, dt := range dtypes {
		fmt.Printf("\n>>> Testing DType: %s (Scale: %.1f)\n", dt.Name, dt.Scale)

		// Prepare scaled tensors
		testTensors := make(map[string]nn.TensorWithShape)
		isUint := dt.Name[0] == 'U'
		for k, v := range baseTensors {
			newVals := make([]float32, len(v.Values))
			for i, val := range v.Values {
				if isUint {
					newVals[i] = (val + 1.0) * dt.Scale
				} else {
					newVals[i] = val * dt.Scale
				}
			}
			testTensors[k] = nn.TensorWithShape{
				Values: newVals,
				Shape:  v.Shape,
				DType:  dt.Name,
			}
		}

		// Save
		fname := fmt.Sprintf("xor_%s.safetensors", dt.Name)
		if err := nn.SaveSafetensors(fname, testTensors); err != nil {
			fmt.Printf("  ❌ Save Failed: %v\n", err)
			continue
		}
		defer os.Remove(fname)

		fi, _ := os.Stat(fname)
		fileSize := fi.Size()

		// Load with memory measurement
		testNet, err := nn.BuildNetworkFromJSON(config)
		if err != nil {
			panic(err)
		}

		runtime.GC()
		var m1, m2 runtime.MemStats
		runtime.ReadMemStats(&m1)

		if err := testNet.LoadWeightsFromSafetensors(fname); err != nil {
			fmt.Printf("  ❌ Load Failed: %v\n", err)
			continue
		}

		runtime.ReadMemStats(&m2)
		ramUsed := m2.HeapAlloc - m1.HeapAlloc

		// Unscale weights
		unscaleWeights(testNet, dt.Scale, isUint)

		// Evaluate
		expected := make([]float64, len(data))
		actual := make([]float64, len(data))

		for i, sample := range data {
			outOrig, _ := gpuNet.Forward(sample.Input)
			outNew, _ := testNet.Forward(sample.Input)
			expected[i] = float64(outOrig[0])
			actual[i] = float64(outNew[0])
		}

		metrics, err := nn.EvaluateModel(expected, actual)
		if err != nil {
			fmt.Printf("  ❌ Evaluation Failed: %v\n", err)
			continue
		}

		metrics.PrintSummary()

		results = append(results, VerificationResult{
			DType:    dt.Name,
			Quality:  metrics.Score,
			AvgDev:   metrics.AverageDeviation,
			FileSize: fileSize,
			RAM:      ramUsed,
		})
	}

	// 6. Final Summary Table
	fmt.Println("\n\n=== NUMERICAL TYPE COMPARISON SUMMARY ===")
	w := tabwriter.NewWriter(os.Stdout, 0, 0, 2, ' ', 0)
	fmt.Fprintln(w, "DType\tQuality Score\tAvg Dev\tFile Size\tRAM Usage")
	fmt.Fprintln(w, "-----\t-------------\t-------\t---------\t---------")
	for _, res := range results {
		fmt.Fprintf(w, "%s\t%.2f%%\t%.4f%%\t%s\t%s\n",
			res.DType,
			res.Quality,
			res.AvgDev,
			formatBytes(res.FileSize),
			formatBytes(int64(res.RAM)),
		)
	}
	w.Flush()

	// 7. Test final predictions
	fmt.Println("\n=== Final XOR Predictions ===")
	for _, sample := range data {
		output, _ := gpuNet.Forward(sample.Input)
		predicted := float32(0)
		if output[0] > 0.5 {
			predicted = 1
		}
		status := "✓"
		if predicted != sample.Target[0] {
			status = "✗"
		}
		fmt.Printf("Input: [%.0f, %.0f] -> Output: %.4f (Predicted: %.0f, Expected: %.0f) %s\n",
			sample.Input[0], sample.Input[1], output[0], predicted, sample.Target[0], status)
	}

	// Cleanup
	gpuNet.ReleaseGPUWeights()
	os.Remove("xor_cpu_model.json")
	os.Remove("xor_gpu_model.json")
	os.Remove(safetensorsPath)
}
