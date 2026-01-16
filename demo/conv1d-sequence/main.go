package main

import (
	"fmt"
	"math/rand"
	"os"
	"runtime"
	"text/tabwriter"

	"github.com/openfluke/loom/nn"
)

type SequenceSample struct {
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

func generateSequenceData(numSamples, seqLen int) []SequenceSample {
	samples := make([]SequenceSample, numSamples)
	rand.Seed(42)

	for i := 0; i < numSamples; i++ {
		input := make([]float32, seqLen)
		pattern := i % 2 // 0: rising, 1: falling

		for j := 0; j < seqLen; j++ {
			if pattern == 0 {
				input[j] = float32(j)/float32(seqLen) + rand.Float32()*0.1
			} else {
				input[j] = 1.0 - float32(j)/float32(seqLen) + rand.Float32()*0.1
			}
		}

		target := make([]float32, 2)
		target[pattern] = 1.0

		samples[i] = SequenceSample{Input: input, Target: target, Label: pattern}
	}
	return samples
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

func verifyReload(name string, original, reloaded *nn.Network, samples []SequenceSample) {
	fmt.Printf("\nVerifying %s consistency:\n", name)
	maxDiff := float32(0)
	for _, s := range samples[:10] {
		o1, _ := original.Forward(s.Input)
		o2, _ := reloaded.Forward(s.Input)
		for j := 0; j < len(o1); j++ {
			d := o1[j] - o2[j]
			if d < 0 {
				d = -d
			}
			if d > maxDiff {
				maxDiff = d
			}
		}
	}
	fmt.Printf("  Max Difference over 10 samples: %.9f\n", maxDiff)
	if maxDiff < 1e-5 {
		fmt.Println("  ✓ Consistency Confirmed")
	} else {
		fmt.Println("  ✗ Consistency WARNING")
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
	fmt.Println("=== Conv1D Demo: Sequence Pattern Detection ===\n")

	seqLen := 32
	trainData := generateSequenceData(100, seqLen)
	testData := generateSequenceData(20, seqLen)
	fmt.Printf("Generated %d training, %d test samples\n\n", len(trainData), len(testData))

	config := `{
		"id": "conv1d_seq",
		"batch_size": 1,
		"grid_rows": 1,
		"grid_cols": 1,
		"layers_per_cell": 6,
		"layers": [
			{"type": "dense", "activation": "none", "input_size": 32, "output_size": 32},
			{"type": "conv1d", "activation": "relu", "input_channels": 1, "filters": 8, "kernel_size": 3, "stride": 1, "padding": 1, "input_length": 32},
			{"type": "conv1d", "activation": "relu", "input_channels": 8, "filters": 16, "kernel_size": 3, "stride": 1, "padding": 1, "input_length": 32},
			{"type": "dense", "activation": "relu", "input_size": 512, "output_size": 64},
			{"type": "dense", "activation": "relu", "input_size": 64, "output_size": 16},
			{"type": "dense", "activation": "softmax", "input_size": 16, "output_size": 2}
		]
	}`

	batches := make([]nn.TrainingBatch, len(trainData))
	for i, sample := range trainData {
		batches[i] = nn.TrainingBatch{Input: sample.Input, Target: sample.Target}
	}

	fmt.Println("=== Training on CPU ===")
	cpuNet, _ := nn.BuildNetworkFromJSON(config)
	cpuNet.InitializeWeights()

	cpuConfig := &nn.TrainingConfig{
		Epochs:          50,
		LearningRate:    0.001,
		UseGPU:          false,
		LossType:        "cross_entropy",
		PrintEveryBatch: 0,
		Verbose:         true,
	}
	cpuNet.Train(batches, cpuConfig)
	cpuNet.SaveModel("conv1d_cpu_model.json", "conv1d_cpu")

	fmt.Println("\n=== Training on GPU ===")
	gpuNet, err := nn.BuildNetworkFromJSON(config)
	if err != nil {
		panic(fmt.Sprintf("Failed to build GPU network: %v", err))
	}
	gpuNet.InitializeWeights()

	gpuConfig := &nn.TrainingConfig{
		Epochs:          50,
		LearningRate:    0.001,
		UseGPU:          true,
		LossType:        "cross_entropy",
		PrintEveryBatch: 0,
		Verbose:         true,
	}
	if _, err := gpuNet.Train(batches, gpuConfig); err != nil {
		panic(fmt.Sprintf("GPU training failed: %v", err))
	}
	gpuNet.WeightsToCPU()
	gpuNet.GPU = false
	gpuNet.SaveModel("conv1d_gpu_model.json", "conv1d_gpu")

	fmt.Println("\n=== Verifying Save/Load ===")
	loadedCPU, _ := nn.LoadModel("conv1d_cpu_model.json", "conv1d_cpu")
	loadedGPU, _ := nn.LoadModel("conv1d_gpu_model.json", "conv1d_gpu")
	verifyReload("CPU JSON", cpuNet, loadedCPU, testData)
	verifyReload("GPU JSON", gpuNet, loadedGPU, testData)

	fmt.Println("\n=== Safetensors ===")
	safetensorsPath := "conv1d_gpu.safetensors"
	gpuNet.SaveWeightsToSafetensors(safetensorsPath)
	safetensorsNet, _ := nn.BuildNetworkFromJSON(config)
	safetensorsNet.LoadWeightsFromSafetensors(safetensorsPath)
	verifyReload("GPU Safetensors", gpuNet, safetensorsNet, testData)

	fmt.Println("\n=== Testing All Numerical Types ===")
	baseBytes, _ := os.ReadFile(safetensorsPath)
	baseTensors, _ := nn.LoadSafetensorsWithShapes(baseBytes)

	dtypes := []struct {
		Name  string
		Scale float32
	}{
		{"F32", 1.0}, {"F64", 1.0}, {"F16", 1.0}, {"BF16", 1.0}, {"F4", 8.0},
		{"I64", 1000.0}, {"I32", 1000.0}, {"I16", 100.0}, {"I8", 10.0},
		{"U64", 1000.0}, {"U32", 1000.0}, {"U16", 100.0}, {"U8", 10.0},
	}

	results := []VerificationResult{}

	for _, dt := range dtypes {
		fmt.Printf("\n>>> Testing DType: %s\n", dt.Name)
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
			testTensors[k] = nn.TensorWithShape{Values: newVals, Shape: v.Shape, DType: dt.Name}
		}

		fname := fmt.Sprintf("conv1d_%s.safetensors", dt.Name)
		if err := nn.SaveSafetensors(fname, testTensors); err != nil {
			fmt.Printf("  ❌ Save Failed\n")
			continue
		}
		defer os.Remove(fname)

		fi, _ := os.Stat(fname)
		runtime.GC()
		var m1, m2 runtime.MemStats
		runtime.ReadMemStats(&m1)

		testNet, _ := nn.BuildNetworkFromJSON(config)
		testNet.LoadWeightsFromSafetensors(fname)
		runtime.ReadMemStats(&m2)

		unscaleWeights(testNet, dt.Scale, isUint)

		expected := make([]float64, len(testData))
		actual := make([]float64, len(testData))
		for i, sample := range testData {
			outOrig, _ := gpuNet.Forward(sample.Input)
			outNew, _ := testNet.Forward(sample.Input)
			expected[i] = float64(outOrig[sample.Label])
			actual[i] = float64(outNew[sample.Label])
		}

		metrics, _ := nn.EvaluateModel(expected, actual)
		metrics.PrintSummary()

		results = append(results, VerificationResult{
			DType: dt.Name, Quality: metrics.Score, AvgDev: metrics.AverageDeviation,
			FileSize: fi.Size(), RAM: m2.HeapAlloc - m1.HeapAlloc,
		})
	}

	fmt.Println("\n\n=== NUMERICAL TYPE COMPARISON SUMMARY ===")
	w := tabwriter.NewWriter(os.Stdout, 0, 0, 2, ' ', 0)
	fmt.Fprintln(w, "DType\tQuality Score\tAvg Dev\tFile Size\tRAM Usage")
	fmt.Fprintln(w, "-----\t-------------\t-------\t---------\t---------")
	for _, res := range results {
		fmt.Fprintf(w, "%s\t%.2f%%\t%.4f%%\t%s\t%s\n",
			res.DType, res.Quality, res.AvgDev,
			formatBytes(res.FileSize), formatBytes(int64(res.RAM)))
	}
	w.Flush()

	correct := 0
	for _, sample := range testData {
		output, _ := gpuNet.Forward(sample.Input)
		predicted := 0
		if output[1] > output[0] {
			predicted = 1
		}
		if predicted == sample.Label {
			correct++
		}
	}
	fmt.Printf("\n=== Test Accuracy: %d/%d (%.1f%%) ===\n", correct, len(testData), 100.0*float64(correct)/float64(len(testData)))

	gpuNet.ReleaseGPUWeights()
	os.Remove("conv1d_cpu_model.json")
	os.Remove("conv1d_gpu_model.json")
	os.Remove(safetensorsPath)
}
