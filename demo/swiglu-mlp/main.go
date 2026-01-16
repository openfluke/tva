package main

import (
	"fmt"
	"math/rand"
	"os"
	"runtime"
	"text/tabwriter"

	"github.com/openfluke/loom/nn"
)

type ClassificationSample struct {
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

func generateData(numSamples, inputSize, numClasses int) []ClassificationSample {
	samples := make([]ClassificationSample, numSamples)
	rand.Seed(46)

	for i := 0; i < numSamples; i++ {
		input := make([]float32, inputSize)
		label := i % numClasses

		for j := 0; j < inputSize; j++ {
			baseVal := rand.Float32() * 0.4
			if (j/16)%numClasses == label {
				baseVal += 0.6
			}
			baseVal += (rand.Float32() - 0.5) * 0.2
			input[j] = baseVal
		}

		target := make([]float32, numClasses)
		target[label] = 1.0
		samples[i] = ClassificationSample{Input: input, Target: target, Label: label}
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

func verifyReload(name string, original, reloaded *nn.Network, samples []ClassificationSample) {
	fmt.Printf("\n%s: ", name)
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
	if maxDiff < 1e-4 {
		fmt.Printf("%.9f ✓\n", maxDiff)
	} else {
		fmt.Printf("%.9f ✗\n", maxDiff)
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
	fmt.Println("=== SwiGLU Demo: Modern Gated MLP ===\n")

	inputSize, numClasses := 256, 10
	trainData := generateData(600, inputSize, numClasses)
	testData := generateData(120, inputSize, numClasses)
	fmt.Printf("Generated %d training, %d test samples\n\n", len(trainData), len(testData))

	config := `{
		"id": "swiglu_test",
		"batch_size": 1,
		"grid_rows": 1,
		"grid_cols": 1,
		"layers_per_cell": 5,
		"layers": [
			{"type": "dense", "activation": "relu", "input_height": 256, "output_height": 512},
			{"type": "swiglu", "input_height": 512, "output_height": 512},
			{"type": "swiglu", "input_height": 512, "output_height": 512},
			{"type": "dense", "activation": "relu", "input_height": 512, "output_height": 128},
			{"type": "dense", "activation": "softmax", "input_height": 128, "output_height": 10}
		]
	}`

	batches := make([]nn.TrainingBatch, len(trainData))
	for i, sample := range trainData {
		batches[i] = nn.TrainingBatch{Input: sample.Input, Target: sample.Target}
	}

	fmt.Println("=== CPU Training ===")
	cpuNet, _ := nn.BuildNetworkFromJSON(config)
	cpuNet.InitializeWeights()
	initialWeights, _ := cpuNet.SaveModelToString("swiglu_init")
	cpuNet.Train(batches, &nn.TrainingConfig{
		Epochs: 40, LearningRate: 0.0005, UseGPU: false, LossType: "cross_entropy", Verbose: true,
	})
	cpuNet.SaveModel("swiglu_cpu_model.json", "swiglu_cpu")

	fmt.Println("\n=== GPU Training ===")
	gpuNet, _ := nn.LoadModelFromString(initialWeights, "swiglu_init")
	gpuNet.Train(batches, &nn.TrainingConfig{
		Epochs: 40, LearningRate: 0.0005, UseGPU: true, LossType: "cross_entropy", Verbose: true,
	})
	gpuNet.WeightsToCPU()
	gpuNet.GPU = false
	gpuNet.SaveModel("swiglu_gpu_model.json", "swiglu_gpu")

	loadedCPU, _ := nn.LoadModel("swiglu_cpu_model.json", "swiglu_cpu")
	loadedGPU, _ := nn.LoadModel("swiglu_gpu_model.json", "swiglu_gpu")
	verifyReload("CPU JSON", cpuNet, loadedCPU, testData)
	verifyReload("GPU JSON", gpuNet, loadedGPU, testData)

	safetensorsPath := "swiglu_gpu.safetensors"
	gpuNet.SaveWeightsToSafetensors(safetensorsPath)
	fmt.Println("Saved " + safetensorsPath)

	fmt.Println("\n=== Multi-Precision Testing ===")
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
		fmt.Printf("\n>>> %s\n", dt.Name)
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

		fname := fmt.Sprintf("swiglu_%s.safetensors", dt.Name)
		nn.SaveSafetensors(fname, testTensors)
		defer os.Remove(fname)
		fileInfo, _ := os.Stat(fname)

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
		fmt.Printf("Quality: %.2f%%\n", metrics.Score)

		results = append(results, VerificationResult{
			DType: dt.Name, Quality: metrics.Score, AvgDev: metrics.AverageDeviation,
			FileSize: fileInfo.Size(), RAM: m2.HeapAlloc - m1.HeapAlloc,
		})
	}

	fmt.Println("\n\n=== NUMERICAL TYPE COMPARISON SUMMARY ===")
	w := tabwriter.NewWriter(os.Stdout, 0, 0, 2, ' ', 0)
	fmt.Fprintln(w, "DType\tQuality Score\tAvg Dev\tFile Size\tRAM Usage")
	fmt.Fprintln(w, "-----\t-------------\t-------\t---------\t---------")
	for _, res := range results {
		fmt.Fprintf(w, "%s\t%.2f%%\t%.4f%%\t%s\t%s\n",
			res.DType, res.Quality, res.AvgDev, formatBytes(res.FileSize), formatBytes(int64(res.RAM)))
	}
	w.Flush()

	correct := 0
	for _, sample := range testData {
		output, _ := gpuNet.Forward(sample.Input)
		predicted := 0
		maxVal := output[0]
		for j := 1; j < len(output); j++ {
			if output[j] > maxVal {
				maxVal = output[j]
				predicted = j
			}
		}
		if predicted == sample.Label {
			correct++
		}
	}
	fmt.Printf("\n=== Test Accuracy: %d/%d (%.1f%%) ===\n", correct, len(testData), 100.0*float64(correct)/float64(len(testData)))

	gpuNet.ReleaseGPUWeights()
	os.Remove("swiglu_cpu_model.json")
	os.Remove("swiglu_gpu_model.json")
	os.Remove(safetensorsPath)
}
