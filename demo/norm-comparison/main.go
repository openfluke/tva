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
	rand.Seed(45)

	for i := 0; i < numSamples; i++ {
		input := make([]float32, inputSize)
		label := i % numClasses

		for j := 0; j < inputSize; j++ {
			baseVal := rand.Float32()*0.5 + 0.25
			if j%numClasses == label {
				baseVal += 0.5
			}
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

func testNormType(normType string) {
	fmt.Printf("\n\n=== Testing %s ===\n", normType)

	var config string
	if normType == "LayerNorm" {
		config = `{"id": "layernorm_test", "batch_size": 1, "grid_rows": 1, "grid_cols": 1, "layers_per_cell": 5,
		"layers": [
			{"type": "dense", "activation": "relu", "input_height": 128, "output_height": 128},
			{"type": "layer_norm", "norm_size": 128, "epsilon": 1e-5},
			{"type": "dense", "activation": "relu", "input_height": 128, "output_height": 64},
			{"type": "layer_norm", "norm_size": 64, "epsilon": 1e-5},
			{"type": "dense", "activation": "softmax", "input_height": 64, "output_height": 10}
		]}`
	} else {
		config = `{"id": "rmsnorm_test", "batch_size": 1, "grid_rows": 1, "grid_cols": 1, "layers_per_cell": 5,
		"layers": [
			{"type": "dense", "activation": "relu", "input_height": 128, "output_height": 128},
			{"type": "rms_norm", "norm_size": 128, "epsilon": 1e-5},
			{"type": "dense", "activation": "relu", "input_height": 128, "output_height": 64},
			{"type": "rms_norm", "norm_size": 64, "epsilon": 1e-5},
			{"type": "dense", "activation": "softmax", "input_height": 64, "output_height": 10}
		]}`
	}

	trainData := generateData(500, 128, 10)
	testData := generateData(100, 128, 10)

	batches := make([]nn.TrainingBatch, len(trainData))
	for i, sample := range trainData {
		batches[i] = nn.TrainingBatch{Input: sample.Input, Target: sample.Target}
	}

	gpuNet, _ := nn.BuildNetworkFromJSON(config)
	gpuNet.InitializeWeights()
	gpuNet.Train(batches, &nn.TrainingConfig{
		Epochs: 30, LearningRate: 0.001, UseGPU: true, LossType: "cross_entropy", Verbose: true,
	})
	gpuNet.WeightsToCPU()
	gpuNet.GPU = false

	safetensorsPath := fmt.Sprintf("%s_model.safetensors", normType)
	gpuNet.SaveWeightsToSafetensors(safetensorsPath)

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
	fmt.Printf("Test Accuracy: %d/%d (%.1f%%)\n", correct, len(testData), 100.0*float64(correct)/float64(len(testData)))

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

		dtFilename := fmt.Sprintf("%s_%s.safetensors", normType, dt.Name)
		nn.SaveSafetensors(dtFilename, testTensors)
		defer os.Remove(dtFilename)
		fileInfo, _ := os.Stat(dtFilename)

		runtime.GC()
		var m1, m2 runtime.MemStats
		runtime.ReadMemStats(&m1)

		reloadNet, _ := nn.BuildNetworkFromJSON(config)
		reloadNet.LoadWeightsFromSafetensors(dtFilename)
		runtime.ReadMemStats(&m2)
		unscaleWeights(reloadNet, dt.Scale, isUint)

		expected := make([]float64, len(testData))
		actual := make([]float64, len(testData))
		for i, sample := range testData {
			outOrig, _ := gpuNet.Forward(sample.Input)
			outNew, _ := reloadNet.Forward(sample.Input)
			expected[i] = float64(outOrig[sample.Label])
			actual[i] = float64(outNew[sample.Label])
		}

		metrics, _ := nn.EvaluateModel(expected, actual)
		results = append(results, VerificationResult{
			DType: dt.Name, Quality: metrics.Score, AvgDev: metrics.AverageDeviation,
			FileSize: fileInfo.Size(), RAM: m2.HeapAlloc - m1.HeapAlloc,
		})
	}

	fmt.Printf("\n\n=== %s NUMERICAL TYPE COMPARISON ===\n", normType)
	w := tabwriter.NewWriter(os.Stdout, 0, 0, 2, ' ', 0)
	fmt.Fprintln(w, "DType\tQuality Score\tAvg Dev\tFile Size\tRAM Usage")
	fmt.Fprintln(w, "-----\t-------------\t-------\t---------\t---------")
	for _, res := range results {
		fmt.Fprintf(w, "%s\t%.2f%%\t%.4f%%\t%s\t%s\n",
			res.DType, res.Quality, res.AvgDev, formatBytes(res.FileSize), formatBytes(int64(res.RAM)))
	}
	w.Flush()

	gpuNet.ReleaseGPUWeights()
	os.Remove(safetensorsPath)
}

func main() {
	fmt.Println("=== Normalization Demo: LayerNorm vs RMSNorm ===\n")
	testNormType("LayerNorm")
	testNormType("RMSNorm")
	fmt.Println("\n\n=== Comparison Complete ===")
}
