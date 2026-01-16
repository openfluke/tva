package main

import (
	"compress/gzip"
	"encoding/binary"
	"fmt"
	"io"
	"math/rand"
	"net/http"
	"os"
	"path/filepath"

	"runtime"
	"text/tabwriter"

	"github.com/openfluke/loom/gpu"
	"github.com/openfluke/loom/nn"
)

const (
	MnistTrainImagesFile = "train-images-idx3-ubyte"
	MnistTrainLabelsFile = "train-labels-idx1-ubyte"
	MnistTestImagesFile  = "t10k-images-idx3-ubyte"
	MnistTestLabelsFile  = "t10k-labels-idx1-ubyte"
	DataDir              = "data"
)

type MNISTSample struct {
	Image []float32
	Label int
}

type VerificationResult struct {
	DType    string
	Quality  float64
	AvgDev   float64
	FileSize int64
	RAM      uint64
}

func main() {
	// NOTE: This demo uses the generic Train() and Forward() methods.
	// Loom automatically routes execution to CPU or GPU based on the
	// TrainingConfig.UseGPU flag and whether weights are mounted to GPU.
	// No direct calls to ForwardCPU/ForwardGPU are needed in user code.

	// 1. Setup Data
	fmt.Println("Setting up MNIST data...")
	if err := ensureData(); err != nil {
		panic(err)
	}

	trainData, err := loadMNIST(filepath.Join(DataDir, MnistTrainImagesFile), filepath.Join(DataDir, MnistTrainLabelsFile), 1000)
	if err != nil {
		panic(err)
	}
	testData, err := loadMNIST(filepath.Join(DataDir, MnistTestImagesFile), filepath.Join(DataDir, MnistTestLabelsFile), 100)
	if err != nil {
		panic(err)
	}

	// 2. Define Network Configuration
	configJSON := `{
		"id": "mnist_cnn",
		"batch_size": 20,
		"grid_rows": 1,
		"grid_cols": 1,
		"layers_per_cell": 6,
		"layers": [
			{
				"type": "dense", "activation": "none",
				"input_height": 784, "output_height": 784
			},
			{
				"type": "conv2d", "activation": "relu",
				"input_height": 28, "input_width": 28, "input_channels": 1,
				"filters": 8, "kernel_size": 3, "stride": 1, "padding": 0,
				"output_height": 26, "output_width": 26
			},
			{
				"type": "conv2d", "activation": "relu",
				"input_height": 26, "input_width": 26, "input_channels": 8,
				"filters": 16, "kernel_size": 3, "stride": 2, "padding": 0,
				"output_height": 12, "output_width": 12
			},
			{
				"type": "dense", "activation": "none",
				"input_height": 2304, "output_height": 64
			},
			{
				"type": "dense", "activation": "none",
				"input_height": 64, "output_height": 10
			},
			{
				"type": "softmax", "activation": "none",
				"input_height": 10
			}
		]
	}`

	batchSize := 20
	trainBatches := createBatches(trainData, batchSize)

	valInputs := make([][]float32, len(testData))
	valTargets := make([]float64, len(testData))
	for i, sample := range testData {
		valInputs[i] = sample.Image
		valTargets[i] = float64(sample.Label)
	}

	// 3. CPU Training
	fmt.Println("\n=== Starting CPU Training ===")
	cpuNet, err := nn.BuildNetworkFromJSON(configJSON)
	if err != nil {
		panic(err)
	}
	cpuNet.InitializeWeights()

	initialWeights, _ := cpuNet.SaveModelToString("mnist_init")

	cpuConfig := &nn.TrainingConfig{
		Epochs:          2, // Reduced for speed in validation test
		LearningRate:    0.01,
		UseGPU:          false,
		LossType:        "cross_entropy",
		PrintEveryBatch: 0,
		Verbose:         true,
	}

	cpuNet.Train(trainBatches, cpuConfig)

	// Save CPU model (JSON)
	fmt.Println("Saving CPU model (JSON)...")
	if err := cpuNet.SaveModel("mnist_cpu_model.json", "mnist_cpu"); err != nil {
		panic(err)
	}

	// 4. GPU Training
	fmt.Println("\n=== Starting GPU Training ===")
	gpuNet, err := nn.LoadModelFromString(initialWeights, "mnist_init")
	if err != nil {
		panic(err)
	}

	gpu.SetAdapterPreference("nvidia")
	gpuNet.BatchSize = batchSize

	gpuConfig := &nn.TrainingConfig{
		Epochs:          2,
		LearningRate:    0.01,
		UseGPU:          true,
		LossType:        "cross_entropy",
		PrintEveryBatch: 0,
		Verbose:         true,
	}

	// Train GPU
	if _, err := gpuNet.Train(trainBatches, gpuConfig); err != nil {
		panic(err)
	}

	// Sync Weights back to CPU for saving
	fmt.Println("Syncing GPU weights to CPU...")
	if err := gpuNet.WeightsToCPU(); err != nil {
		panic(fmt.Sprintf("Failed to sync weights: %v", err))
	}
	// Switch gpuNet to CPU mode for verification (so we compare CPU execution of original vs reloaded)
	gpuNet.GPU = false

	// Save GPU model (JSON) and Safetensors
	fmt.Println("Saving GPU model (JSON)...")
	if err := gpuNet.SaveModel("mnist_gpu_model.json", "mnist_gpu"); err != nil {
		panic(err)
	}

	// 5. Verify Save/Load Consistency
	fmt.Println("\n=== Verifying Save/Load Consistency ===")

	// Load CPU model from JSON
	loadedCPU, err := nn.LoadModel("mnist_cpu_model.json", "mnist_cpu")
	if err != nil {
		panic(err)
	}
	// Load GPU model from JSON
	loadedGPU, err := nn.LoadModel("mnist_gpu_model.json", "mnist_gpu")
	if err != nil {
		panic(err)
	}

	// Compare outputs on 10 samples
	fmt.Println("Comparing outputs of re-loaded models on 10 samples...")
	for i := 0; i < 10; i++ {
		input := testData[i].Image

		outCPU, _ := loadedCPU.Forward(input)
		outGPU, _ := loadedGPU.Forward(input) // loadedGPU running on CPU now (default state after load)

		// Basic check if outputs are close
		diff := float32(0)
		for j := 0; j < 10; j++ {
			d := outCPU[j] - outGPU[j]
			if d < 0 {
				d = -d
			}
			diff += d
		}

		fmt.Printf("Sample %d Diff: %.6f\n", i, diff)
		// They won't be identical because training diverges slightly due to floating point ordering/precision on GPU
		// But they should be deterministic per-backend if re-loaded.

		// Wait, user asked to compare: "save and relaod them iwht both the normal and the safetensor and using 10 samples we get exactly the same outputs upon saving and reloading"
		// Only check: Reloaded(Model) == Original(Model)
	}

	// Verification 1: Original CPU vs Reloaded CPU
	verifyReload("CPU JSON", cpuNet, loadedCPU, testData[:10])

	// Verification 2: Original GPU (synced) vs Reloaded GPU
	verifyReload("GPU JSON", gpuNet, loadedGPU, testData[:10])

	// Verification 3: Safetensors
	fmt.Println("\n=== Verifying Safetensors ===")

	// Save GPU model to Safetensors
	safetensorsPath := "mnist_gpu.safetensors"
	if err := gpuNet.SaveWeightsToSafetensors(safetensorsPath); err != nil {
		panic(fmt.Sprintf("Failed to save safetensors: %v", err))
	}
	fmt.Println("Saved " + safetensorsPath)

	// Load into a fresh network
	safetensorsNet, err := nn.BuildNetworkFromJSON(configJSON)
	if err != nil {
		panic(err)
	}
	if err := safetensorsNet.LoadWeightsFromSafetensors(safetensorsPath); err != nil {
		panic(fmt.Sprintf("Failed to load safetensors: %v", err))
	}

	verifyReload("GPU Safetensors", gpuNet, safetensorsNet, testData[:10])

	// Verification 3b: All Numerical Types
	fmt.Println("\n=== Verifying All Numerical Types ===")

	// Load base tensors to manipulate
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
		{"I64", 1000000.0},
		{"I32", 1000000.0},
		{"I16", 1000.0},
		{"I8", 100.0},
		{"U64", 1000000.0},
		{"U32", 1000000.0},
		{"U16", 1000.0},
		{"U8", 100.0},
	}

	results := []VerificationResult{}

	for _, dt := range dtypes {
		fmt.Printf("\n>>> Testing DType: %s (Scale: %.1f)\n", dt.Name, dt.Scale)

		// Prepare quantized/converted tensors
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
		fname := fmt.Sprintf("mnist_%s.safetensors", dt.Name)
		if err := nn.SaveSafetensors(fname, testTensors); err != nil {
			fmt.Printf("  ❌ Save Failed: %v\n", err)
			continue
		}
		defer os.Remove(fname)

		fi, _ := os.Stat(fname)
		fileSize := fi.Size()

		// Load with Memory Measurement
		testNet, err := nn.BuildNetworkFromJSON(configJSON)
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

		// Unscale weights if needed
		unscaleWeights(testNet, dt.Scale, isUint)

		// Evaluate using Deviation Metrics
		expected := make([]float64, 100)
		actual := make([]float64, 100)
		inputs := make([][]float32, 100)

		for i := 0; i < 100; i++ {
			inputs[i] = testData[i].Image
			outOrig, _ := gpuNet.Forward(inputs[i])
			outNew, _ := testNet.Forward(inputs[i])

			maxIdx := 0
			for j := range outOrig {
				if outOrig[j] > outOrig[maxIdx] {
					maxIdx = j
				}
			}
			expected[i] = float64(outOrig[maxIdx])
			actual[i] = float64(outNew[maxIdx])
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

	// Final Summary Table
	fmt.Println("\n=== NUMERICAL TYPE COMPARISON SUMMARY ===")
	w := tabwriter.NewWriter(os.Stdout, 0, 0, 2, ' ', tabwriter.Debug)
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

	// Clean up
	gpuNet.ReleaseGPUWeights()
	os.Remove("mnist_cpu_model.json")
	os.Remove("mnist_gpu_model.json")
	os.Remove(safetensorsPath)
}

func unscaleWeights(n *nn.Network, scale float32, isUint bool) {
	invScale := 1.0 / scale
	for i := range n.Layers {
		l := &n.Layers[i]
		// Helper to scale slice
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

func verifyReload(name string, original, reloaded *nn.Network, samples []MNISTSample) {
	fmt.Printf("\nVerifying %s consistency:\n", name)
	maxDiff := float32(0)
	for i, s := range samples {
		o1, _ := original.Forward(s.Image)
		o2, _ := reloaded.Forward(s.Image)

		for j := 0; j < len(o1); j++ {
			diff := o1[j] - o2[j]
			if diff < 0 {
				diff = -diff
			}
			if diff > maxDiff {
				maxDiff = diff
			}
		}
		if i == 0 {
			fmt.Printf("  Sample 0: Orig %v vs Load %v\n", o1[:3], o2[:3])
		}
	}
	fmt.Printf("  Max Difference over %d samples: %.9f\n", len(samples), maxDiff)
	if maxDiff < 1e-6 {
		fmt.Println("  ✓ Consistency Confirmed")
	} else {
		fmt.Println("  ✗ Consistency FAILED")
	}
}

func createBatches(data []MNISTSample, batchSize int) []nn.TrainingBatch {
	// Simple batch creation without shuffle for deterministic debugging if needed
	// But training needs shuffle. We'll use random shuffle.
	rand.Seed(123) // Fixed seed
	shuffled := make([]MNISTSample, len(data))
	copy(shuffled, data)
	rand.Shuffle(len(shuffled), func(i, j int) {
		shuffled[i], shuffled[j] = shuffled[j], shuffled[i]
	})

	numBatches := len(shuffled) / batchSize
	batches := make([]nn.TrainingBatch, numBatches)

	for b := 0; b < numBatches; b++ {
		input := make([]float32, batchSize*784)
		target := make([]float32, batchSize*10)

		for i := 0; i < batchSize; i++ {
			idx := b*batchSize + i
			sample := shuffled[idx]
			copy(input[i*784:], sample.Image)
			target[i*10+sample.Label] = 1.0
		}
		batches[b] = nn.TrainingBatch{Input: input, Target: target}
	}
	return batches
}

func ensureData() error {
	if _, err := os.Stat(DataDir); os.IsNotExist(err) {
		os.Mkdir(DataDir, 0755)
	}
	files := map[string]string{
		MnistTrainImagesFile: "https://ossci-datasets.s3.amazonaws.com/mnist/train-images-idx3-ubyte.gz",
		MnistTrainLabelsFile: "https://ossci-datasets.s3.amazonaws.com/mnist/train-labels-idx1-ubyte.gz",
		MnistTestImagesFile:  "https://ossci-datasets.s3.amazonaws.com/mnist/t10k-images-idx3-ubyte.gz",
		MnistTestLabelsFile:  "https://ossci-datasets.s3.amazonaws.com/mnist/t10k-labels-idx1-ubyte.gz",
	}
	for filename, url := range files {
		path := filepath.Join(DataDir, filename)
		if _, err := os.Stat(path); os.IsNotExist(err) {
			if err := downloadAndExtract(url, path); err != nil {
				return err
			}
		}
	}
	return nil
}

func downloadAndExtract(url, destPath string) error {
	resp, err := http.Get(url)
	if err != nil {
		return err
	}
	defer resp.Body.Close()
	gzReader, err := gzip.NewReader(resp.Body)
	if err != nil {
		return err
	}
	defer gzReader.Close()
	outFile, err := os.Create(destPath)
	if err != nil {
		return err
	}
	defer outFile.Close()
	_, err = io.Copy(outFile, gzReader)
	return err
}

func loadMNIST(imageFile, labelFile string, maxCount int) ([]MNISTSample, error) {
	imgF, err := os.Open(imageFile)
	if err != nil {
		return nil, err
	}
	defer imgF.Close()
	var magic, numImgs, rows, cols int32
	binary.Read(imgF, binary.BigEndian, &magic)
	binary.Read(imgF, binary.BigEndian, &numImgs)
	binary.Read(imgF, binary.BigEndian, &rows)
	binary.Read(imgF, binary.BigEndian, &cols)
	lblF, err := os.Open(labelFile)
	if err != nil {
		return nil, err
	}
	defer lblF.Close()
	var lMagic, lNumItems int32
	binary.Read(lblF, binary.BigEndian, &lMagic)
	binary.Read(lblF, binary.BigEndian, &lNumItems)
	count := int(numImgs)
	if maxCount > 0 && maxCount < count {
		count = maxCount
	}
	samples := make([]MNISTSample, count)
	imgSize := int(rows * cols)
	buf := make([]byte, imgSize)
	lBuf := make([]byte, 1)
	for i := 0; i < count; i++ {
		imgF.Read(buf)
		lblF.Read(lBuf)
		imgFloats := make([]float32, imgSize)
		for j := 0; j < imgSize; j++ {
			imgFloats[j] = float32(buf[j]) / 255.0
		}
		samples[i] = MNISTSample{Image: imgFloats, Label: int(lBuf[0])}
	}
	return samples, nil
}
