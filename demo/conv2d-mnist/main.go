package main

import (
	"compress/gzip"
	"encoding/binary"
	"fmt"
	"io"
	"net/http"
	"os"
	"path/filepath"

	"github.com/openfluke/loom/gpu"
	"github.com/openfluke/loom/nn"
)

const (
	MnistTrainImagesFile = "train-images-idx3-ubyte"
	MnistTrainLabelsFile = "train-labels-idx1-ubyte"
	MnistTestImagesFile  = "t10k-images-idx3-ubyte"
	MnistTestLabelsFile  = "t10k-labels-idx1-ubyte"
	DataDir              = "data"
	Epochs               = 20
)

type MNISTSample struct {
	Image []float32
	Label int
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
		"layers_per_cell": 5,
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
				"type": "dense", "activation": "sigmoid",
				"input_height": 64, "output_height": 10
			}
		]
	}`

	batchSize := 20

	// Prepare training data for new generic TrainLabels
	trainInputs := make([][]float32, len(trainData))
	trainLabels := make([]int, len(trainData))
	for i, s := range trainData {
		trainInputs[i] = s.Image
		trainLabels[i] = s.Label
	}

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

	// Evaluate before CPU training
	fmt.Println("\n--- Pre-Training CPU Evaluation ---")
	metricsBefore, err := cpuNet.EvaluateNetwork(valInputs, valTargets)
	metricsBefore.PrintSummary()

	//initialWeights, _ := cpuNet.SaveModelToString("mnist_init")

	cpuConfig := &nn.TrainingConfig{
		Epochs:          Epochs, // Reduced for speed in validation test
		LearningRate:    0.05,
		UseGPU:          false,
		LossType:        "mse",
		PrintEveryBatch: 0,
		Verbose:         true,
	}

	cpuNet.TrainLabels(trainInputs, trainLabels, cpuConfig)

	// Evaluate after CPU training
	fmt.Println("\n--- Post-Training CPU Evaluation ---")
	metricsAfter, err := cpuNet.EvaluateNetwork(valInputs, valTargets)
	metricsAfter.PrintSummary()

	// Show Deviation Comparison
	nn.PrintDeviationComparisonTable("CPU Results: MNIST", metricsBefore, metricsAfter)

	// Save CPU model (JSON)
	fmt.Println("Saving CPU model (JSON)...")
	if err := cpuNet.SaveModel("mnist_cpu_model.json", "mnist_cpu"); err != nil {
		panic(err)
	}

	// 4. GPU Training
	fmt.Println("\n=== Starting GPU Training ===")
	// Initialize GPU network freshly (random weights) to prove independent learning
	gpuNet, err := nn.BuildNetworkFromJSON(configJSON)
	if err != nil {
		panic(err)
	}
	gpuNet.InitializeWeights()

	gpu.SetAdapterPreference("nvidia")
	gpuNet.BatchSize = batchSize

	gpuConfig := &nn.TrainingConfig{
		Epochs:          Epochs,
		LearningRate:    0.05,
		UseGPU:          true,
		LossType:        "mse",
		PrintEveryBatch: 0,
		Verbose:         true,
	}

	// Evaluate before GPU training
	// We must enable GPU mode and mount weights explicitly for evaluation to use GPU
	gpuNet.SetGPU(true)
	gpuNet.WeightsToGPU()

	fmt.Println("\n--- Pre-Training GPU Evaluation ---")
	metricsGPUBefore, err := gpuNet.EvaluateNetwork(valInputs, valTargets)
	metricsGPUBefore.PrintSummary()

	// Train GPU
	if _, err := gpuNet.TrainLabels(trainInputs, trainLabels, gpuConfig); err != nil {
		panic(err)
	}

	// Evaluate after GPU training
	fmt.Println("\n--- Post-Training GPU Evaluation ---")
	metricsGPUAfter, err := gpuNet.EvaluateNetwork(valInputs, valTargets)
	metricsGPUAfter.PrintSummary()

	// Show Deviation Comparison
	nn.PrintDeviationComparisonTable("GPU Results: MNIST", metricsGPUBefore, metricsGPUAfter)

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

	// Prepare test inputs
	verifyInputs := make([][]float32, 10)
	for i := 0; i < 10; i++ {
		verifyInputs[i] = testData[i].Image
	}

	// Verification 1: CPU JSON
	resultCPUJSON, err := nn.VerifySaveLoadConsistency(cpuNet, "json", verifyInputs, 1e-6)
	if err != nil {
		panic(err)
	}
	resultCPUJSON.PrintConsistencyResult()

	// Verification 2: GPU JSON
	resultGPUJSON, err := nn.VerifySaveLoadConsistency(gpuNet, "json", verifyInputs, 1e-6)
	if err != nil {
		panic(err)
	}
	resultGPUJSON.PrintConsistencyResult()

	// Verification 3: Safetensors (in-memory)
	resultSafetensors, err := nn.VerifySaveLoadConsistency(gpuNet, "safetensors", verifyInputs, 1e-6)
	if err != nil {
		panic(err)
	}
	resultSafetensors.PrintConsistencyResult()

	// Verification 4: All Numerical Types
	fmt.Println("\n=== Benchmarking All Numerical Types ===")

	// Define dtypes and their scales (grouped by type: floats, signed ints, unsigned ints)
	dtypes := []string{
		// Float types (largest to smallest precision)
		"F64", "F32", "F16", "BF16", "F4",
		// Signed integer types (largest to smallest)
		"I64", "I32", "I16", "I8",
		// Unsigned integer types (largest to smallest)
		"U64", "U32", "U16", "U8",
	}
	scales := []float32{
		// Float scales
		1.0, 1.0, 1.0, 1.0, 8.0,
		// Signed int scales
		1000000.0, 1000000.0, 1000.0, 100.0,
		// Unsigned int scales
		1000000.0, 1000000.0, 1000.0, 100.0,
	}

	// Prepare benchmark inputs (use 100 test samples)
	benchInputs := make([][]float32, 100)
	benchExpected := make([]float64, 100)
	for i := 0; i < 100; i++ {
		benchInputs[i] = testData[i].Image
		benchExpected[i] = float64(testData[i].Label)
	}

	// Run benchmark
	benchmark, err := nn.BenchmarkNumericalTypes(gpuNet, dtypes, scales, benchInputs, benchExpected)
	if err != nil {
		panic(err)
	}

	// Print results
	benchmark.PrintNumericalTypeSummary()

	// Show best options
	fmt.Println("\n=== RECOMMENDATIONS ===")
	if best := benchmark.GetBestByQuality(); best != nil {
		fmt.Printf("Best Quality: %s (%.2f%% score)\n", best.DType, best.QualityScore)
	}
	if smallest := benchmark.GetSmallest(); smallest != nil {
		fmt.Printf("Smallest Size: %s (%s)\n", smallest.DType, formatBytes(smallest.MemoryBytes))
	}
	if tradeoff := benchmark.GetBestTradeoff(); tradeoff != nil {
		fmt.Printf("Best Tradeoff: %s (%.2f%% quality, %s)\n",
			tradeoff.DType, tradeoff.QualityScore, formatBytes(tradeoff.MemoryBytes))
	}

	// Clean up
	gpuNet.ReleaseGPUWeights()
	os.Remove("mnist_cpu_model.json")
	os.Remove("mnist_gpu_model.json")
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
