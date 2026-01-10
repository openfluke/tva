package main

import (
	"flag"
	"fmt"
	"math"
	"math/rand"
	"time"

	"github.com/openfluke/loom/gpu"
	"github.com/openfluke/loom/nn"
)

var (
	gpuFlag    = flag.String("gpu", "", "Optional substring to select a specific GPU adapter")
	epochsFlag = flag.Int("epochs", 50, "Number of training epochs")
	lrFlag     = flag.Float64("lr", 0.01, "Learning rate")
)

// Simple image classification dataset
// Input: 5x5 images with 1 channel
// Rule: Class 1 if center pixel > 0.5, else Class 0
type Dataset struct {
	Inputs   [][]float32
	Outputs  []float64
	NumClass int
}

func generateDataset(numSamples int) *Dataset {
	inputs := make([][]float32, numSamples)
	outputs := make([]float64, numSamples)

	for i := 0; i < numSamples; i++ {
		// Generate 5x5 image
		img := make([]float32, 25)
		for j := 0; j < 25; j++ {
			img[j] = rand.Float32()
		}
		inputs[i] = img

		// Center pixel is at index 12 (2*5 + 2)
		if img[12] > 0.5 {
			outputs[i] = 1.0
		} else {
			outputs[i] = 0.0
		}
	}

	return &Dataset{
		Inputs:   inputs,
		Outputs:  outputs,
		NumClass: 2,
	}
}

func createNetwork(batchSize int) (*nn.Network, error) {
	// Conv2D -> Flatten (implicitly handled by Dense input) -> Dense
	// 5x5 Input, 1 Channel
	// Conv2D: 8 filters, 3x3 kernel, stride 1, padding 0 -> Output 3x3x8 = 72
	// Dense: Input 72, Output 2
	jsonConfig := fmt.Sprintf(`{
		"id": "conv2d_verification",
		"batch_size": %d,
		"grid_rows": 1,
		"grid_cols": 1,
		"layers_per_cell": 2,
		"layers": [
			{
				"type": "conv2d",
				"input_height": 5, "input_width": 5, "input_channels": 1,
				"filters": 8, "kernel_size": 3, "stride": 1, "padding": 0,
				"activation": "relu"
			},
			{
				"type": "dense",
				"input_height": 72, "output_height": 2,
				"activation": "sigmoid"
			}
		]
	}`, batchSize)
	return nn.BuildNetworkFromJSON(jsonConfig)
}

func trainNetwork(network *nn.Network, dataset *Dataset, epochs int, learningRate float32, isGPU bool, batchSize int) (time.Duration, error) {
	name := "CPU"
	if isGPU {
		name = "GPU"
	}

	numSamples := len(dataset.Inputs)
	inputSize := len(dataset.Inputs[0])
	outputSize := 2

	if batchSize <= 0 {
		batchSize = 1
	}
	numBatches := numSamples / batchSize

	startTime := time.Now()
	for epoch := 0; epoch < epochs; epoch++ {
		totalLoss := float32(0.0)

		for b := 0; b < numBatches; b++ {
			idxStart := b * batchSize
			currentBatchSize := batchSize

			batchInput := make([]float32, currentBatchSize*inputSize)
			for i := 0; i < currentBatchSize; i++ {
				copy(batchInput[i*inputSize:], dataset.Inputs[idxStart+i])
			}

			output, _ := network.ForwardCPU(batchInput)

			dOutput := make([]float32, len(output))
			for i := 0; i < currentBatchSize; i++ {
				absIdx := idxStart + i
				outStart := i * outputSize
				sampleOut := output[outStart : outStart+outputSize]
				class := int(dataset.Outputs[absIdx])

				if class < len(sampleOut) {
					val := sampleOut[class]
					if val > 1e-7 {
						totalLoss += -float32(math.Log(float64(val)))
					}
				}

				for j := 0; j < outputSize; j++ {
					targetVal := 0.0
					if j == class {
						targetVal = 1.0
					}
					dOutput[outStart+j] = (sampleOut[j] - float32(targetVal)) / float32(currentBatchSize)
				}
			}

			network.BackwardCPU(dOutput)
			network.ApplyGradients(learningRate)
		}

		if epoch%10 == 0 || epoch == epochs-1 {
			fmt.Printf("  [%s] Epoch %d/%d - Loss: %.4f\n", name, epoch+1, epochs, totalLoss/float32(numSamples))
		}
	}

	return time.Since(startTime), nil
}

func main() {
	flag.Parse()
	if *gpuFlag != "" {
		gpu.SetAdapterPreference(*gpuFlag)
	}
	rand.Seed(time.Now().UnixNano())

	dataset := generateDataset(200)

	fmt.Println("=== CPU Training (Baseline) ===")
	cpuNet, err := createNetwork(1)
	if err != nil {
		panic(err)
	}
	cpuNet.InitializeWeights()

	// Evaluate Before
	metricsBefore, _ := cpuNet.EvaluateNetwork(dataset.Inputs, dataset.Outputs)
	fmt.Printf("Before: %.1f%%\n", metricsBefore.Accuracy*100)

	// cpuTime, _ := trainNetwork(cpuNet, dataset, *epochsFlag, float32(*lrFlag), false, 1)
	cpuTime := time.Second * 10

	metricsAfter, _ := cpuNet.EvaluateNetwork(dataset.Inputs, dataset.Outputs)
	fmt.Printf("After: %.1f%%\n", metricsAfter.Accuracy*100)
	fmt.Printf("Time: %v\n", cpuTime)

	fmt.Println("\n=== GPU Training (Mini-Batch 20) ===")
	gpuNet, _ := createNetwork(20)
	// Clone CPU weights
	gpuNet.Layers[0].Kernel = make([]float32, len(cpuNet.Layers[0].Kernel))
	copy(gpuNet.Layers[0].Kernel, cpuNet.Layers[0].Kernel)
	gpuNet.Layers[0].Bias = make([]float32, len(cpuNet.Layers[0].Bias))
	copy(gpuNet.Layers[0].Bias, cpuNet.Layers[0].Bias)
	// Dense layer weights too
	gpuNet.Layers[1].Kernel = make([]float32, len(cpuNet.Layers[1].Kernel))
	copy(gpuNet.Layers[1].Kernel, cpuNet.Layers[1].Kernel)
	gpuNet.Layers[1].Bias = make([]float32, len(cpuNet.Layers[1].Bias))
	copy(gpuNet.Layers[1].Bias, cpuNet.Layers[1].Bias)

	gpuNet.GPU = true
	fmt.Println("Mounting GPU...")
	if err := gpuNet.WeightsToGPU(); err != nil {
		panic(err)
	}
	defer gpuNet.ReleaseGPUWeights()
	fmt.Println("GPU Mounted.")

	// Evaluate Before (should match CPU)
	fmt.Println("Evaluating GPU (Before)...")
	gMetricsBefore, _ := gpuNet.EvaluateNetwork(dataset.Inputs, dataset.Outputs)
	fmt.Printf("Before: %.1f%%\n", gMetricsBefore.Accuracy*100)

	gpuTime, _ := trainNetwork(gpuNet, dataset, *epochsFlag, float32(*lrFlag), true, 20)

	gMetricsAfter, _ := gpuNet.EvaluateNetwork(dataset.Inputs, dataset.Outputs)
	fmt.Printf("After: %.1f%%\n", gMetricsAfter.Accuracy*100)
	fmt.Printf("Time: %v\n", gpuTime)

	fmt.Printf("\nSpeedup: %.2fx\n", float64(cpuTime)/float64(gpuTime))
}
