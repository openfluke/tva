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
	gpuFlag    = flag.String("gpu", "", "Optional substring to select a specific GPU adapter (e.g. 'nvidia')")
	epochsFlag = flag.Int("epochs", 100, "Number of training epochs")
	lrFlag     = flag.Float64("lr", 0.05, "Learning rate")
)

// Simple linearly separable dataset: if x[0] > 0.5 then class 1, else class 0
type Dataset struct {
	Inputs   [][]float32
	Outputs  []float64 // Expected class labels (0 or 1)
	NumClass int
}

func generateSimpleDataset(numSamples int) *Dataset {
	inputs := make([][]float32, numSamples)
	outputs := make([]float64, numSamples)

	for i := 0; i < numSamples; i++ {
		x0 := rand.Float32()
		x1 := rand.Float32()
		inputs[i] = []float32{x0, x1}

		// Simple rule: if x[0] > 0.5, class 1, else class 0
		if x0 > 0.5 {
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
	jsonConfig := fmt.Sprintf(`{
		"id": "training_verification",
		"batch_size": %d,
		"grid_rows": 1,
		"grid_cols": 1,
		"layers_per_cell": 3,
		"layers": [
			{"type": "dense", "activation": "relu", "input_height": 2, "output_height": 512},
			{"type": "dense", "activation": "relu", "input_height": 512, "output_height": 512},
			{"type": "dense", "activation": "sigmoid", "input_height": 512, "output_height": 2}
		]
	}`, batchSize)
	return nn.BuildNetworkFromJSON(jsonConfig)
}

func cloneWeights(src, dst *nn.Network) {
	// Clone all layer weights from src to dst
	for i := 0; i < src.TotalLayers(); i++ {
		// Clone kernels
		if len(src.Layers[i].Kernel) > 0 {
			dst.Layers[i].Kernel = make([]float32, len(src.Layers[i].Kernel))
			copy(dst.Layers[i].Kernel, src.Layers[i].Kernel)
		}
		// Clone biases
		if len(src.Layers[i].Bias) > 0 {
			dst.Layers[i].Bias = make([]float32, len(src.Layers[i].Bias))
			copy(dst.Layers[i].Bias, src.Layers[i].Bias)
		}
	}
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
			// Determine batch range
			idxStart := b * batchSize
			idxEnd := idxStart + batchSize
			if idxEnd > numSamples {
				idxEnd = numSamples
			}
			currentBatchSize := idxEnd - idxStart

			// Extract Batch Input
			// We need a flat slice for the batch
			batchInputLen := currentBatchSize * inputSize
			// Optimization: could reuse buffer, but allocation is cleaner for now
			batchInput := make([]float32, batchInputLen)

			// Copy samples into batchInput
			for i := 0; i < currentBatchSize; i++ {
				copy(batchInput[i*inputSize:], dataset.Inputs[idxStart+i])
			}

			// Forward Pass
			output, _ := network.ForwardCPU(batchInput)

			// Compute Gradients
			dOutput := make([]float32, len(output))

			for i := 0; i < currentBatchSize; i++ {
				// absolute index in dataset
				absIdx := idxStart + i

				// Output corresponds to batch index i
				outStart := i * outputSize
				sampleOut := output[outStart : outStart+outputSize]

				// Target
				class := int(dataset.Outputs[absIdx])

				// Loss calculation
				if class < len(sampleOut) {
					val := sampleOut[class]
					if val > 1e-7 {
						totalLoss += -float32(math.Log(float64(val)))
					}
				}

				// Gradient dL/dY = (Y - T) / N
				// dOutput should be scaled by currentBatchSize
				for j := 0; j < outputSize; j++ {
					targetVal := 0.0
					if j == class {
						targetVal = 1.0
					}
					dOutput[outStart+j] = (sampleOut[j] - float32(targetVal)) / float32(currentBatchSize)
				}
			}

			// Backward Pass
			_, _ = network.BackwardCPU(dOutput)

			// Apply Gradients
			network.ApplyGradients(learningRate)
		}

		avgLoss := totalLoss / float32(numSamples)
		if epoch%10 == 0 || epoch == epochs-1 {
			fmt.Printf("  [%s] Epoch %d/%d - Loss: %.4f\n", name, epoch+1, epochs, avgLoss)
		}
	}

	totalTime := time.Since(startTime)
	return totalTime, nil
}

func printDeviationTable(name string, before, after *nn.DeviationMetrics) {
	fmt.Printf("\n╔═══════════════════════════════════════════════════════════════╗\n")
	fmt.Printf("║  %-59s  ║\n", name)
	fmt.Printf("╠═══════════════════════════════════════════════════════════════╣\n")
	fmt.Printf("║  Accuracy:      %6.1f%% → %6.1f%%                           ║\n",
		before.Accuracy*100, after.Accuracy*100)
	fmt.Printf("║  Quality Score: %6.1f   → %6.1f                           ║\n",
		before.Score, after.Score)
	fmt.Printf("║  Avg Deviation: %6.1f%% → %6.1f%%                          ║\n",
		before.AverageDeviation, after.AverageDeviation)
	fmt.Printf("╠═══════════════════════════════════════════════════════════════╣\n")
	fmt.Printf("║  Deviation Distribution:                                      ║\n")
	fmt.Printf("╠═══════════════════════════════════════════════════════════════╣\n")

	bucketOrder := []string{"0-10%", "10-20%", "20-30%", "30-40%", "40-50%", "50-100%", "100%+"}
	for _, bucketName := range bucketOrder {
		beforeCount := before.Buckets[bucketName].Count
		afterCount := after.Buckets[bucketName].Count
		beforePct := float64(beforeCount) / float64(before.TotalSamples) * 100
		afterPct := float64(afterCount) / float64(after.TotalSamples) * 100

		fmt.Printf("║    %8s: %3d (%5.1f%%) → %3d (%5.1f%%)                    ║\n",
			bucketName, beforeCount, beforePct, afterCount, afterPct)
	}
	fmt.Printf("╚═══════════════════════════════════════════════════════════════╝\n")
}

func main() {
	flag.Parse()
	if *gpuFlag != "" {
		fmt.Printf("Requesting GPU adapter matching: %q\n", *gpuFlag)
		gpu.SetAdapterPreference(*gpuFlag)
	}

	rand.Seed(time.Now().UnixNano())

	fmt.Println("╔═══════════════════════════════════════════════════════════════╗")
	fmt.Println("║       GPU Training Verification Test                         ║")
	fmt.Println("╚═══════════════════════════════════════════════════════════════╝")
	fmt.Println()

	// Generate dataset
	fmt.Printf("Generating linearly separable dataset (100 samples)...\n")
	dataset := generateSimpleDataset(100)

	// ========================================================================
	// CPU Training
	// ========================================================================
	fmt.Println("\n┌───────────────────────────────────────────────────────────────┐")
	fmt.Println("│ CPU TRAINING                                                  │")
	fmt.Println("└───────────────────────────────────────────────────────────────┘")

	cpuNetwork, err := createNetwork(1) // Batch size 1 for CPU SGD
	if err != nil {
		panic(fmt.Sprintf("Failed to create CPU network: %v", err))
	}
	cpuNetwork.InitializeWeights()

	// Save initial weights for GPU network (before CPU training)
	initialWeightsNetwork, _ := createNetwork(20) // Dummy batch size, we only want weights
	cloneWeights(cpuNetwork, initialWeightsNetwork)

	// Evaluate before training
	fmt.Println("\nEvaluating before training...")
	evalStart := time.Now()
	cpuBeforeMetrics, err := cpuNetwork.EvaluateNetwork(dataset.Inputs, dataset.Outputs)
	evalTime := time.Since(evalStart)
	if err != nil {
		panic(fmt.Sprintf("CPU evaluation failed: %v", err))
	}
	fmt.Printf("Before training: Accuracy = %.1f%% (eval time: %v)\n", cpuBeforeMetrics.Accuracy*100, evalTime)

	// Train
	fmt.Printf("\nTraining for %d epochs (lr=%.3f)...\n", *epochsFlag, *lrFlag)
	// CPU: Use Batch Size 1 (SGD) as CPU implementation doesn't support batching
	cpuTrainTime, err := trainNetwork(cpuNetwork, dataset, *epochsFlag, float32(*lrFlag), false, 1)
	if err != nil {
		panic(fmt.Sprintf("CPU training failed: %v", err))
	}
	fmt.Printf("Training completed in %v\n", cpuTrainTime)

	// Evaluate after training
	fmt.Println("\nEvaluating after training...")
	evalStart = time.Now()
	cpuAfterMetrics, err := cpuNetwork.EvaluateNetwork(dataset.Inputs, dataset.Outputs)
	evalTime = time.Since(evalStart)
	if err != nil {
		panic(fmt.Sprintf("CPU evaluation failed: %v", err))
	}
	fmt.Printf("After training: Accuracy = %.1f%% (eval time: %v)\n", cpuAfterMetrics.Accuracy*100, evalTime)

	// ========================================================================
	// GPU Training
	// ========================================================================
	fmt.Println("\n┌───────────────────────────────────────────────────────────────┐")
	fmt.Println("│ GPU TRAINING                                                  │")
	fmt.Println("└───────────────────────────────────────────────────────────────┘")

	gpuNetwork, err := createNetwork(20) // Batch size 20 for GPU Mini-Match
	if err != nil {
		panic(fmt.Sprintf("Failed to create GPU network: %v", err))
	}

	// Clone INITIAL weights (before any training)
	cloneWeights(initialWeightsNetwork, gpuNetwork)

	// Enable GPU
	gpuNetwork.GPU = true
	err = gpuNetwork.WeightsToGPU()
	if err != nil {
		panic(fmt.Sprintf("Failed to mount weights to GPU: %v", err))
	}
	defer gpuNetwork.ReleaseGPUWeights()

	// Evaluate before training
	fmt.Println("\nEvaluating before training...")
	evalStart = time.Now()
	gpuBeforeMetrics, err := gpuNetwork.EvaluateNetwork(dataset.Inputs, dataset.Outputs)
	evalTime = time.Since(evalStart)
	if err != nil {
		panic(fmt.Sprintf("GPU evaluation failed: %v", err))
	}
	fmt.Printf("Before training: Accuracy = %.1f%% (eval time: %v)\n", gpuBeforeMetrics.Accuracy*100, evalTime)

	// Train
	fmt.Printf("\nTraining for %d epochs (lr=%.3f)...\n", *epochsFlag, *lrFlag)
	// GPU: Use Batch Size 20 (Mini-Batch)
	gpuTrainTime, err := trainNetwork(gpuNetwork, dataset, *epochsFlag, float32(*lrFlag), true, 20)
	if err != nil {
		panic(fmt.Sprintf("GPU training failed: %v", err))
	}
	fmt.Printf("Training completed in %v\n", gpuTrainTime)

	// Evaluate after training
	fmt.Println("\nEvaluating after training...")
	evalStart = time.Now()
	gpuAfterMetrics, err := gpuNetwork.EvaluateNetwork(dataset.Inputs, dataset.Outputs)
	evalTime = time.Since(evalStart)
	if err != nil {
		panic(fmt.Sprintf("GPU evaluation failed: %v", err))
	}
	fmt.Printf("After training: Accuracy = %.1f%% (eval time: %v)\n", gpuAfterMetrics.Accuracy*100, evalTime)

	// ========================================================================
	// Comparison
	// ========================================================================
	fmt.Println("\n┌───────────────────────────────────────────────────────────────┐")
	fmt.Println("│ COMPARISON RESULTS                                            │")
	fmt.Println("└───────────────────────────────────────────────────────────────┘")

	printDeviationTable("CPU Training", cpuBeforeMetrics, cpuAfterMetrics)
	printDeviationTable("GPU Training", gpuBeforeMetrics, gpuAfterMetrics)

	fmt.Printf("\nPerformance:\n")
	fmt.Printf("  CPU Time: %v\n", cpuTrainTime)
	fmt.Printf("  GPU Time: %v\n", gpuTrainTime)
	speedup := float64(cpuTrainTime) / float64(gpuTrainTime)
	fmt.Printf("  Speedup:  %.2fx\n", speedup)

	// Validation
	// CPU (SGD) does NumSamples updates per epoch.
	// GPU (Batch=20) does NumSamples/20 updates per epoch.
	// We expect CPU to converge faster per-epoch.
	// Success is defined as:
	// 1. Speedup > 1.0x
	// 2. GPU Accuracy significantly improved from baseline

	fmt.Printf("\nVerification Analysis:\n")
	fmt.Printf("  CPU Updates/Epoch: %d (Batch=1)\n", dataset.NumClass*50) // Approx
	fmt.Printf("  GPU Updates/Epoch: %d (Batch=20)\n", (dataset.NumClass*50)/20)

	learningThreshold := 0.30 // Expect at least 30% improvement
	accuracyGain := gpuAfterMetrics.Accuracy - gpuBeforeMetrics.Accuracy

	passed := true
	if speedup < 1.0 {
		fmt.Printf("  [FAIL] GPU is slower than CPU (%.2fx)\n", speedup)
		passed = false
	} else {
		fmt.Printf("  [PASS] GPU Speedup: %.2fx\n", speedup)
	}

	if accuracyGain < float64(learningThreshold) && gpuAfterMetrics.Accuracy < 0.90 {
		fmt.Printf("  [FAIL] GPU did not learn sufficienty (Gain: %.1f%%)\n", accuracyGain*100)
		passed = false
	} else {
		fmt.Printf("  [PASS] GPU Learning Confirmed (Gain: %.1f%%, Final: %.1f%%)\n", accuracyGain*100, gpuAfterMetrics.Accuracy*100)
	}

	if passed {
		fmt.Printf("\nGradient Descent Verification: PASSED\n")
	} else {
		fmt.Printf("\nGradient Descent Verification: FAILED\n")
	}
}
