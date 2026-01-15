package main

import (
	"fmt"
	"math"
	"math/rand"
	"time"

	"github.com/openfluke/loom/nn"
)

// Simple test: Can MHA learn to copy its input?
// Input: [batch, seq=1, embed_dim]
// Output: should match input
func main() {
	rand.Seed(time.Now().UnixNano())

	embedDim := 64
	numHeads := 4
	batchSize := 32
	epochs := 50
	lr := 0.01

	fmt.Println("╔═══════════════════════════════════════════════════════════════╗")
	fmt.Println("║           MHA Layer Learning Test (CPU & GPU)                ║")
	fmt.Println("╚═══════════════════════════════════════════════════════════════╝")
	fmt.Printf("\nTask: Learn simple pattern (if x[0] > 0.5 then class 1)\n")
	fmt.Printf("Config: embed_dim=%d, num_heads=%d, batch=%d\n\n", embedDim, numHeads, batchSize)

	// Create network with MHA layer
	jsonConfig := fmt.Sprintf(`{
		"id": "mha_test",
		"batch_size": %d,
		"grid_rows": 1,
		"grid_cols": 1,
		"layers_per_cell": 3,
		"layers": [
			{"type": "dense", "activation": "relu", "input_height": 2, "output_height": %d},
			{"type": "multi_head_attention", "d_model": %d, "num_heads": %d, "seq_length": 1},
			{"type": "dense", "activation": "sigmoid", "input_height": %d, "output_height": 2}
		]
	}`, batchSize, embedDim, embedDim, numHeads, embedDim)

	// Test CPU version
	fmt.Println("═══════════════════════════════════════════════════════════════")
	fmt.Println("CPU Training")
	fmt.Println("═══════════════════════════════════════════════════════════════")

	cpuNet, err := nn.BuildNetworkFromJSON(jsonConfig)
	if err != nil {
		fmt.Printf("Error creating CPU network: %v\n", err)
		return
	}
	cpuNet.InitializeWeights()

	// Generate simple dataset
	numSamples := 100
	inputs := make([][]float32, numSamples)
	labels := make([]float64, numSamples)

	for i := 0; i < numSamples; i++ {
		x0 := rand.Float32()
		x1 := rand.Float32()
		inputs[i] = []float32{x0, x1}
		if x0 > 0.5 {
			labels[i] = 1.0
		} else {
			labels[i] = 0.0
		}
	}

	// Train CPU
	cpuStartAcc := evaluateAccuracy(cpuNet, inputs, labels)
	fmt.Printf("Initial accuracy: %.1f%%\n", cpuStartAcc*100)

	cpuStartTime := time.Now()
	for epoch := 0; epoch < epochs; epoch++ {
		totalLoss := float32(0.0)

		// Mini-batch training
		for i := 0; i < numSamples; i += batchSize {
			endIdx := i + batchSize
			if endIdx > numSamples {
				endIdx = numSamples
			}
			currentBatch := endIdx - i

			// Prepare batch
			batchInput := make([]float32, currentBatch*2)
			for j := 0; j < currentBatch; j++ {
				copy(batchInput[j*2:], inputs[i+j])
			}

			// Forward
			output, _ := cpuNet.ForwardCPU(batchInput)

			// Compute gradients
			dOutput := make([]float32, len(output))
			for j := 0; j < currentBatch; j++ {
				class := int(labels[i+j])
				outStart := j * 2

				// Loss
				if class < 2 {
					val := output[outStart+class]
					if val > 1e-7 {
						totalLoss += -float32(math.Log(float64(val)))
					}
				}

				// Gradient
				for k := 0; k < 2; k++ {
					targetVal := 0.0
					if k == class {
						targetVal = 1.0
					}
					dOutput[outStart+k] = (output[outStart+k] - float32(targetVal)) / float32(currentBatch)
				}
			}

			// Backward
			cpuNet.BackwardCPU(dOutput)
			cpuNet.ApplyGradients(float32(lr))
		}

		avgLoss := totalLoss / float32(numSamples)
		if epoch%10 == 0 || epoch == epochs-1 {
			acc := evaluateAccuracy(cpuNet, inputs, labels)
			fmt.Printf("Epoch %2d/%d - Loss: %.4f, Acc: %.1f%%\n", epoch+1, epochs, avgLoss, acc*100)
		}
	}
	cpuTime := time.Since(cpuStartTime)
	cpuEndAcc := evaluateAccuracy(cpuNet, inputs, labels)

	fmt.Printf("\nCPU Result: %.1f%% → %.1f%% in %v\n", cpuStartAcc*100, cpuEndAcc*100, cpuTime)

	// Test GPU version
	fmt.Println("\n═══════════════════════════════════════════════════════════════")
	fmt.Println("GPU Training")
	fmt.Println("═══════════════════════════════════════════════════════════════")

	gpuNet, err := nn.BuildNetworkFromJSON(jsonConfig)
	if err != nil {
		fmt.Printf("Error creating GPU network: %v\n", err)
		return
	}

	// Copy weights from CPU network
	for i := 0; i < cpuNet.TotalLayers(); i++ {
		if len(cpuNet.Layers[i].Kernel) > 0 {
			gpuNet.Layers[i].Kernel = make([]float32, len(cpuNet.Layers[i].Kernel))
			copy(gpuNet.Layers[i].Kernel, cpuNet.Layers[i].Kernel)
		}
		if len(cpuNet.Layers[i].Bias) > 0 {
			gpuNet.Layers[i].Bias = make([]float32, len(cpuNet.Layers[i].Bias))
			copy(gpuNet.Layers[i].Bias, cpuNet.Layers[i].Bias)
		}
	}

	gpuNet.GPU = true
	err = gpuNet.WeightsToGPU()
	if err != nil {
		fmt.Printf("Failed to mount GPU: %v\n", err)
		fmt.Println("\n⚠️ GPU test skipped")
		return
	}
	defer gpuNet.ReleaseGPUWeights()

	gpuStartAcc := evaluateAccuracy(gpuNet, inputs, labels)
	fmt.Printf("Initial accuracy: %.1f%%\n", gpuStartAcc*100)

	gpuStartTime := time.Now()
	for epoch := 0; epoch < epochs; epoch++ {
		totalLoss := float32(0.0)

		for i := 0; i < numSamples; i += batchSize {
			endIdx := i + batchSize
			if endIdx > numSamples {
				endIdx = numSamples
			}
			currentBatch := endIdx - i

			batchInput := make([]float32, currentBatch*2)
			for j := 0; j < currentBatch; j++ {
				copy(batchInput[j*2:], inputs[i+j])
			}

			output, _ := gpuNet.ForwardCPU(batchInput)

			dOutput := make([]float32, len(output))
			for j := 0; j < currentBatch; j++ {
				class := int(labels[i+j])
				outStart := j * 2

				if class < 2 {
					val := output[outStart+class]
					if val > 1e-7 {
						totalLoss += -float32(math.Log(float64(val)))
					}
				}

				for k := 0; k < 2; k++ {
					targetVal := 0.0
					if k == class {
						targetVal = 1.0
					}
					dOutput[outStart+k] = (output[outStart+k] - float32(targetVal)) / float32(currentBatch)
				}
			}

			gpuNet.BackwardCPU(dOutput)
			gpuNet.ApplyGradients(float32(lr) * 5.0) // GPU uses higher LR
		}

		avgLoss := totalLoss / float32(numSamples)
		if epoch%10 == 0 || epoch == epochs-1 {
			acc := evaluateAccuracy(gpuNet, inputs, labels)
			fmt.Printf("Epoch %2d/%d - Loss: %.4f, Acc: %.1f%%\n", epoch+1, epochs, avgLoss, acc*100)
		}
	}
	gpuTime := time.Since(gpuStartTime)
	gpuEndAcc := evaluateAccuracy(gpuNet, inputs, labels)

	fmt.Printf("\nGPU Result: %.1f%% → %.1f%% in %v\n", gpuStartAcc*100, gpuEndAcc*100, gpuTime)

	// Summary
	fmt.Println("\n╔═══════════════════════════════════════════════════════════════╗")
	fmt.Println("║ SUMMARY                                                       ║")
	fmt.Println("╚═══════════════════════════════════════════════════════════════╝")
	fmt.Printf("CPU: %.1f%% → %.1f%% (%s)\n", cpuStartAcc*100, cpuEndAcc*100,
		learningStatus(cpuStartAcc, cpuEndAcc))
	fmt.Printf("GPU: %.1f%% → %.1f%% (%s)\n", gpuStartAcc*100, gpuEndAcc*100,
		learningStatus(gpuStartAcc, gpuEndAcc))

	if cpuEndAcc < 0.7 && gpuEndAcc < 0.7 {
		fmt.Println("\n⚠️ ISSUE: MHA layer not learning on EITHER CPU or GPU!")
		fmt.Println("This suggests a fundamental problem with the MHA implementation,")
		fmt.Println("not a GPU-specific bug.")
	}
}

func evaluateAccuracy(net *nn.Network, inputs [][]float32, labels []float64) float64 {
	correct := 0
	for i, input := range inputs {
		output, _ := net.ForwardCPU(input)
		predicted := 0
		if output[1] > output[0] {
			predicted = 1
		}
		if predicted == int(labels[i]) {
			correct++
		}
	}
	return float64(correct) / float64(len(inputs))
}

func learningStatus(start, end float64) string {
	if end > 0.9 {
		return "✓ LEARNING"
	} else if end > start+0.1 {
		return "~ SOME LEARNING"
	} else {
		return "✗ NOT LEARNING"
	}
}
