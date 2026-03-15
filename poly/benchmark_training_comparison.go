package main

import (
	"fmt"
	"math/rand"
	"time"

	"github.com/openfluke/loom/poly"
)

func main() {
	fmt.Println("=== M-POLY-VTD Multi-Architecture Training Showdown ===")
	
	// Set seed for reproducibility
	rand.Seed(42)

	config := &poly.TrainingConfig{
		Epochs:       50,
		LearningRate: 0.1,
		LossType:     "mse",
		Verbose:      false,
	}

	// 1. MLP
	runArchitectureTest("Dense MLP (XOR-like)", createDenseNetwork, generateDenseData, config)

	// 2. CNNs
	runArchitectureTest("CNN 1D Classifier", createCNN1DNetwork, generateCNN1DData, config)
	runArchitectureTest("CNN 2D Classifier", createCNN2DNetwork, generateCNN2DData, config)
	runArchitectureTest("CNN 3D Classifier", createCNN3DNetwork, generateCNN3DData, config)

	// 3. Normalization
	runArchitectureTest("RMSNorm MLP", createNormalizedNetwork, generateDenseData, config)
}

func runArchitectureTest(name string, netFactory func() *poly.VolumetricNetwork, dataFactory func(int, int) []poly.TrainingBatch[float32], config *poly.TrainingConfig) {
	fmt.Printf("\n--- Testing Architecture: %s ---\n", name)

	numBatches := 10
	batchSize := 8
	batches := dataFactory(numBatches, batchSize)
	
	nCPU := netFactory()
	nGPU := netFactory()
	
	// Parity Sync
	for i := range nCPU.Layers {
		if nCPU.Layers[i].WeightStore != nil {
			copy(nGPU.Layers[i].WeightStore.Master, nCPU.Layers[i].WeightStore.Master)
		}
	}

	// 1. CPU
	fmt.Print("[CPU] Training...")
	startCPU := time.Now()
	resCPU, err := poly.Train(nCPU, batches, config)
	if err != nil { fmt.Printf(" Error: %v\n", err); return }
	cpuDuration := time.Since(startCPU)
	fmt.Printf(" Done (%v)\n", cpuDuration)

	// 2. GPU
	fmt.Print("[GPU] Training...")
	config.UseGPU = true
	startGPU := time.Now()
	resGPU, err := poly.Train(nGPU, batches, config)
	if err != nil { fmt.Printf(" Error: %v\n", err); return }
	gpuDuration := time.Since(startGPU)
	fmt.Printf(" Done (%v)\n", gpuDuration)
	config.UseGPU = false // Reset for next test

	// Results
	initialLoss := resCPU.LossHistory[0]
	finalLossCPU := resCPU.LossHistory[len(resCPU.LossHistory)-1]
	finalLossGPU := resGPU.LossHistory[len(resGPU.LossHistory)-1]
	
	cpuImprov := (initialLoss - finalLossCPU) / initialLoss * 100
	gpuImprov := (initialLoss - finalLossGPU) / initialLoss * 100

	fmt.Printf("| Metric      | CPU         | GPU         | Improvement |\n")
	fmt.Printf("|-------------|-------------|-------------|-------------|\n")
	fmt.Printf("| Final Loss  | %-11.6f | %-11.6f | CPU: %.1f%% |\n", finalLossCPU, finalLossGPU, cpuImprov)
	fmt.Printf("| Gain %%      |             |             | GPU: %.1f%% |\n", gpuImprov)
	
	speedup := float64(cpuDuration) / float64(gpuDuration)
	fmt.Printf("Speedup: %.2fx\n", speedup)
}

// Architecture Factories

func createDenseNetwork() *poly.VolumetricNetwork {
	n := poly.NewVolumetricNetwork(1, 1, 1, 2)
	setupDense(n.GetLayer(0, 0, 0, 0), 2, 8, poly.ActivationReLU)
	setupDense(n.GetLayer(0, 0, 0, 1), 8, 1, poly.ActivationLinear)
	randomizeNetwork(n)
	return n
}

func createCNN2DNetwork() *poly.VolumetricNetwork {
	// Simple CNN: 1x8x8 Input -> CNN2D (3x3, filters=4) -> Flatten -> Dense
	n := poly.NewVolumetricNetwork(1, 1, 1, 2)
	
	l0 := n.GetLayer(0, 0, 0, 0)
	l0.Type = poly.LayerCNN2
	l0.InputChannels = 1
	l0.InputHeight = 8
	l0.InputWidth = 8
	l0.Filters = 4
	l0.KernelSize = 3
	l0.Stride = 1
	l0.Padding = 1
	l0.OutputHeight = 8
	l0.OutputWidth = 8
	l0.Activation = poly.ActivationReLU
	l0.WeightStore = poly.NewWeightStore(4 * 1 * 3 * 3)
	
	l1 := n.GetLayer(0, 0, 0, 1)
	setupDense(l1, 4*8*8, 1, poly.ActivationLinear)
	
	randomizeNetwork(n)
	return n
}

func createNormalizedNetwork() *poly.VolumetricNetwork {
	n := poly.NewVolumetricNetwork(1, 1, 1, 3)
	setupDense(n.GetLayer(0, 0, 0, 0), 2, 16, poly.ActivationLinear)
	
	lNorm := n.GetLayer(0, 0, 0, 1)
	lNorm.Type = poly.LayerRMSNorm
	lNorm.InputHeight = 16
	lNorm.OutputHeight = 16
	lNorm.WeightStore = poly.NewWeightStore(16)
	
	setupDense(n.GetLayer(0, 0, 0, 2), 16, 1, poly.ActivationLinear)
	
	randomizeNetwork(n)
	return n
}

// Data Generators

func generateDenseData(numBatches, batchSize int) []poly.TrainingBatch[float32] {
	batches := make([]poly.TrainingBatch[float32], numBatches)
	for i := 0; i < numBatches; i++ {
		input := poly.NewTensor[float32](batchSize, 2)
		target := poly.NewTensor[float32](batchSize, 1)
		for j := 0; j < batchSize; j++ {
			x1, x2 := rand.Float32(), rand.Float32()
			input.Data[j*2] = x1
			input.Data[j*2+1] = x2
			if x1+x2 > 1.0 { target.Data[j] = 1.0 } else { target.Data[j] = 0.0 }
		}
		batches[i] = poly.TrainingBatch[float32]{Input: input, Target: target}
	}
	return batches
}

func generateCNN2DData(numBatches, batchSize int) []poly.TrainingBatch[float32] {
	batches := make([]poly.TrainingBatch[float32], numBatches)
	for i := 0; i < numBatches; i++ {
		input := poly.NewTensor[float32](batchSize, 1, 8, 8)
		target := poly.NewTensor[float32](batchSize, 1)
		for j := 0; j < batchSize; j++ {
			sum := float32(0)
			for k := 0; k < 64; k++ {
				v := rand.Float32()
				input.Data[j*64+k] = v
				sum += v
			}
			if sum > 32.0 { target.Data[j] = 1.0 } else { target.Data[j] = 0.0 }
		}
		batches[i] = poly.TrainingBatch[float32]{Input: input, Target: target}
	}
	return batches
}

func createCNN1DNetwork() *poly.VolumetricNetwork {
	n := poly.NewVolumetricNetwork(1, 1, 1, 2)
	l0 := n.GetLayer(0, 0, 0, 0)
	l0.Type = poly.LayerCNN1
	l0.InputChannels = 1
	l0.InputHeight = 16
	l0.Filters = 4
	l0.KernelSize = 3
	l0.Stride = 1
	l0.Padding = 1
	l0.OutputHeight = 16
	l0.Activation = poly.ActivationReLU
	l0.WeightStore = poly.NewWeightStore(4 * 1 * 3)
	
	l1 := n.GetLayer(0, 0, 0, 1)
	setupDense(l1, 4*16, 1, poly.ActivationLinear)
	
	randomizeNetwork(n)
	return n
}

func generateCNN1DData(numBatches, batchSize int) []poly.TrainingBatch[float32] {
	return createSyntheticData(numBatches, batchSize, []int{1, 16}, []int{1})
}

func createCNN3DNetwork() *poly.VolumetricNetwork {
	n := poly.NewVolumetricNetwork(1, 1, 1, 2)
	l0 := n.GetLayer(0, 0, 0, 0)
	l0.Type = poly.LayerCNN3
	l0.InputChannels = 1
	l0.InputDepth = 4
	l0.InputHeight = 4
	l0.InputWidth = 4
	l0.Filters = 2
	l0.KernelSize = 3
	l0.Stride = 1
	l0.Padding = 1
	l0.OutputDepth = 4
	l0.OutputHeight = 4
	l0.OutputWidth = 4
	l0.Activation = poly.ActivationReLU
	l0.WeightStore = poly.NewWeightStore(2 * 1 * 3 * 3 * 3)
	
	l1 := n.GetLayer(0, 0, 0, 1)
	setupDense(l1, 2*4*4*4, 1, poly.ActivationLinear)
	
	randomizeNetwork(n)
	return n
}

func generateCNN3DData(numBatches, batchSize int) []poly.TrainingBatch[float32] {
	return createSyntheticData(numBatches, batchSize, []int{1, 4, 4, 4}, []int{1})
}

func createSyntheticData(numBatches, batchSize int, inputShape, targetShape []int) []poly.TrainingBatch[float32] {
	batches := make([]poly.TrainingBatch[float32], numBatches)
	for b := 0; b < numBatches; b++ {
		fullInShape := append([]int{batchSize}, inputShape...)
		fullTargetShape := append([]int{batchSize}, targetShape...)
		
		input := poly.NewTensor[float32](fullInShape...)
		target := poly.NewTensor[float32](fullTargetShape...)
		
		for i := range input.Data { input.Data[i] = rand.Float32() }
		for i := range target.Data { target.Data[i] = rand.Float32() }
		
		batches[b] = poly.TrainingBatch[float32]{
			Input:  input,
			Target: target,
		}
	}
	return batches
}

// Helpers

func randomizeNetwork(n *poly.VolumetricNetwork) {
	for i := range n.Layers {
		l := &n.Layers[i]
		if l.WeightStore != nil {
			for j := range l.WeightStore.Master {
				l.WeightStore.Master[j] = (rand.Float32()*2 - 1) * 0.1
			}
		}
	}
}

func setupDense(l *poly.VolumetricLayer, in, out int, act poly.ActivationType) {
	l.Type = poly.LayerDense
	l.InputHeight = in
	l.OutputHeight = out
	l.Activation = act
	l.WeightStore = poly.NewWeightStore(in*out + out)
}
