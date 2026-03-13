package main

import (
	"fmt"
	"github.com/openfluke/loom/poly"
)

func main() {
	fmt.Println("=== M-POLY-VTD Training Convergence Verification ===")

	// 1. Setup a simple network (1 hidden layer)
	inSize := 4
	hiddenSize := 8
	outSize := 2
	net := poly.NewVolumetricNetwork(1, 1, 1, 2)
	
	// Layer 0: Hidden
	l0 := net.GetLayer(0, 0, 0, 0)
	l0.Type = poly.LayerDense
	l0.Activation = poly.ActivationLinear
	l0.InputHeight = inSize
	l0.OutputHeight = hiddenSize
	l0.WeightStore = poly.NewWeightStore(inSize * hiddenSize)
	l0.WeightStore.Randomize(42, 0.1)
	
	// Layer 1: Output
	l1 := net.GetLayer(0, 0, 0, 1)
	l1.Type = poly.LayerDense
	l1.Activation = poly.ActivationLinear
	l1.InputHeight = hiddenSize
	l1.OutputHeight = outSize
	l1.WeightStore = poly.NewWeightStore(hiddenSize * outSize)
	l1.WeightStore.Randomize(43, 0.1)

	// 2. Create Dummy Training Data (Identity-ish)
	batch := poly.TrainingBatch[float32]{
		Input:  poly.NewTensorFromSlice([]float32{1, 0, 0, 0}, 1, 4),
		Target: poly.NewTensorFromSlice([]float32{1, 0}, 1, 2),
	}
	batches := []poly.TrainingBatch[float32]{batch}

	// 3. Train
	config := poly.DefaultTrainingConfig()
	config.Epochs = 200
	config.LearningRate = 0.1
	
	res, err := poly.Train(net, batches, config)
	if err != nil {
		fmt.Printf("Training failed: %v\n", err)
		return
	}

	// 4. Verify Loss Reduction
	if len(res.LossHistory) < 2 {
		fmt.Println("Error: No loss history generated.")
		return
	}

	firstLoss := res.LossHistory[0]
	lastLoss := res.LossHistory[len(res.LossHistory)-1]
	
	fmt.Printf("\nVerification Results:\n")
	fmt.Printf("Initial Loss: %.8f\n", firstLoss)
	fmt.Printf("Final Loss:   %.8f\n", lastLoss)
	
	if lastLoss < firstLoss {
		fmt.Println("✓ SUCCESS: Loss decreased. The network is learning.")
	} else {
		fmt.Println("✗ FAILURE: Loss did not decrease.")
	}
}
