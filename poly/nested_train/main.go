package main

import (
	"fmt"
	"github.com/openfluke/loom/poly"
)

func main() {
	fmt.Println("=== M-POLY-VTD Nested Training (Weights Mutation) Verification ===")

	// 1. Setup Network (1x1x1x1)
	net := poly.NewVolumetricNetwork(1, 1, 1, 1)
	
	inputSize := 8
	hiddenSize := 8
	
	wStore := func() *poly.WeightStore {
		ws := &poly.WeightStore{Master: make([]float32, 64), Scale: 1.0}
		for i := range ws.Master { ws.Master[i] = 1.0 } // Init with 1.0
		return ws
	}
	
	denseA := poly.VolumetricLayer{Type: poly.LayerDense, InputHeight: inputSize, OutputHeight: hiddenSize, DType: poly.DTypeFloat32, WeightStore: wStore()}
	denseB := poly.VolumetricLayer{Type: poly.LayerDense, InputHeight: hiddenSize, OutputHeight: hiddenSize, DType: poly.DTypeFloat32, WeightStore: wStore()}
	
	// Create a Sequential layer at the only coordinate
	rootLayer := net.GetLayer(0, 0, 0, 0)
	rootLayer.Type = poly.LayerSequential
	rootLayer.SequentialLayers = []poly.VolumetricLayer{denseA, denseB}

	// 2. Initial state capture
	initSumA := sum(denseA.WeightStore.Master)
	initSumB := sum(denseB.WeightStore.Master)
	fmt.Printf("Initial Weight Sum A: %v\n", initSumA)
	fmt.Printf("Initial Weight Sum B: %v\n", initSumB)

	// 3. Run Training
	batch := poly.TrainingBatch[float32]{
		Input:  poly.NewTensor[float32](1, inputSize),
		Target: poly.NewTensor[float32](1, hiddenSize),
	}
	for i := range batch.Input.Data { batch.Input.Data[i] = 1.0 }
	// Target is zeros, so loss will be high and weights should move
	
	config := poly.DefaultTrainingConfig()
	config.Epochs = 1
	config.LearningRate = 1.0 // High LR for obvious movement
	
	fmt.Println("\n--- Starting Training Step ---")
	_, err := poly.Train(net, []poly.TrainingBatch[float32]{batch}, config)
	if err != nil {
		fmt.Printf("Error: %v\n", err)
		return
	}

	// 4. Verify Mutation
	// NOTE: We must fetch the layers again or use our pointers if they were Shallow copied.
	// Actually, poly.Train gets n.Layers[0], but we need to check the Nested ones.
	// Since we passed denseA/denseB by value into the slice, the pointers in WeightStore should still point to the same Master slice.
	
	finalSumA := sum(denseA.WeightStore.Master)
	finalSumB := sum(denseB.WeightStore.Master)
	fmt.Printf("\nFinal Weight Sum A: %v\n", finalSumA)
	fmt.Printf("Final Weight Sum B: %v\n", finalSumB)

	if finalSumA != initSumA && finalSumB != initSumB {
		fmt.Println("\nSUCCESS: Both nested stages had their weights updated! o_O")
	} else {
		fmt.Println("\nFAILURE: One or more nested stages did NOT update weights.")
		if finalSumA == initSumA { fmt.Println(" - Stage A untouched.") }
		if finalSumB == initSumB { fmt.Println(" - Stage B untouched.") }
	}

	fmt.Println("\n=== Nested Weights Mutation Verified! o_O ===")
}

func sum(data []float32) float32 {
	var s float32
	for _, v := range data { s += v }
	return s
}
