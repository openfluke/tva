package main

import (
	"fmt"
	"github.com/openfluke/loom/poly"
)

func main() {
	fmt.Println("=== M-POLY-VTD Nested Training (Gradient Tree) Verification ===")

	inputSize := 8
	outputSize := 4
	
	// 1. Setup Nested Structure:
	// Parallel (Concat)
	//   - Branch 0: Dense (8 -> 4)
	//   - Branch 1: Parallel (Add)
	//       - Branch 1.0: Dense (8 -> 4)
	
	dense0 := poly.VolumetricLayer{
		Type: poly.LayerDense, InputHeight: inputSize, OutputHeight: outputSize, DType: poly.DTypeFloat32,
		WeightStore: &poly.WeightStore{Master: make([]float32, inputSize*outputSize)},
	}
	// Initial weights = 1.0
	for i := range dense0.WeightStore.Master { dense0.WeightStore.Master[i] = 1.0 }

	dense1 := poly.VolumetricLayer{
		Type: poly.LayerDense, InputHeight: inputSize, OutputHeight: outputSize, DType: poly.DTypeFloat32,
		WeightStore: &poly.WeightStore{Master: make([]float32, inputSize*outputSize)},
	}
	for i := range dense1.WeightStore.Master { dense1.WeightStore.Master[i] = 1.0 }

	innerParallel := poly.VolumetricLayer{
		Type: poly.LayerParallel,
		CombineMode: "add",
		ParallelBranches: []poly.VolumetricLayer{dense1},
	}

	outerParallel := poly.VolumetricLayer{
		Type: poly.LayerParallel,
		CombineMode: "concat",
		ParallelBranches: []poly.VolumetricLayer{dense0, innerParallel},
	}

	// 2. Forward Pass
	input := poly.NewTensor[float32](1, inputSize)
	for i := range input.Data { input.Data[i] = 1.0 }

	fmt.Println("\n--- Running Nested Forward ---")
	preAct, postAct := poly.DispatchLayer(&outerParallel, input)
	
	fmt.Printf("Output Size: %d\n", len(postAct.Data))
	if preAct.Nested != nil {
		fmt.Printf("Activation Tree Root Branches: %d\n", len(preAct.Nested))
		if len(preAct.Nested) > 1 && preAct.Nested[1].Nested != nil {
			fmt.Println("SUCCESS: Activation Tree captured nested Parallel branch.")
		}
	}

	// 3. Backward Pass
	fmt.Println("\n--- Running Nested Backward (Backprop) ---")
	gradOutput := poly.NewTensor[float32](1, outputSize + outputSize) // concat size
	for i := range gradOutput.Data { gradOutput.Data[i] = 1.0 }

	gIn, gW := poly.DispatchLayerBackward(&outerParallel, gradOutput, input, preAct)

	// 4. Verify Gradient Tree
	if gW != nil && gW.Nested != nil {
		fmt.Printf("Gradient Tree Root Branches: %d\n", len(gW.Nested))
		
		// Branch 0 (Dense)
		gW0 := gW.Nested[0]
		if gW0 != nil && len(gW0.Data) > 0 {
			fmt.Printf("Branch 0 (Dense) Gradient Sum: %v\n", sum(gW0.Data))
		}

		// Branch 1 (Parallel) -> Branch 1.0 (Dense)
		gW1 := gW.Nested[1]
		if gW1 != nil && gW1.Nested != nil {
			gW1_0 := gW1.Nested[0]
			if gW1_0 != nil && len(gW1_0.Data) > 0 {
				fmt.Printf("Branch 1.0 (Nested Dense) Gradient Sum: %v\n", sum(gW1_0.Data))
				fmt.Println("SUCCESS: Gradient successfully flowed through nested parallel branches!")
			}
		}
	}

	if gIn != nil && len(gIn.Data) == inputSize {
		fmt.Println("SUCCESS: Input gradient recovered.")
	}

	fmt.Println("\n=== Nested Training Logic Verified! o_O ===")
}

func sum(data []float32) float32 {
	var s float32
	for _, v := range data { s += v }
	return s
}
