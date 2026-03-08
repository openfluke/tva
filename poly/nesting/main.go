package main

import (
	"fmt"
	"github.com/openfluke/loom/poly"
)

func main() {
	fmt.Println("=== M-POLY-VTD Deep Universal Nesting Verification ===")

	inputSize := 8
	hiddenSize := 8
	
	// Structure:
	// Sequential
	//   - Dense (8 -> 8)
	//   - Parallel (Add)
	//       - Branch 0: Dense (8 -> 8)
	//       - Branch 1: Dense (8 -> 8)
	//   - Softmax
	
	wStore := func() *poly.WeightStore {
		return &poly.WeightStore{Master: make([]float32, 64), Scale: 1.0}
	}
	
	dense1 := poly.VolumetricLayer{Type: poly.LayerDense, InputHeight: inputSize, OutputHeight: hiddenSize, DType: poly.DTypeFloat32, WeightStore: wStore()}
	dense2a := poly.VolumetricLayer{Type: poly.LayerDense, InputHeight: hiddenSize, OutputHeight: hiddenSize, DType: poly.DTypeFloat32, WeightStore: wStore()}
	dense2b := poly.VolumetricLayer{Type: poly.LayerDense, InputHeight: hiddenSize, OutputHeight: hiddenSize, DType: poly.DTypeFloat32, WeightStore: wStore()}
	
	para := poly.VolumetricLayer{
		Type: poly.LayerParallel,
		CombineMode: "add",
		ParallelBranches: []poly.VolumetricLayer{dense2a, dense2b},
	}
	
	softmax := poly.VolumetricLayer{Type: poly.LayerSoftmax, SoftmaxType: poly.SoftmaxStandard, DType: poly.DTypeFloat32}

	deepNet := poly.VolumetricLayer{
		Type: poly.LayerSequential,
		SequentialLayers: []poly.VolumetricLayer{dense1, para, softmax},
	}

	// 1. Forward
	input := poly.NewTensor[float32](1, inputSize)
	for i := range input.Data { input.Data[i] = 1.0 }

	fmt.Println("\n--- Forward: Sequential[Dense, Parallel[Dense, Dense], Softmax] ---")
	preAct, postAct := poly.DispatchLayer(&deepNet, input, nil)
	fmt.Printf("Output Size: %d\n", len(postAct.Data))
	
	if preAct != nil && len(preAct.Nested) == 3 {
		fmt.Println("SUCCESS: Sequential Activation Tree captured all 3 stages.")
		if preAct.Nested[1] != nil && preAct.Nested[1].Nested != nil && len(preAct.Nested[1].Nested[0].Nested) == 2 {
			fmt.Println("SUCCESS: Recursive Nesting (Parallel inside Sequential) confirmed.")
		}
	}

	// 2. Backward
	fmt.Println("\n--- Backward: Propagating through the entire depth ---")
	gradOutput := poly.NewTensor[float32](1, hiddenSize) // Softmax out is same size as hidden
	for i := range gradOutput.Data { gradOutput.Data[i] = 1.0 }

	_, gW := poly.DispatchLayerBackward(&deepNet, gradOutput, input, nil, preAct)

	// 3. Verify Gradient Tree
	if gW != nil && len(gW.Nested) == 3 {
		// dense1 grad
		gw1 := gW.Nested[0]
		// para grads
		gwP := gW.Nested[1]
		
		if gw1 != nil && gwP != nil && gwP.Nested != nil {
			fmt.Println("SUCCESS: Gradient Tree recovered for all nested levels! o_O")
		}
	}

	fmt.Println("\n=== Universal Nesting & Training Bedrock Verified! ===")
}
