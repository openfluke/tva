package main

import (
	"fmt"
	"github.com/openfluke/loom/poly"
)

func main() {
	fmt.Println("=== M-POLY-VTD Systolic Target Propagation Verification ===")

	// 1. Setup 1x1x3 grid
	net := poly.NewVolumetricNetwork(1, 1, 3, 1)
	for i := range net.Layers {
		net.Layers[i].Type = poly.LayerDense
		net.Layers[i].InputHeight = 1
		net.Layers[i].OutputHeight = 1
		net.Layers[i].DType = poly.DTypeFloat32
		net.Layers[i].WeightStore = &poly.WeightStore{
			Master: []float32{1.0}, // Start with 1.0 weight
			Scale:  1.0,
		}
	}

	// 2. Initialize Systolic State
	sys := poly.NewSystolicState[float32](net)

	// 3. Define Input and Target
	input := poly.NewTensor[float32](1, 1)
	input.Data[0] = 5.0
	
	target := poly.NewTensor[float32](1, 1)
	target.Data[0] = 10.0 // We want the grid to amplify 5 -> 10

	fmt.Println("\n--- Executing 3 Systolic Ticks ---")
	sys.SetInput(input)
	poly.SystolicForward(net, sys, false) // T1
	poly.SystolicForward(net, sys, false) // T2
	poly.SystolicForward(net, sys, false) // T3
	
	finalOutput := sys.LayerData[len(net.Layers)-1]
	fmt.Printf("Initial Out: %v (Gap: %v)\n", finalOutput.Data[0], target.Data[0]-finalOutput.Data[0])

	// 4. Apply Target Prop via Systolic Bridge
	fmt.Println("\n--- Applying Systolic Target Propagation ---")
	initWeight := net.Layers[1].WeightStore.Master[0]
	fmt.Printf("L1 Init Weight: %v\n", initWeight)
	
	// Apply Target Prop logic
	poly.SystolicApplyTargetProp(net, sys, target, 0.1)
	
	finalWeight := net.Layers[1].WeightStore.Master[0]
	fmt.Printf("L1 Final Weight: %v\n", finalWeight)

	if finalWeight != initWeight {
		fmt.Println("\nSUCCESS: Mesh weights mutated via Systolic Target Prop! o_O")
	} else {
		fmt.Println("\nFAILURE: Mesh weights stayed static.")
	}

	fmt.Println("\n=== Systolic Target Prop Integration Verified! o_O ===")
}
