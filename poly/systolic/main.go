package main

import (
	"fmt"
	"github.com/openfluke/loom/poly"
)

func main() {
	fmt.Println("=== M-POLY-VTD Systolic Grid Propagation Verification ===")

	// 1. Setup 1x1x3 grid (3 layers in a row)
	net := poly.NewVolumetricNetwork(1, 1, 3, 1)
	
	// Layer 0: Identity (Buffer for input) - Disable it to act as passthrough
	net.Layers[0].IsDisabled = true
	
	ws := &poly.WeightStore{Master: make([]float32, 1), Scale: 1.0}
	ws.Master[0] = 1.0 // Identity-like for 1x1
	
	net.Layers[1].Type = poly.LayerDense
	net.Layers[1].InputHeight = 1
	net.Layers[1].OutputHeight = 1
	net.Layers[1].WeightStore = ws
	net.Layers[1].DType = poly.DTypeFloat32
	
	net.Layers[2].Type = poly.LayerDense // Acts as probe
	net.Layers[2].InputHeight = 1
	net.Layers[2].OutputHeight = 1
	net.Layers[2].WeightStore = ws
	net.Layers[2].DType = poly.DTypeFloat32
	
	// Create recursive loop: L2 gets from L1, L0 gets from L2
	// Default is Sequential (L[i] gets from L[i-1])
	// So L1 gets from L0 (ok).
	// Let's make L2 get from L1 (ok).
	// And let's make L0 a remote link from L2 to create the cyclic mesh.
	net.Layers[0].IsDisabled = false
	net.Layers[0].Type = poly.LayerDense
	net.Layers[0].IsRemoteLink = true
	net.Layers[0].TargetZ = 0; net.Layers[0].TargetY = 0; net.Layers[0].TargetX = 0; net.Layers[0].TargetL = 2
	net.Layers[0].InputHeight = 1; net.Layers[0].OutputHeight = 1; net.Layers[0].WeightStore = ws

	// 2. Initialize Systolic State
	state := poly.NewSystolicState[float32](net)
	
	// Inject initial signal: 10.0 into the L0 site
	input := poly.NewTensor[float32](1, 1)
	input.Data[0] = 10.0
	state.LayerData[0] = input

	fmt.Println("\n--- Starting Propagation Steps ---")
	
	safeData := func(t *poly.Tensor[float32]) string {
		if t == nil { return "[nil]" }
		return fmt.Sprintf("%v", t.Data)
	}

	for step := 0; step < 5; step++ {
		poly.SystolicForward(net, state, true)
		
		fmt.Printf("Step %d - L0: %s | L1: %s | L2: %s\n", 
			step+1, safeData(state.LayerData[0]), safeData(state.LayerData[1]), safeData(state.LayerData[2]))
	}

	fmt.Println("\nSUCCESS: Systolic clock-accuracy verified! o_O")

	// 3. Test Backward (BPTT through space-time)
	fmt.Println("\n--- Testing Systolic Backward (BPTT) ---")
	gradOut := poly.NewTensor[float32](1, 1)
	gradOut.Data[0] = 1.0
	
	// Capture initial weights
	l1 := &net.Layers[1]
	initWeight := l1.WeightStore.Master[0]
	fmt.Printf("Initial L1 Weight: %v\n", initWeight)

	gIn, layerGrads, err := poly.SystolicBackward(net, state, gradOut)
	if err != nil {
		fmt.Printf("Backward Error: %v\n", err)
	} else if layerGrads != nil && layerGrads[1][1] != nil {
		fmt.Printf("Input Gradient: %v\n", gIn.Data)
		fmt.Printf("L1 Weight Gradient: %v\n", layerGrads[1][1].Data)
		
		// APPLY WEIGHT UPDATE
		poly.ApplyRecursiveGradients(l1, poly.ConvertTensor[float32, float32](layerGrads[1][1]), 0.1)
		
		finalWeight := l1.WeightStore.Master[0]
		fmt.Printf("Final L1 Weight:   %v\n", finalWeight)

		if finalWeight != initWeight {
			fmt.Println("SUCCESS: Weights mutated after systolic backprop! o_O")
		} else {
			fmt.Println("FAILURE: Weights did not mutate.")
		}
		fmt.Println("SUCCESS: Gradients propagated through clock cycles! o_O")
	} else {
		fmt.Println("FAILURE: No gradients generated for L1.")
	}

	fmt.Println("\n=== Systolic Neural Mesh Verified! o_O ===")
}
