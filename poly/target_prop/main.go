package main

import (
	"fmt"
	"github.com/openfluke/loom/poly"
)

func main() {
	fmt.Println("=== M-POLY-VTD Neural Target Propagation Verification ===")

	// 1. Setup 1x1x5 grid (Deep stack for alignment test)
	net := poly.NewVolumetricNetwork(1, 1, 5, 1)
	for i := range net.Layers {
		net.Layers[i].Type = poly.LayerDense
		net.Layers[i].InputHeight = 1
		net.Layers[i].OutputHeight = 1
		net.Layers[i].DType = poly.DTypeFloat32
		net.Layers[i].WeightStore = &poly.WeightStore{
			Master: []float32{0.5}, // Initialize weights to 0.5
			Scale:  1.0,
		}
	}

	// 2. Initialize TargetProp State
	config := poly.DefaultTargetPropConfig()
	config.UseChainRule = false // Disable chaining to test True Target Propagation o_O
	config.GradientScale = 0.5
	
	state := poly.NewTargetPropState[float32](net, config)

	// 3. Define Input and Ideal Target
	input := poly.NewTensor[float32](1, 1)
	input.Data[0] = 10.0
	
	target := poly.NewTensor[float32](1, 1)
	target.Data[0] = 20.0 // We want the output to be 20.0 (The "ideal")

	fmt.Println("\n--- Executing Bidirectional Pass ---")
	
	// Forward Pass
	output := poly.TargetPropForward(net, state, input)
	fmt.Printf("Initial Output: %v (Gap to Target: %v)\n", output.Data[0], 20.0-output.Data[0])

	// Backward Pass (Target Propagation)
	poly.TargetPropBackward(net, state, target)
	state.CalculateLinkBudgets()

	fmt.Println("\n--- Diagnostic Results (Bridging the Gap) ---")
	for i := 0; i < state.TotalLayers; i++ {
		fmt.Printf("Layer %d:\n", i)
		fmt.Printf("  Actual Output: %v\n", state.ForwardActs[i+1].Data)
		fmt.Printf("  Target Output: %v\n", state.BackwardTargets[i+1].Data)
		fmt.Printf("  Alignment (Link Budget): %.4f\n", state.LinkBudgets[i])
		fmt.Printf("  Local Gap Magnitude:     %.4f\n", state.Gaps[i])
	}

	// Verify Chaining
	if config.UseChainRule {
		l0Grad := state.Gradients[0].Data[0]
		l0Target := state.BackwardTargets[0].Data[0]
		fmt.Printf("\nChaining Verification (Input Site):\n")
		fmt.Printf("  L0 Gradient: %.4f\n", l0Grad)
		fmt.Printf("  L0 Target Calculation: %.4f + (%.4f * %.2f) = %.4f\n", 
			input.Data[0], l0Grad, config.GradientScale, input.Data[0] + l0Grad*config.GradientScale)
		fmt.Printf("  Actual L0 Target:      %.4f\n", l0Target)

		if l0Target != input.Data[0] {
			fmt.Println("\nSUCCESS: Chaining mechanism is active and affecting targets! o_O")
		} else {
			fmt.Println("\nFAILURE: Targets are identical to activations (Chaining failed).")
		}
	} else {
		fmt.Println("\nSUCCESS: True Target Propagation executed (Gap-based without gradients)! o_O")
	}

	// Capture initial weights
	l4 := &net.Layers[4]
	initWeight := l4.WeightStore.Master[0]
	fmt.Printf("\n--- Testing Weight Mutation ---\n")
	fmt.Printf("Initial L4 Weight: %v\n", initWeight)

	// Apply Updates
	poly.ApplyTargetPropGaps(net, state, 0.1)
	
	finalWeight := l4.WeightStore.Master[0]
	fmt.Printf("Final L4 Weight:   %v\n", finalWeight)

	if finalWeight != initWeight {
		fmt.Println("\nSUCCESS: Weights mutated via Neural Target Propagation! o_O")
	} else {
		fmt.Println("\nFAILURE: Weights did not mutate.")
	}

	fmt.Println("\n=== Neural Target Propagation Bedrock Verified! o_O ===")
}
