package main

import (
	"fmt"
	"math"
	"strings"
	"time"

	"github.com/openfluke/loom/nn"
)

// ═══════════════════════════════════════════════════════════════════════════════
// ⏳ THE CHRONOS PARADOX: A TEST OF IMPLICIT RECURRENCE
// ═══════════════════════════════════════════════════════════════════════════════
//
// CHALLENGE: A Feed-Forward Network usually has NO memory.
// But Loom's Step Tween Chain uses "Double Buffering" (Pipeline Execution).
// This implies that a signal takes exactly 1 step to move 1 layer.
//
// In a 12-Layer Deep Network, the output should reflect the input from 12 steps ago.
// If we CUT the sensor input (Occlusion), the "Ghost" of the object should
// persist inside the network for 12 steps before the output collapses.
//
// This proves "Object Permanence" (Subjective Time) without RNNs.
//
// ═══════════════════════════════════════════════════════════════════════════════

const (
	DeepLayers     = 12
	InputWindow    = 1
	StepInterval   = 20 * time.Millisecond // Slower tick to let us see it
	TrainingSteps  = 300
	OcclusionSteps = 20
)

func main() {
	fmt.Println("⏳ The Chronos Paradox: Initializing Object Permanence Test...")

	// 1. Build Deep Architecture (12 Layers)
	// Input -> Dense -> Dense ... -> Output
	// We use Identity activation (or close to it) to make the signal propagation clear?
	// Tanh is fine.
	net := nn.NewNetwork(InputWindow, 1, 1, 1)

	// Layer 0: Input (Size 1)
	net.SetLayer(0, 0, 0, nn.InitDenseLayer(1, 16, nn.ActivationTanh))

	// Layers 1-11: Deep Hidden Layers (The "Time Tunnel")
	// Note: Maximum layers per cell is usually limited. We need to check if 12 fits?
	// Usually Loom supports flexible depth. We'll use a sequential approach if needed.
	// But `SetLayer` targets indices. `NewNetwork` sets `LayersPerCell`?
	// Wait, `NewNetwork` takes `layersPerCell` in arg 4?
	// Checking signature: NewNetwork(inputSize, gridRows, gridCols, layersPerCell)
	// We need 12 layers + Input + Output? No, Total Layers.
	// Let's remake the network with enough capacity.

	// Re-initializing network with correct depth
	net = nn.NewNetwork(1, 1, 1, DeepLayers+2)

	net.SetLayer(0, 0, 0, nn.InitDenseLayer(1, 16, nn.ActivationLeakyReLU)) // Input Layer

	for i := 1; i <= DeepLayers; i++ {
		// Linear-ish layers to preserve signal fidelity for this demo
		net.SetLayer(0, 0, i, nn.InitDenseLayer(16, 16, nn.ActivationLeakyReLU))
	}

	// Output Layer
	net.SetLayer(0, 0, DeepLayers+1, nn.InitDenseLayer(16, 1, nn.ActivationType(-1))) // Linear Output

	net.InitializeWeights()

	// 2. Setup STC
	tweenConfig := &nn.TweenConfig{
		//UseChainRule:    true,
		FrontierEnabled: true,
		DenseRate:       0.5,
		Momentum:        0.0, // No momentum to ensure delay is purely structural, not weight-based
	}
	ts := nn.NewGenericTweenState[float32](net, tweenConfig)

	// 3. Loop
	inputBuffer := make([]float32, 1)

	fmt.Printf("\n%-8s | %-10s | %-10s | %-10s | %s\n", "Step", "State", "Input", "Output", "Status")
	fmt.Println(strings.Repeat("-", 65))

	startTime := time.Now()
	ticker := time.NewTicker(StepInterval)
	defer ticker.Stop()

	// Phase 1: Training (Sync Network)
	// We need to run enough steps to fill the pipeline first!
	// Step 0 input reaches output at Step 12.

	occluded := false
	occlusionStepStart := 0

	for step := 0; step < TrainingSteps+OcclusionSteps; step++ {
		<-ticker.C
		t := float64(step) * 0.2

		// Signal Gen: Simple Sine Wave
		realSignal := float32(0.5 + 0.4*math.Sin(t))

		// Input Handling
		var inputVal float32
		stateStr := "TRAINING"

		if step >= TrainingSteps {
			// OCCLUSION START
			occluded = true
			if occlusionStepStart == 0 {
				occlusionStepStart = step
				fmt.Println(">>> ✂️  SENSORS CUT! (Input = 0) ✂️")
			}
			inputVal = 0.0 // BLINDNESS
			stateStr = "OCCLUDED"
		} else {
			inputVal = realSignal
		}

		// Update Input
		inputBuffer[0] = inputVal
		inputT := nn.NewTensorFromSlice(inputBuffer, 1)

		// Forward
		predT := ts.ForwardPass(net, inputT, nil)
		prediction := predT.Data[0]

		// Target: We train to reproduce the CURRENT signal,
		// ignoring the implicit delay. The network will learn to predict/pass-through.
		target := realSignal

		// Backward
		if !occluded {
			ts.BackwardPassRegression(net, []float32{target})
			ts.TweenWeightsChainRule(net, 0.5) // High rate
		}

		// Analysis
		// We want to see if prediction PERSISTS (Memory) vs COLLAPSES (Amnesia)

		status := "✅ OK"

		if occluded {
			// Check against 0.0 (Collapse)
			if math.Abs(float64(prediction)) < 0.01 {
				status = "❌ COLLAPSED"
			} else {
				status = "👻 GHOST"
			}
		} else {
			// Training Check
			diff := prediction - realSignal
			loss := diff * diff
			if loss > 0.1 {
				status = "❌ LOST"
			}
		}

		stepRel := step
		if occluded {
			stepRel = step - occlusionStepStart
		}

		if occluded || step%20 == 0 {
			fmt.Printf("%-8d | %-10s | %-10.4f | %-10.4f | %s\n",
				stepRel, stateStr, inputVal, prediction, status)
		}

		// Termination check
		// We expect it to collapse around step 12.
		if occluded && status == "❌ COLLAPSED" {
			fmt.Printf("\n💥 SIGNAL COLLAPSE at Step +%d (Expected approx ~%d)\n", stepRel, DeepLayers)
			// Don't break immediately, let's see if it stays collapsed.
			// Actually we can stop after collapse to keep log short.
			if stepRel > 2 {
				break
			}
		}
	}

	fmt.Println("\n🏁 Chronos Paradox Test Complete.")
	elapsed := time.Since(startTime)
	fmt.Printf("Total Time: %s\n", elapsed)
}
