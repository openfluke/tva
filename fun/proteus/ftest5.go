package main

import (
	"fmt"
	"math"
	"strings"
	"time"

	"github.com/openfluke/loom/nn"
)

// ═══════════════════════════════════════════════════════════════════════════════
// THE PROTEUS SIGNAL: A TEST OF CONTINUOUS ADAPTATION
// ═══════════════════════════════════════════════════════════════════════════════
//
// CHALLENGE: The target signal "morphs" continuously. The network must adapt
// in real-time without stopping to retraining.
//
// Phases:
//   1. 0-5s:   Sine Wave
//   2. 5-10s:  Frequency Shift (2x Speed)
//   3. 10-15s: Square Wave
//   4. 15-20s: Composite (Sine + Square)
//
// ═══════════════════════════════════════════════════════════════════════════════

const (
	InputWindow         = 32
	PhaseDuration       = 5 * time.Second
	StepInterval        = 10 * time.Millisecond
	TotalPhases         = 4
	AdaptationThreshold = 0.05 // Loss must drop below this to be considered "LOCKED"
)

func main() {
	fmt.Println("🌊 The Proteus Signal: Initializing Continuous Adaptation Test...")

	// 1. Build Architecture
	// Standard Dense Network (Plasticity comes from the Engine, not special architecture)
	net := nn.NewNetwork(InputWindow, 1, 1, 1)

	// Layer 0: Input
	net.SetLayer(0, 0, 0, nn.InitDenseLayer(InputWindow, 64, nn.ActivationTanh))
	// Layer 1: Hidden (Wide for flexibility)
	net.SetLayer(0, 1, 0, nn.InitDenseLayer(64, 32, nn.ActivationLeakyReLU))
	// Layer 2: Output
	net.SetLayer(0, 2, 0, nn.InitDenseLayer(32, 1, nn.ActivationType(-1)))

	net.InitializeWeights()

	// 2. Setup Step Tween Chain (High Plasticity)
	tweenConfig := &nn.TweenConfig{
		FrontierEnabled:   true, // Enable Frontier Plasticity
		FrontierThreshold: 0.5,  // Aggressive frontier
		DenseRate:         0.2,  // High learning rate for rapid adaptation
		Momentum:          0.4,  // Lower momentum to allow quick direction changes
	}
	ts := nn.NewGenericTweenState[float32](net, tweenConfig)

	// 3. Main Loop
	inputBuffer := make([]float32, InputWindow)

	// Phase Info
	phaseNames := []string{"Sine Wave", "High Freq Sine", "Square Wave", "Composite Wave"}
	currentPhaseIdx := 0

	fmt.Printf("\n%-10s | %-16s | %-10s | %s\n", "Time", "Signal", "Loss", "Status")
	fmt.Println(strings.Repeat("-", 70))

	startTime := time.Now()
	phaseStart := startTime

	// Stats
	morphTime := startTime
	hasAdapted := false

	ticker := time.NewTicker(StepInterval)
	defer ticker.Stop()

	stepCount := 0

	for {
		now := time.Now()
		elapsed := now.Sub(startTime)
		phaseElapsed := now.Sub(phaseStart)

		// Check Phase Rotation
		if phaseElapsed >= PhaseDuration {
			currentPhaseIdx++
			if currentPhaseIdx >= len(phaseNames) {
				break
			}
			phaseStart = now
			morphTime = now
			hasAdapted = false
			fmt.Printf(">>> MORPH EVENT: %s\n", phaseNames[currentPhaseIdx])
		}

		<-ticker.C
		stepCount++
		t := float64(stepCount) * 0.1

		// 1. Generate Signal (Proteus)
		var target float32
		var val float64

		switch currentPhaseIdx {
		case 0: // Sine
			val = math.Sin(t)
		case 1: // High Freq
			val = math.Sin(t * 2.5)
		case 2: // Square
			s := math.Sin(t)
			if s > 0 {
				val = 0.8
			} else {
				val = -0.8
			}
		case 3: // Composite
			val = (math.Sin(t) + (math.Sin(t*3) * 0.5)) / 1.5
		}
		target = float32(val)

		// Context Window Update
		copy(inputBuffer, inputBuffer[1:])
		inputBuffer[InputWindow-1] = target

		// 2. Train (Step Tween Chain)
		inputT := nn.NewTensorFromSlice(inputBuffer, InputWindow)

		// Forward
		predT := ts.ForwardPass(net, inputT, nil)
		prediction := predT.Data[0]

		// Backward
		ts.BackwardPassRegression(net, []float32{target})

		// Update Weights (Online Learning)
		ts.TweenWeightsChainRule(net, 0.2) // High rate for adaptation

		loss := (target - prediction) * (target - prediction)

		// 3. Monitoring
		status := "ADAPTING..."
		if loss < AdaptationThreshold {
			status = "LOCKED"
			if !hasAdapted {
				hasAdapted = true
				latency := time.Since(morphTime)
				status = fmt.Sprintf("LOCKED (Latency: %s)", latency.Round(time.Millisecond))
				// Log the lock-on event clearly
				fmt.Printf("%-10s | %-16s | %-10.4f | %s\n",
					elapsed.Round(100*time.Millisecond).String(),
					phaseNames[currentPhaseIdx],
					loss,
					status)
			}
		}

		if stepCount%50 == 0 && !hasAdapted {
			fmt.Printf("%-10s | %-16s | %-10.4f | %s\n",
				elapsed.Round(100*time.Millisecond).String(),
				phaseNames[currentPhaseIdx],
				loss,
				status)
		}
	}

	fmt.Println("\n🏁 Proteus Signal Test Complete.")
	// Ideally we'd print summary stats, but the real-time log is the proof.
}
