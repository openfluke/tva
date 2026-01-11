package main

import (
	"fmt"
	"math"
	"math/rand"
	"strings"
	"time"

	"github.com/openfluke/loom/nn"
)

// ═══════════════════════════════════════════════════════════════════════════════
// HARMONIC CHAMELEON: THE "IMPOSSIBLE" ADAPTATION BENCHMARK
// ═══════════════════════════════════════════════════════════════════════════════
//
// CHALLENGE: Predict a noisy time-series signal where the underlying rule
// shifts abruptly every 5 seconds. The network is NEVER notified of the shift.
//
// Rules:
//   1. Power Shift: $y = x^2$
//   2. Rectification: $y = |x|$
//   3. Frequency Fold: $y = \sin(5x)$
//   4. Binarization: $y = \text{sign}(x)$
//
// Logic:
//   If the Step Tween Chain can recover accuracy within 1s of an unannounced
//   rule shift, it demonstrates "Neural Fluid Dynamics" - the ability to
//   re-equilibrate logic in real-time.
// ═══════════════════════════════════════════════════════════════════════════════

const (
	InputWindow    = 16
	RuleDuration   = 5 * time.Second
	TotalSteps     = 3000 // Total iterations
	StepInterval   = 10 * time.Millisecond
	AccuracyWindow = 50 // Rolling window for accuracy calculation
)

type RuleType int

const (
	RulePower RuleType = iota
	RuleRectify
	RuleSineFold
	RuleBinarize
)

var ruleNames = map[RuleType]string{
	RulePower:    "Power Shift (x^2)",
	RuleRectify:  "Rectification (|x|)",
	RuleSineFold: "Frequency Fold (sin 5x)",
	RuleBinarize: "Binarization (sign x)",
}

func main() {
	fmt.Println("🦎 Harmonic Chameleon: Initializing Impossible Task...")

	// 1. Build a high-capacity architecture for rule-switching
	// We create a deep 3-layer grid.
	net := nn.NewNetwork(InputWindow, 1, 3, 1)

	// Customize layers for specific task
	// Layer 0: Large Dense
	net.SetLayer(0, 0, 0, nn.InitDenseLayer(InputWindow, 32, nn.ActivationScaledReLU))
	// Layer 1: Parallel/Ensemble-like (Simulated with Dense for now, or just a large hidden)
	net.SetLayer(0, 1, 0, nn.InitDenseLayer(32, 32, nn.ActivationLeakyReLU))
	// Layer 2: Output
	net.SetLayer(0, 2, 0, nn.InitDenseLayer(32, 1, nn.ActivationTanh))

	net.InitializeWeights()

	// 2. Setup Step Tween Chain State
	tweenConfig := &nn.TweenConfig{
		FrontierEnabled:   true,
		FrontierThreshold: 0.55,
		IgnoreThreshold:   0.01,
		DenseRate:         1.0,
	}
	ts := nn.NewGenericTweenState[float32](net, tweenConfig)

	// 2. Main Simulation Loop
	startTime := time.Now()
	inputBuffer := make([]float32, InputWindow)

	// Rolling accuracy tracker
	rollingErrors := make([]float32, 0, AccuracyWindow)

	fmt.Printf("\n%-12s | %-25s | %-10s | %-10s\n", "Time", "Active Rule", "Loss (MSE)", "Adaptation")
	fmt.Println(strings.Repeat("-", 70))

	for step := 0; step < TotalSteps; step++ {
		elapsed := time.Since(startTime)

		// Determine which rule is active
		ruleCycle := int(elapsed/RuleDuration) % 4
		activeRule := RuleType(ruleCycle)

		// Generate raw oscillating signal
		t := float64(step) * 0.1
		rawSignal := float32(math.Sin(t)) + (rand.Float32() * 0.02) // Add tiny noise

		// Shift window
		copy(inputBuffer, inputBuffer[1:])
		inputBuffer[InputWindow-1] = rawSignal

		// Calculate target based on active rule
		x := rawSignal
		var target float32
		switch activeRule {
		case RulePower:
			target = x * x
		case RuleRectify:
			target = float32(math.Abs(float64(x)))
		case RuleSineFold:
			target = float32(math.Sin(float64(x) * 5.0))
		case RuleBinarize:
			if x >= 0 {
				target = 1.0
			} else {
				target = -1.0
			}
		}

		// 3. Step Tween Chain: Live Adaptation
		inputT := nn.NewTensorFromSlice(inputBuffer, InputWindow)

		// ONE cycle of Forward -> Backward -> Weight Update
		predT := ts.ForwardPass(net, inputT, nil)
		prediction := predT.Data[0]

		// Calculate loss
		diff := target - prediction
		loss := diff * diff

		// Step Tween Chain Logic
		ts.BackwardPassRegression(net, []float32{target})
		ts.TweenWeightsChainRule(net, 0.05) // Fast adaptation rate

		// 4. Reporting
		rollingErrors = append(rollingErrors, loss)
		if len(rollingErrors) > AccuracyWindow {
			rollingErrors = rollingErrors[1:]
		}

		avgLoss := float32(0)
		for _, e := range rollingErrors {
			avgLoss += e
		}
		avgLoss /= float32(len(rollingErrors))

		if step%100 == 0 {
			adaptation := "STABLE"
			if avgLoss > 0.05 {
				adaptation = "ADAPTING..."
			}
			if avgLoss > 0.5 {
				adaptation = "SHIFTING!"
			}

			fmt.Printf("%-12s | %-25s | %-10.4f | %-10s\n",
				elapsed.Round(100*time.Millisecond).String(),
				ruleNames[activeRule],
				avgLoss,
				adaptation)
		}

		time.Sleep(StepInterval)
	}

	fmt.Println("\n🏁 Benchmark Complete.")
	finalLoss := float32(0)
	for _, e := range rollingErrors {
		finalLoss += e
	}
	fmt.Printf("Final Average Loss (Rolling): %.4f\n", finalLoss/float32(len(rollingErrors)))
}
