package main

import (
	"fmt"
	"math"
	"strings"
	"time"

	"github.com/openfluke/loom/nn"
)

// ═══════════════════════════════════════════════════════════════════════════════
// THE PHOENIX RESILIENCE: A TEST OF NEURAL NEUROPLASTICITY
// ═══════════════════════════════════════════════════════════════════════════════
//
// CHALLENGE: Demonstrate biological-like resilience by simulating "brain damage"
// to a running network and observing how the Step Tween Chain recovers.
//
// Scenario:
//   1. Network trains on a complex multi-harmonic signal.
//   2. At 8 seconds, we simulate catastrophic injury by ZEROING Expert A.
//   3. We observe if the network can "re-route" intelligence and recover.
// ═══════════════════════════════════════════════════════════════════════════════

const (
	InputWindow    = 24
	InjuryTime     = 8 * time.Second
	TotalDuration  = 20 * time.Second
	StepInterval   = 10 * time.Millisecond
	AccuracyWindow = 50
)

func main() {
	fmt.Println("🦅 The Phoenix Resilience: Initializing Neuroplasticity Test...")

	// 1. Build a Parallel architecture (2 Experts)
	// Manual initialization since InitParallelLayer is not in the public API yet
	branchA := nn.InitDenseLayer(InputWindow, 32, nn.ActivationScaledReLU)
	branchB := nn.InitDenseLayer(InputWindow, 32, nn.ActivationScaledReLU)

	parallelLayer := nn.LayerConfig{
		Type:             nn.LayerParallel,
		CombineMode:      "add",
		ParallelBranches: []nn.LayerConfig{branchA, branchB},
	}

	net := nn.NewNetwork(InputWindow, 1, 3, 1)

	// Complex Parallel structure in the middle
	net.SetLayer(0, 0, 0, nn.InitDenseLayer(InputWindow, InputWindow, nn.ActivationScaledReLU))
	net.SetLayer(0, 1, 0, parallelLayer)
	net.SetLayer(0, 2, 0, nn.InitDenseLayer(32, 1, nn.ActivationTanh))

	net.InitializeWeights()

	// 2. Setup Step Tween Chain State
	tweenConfig := &nn.TweenConfig{
		FrontierEnabled:   true,
		FrontierThreshold: 0.5,
		IgnoreThreshold:   0.005,
		DenseRate:         1.0,
	}
	ts := nn.NewGenericTweenState[float32](net, tweenConfig)

	// 3. Main Simulation Loop
	startTime := time.Now()
	inputBuffer := make([]float32, InputWindow)
	rollingErrors := make([]float32, 0, AccuracyWindow)

	injured := false

	fmt.Printf("\n%-12s | %-15s | %-10s | %-12s | %-10s\n", "Time", "Expert Status", "Loss (MSE)", "Healing", "Expert A L2")
	fmt.Println(strings.Repeat("-", 75))

	ticker := time.NewTicker(StepInterval)
	defer ticker.Stop()

	stepCount := 0
	for elapsed := time.Since(startTime); elapsed < TotalDuration; elapsed = time.Since(startTime) {
		<-ticker.C
		stepCount++

		// Generate signal: Sin(x) + Cos(2x)
		t := float64(stepCount) * 0.1
		rawSignal := float32(math.Sin(t) + 0.5*math.Cos(t*2.0))

		copy(inputBuffer, inputBuffer[1:])
		inputBuffer[InputWindow-1] = rawSignal

		// Target is the signal one step ahead
		target := float32(math.Sin(t+0.1) + 0.5*math.Cos((t+0.1)*2.0))

		// 4. Simulate Injury
		if !injured && elapsed > InjuryTime {
			fmt.Println("\n💥 CRITICAL INJURY: Expert A flattened to zero!")
			injured = true

			// Find the Parallel layer and wipe branchA weights
			parallelCfg := net.GetLayer(0, 1, 0)
			for i := range parallelCfg.ParallelBranches[0].Kernel {
				parallelCfg.ParallelBranches[0].Kernel[i] = 0
			}
			for i := range parallelCfg.ParallelBranches[0].Bias {
				parallelCfg.ParallelBranches[0].Bias[i] = 0
			}
		}

		// 5. Training Step
		inputT := nn.NewTensorFromSlice(inputBuffer, InputWindow)
		predT := ts.ForwardPass(net, inputT, nil)
		prediction := predT.Data[0]

		loss := (target - prediction) * (target - prediction)
		ts.BackwardPassRegression(net, []float32{target})

		// Biological adaptation rate: High during shift
		adaptationRate := float32(0.05)
		if injured && elapsed < InjuryTime+2*time.Second {
			adaptationRate = 0.15 // Surge of plasticity during trauma
		}
		ts.TweenWeightsChainRule(net, adaptationRate)

		// 6. Metrics
		rollingErrors = append(rollingErrors, loss)
		if len(rollingErrors) > AccuracyWindow {
			rollingErrors = rollingErrors[1:]
		}

		avgLoss := float32(0)
		for _, e := range rollingErrors {
			avgLoss += e
		}
		avgLoss /= float32(len(rollingErrors))

		if stepCount%100 == 0 {
			status := "HEALTHY"
			if injured {
				status = "INJURED"
			}

			healing := "STABLE"
			if injured && avgLoss > 0.1 {
				healing = "HEALING..."
			}
			if injured && avgLoss < 0.05 {
				healing = "RECOVERED"
			}

			// Calculate L2 of Expert A to see it regrow
			l2 := float32(0)
			pCfg := net.GetLayer(0, 1, 0)
			for _, w := range pCfg.ParallelBranches[0].Kernel {
				l2 += w * w
			}

			fmt.Printf("%-12s | %-15s | %-10.4f | %-12s | %.4f\n",
				elapsed.Round(100*time.Millisecond).String(),
				status,
				avgLoss,
				healing,
				l2)
		}
	}

	fmt.Println("\n🏁 Phoenix Resilience Test Complete.")
}
