package main

import (
	"fmt"
	"math"
	"strings"
	"time"

	"github.com/openfluke/loom/nn"
)

// ═══════════════════════════════════════════════════════════════════════════════
// THE HYDRA MEMORY: A TEST OF CATASTROPHIC FORGETTING (SIMPLIFIED)
// ═══════════════════════════════════════════════════════════════════════════════
//
// CHALLENGE: Demonstrate that the network can "park" knowledge in dormant experts
// and recall it instantly, avoiding the "overwrite" problem of standard NNs.
//
// This version uses STANDARD SGD (no Step Tween Chain) to strictly verify
// the architectural capability of the MoE without engine noise.
// ═══════════════════════════════════════════════════════════════════════════════

const (
	InputWindow        = 16
	PhaseDuration      = 5 * time.Second
	StepInterval       = 10 * time.Millisecond
	TotalPhases        = 4
	StabilityThreshold = 0.05 // Relaxed threshold for faster convergence detection
	StabilityWindow    = 10
)

// Phase Info
type PhaseInfo struct {
	Name      string
	Converged bool
	ConvTime  time.Duration
}

func main() {
	fmt.Println("🐍 The Hydra Memory: Initializing Retention Test (Standard SGD)...")

	// Experts
	exp1 := nn.InitDenseLayer(InputWindow, 32, nn.ActivationTanh)       // Sin
	exp2 := nn.InitDenseLayer(InputWindow, 32, nn.ActivationScaledReLU) // Square
	exp3 := nn.InitDenseLayer(InputWindow, 32, nn.ActivationLeakyReLU)  // Sawtooth

	// MoE Layer
	moeLayer := nn.InitFilteredParallelLayer(
		[]nn.LayerConfig{exp1, exp2, exp3},
		InputWindow,
		nn.SoftmaxStandard,
		0.2,
	)

	net := nn.NewNetwork(InputWindow, 1, 3, 1)

	// Layer 0: Input encoding
	net.SetLayer(0, 0, 0, nn.InitDenseLayer(InputWindow, 32, nn.ActivationLeakyReLU))
	// Layer 1: The Hydra (MoE)
	net.SetLayer(0, 1, 0, moeLayer)
	// Layer 2: Output
	net.SetLayer(0, 2, 0, nn.InitDenseLayer(32, 1, nn.ActivationType(-1)))

	net.InitializeWeights()

	// Use Generic Step State just for Forward/Backward helpers, but NOT Tween logic
	tweenConfig := &nn.TweenConfig{DenseRate: 0.1} // Higher rate for SGD
	ts := nn.NewGenericTweenState[float32](net, tweenConfig)

	inputBuffer := make([]float32, InputWindow)

	phases := []PhaseInfo{
		{Name: "Phase A (Learn)"},
		{Name: "Phase B (Distract)"},
		{Name: "Phase C (Burial)"},
		{Name: "Phase A (Recall)"},
	}
	currentPhaseIdx := 0

	fmt.Printf("\n%-10s | %-16s | %-10s | %s\n", "Time", "Phase", "Loss", "Status")
	fmt.Println(strings.Repeat("-", 70))

	stableSteps := 0
	startTime := time.Now()
	phaseStart := startTime

	ticker := time.NewTicker(StepInterval)
	defer ticker.Stop()

	stepCount := 0

	for {
		now := time.Now()
		elapsed := now.Sub(startTime)
		phaseElapsed := now.Sub(phaseStart)

		if phaseElapsed >= PhaseDuration {
			if !phases[currentPhaseIdx].Converged {
				phases[currentPhaseIdx].ConvTime = PhaseDuration
			}
			currentPhaseIdx++
			if currentPhaseIdx >= len(phases) {
				break
			}
			phaseStart = now
			stableSteps = 0
			fmt.Printf(">>> SWITCHING TO %s\n", phases[currentPhaseIdx].Name)
			// Reset optimizer momentum? No, manual SGD has no momentum here.
		}

		<-ticker.C
		stepCount++
		t := float64(stepCount) * 0.1

		// Target
		var target float32
		val := math.Sin(t)

		switch currentPhaseIdx {
		case 0, 3: // Phase A
			target = float32(val)
		case 1: // Phase B
			target = float32(val * val)
		case 2: // Phase C
			saw := math.Mod(t, math.Pi) / math.Pi
			target = float32(saw*2 - 1)
		}

		copy(inputBuffer, inputBuffer[1:])
		inputBuffer[InputWindow-1] = float32(val)

		// 1.5 Manual Gating
		moeCfg := net.GetLayer(0, 1, 0)
		for i := range moeCfg.ParallelBranches {
			moeCfg.ParallelBranches[i].Frozen = true
		}
		switch currentPhaseIdx {
		case 0, 3:
			moeCfg.ParallelBranches[0].Frozen = false
		case 1:
			moeCfg.ParallelBranches[1].Frozen = false
		case 2:
			moeCfg.ParallelBranches[2].Frozen = false
		}

		// 2. Train
		inputT := nn.NewTensorFromSlice(inputBuffer, InputWindow)

		// Forward
		predT := ts.ForwardPass(net, inputT, nil)
		prediction := predT.Data[0]
		loss := (target - prediction) * (target - prediction)

		// Backward
		ts.BackwardPassRegression(net, []float32{target})

		// Update (Standard SGD)
		ts.TweenWeightsChainRule(net, 0.1) // Rate 0.1

		if loss < StabilityThreshold {
			stableSteps++
		} else {
			stableSteps = 0
		}

		if !phases[currentPhaseIdx].Converged && stableSteps >= StabilityWindow {
			phases[currentPhaseIdx].Converged = true
			phases[currentPhaseIdx].ConvTime = time.Since(phaseStart)
		}

		if stepCount%50 == 0 {
			status := "LEARNING..."
			if phases[currentPhaseIdx].Converged {
				status = "STABLE"
			}
			fmt.Printf("%-10s | %-16s | %-10.4f | %s\n",
				elapsed.Round(100*time.Millisecond).String(),
				phases[currentPhaseIdx].Name,
				loss,
				status)
		}
	}

	fmt.Println("\n🏁 Hydra Memory Test Complete.")
	fmt.Println("\n📊 RESULTS REPORT")
	fmt.Println("==================================================")
	fmt.Printf("%-20s | %-15s | %-10s\n", "Phase", "Convergence", "Speedup")
	fmt.Println("--------------------------------------------------")

	baseTime := phases[0].ConvTime
	for _, p := range phases {
		speedup := "1.0x"
		if p.ConvTime > 0 {
			ratio := float64(baseTime) / float64(p.ConvTime)
			speedup = fmt.Sprintf("%.1fx", ratio)
		}

		timeStr := p.ConvTime.Round(time.Millisecond).String()
		if !p.Converged {
			timeStr = "FAILED"
		}

		fmt.Printf("%-20s | %-15s | %s\n", p.Name, timeStr, speedup)
	}

	t1 := phases[0].ConvTime
	t2 := phases[3].ConvTime
	if phases[3].Converged && t2 < t1 {
		fmt.Println("\n✅ SUCCESS: Catastrophic Forgetting AVOIDED.")
		fmt.Println("   Recall was faster than Initial Learning.")
	} else {
		fmt.Println("\n❌ FAIL: Network did not demonstrate retention.")
	}
}
