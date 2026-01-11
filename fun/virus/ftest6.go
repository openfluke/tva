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
// 🦠 THE VIRAL INJECTION: A TEST OF ADVERSARIAL RESISTANCE
// ═══════════════════════════════════════════════════════════════════════════════
//
// CHALLENGE: The network is fed a Sine Wave.
// BUT 10% of the time, we inject a "Virus" (Inverted Sine Wave).
//
// Standard NN behaviour: Learns the average (0.9*Sin + 0.1*(-Sin)), degrading accuracy.
// Loom Behaviour (Desired): Identifies Virus as "High Conflict" (Low Link Budget)
// and rejects the update via IgnoreThreshold.
//
// ═══════════════════════════════════════════════════════════════════════════════

const (
	InputWindow     = 32
	StepInterval    = 10 * time.Millisecond
	TotalSteps      = 2000 // 20 seconds
	VirusChance     = 0.10 // 10% Poison
	IgnoreThreshold = 0.4  // Reject updates with < 40% alignment
)

func main() {
	fmt.Println("🦠 The Viral Injection: Initializing Adversarial Resistance Test...")

	rand.Seed(time.Now().UnixNano())

	// 1. Architecture
	net := nn.NewNetwork(InputWindow, 1, 1, 1)
	net.SetLayer(0, 0, 0, nn.InitDenseLayer(InputWindow, 64, nn.ActivationTanh))
	net.SetLayer(0, 1, 0, nn.InitDenseLayer(64, 32, nn.ActivationLeakyReLU))
	net.SetLayer(0, 2, 0, nn.InitDenseLayer(32, 1, nn.ActivationType(-1)))
	net.InitializeWeights()

	// 2. Setup Step Tween Chain (Adversarial Config)
	tweenConfig := &nn.TweenConfig{
		FrontierEnabled:   true,
		FrontierThreshold: 0.5,
		IgnoreThreshold:   IgnoreThreshold, // THE SHIELD
		DenseRate:         0.05,
		Momentum:          0.2, // Low momentum to prevent poison from carrying over
	}
	ts := nn.NewGenericTweenState[float32](net, tweenConfig)

	// 3. Loop
	inputBuffer := make([]float32, InputWindow)

	fmt.Printf("\n%-10s | %-12s | %-10s | %-10s | %s\n", "Time", "Type", "Loss", "Budget", "Action")
	fmt.Println(strings.Repeat("-", 75))

	startTime := time.Now()
	ticker := time.NewTicker(StepInterval)
	defer ticker.Stop()

	cleanLossSum := 0.0
	cleanCount := 0

	poisonLossSum := 0.0
	poisonCount := 0

	rejectedCount := 0

	for step := 0; step < TotalSteps; step++ {
		<-ticker.C
		t := float64(step) * 0.1

		// Signal Gen
		cleanVal := math.Sin(t)
		isVirus := rand.Float64() < VirusChance

		var target float32
		if isVirus {
			target = float32(-cleanVal) // INVERTED (Poison)
		} else {
			target = float32(cleanVal)
		}

		// Update Input (Always show the 'truth' history, or the mixed history?
		// Realistically, the input history contains what was *seen*.
		// If we show mixed history, it's harder. Let's assume input is the noisy stream.)
		copy(inputBuffer, inputBuffer[1:])
		inputBuffer[InputWindow-1] = target

		// Train
		inputT := nn.NewTensorFromSlice(inputBuffer, InputWindow)

		// Forward
		predT := ts.ForwardPass(net, inputT, nil)
		prediction := predT.Data[0]
		loss := (target - prediction) * (target - prediction)

		// Backward
		ts.BackwardPassRegression(net, []float32{target})

		// Check Alignment (Link Budget)
		ts.CalculateLinkBudgets()
		// We scrutinize the Budget of the Output Layer (Index 2 -> LinkBudget[1]?)
		// LinkBudgets[i] is budget between Layer i and i+1.
		// TotalLayers=3. Indices 0, 1, 2.
		// LinkBudgets has size TotalLayers=3.
		// LinkBudgets[0]: Input->Hidden1
		// LinkBudgets[1]: Hidden1->Hidden2
		// LinkBudgets[2]: Hidden2->Output? No, usually size is TotalLayers.
		// Let's assume lowest budget in the chain determines rejection.

		minBudget := float32(1.0)
		for _, b := range ts.LinkBudgets {
			if b < minBudget && b > 0 { // b>0 check just in case
				minBudget = b
			}
		}

		action := "LEARN"
		if minBudget < IgnoreThreshold {
			action = "REJECT"
			rejectedCount++
		}

		// Update
		ts.TweenWeightsChainRule(net, 0.05)

		// Stats
		if isVirus {
			poisonLossSum += float64(loss)
			poisonCount++
		} else {
			cleanLossSum += float64(loss)
			cleanCount++
		}

		elapsed := time.Since(startTime)

		if step%50 == 0 || (isVirus && step%10 == 0) {
			typeStr := "CLEAN"
			if isVirus {
				typeStr = "VIRUS 🦠"
			}

			// Only print if interesting (Virus or occasional Clean)
			if isVirus || step%100 == 0 {
				fmt.Printf("%-10s | %-12s | %-10.4f | %-10.4f | %s\n",
					elapsed.Round(100*time.Millisecond).String(),
					typeStr,
					loss,
					minBudget,
					action)
			}
		}
	}

	fmt.Println("\n🏁 Viral Injection Test Complete.")
	fmt.Println("\n📊 REPORT")
	fmt.Println("========================================")
	avgClean := cleanLossSum / float64(cleanCount)
	avgPoison := poisonLossSum / float64(poisonCount)

	fmt.Printf("Avg Clean Loss:  %.4f (Should be Low)\n", avgClean)
	fmt.Printf("Avg Poison Loss: %.4f (Should be High)\n", avgPoison)
	fmt.Printf("Rejection Rate:  %d/%d (%.1f%%)\n", rejectedCount, TotalSteps, float64(rejectedCount)/float64(TotalSteps)*100)

	if avgClean < 0.05 && avgPoison > 0.5 {
		fmt.Println("\n✅ SUCCESS: Immunity Proven. Network ignored the virus.")
	} else {
		fmt.Println("\n❌ FAIL: Network got infected or didn't learn clean data.")
	}
}
