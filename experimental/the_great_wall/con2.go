package main

import (
	"fmt"
	"math"
	"math/rand"
	"strings"
	"time"

	"github.com/openfluke/loom/nn"
)

// CON2: Breaking Scaling Laws
// Goal: Prove that StepTweenChain bypasses the "Depth Barrier" where Backprop plateaus.
// Setup: A 50-layer deep network task.
// Task: Continuous adaptation to a rotating "Hidden Manifold".

const (
	CON2InputDim     = 64
	CON2HiddenLayers = 50 // Deep enough to kill standard gradients
	CON2OutputDim    = 2
	CON2LearningRate = 0.05
	CON2TotalSteps   = 10000
)

func main() {
	rand.Seed(time.Now().UnixNano())

	fmt.Println("╔════════════════════════════════════════════════════════════════╗")
	fmt.Println("║   EXPERIMENT CON2: Breaking Neural Scaling Laws                ║")
	fmt.Println("║   Task: 50-Layer Deep Adaptation | Metric: Step Accuracy       ║")
	fmt.Println("╚════════════════════════════════════════════════════════════════╝")

	// 1. Build a Deep Bicameral Node
	fmt.Println("🧠 Initializing 50-Layer StepTween Node...")
	net, tweenState := buildDeepNode()

	// 2. Run the Scaling Challenge
	runScalingTest(net, tweenState)
}

func buildDeepNode() (*nn.Network, *nn.TweenState) {
	// Build a massive sequential stack: Input -> 50x Dense -> Output
	net := nn.NewNetwork(CON2InputDim, 1, 1, CON2HiddenLayers+2)

	// Input Layer
	net.SetLayer(0, 0, 0, nn.InitDenseLayer(CON2InputDim, 32, nn.ActivationTanh))

	// The "Wall" of Hidden Layers
	for i := 1; i <= CON2HiddenLayers; i++ {
		net.SetLayer(0, 0, i, nn.InitDenseLayer(32, 32, nn.ActivationTanh))
	}

	// Output Readout
	net.SetLayer(0, 0, CON2HiddenLayers+1, nn.InitDenseLayer(32, CON2OutputDim, nn.ActivationSigmoid))

	net.InitializeWeights()

	ts := nn.NewTweenState(net, nil)
	ts.Config.UseChainRule = true
	ts.Config.DepthScaleFactor = 1.1 // Boost deeper layers

	return net, ts
}

func runScalingTest(net *nn.Network, ts *nn.TweenState) {
	windowSize := 500
	var accuracies []float32

	fmt.Printf("%-10s | %-10s | %-10s | %-12s | %s\n", "Step", "Accuracy", "Avg Gap", "Link Budget", "Status")
	fmt.Println(strings.Repeat("-", 80))

	for t := 0; t < CON2TotalSteps; t++ {
		// Generate dynamic manifold data (Task shifts every 2000 steps)
		input, target := generateManifoldData(t)

		// Predict
		out, _ := net.ForwardCPU(input)
		pred := 0
		if out[1] > out[0] {
			pred = 1
		}

		// Step Acc
		acc := float32(0)
		if pred == target {
			acc = 1.0
		}
		accuracies = append(accuracies, acc)

		// Neural Tweening Step
		ts.TweenStep(net, input, target, CON2OutputDim, CON2LearningRate)

		// Logging
		if t%windowSize == 0 && t > 0 {
			avgAcc := calcAvg(accuracies[len(accuracies)-windowSize:])
			avgGap, _, _ := ts.GetBudgetSummary()
			_, _, maxBudget := ts.GetBudgetSummary()

			status := "🟢 BREAKING LINE"
			if avgAcc < 0.6 {
				status = "🟡 STRUGGLING"
			}
			if avgAcc < 0.51 {
				status = "🔴 PLATEAU"
			}

			fmt.Printf("%-10d | %-10.2f%% | %-10.4f | %-12.4f | %s\n",
				t, avgAcc*100, avgGap, maxBudget, status)
		}
	}
}

// Data generator for a "Rotating Manifold" that shifts over time
func generateManifoldData(step int) ([]float32, int) {
	v := make([]float32, CON2InputDim)
	phase := float64(step) / 2000.0 // Phase shifts every 2k steps

	target := 0
	if rand.Float64() > 0.5 {
		target = 1
	}

	for i := range v {
		// Add some non-linear harmonic noise based on phase
		noise := math.Sin(float64(i) * phase)
		if target == 1 {
			v[i] = float32(rand.NormFloat64()*0.2 + 0.5 + noise)
		} else {
			v[i] = float32(rand.NormFloat64()*0.2 - 0.5 + noise)
		}
	}
	return v, target
}

func calcAvg(vals []float32) float32 {
	sum := float32(0)
	for _, v := range vals {
		sum += v
	}
	return sum / float32(len(vals))
}
