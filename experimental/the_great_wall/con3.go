package main

import (
	"fmt"
	"math"
	"math/rand"
	"sync"
	"time"

	"github.com/openfluke/loom/nn"
)

// CON2-B: The Depth Showdown
// Task: Track a rotating manifold through a 50-layer "Gradient Graveyard".
// Setup: Comparison of 6 training modes in parallel.

const (
	CON2InputDim     = 64
	CON2HiddenLayers = 50
	CON2OutputDim    = 2
	CON2LearningRate = 0.0001
	CON2TotalSteps   = 50 // Focused run
)

type ModeResult struct {
	Mode     string
	Accuracy float32
	Gap      float32
	Status   string
}

func main() {
	rand.Seed(time.Now().UnixNano())

	fmt.Println("╔════════════════════════════════════════════════════════════════╗")
	fmt.Println("║   EXPERIMENT CON2-B: THE DEPTH SHOWDOWN                        ║")
	fmt.Println("║   50 Layers | Rotating Manifold | Comparing All Modes          ║")
	fmt.Println("╚════════════════════════════════════════════════════════════════╝")

	modes := []string{"NormalBP", "StepBP", "Tween", "StepTweenChain"}
	var wg sync.WaitGroup
	results := make(chan ModeResult, len(modes))

	for _, mode := range modes {
		wg.Add(1)
		go func(m string) {
			defer wg.Done()
			results <- runMode(m)
		}(mode)
	}

	wg.Wait()
	close(results)

	fmt.Printf("\n%-18s | %-10s | %-10s | %s\n", "Mode", "Accuracy", "Avg Gap", "Final Verdict")
	fmt.Println("--------------------------------------------------------------------------------")
	for res := range results {
		fmt.Printf("%-18s | %-9.2f%% | %-10.4f | %s\n", res.Mode, res.Accuracy*100, res.Gap, res.Status)
	}
}

func runMode(mode string) ModeResult {
	net, ts, state := setupNet(mode)
	correct := 0
	totalGap := float32(0)

	for t := 0; t < CON2TotalSteps; t++ {
		input, target := generateManifoldData(t)

		var output []float32
		if state != nil {
			state.SetInput(input)
			for i := 0; i < net.TotalLayers(); i++ {
				net.StepForward(state)
			}
			output = state.GetOutput()
		} else {
			output, _ = net.ForwardCPU(input)
		}

		// Calculate accuracy
		pred := 0
		if output[1] > output[0] {
			pred = 1
		}
		if pred == target {
			correct++
		}

		// Apply Training
		applyTraining(mode, net, ts, state, input, target, output)

		if ts != nil {
			g, _, _ := ts.GetBudgetSummary()
			totalGap += g
		}
	}

	acc := float32(correct) / float32(CON2TotalSteps)
	verdict := "🔴 PLATEAU"
	if acc > 0.6 {
		verdict = "🟡 ADAPTING"
	}
	if acc > 0.8 {
		verdict = "🟢 BREAKING LINE"
	}

	return ModeResult{mode, acc, totalGap / float32(CON2TotalSteps), verdict}
}

func setupNet(mode string) (*nn.Network, *nn.TweenState, *nn.StepState) {
	net := nn.NewNetwork(CON2InputDim, 1, 1, CON2HiddenLayers+2)
	for i := 0; i <= CON2HiddenLayers+1; i++ {
		in := 32
		out := 32
		if i == 0 {
			in = CON2InputDim
		}
		if i == CON2HiddenLayers+1 {
			out = CON2OutputDim
		}
		net.SetLayer(0, 0, i, nn.InitDenseLayer(in, out, nn.ActivationLeakyReLU))
	}
	net.InitializeWeights()

	var ts *nn.TweenState
	var state *nn.StepState

	if mode == "StepTweenChain" || mode == "Tween" {
		ts = nn.NewTweenState(net, nil)
		ts.Config.UseChainRule = (mode == "StepTweenChain")
		//ts.Config.DepthScaleFactor = 120000.0 // Guerilla Boost
	}
	if mode == "StepBP" || mode == "StepTweenChain" {
		state = net.InitStepState(CON2InputDim)
	}

	return net, ts, state
}

func applyTraining(mode string, net *nn.Network, ts *nn.TweenState, state *nn.StepState, input []float32, target int, output []float32) {
	targetVec := []float32{0, 0}
	targetVec[target] = 1.0

	switch mode {
	case "NormalBP":
		batch := []nn.TrainingBatch{{Input: input, Target: targetVec}}
		net.Train(batch, &nn.TrainingConfig{Epochs: 1, LearningRate: CON2LearningRate})
	case "StepBP":
		grad := make([]float32, 2)
		grad[0] = output[0] - targetVec[0]
		grad[1] = output[1] - targetVec[1]
		net.StepBackward(state, grad)
		net.ApplyGradients(CON2LearningRate)
	case "Tween":
		ts.TweenStep(net, input, target, CON2OutputDim, CON2LearningRate)
	case "StepTweenChain":
		ts.TweenStep(net, input, target, CON2OutputDim, CON2LearningRate)
	}
}

func generateManifoldData(step int) ([]float32, int) {
	v := make([]float32, CON2InputDim)
	phase := float64(step) / 1000.0
	target := step % 2
	for i := range v {
		noise := math.Sin(float64(i) * phase)
		if target == 1 {
			v[i] = float32(rand.NormFloat64()*0.1 + 0.3 + noise)
		} else {
			v[i] = float32(rand.NormFloat64()*0.1 - 0.3 + noise)
		}
	}
	return v, target
}
