package main

import (
	"fmt"
	"math"
	"math/rand"
	"sync"
	"time"

	"github.com/openfluke/loom/nn"
)

// CON4: The Deviation Audit
// Goal: Use MNIST-style "Deviation Buckets" to see if StepTweenChain
// can bend the manifold through 50 layers where others fail.

const (
	CON2InputDim     = 64
	CON2HiddenLayers = 50
	CON2OutputDim    = 2
	CON2LearningRate = 0.1
	CON2TotalSteps   = 1000
	EvalSamples      = 200
)

type ModeResult struct {
	Mode         string
	FinalMetrics *nn.DeviationMetrics
	PreMetrics   *nn.DeviationMetrics
	Status       string
}

func main() {
	rand.Seed(time.Now().UnixNano())

	fmt.Println("╔════════════════════════════════════════════════════════════════╗")
	fmt.Println("║   EXPERIMENT CON4: THE FULL DEVIATION AUDIT                    ║")
	fmt.Println("║   50 Layers | Manifold Tracking | Deviation Bucket Analysis     ║")
	fmt.Println("╚════════════════════════════════════════════════════════════════╝")

	// 1. Prepare Validation Set (Rotating Manifold)
	valInputs := make([][]float32, EvalSamples)
	valTargets := make([]float64, EvalSamples)
	for i := 0; i < EvalSamples; i++ {
		input, target := generateManifoldData(i + 10000)
		valInputs[i] = input
		valTargets[i] = float64(target)
	}

	modes := []string{"NormalBP", "StepBP", "Tween", "StepTweenChain"}
	var wg sync.WaitGroup
	results := make(chan ModeResult, len(modes))

	for _, mode := range modes {
		wg.Add(1)
		go func(m string) {
			defer wg.Done()
			results <- runModeWithEval(m, valInputs, valTargets)
		}(mode)
	}

	wg.Wait()
	close(results)

	// 2. Output Detailed Comparison Tables
	for res := range results {
		fmt.Printf("\n📊 AUDIT REPORT: [%s]\n", res.Mode)
		if res.PreMetrics != nil && res.FinalMetrics != nil {
			// Using the logic from your provided eval source
			nn.PrintDeviationComparisonTable(res.Mode+" Scaling Analysis", res.PreMetrics, res.FinalMetrics)
		}
	}
}

func runModeWithEval(mode string, valIn [][]float32, valTar []float64) ModeResult {
	net, ts, state := setupNet(mode)

	// Pre-Training Eval
	preMetrics, _ := net.EvaluateNetwork(valIn, valTar)

	// Training Loop
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
			output, _ = net.Forward(input)
		}

		applyTraining(mode, net, ts, state, input, target, output)
	}

	// Post-Training Eval
	finalMetrics, _ := net.EvaluateNetwork(valIn, valTar)

	status := "🔴 PLATEAU"
	if finalMetrics.Accuracy > 0.6 {
		status = "🟢 ADAPTING"
	}

	return ModeResult{mode, finalMetrics, preMetrics, status}
}

func setupNet(mode string) (*nn.Network, *nn.TweenState, *nn.StepState) {
	net := nn.NewNetwork(CON2InputDim, 1, 1, CON2HiddenLayers+2)
	for i := 0; i <= CON2HiddenLayers+1; i++ {
		in, out := 32, 32
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
		ts.Config.DepthScaleFactor = 80.0 // Shouting through the grave
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
		net.Train([]nn.TrainingBatch{{Input: input, Target: targetVec}}, &nn.TrainingConfig{Epochs: 1, LearningRate: CON2LearningRate, LossType: "mse"})
	case "StepBP":
		grad := []float32{output[0] - targetVec[0], output[1] - targetVec[1]}
		net.StepBackward(state, grad)
		net.ApplyGradients(CON2LearningRate)
	case "Tween", "StepTweenChain":
		ts.TweenStep(net, input, target, CON2OutputDim, CON2LearningRate)
	}
}

func generateManifoldData(step int) ([]float32, int) {
	v := make([]float32, CON2InputDim)
	phase := float64(step) / 1000.0
	target := step % 2
	for i := range v {
		noise := math.Sin(float64(i) * phase)
		v[i] = float32(rand.NormFloat64()*0.1 + noise)
		if target == 1 {
			v[i] += 0.3
		} else {
			v[i] -= 0.3
		}
	}
	return v, target
}
