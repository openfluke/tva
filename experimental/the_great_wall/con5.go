package main

import (
	"fmt"
	"math"
	"math/rand"
	"sync"
	"time"

	"github.com/openfluke/loom/nn"
)

// CON5: The KMeans Anchor Experiment
// Logic: A 50-layer graveyard "Relay" using an unsupervised KMeans middle-layer.

const (
	CON5InputDim     = 64
	CON5HiddenLayers = 50
	CON5OutputDim    = 2
	CON5LearningRate = 0.0005
	CON5TotalSteps   = 2000
	EvalSamples      = 200
)

type ModeResult struct {
	Mode         string
	FinalMetrics *nn.DeviationMetrics
	PreMetrics   *nn.DeviationMetrics
}

func main() {
	rand.Seed(time.Now().UnixNano())

	fmt.Println("╔════════════════════════════════════════════════════════════════╗")
	fmt.Println("║   EXPERIMENT CON5: THE KMEANS ANCHOR (RELOADED)                ║")
	fmt.Println("║   50 Layers | LayerKMeans Anchor | Deviation Audit             ║")
	fmt.Println("╚════════════════════════════════════════════════════════════════╝")

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
			results <- runMode(m, valInputs, valTargets)
		}(mode)
	}

	wg.Wait()
	close(results)

	for res := range results {
		fmt.Printf("\n📊 CON5 AUDIT: [%s]\n", res.Mode)
		if res.PreMetrics != nil && res.FinalMetrics != nil {
			nn.PrintDeviationComparisonTable(res.Mode+" KMeans Bypass Analysis", res.PreMetrics, res.FinalMetrics)
		}
	}
}

func runMode(mode string, valIn [][]float32, valTar []float64) ModeResult {
	net, ts, state := setupNet(mode)
	preMetrics, _ := net.EvaluateNetwork(valIn, valTar)

	for t := 0; t < CON5TotalSteps; t++ {
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
		applyTraining(mode, net, ts, state, input, target, output)
	}

	finalMetrics, _ := net.EvaluateNetwork(valIn, valTar)
	return ModeResult{mode, finalMetrics, preMetrics}
}

func setupNet(mode string) (*nn.Network, *nn.TweenState, *nn.StepState) {
	net := nn.NewNetwork(CON5InputDim, 1, 1, CON5HiddenLayers+2)

	for i := 0; i <= CON5HiddenLayers+1; i++ {
		in, out := 32, 32
		if i == 0 {
			in = CON5InputDim
		}
		if i == CON5HiddenLayers+1 {
			out = CON5OutputDim
		}

		if i == 25 {
			// THE ANCHOR: Takes 32 -> Transforms to 16 Centroids
			prev_conf := nn.InitDenseLayer(32, 32, nn.ActivationLeakyReLU)
			km_conf := nn.InitKMeansLayer(16, prev_conf, "probabilities")
			km_conf.KMeansLearningRate = float32(CON5LearningRate)
			net.SetLayer(0, 0, i, km_conf)
		} else if i == 26 {
			// THE ADAPTER: Picks up the 16 centroids and re-expands to 32
			net.SetLayer(0, 0, i, nn.InitDenseLayer(16, 32, nn.ActivationLeakyReLU))
		} else {
			net.SetLayer(0, 0, i, nn.InitDenseLayer(in, out, nn.ActivationLeakyReLU))
		}
	}
	net.InitializeWeights()

	var ts *nn.TweenState
	var state *nn.StepState
	if mode == "StepTweenChain" || mode == "Tween" {
		ts = nn.NewTweenState(net, nil)
		ts.Config.UseChainRule = (mode == "StepTweenChain")
		ts.Config.DepthScaleFactor = 200.0
	}
	if mode == "StepBP" || mode == "StepTweenChain" {
		state = net.InitStepState(CON5InputDim)
	}
	return net, ts, state
}

func applyTraining(mode string, net *nn.Network, ts *nn.TweenState, state *nn.StepState, input []float32, target int, output []float32) {
	targetVec := []float32{0, 0}
	targetVec[target] = 1.0

	switch mode {
	case "NormalBP":
		net.Train([]nn.TrainingBatch{{Input: input, Target: targetVec}}, &nn.TrainingConfig{Epochs: 1, LearningRate: float32(CON5LearningRate)})
	case "StepBP":
		grad := []float32{output[0] - targetVec[0], output[1] - targetVec[1]}
		net.StepBackward(state, grad)
		net.ApplyGradients(float32(CON5LearningRate))
	case "Tween", "StepTweenChain":
		ts.TweenStep(net, input, target, CON5OutputDim, float32(CON5LearningRate))
	}
}

func generateManifoldData(step int) ([]float32, int) {
	v := make([]float32, CON5InputDim)
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
