package main

import (
	"fmt"
	"math"
	"math/rand"
	"sync"
	"time"

	"github.com/openfluke/loom/nn"
)

const (
	CON6InputDim     = 64
	CON6HiddenLayers = 50
	CON6OutputDim    = 2
	CON6LearningRate = 0.0005
	CON6TotalSteps   = 1500 // Faster iteration
	EvalSamples      = 200
)

type LayerModeKey struct {
	Layer nn.LayerType
	Mode  string
}

var layerTypeNames = map[nn.LayerType]string{
	nn.LayerDense:  "Dense",
	nn.LayerConv1D: "Conv1D",
	nn.LayerLSTM:   "LSTM",
	nn.LayerSwiGLU: "SwiGLU",
	nn.LayerKMeans: "KMeans-Anchor",
}

type ModeResult struct {
	Key          LayerModeKey
	FinalMetrics *nn.DeviationMetrics
	PreMetrics   *nn.DeviationMetrics
}

func main() {
	rand.Seed(time.Now().UnixNano())

	fmt.Println("╔══════════════════════════════════════════════════════════════════════════════════════╗")
	fmt.Println("║   🛡️  EXPERIMENT CON6-B: STABLE DEPTH AUDIT                                         ║")
	fmt.Println("║   50 Layers | 4 Stable Layer Types | 4 Modes | Breaking the 50% Barrier            ║")
	fmt.Println("╚══════════════════════════════════════════════════════════════════════════════════════╝")

	valInputs, valTargets := generateEvalSet(EvalSamples)

	modes := []string{"NormalBP", "StepBP", "Tween", "StepTweenChain"}
	layers := []nn.LayerType{
		nn.LayerDense,
		nn.LayerConv1D,
		nn.LayerLSTM,
		nn.LayerSwiGLU,
		nn.LayerKMeans,
	}

	results := make(chan ModeResult, len(modes)*len(layers))
	var wg sync.WaitGroup

	for _, l := range layers {
		for _, m := range modes {
			wg.Add(1)
			go func(layer nn.LayerType, mode string) {
				defer wg.Done()
				results <- runUniversalMode(layer, mode, valInputs, valTargets)
			}(l, m)
		}
	}

	wg.Wait()
	close(results)
	printUniversalTable(results)
}

func runUniversalMode(lType nn.LayerType, mode string, valIn [][]float32, valTar []float64) ModeResult {
	net, ts, state := setupUniversalNet(lType, mode)
	preMetrics, _ := net.EvaluateNetwork(valIn, valTar)

	for t := 0; t < CON6TotalSteps; t++ {
		input, target := generateManifoldData(t)
		var output []float32
		if state != nil {
			state.SetInput(input)
			// Safety check: ensure we don't over-step
			for i := 0; i < net.TotalLayers(); i++ {
				net.StepForward(state)
			}
			output = state.GetOutput()
		} else {
			output, _ = net.ForwardCPU(input)
		}

		if len(output) == CON6OutputDim {
			applyUniversalTraining(mode, net, ts, state, input, target, output)
		}
	}

	finalMetrics, _ := net.EvaluateNetwork(valIn, valTar)
	return ModeResult{Key: LayerModeKey{lType, mode}, FinalMetrics: finalMetrics, PreMetrics: preMetrics}
}

func setupUniversalNet(lType nn.LayerType, mode string) (*nn.Network, *nn.TweenState, *nn.StepState) {
	net := nn.NewNetwork(CON6InputDim, 1, 1, CON6HiddenLayers+2)

	for i := 0; i <= CON6HiddenLayers+1; i++ {
		in, out := 32, 32
		if i == 0 {
			in = CON6InputDim
		}
		if i == CON6HiddenLayers+1 {
			out = CON6OutputDim
		}

		var layer nn.LayerConfig
		if lType == nn.LayerKMeans && i == 25 {
			prev := nn.InitDenseLayer(32, 32, nn.ActivationLeakyReLU)
			layer = nn.InitKMeansLayer(16, prev, "probabilities")
		} else if lType == nn.LayerKMeans && i == 26 {
			layer = nn.InitDenseLayer(16, 32, nn.ActivationLeakyReLU)
		} else {
			switch lType {
			case nn.LayerConv1D:
				layer = nn.InitConv1DLayer(in, 1, 3, 1, 1, 1, nn.ActivationLeakyReLU)
				layer.OutputHeight = out
			case nn.LayerLSTM:
				layer = nn.InitLSTMLayer(in, out, 1, 1)
			case nn.LayerSwiGLU:
				layer = nn.InitSwiGLUBrain(in, 0.2)
				layer.OutputHeight = out
			default:
				layer = nn.InitDenseLayer(in, out, nn.ActivationLeakyReLU)
			}
		}
		net.SetLayer(0, 0, i, layer)
	}
	net.InitializeWeights()

	var ts *nn.TweenState
	var state *nn.StepState
	if mode == "StepTweenChain" || mode == "Tween" {
		ts = nn.NewTweenState(net, nil)
		ts.Config.UseChainRule = (mode == "StepTweenChain")
		ts.Config.DepthScaleFactor = 250.0
	}
	if mode == "StepBP" || mode == "StepTweenChain" {
		state = net.InitStepState(CON6InputDim)
	}
	return net, ts, state
}

func applyUniversalTraining(mode string, net *nn.Network, ts *nn.TweenState, state *nn.StepState, input []float32, target int, output []float32) {
	targetVec := []float32{0, 0}
	targetVec[target] = 1.0

	switch mode {
	case "NormalBP":
		net.Train([]nn.TrainingBatch{{Input: input, Target: targetVec}}, &nn.TrainingConfig{Epochs: 1, LearningRate: float32(CON6LearningRate)})
	case "StepBP":
		grad := []float32{output[0] - targetVec[0], output[1] - targetVec[1]}
		net.StepBackward(state, grad)
		net.ApplyGradients(float32(CON6LearningRate))
	case "Tween", "StepTweenChain":
		ts.TweenStep(net, input, target, CON6OutputDim, float32(CON6LearningRate))
	}
}

func generateEvalSet(samples int) ([][]float32, []float64) {
	inputs := make([][]float32, samples)
	targets := make([]float64, samples)
	for i := 0; i < samples; i++ {
		in, tar := generateManifoldData(i + 10000)
		inputs[i] = in
		targets[i] = float64(tar)
	}
	return inputs, targets
}

func generateManifoldData(step int) ([]float32, int) {
	v := make([]float32, CON6InputDim)
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

func printUniversalTable(results chan ModeResult) {
	fmt.Println("\n╔═══════════════════════════╦══════════════════╦══════════╦══════════╦══════════════╗")
	fmt.Printf("║ %-25s ║ %-16s ║ %-8s ║ %-8s ║ %-12s ║\n", "Layer Type", "Mode", "Pre Acc", "Post Acc", "Verdict")
	fmt.Println("╠═══════════════════════════╬══════════════════╬══════════╬══════════╬══════════════╣")

	for res := range results {
		verdict := "🔴 PLATEAU"
		if res.FinalMetrics != nil && res.PreMetrics != nil {
			if res.FinalMetrics.Accuracy > res.PreMetrics.Accuracy+0.01 {
				verdict = "🟢 ADAPTING"
			}
			fmt.Printf("║ %-25s ║ %-16s ║ %7.1f%% ║ %7.1f%% ║ %-12s ║\n",
				layerTypeNames[res.Key.Layer], res.Key.Mode,
				res.PreMetrics.Accuracy*100, res.FinalMetrics.Accuracy*100, verdict)
		}
	}
	fmt.Println("╚═══════════════════════════╩══════════════════╩══════════╩══════════╩══════════════╝")
}
