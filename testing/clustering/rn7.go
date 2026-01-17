package main

import (
	"fmt"
	"math"
	"math/rand"
	"time"

	"github.com/openfluke/loom/nn"
)

// EXPERIMENT RN7: The "Open-World" Adversary
// Scenario: Two 2D clusters (Squares at 0.2, Circles at 0.8)
// Adversaries:
//   - Ambiguity: (0.5, 0.5) - Exactly between classes
//   - Alien: (2.0, 2.0) - Geographically distant from training manifold
//
// Goal: Demonstrate that KMeans interpretive layers provide a "Geometric Truth"
// through distance-based Trust Scores, whereas standard MLPs provide overconfident
// Softmax probabilities for OOD (Alien) data.

const (
	RN7InputDim        = 2
	RN7NumClasses      = 2
	RN7NumSamples      = 800
	RN7LearningRate    = float32(0.15)
	RN7Epochs          = 120
	RN7CertaintyRadius = 0.25
)

type RN7TrainingMode int

const (
	RN7ModeNormalBP RN7TrainingMode = iota
	RN7ModeStepBP
	RN7ModeTween
	RN7ModeTweenChain
	RN7ModeStepTween
	RN7ModeStepTweenChain
	RN7ModeStandardDense
)

var rn7ModeNames = map[RN7TrainingMode]string{
	RN7ModeNormalBP:       "NormalBP",
	RN7ModeStepBP:         "StepBP",
	RN7ModeTween:          "Tween",
	RN7ModeTweenChain:     "TweenChain",
	RN7ModeStepTween:      "StepTween",
	RN7ModeStepTweenChain: "StepTweenChain",
	RN7ModeStandardDense:  "StandardDense",
}

func main() {
	rand.Seed(time.Now().UnixNano())

	fmt.Println("╔════════════════════════════════════════════════════════════════╗")
	fmt.Println("║   EXPERIMENT RN7:  The \"Open-World\" Adversary (Geometric)      ║")
	fmt.Println("╚════════════════════════════════════════════════════════════════╝")
	fmt.Printf("Scenario: Squares (0.2, 0.2) vs Circles (0.8, 0.8)\n")
	fmt.Printf("Adversaries: Ambiguity (0.5, 0.5) | Alien (2.0, 2.0)\n\n")

	// 1. Data Generation
	trainData, trainLabels := generateRN7Data(RN7NumSamples)

	modes := []RN7TrainingMode{
		RN7ModeNormalBP, RN7ModeStepBP,
		RN7ModeTween, RN7ModeTweenChain,
		RN7ModeStepTween, RN7ModeStepTweenChain,
		RN7ModeStandardDense,
	}

	for _, m := range modes {
		name := rn7ModeNames[m]
		fmt.Printf("🏃 Training %-16s ", name)

		// 2. Network Selection
		var net *nn.Network
		if m == RN7ModeStandardDense {
			net = nn.NewNetwork(2, 1, 1, 3)
			c1 := nn.InitDenseLayer(2, 32, nn.ActivationLeakyReLU)
			c2 := nn.InitDenseLayer(32, 2, nn.ActivationType(-1)) // Linear projection
			c3 := nn.InitSoftmaxLayer()
			net.SetLayer(0, 0, 0, c1)
			net.SetLayer(0, 0, 1, c2)
			net.SetLayer(0, 0, 2, c3)
		} else {
			// Loom Hybrid: KMeans(2) with 2->2 LeakyReLU SubNetwork
			sub := nn.InitDenseLayer(2, 2, nn.ActivationLeakyReLU)
			km := nn.InitKMeansLayer(2, sub, "probabilities")
			km.KMeansTemperature = 0.5 // Sharper separation

			head := nn.InitSoftmaxLayer()

			net = nn.NewNetwork(2, 1, 1, 2)
			net.SetLayer(0, 0, 0, km)
			net.SetLayer(0, 0, 1, head)
		}
		net.InitializeWeights()

		// 3. Train
		trainRN7(net, m, trainData, trainLabels)

		// 4. Quick Accuracy Check
		correct := 0
		for i := 0; i < len(trainData); i++ {
			out, _ := net.ForwardCPU(trainData[i])
			if rn7argmax(out) == trainLabels[i] {
				correct++
			}
		}
		acc := float64(correct) / float64(len(trainData))
		fmt.Printf("DONE (Acc: %.2f%%)\n", acc*100)

		// 5. Adversarial Audit
		performAudit(net, m)
	}

	fmt.Println("\n🏁 Experiment Complete.")
}

func rn7argmax(v []float32) int {
	mi, mv := 0, v[0]
	for i, val := range v {
		if val > mv {
			mi, mv = i, val
		}
	}
	return mi
}

func generateRN7Data(n int) ([][]float32, []int) {
	data := make([][]float32, n)
	labels := make([]int, n)
	noise := float32(0.08)

	for i := 0; i < n; i++ {
		class := rand.Intn(2)
		labels[i] = class
		x, y := float32(0.2), float32(0.2)
		if class == 1 {
			x, y = 0.8, 0.8
		}
		data[i] = []float32{
			x + (rand.Float32()*2-1)*noise,
			y + (rand.Float32()*2-1)*noise,
		}
	}
	return data, labels
}

func trainRN7(net *nn.Network, m RN7TrainingMode, data [][]float32, labels []int) {
	var ts *nn.TweenState
	if m >= RN7ModeTween && m != RN7ModeStandardDense {
		ts = nn.NewTweenState(net, nil)
		if m == RN7ModeTweenChain || m == RN7ModeStepTweenChain {
			ts.Config.UseChainRule = true
		}
	}

	state := net.InitStepState(RN7InputDim)
	for epoch := 0; epoch < RN7Epochs; epoch++ {
		indices := rand.Perm(len(data))
		for _, idx := range indices {
			input := data[idx]
			target := []float32{0, 0}
			target[labels[idx]] = 1.0

			switch m {
			case RN7ModeNormalBP, RN7ModeStepBP, RN7ModeStandardDense:
				state.SetInput(input)
				net.StepForward(state)
				output := state.GetOutput()
				grad := make([]float32, 2)
				for j := range grad {
					grad[j] = output[j] - target[j]
				}
				net.StepBackward(state, grad)
				net.ApplyGradients(RN7LearningRate)
			case RN7ModeTween, RN7ModeTweenChain, RN7ModeStepTween, RN7ModeStepTweenChain:
				// 1. Structural update (KMeans Internal transformation)
				ts.TweenStep(net, input, labels[idx], 2, RN7LearningRate)

				// 2. Behavioral update (KMeans Centers)
				state.SetInput(input)
				net.StepForward(state)
				output := state.GetOutput()
				grad := make([]float32, 2)
				for j := range grad {
					grad[j] = output[j] - target[j]
				}
				net.StepBackward(state, grad)
				net.ApplyGradients(RN7LearningRate)
			}
		}
	}
}

func performAudit(net *nn.Network, m RN7TrainingMode) {
	adversaries := []struct {
		Name  string
		Point []float32
	}{
		{"Square (Base)", []float32{0.2, 0.2}},
		{"Circle (Base)", []float32{0.8, 0.8}},
		{"Ambiguity (Mid)", []float32{0.5, 0.5}},
		{"Alien (Geog)", []float32{2.0, 2.0}},
	}

	fmt.Println("    ╠══════════════════╦═════════════╦══════════════╦═══════════════╦═════════════╣")
	fmt.Println("    ║ Input            ║ Class 0 Prob║ Class 1 Prob ║ Trust (Geom)  ║ MinDist     ║")
	fmt.Println("    ╠══════════════════╬═════════════╬══════════════╬═══════════════╬═════════════╣")

	for _, adv := range adversaries {
		output, _ := net.ForwardCPU(adv.Point)

		trust := float32(1.0)
		minDist := float32(0)
		if m != RN7ModeStandardDense {
			km := net.GetLayer(0, 0, 0)
			if km.Type == nn.LayerKMeans {
				minDist = float32(1e10)
				for i := 0; i < km.NumClusters; i++ {
					dist := float32(0)
					center := km.ClusterCenters[i*km.ClusterDim : (i+1)*km.ClusterDim]
					features, _ := km.SubNetwork.ForwardCPU(adv.Point)
					for j := range features {
						diff := features[j] - center[j]
						dist += diff * diff
					}
					dist = float32(math.Sqrt(float64(dist)))
					if dist < minDist {
						minDist = dist
					}
				}
				// Trust Score: 1.0 at 0 distance, 0.5 at RN7CertaintyRadius
				trust = float32(1.0 / (1.0 + math.Pow(float64(minDist/RN7CertaintyRadius), 4)))
			}
		} else {
			trust = 1.0
		}

		fmt.Printf("    ║ %-16s ║   %8.2f%% ║   %8.2f%% ║      %8.2f%% ║   %8.3f  ║\n",
			adv.Name, output[0]*100, output[1]*100, trust*100, minDist)
	}
	fmt.Println("    ╚══════════════════╩═════════════╩══════════════╩═══════════════╩═════════════╝")
}
