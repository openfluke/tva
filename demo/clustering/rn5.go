package main

import (
	"fmt"
	"math"
	"math/rand"
	"time"

	"github.com/openfluke/loom/nn"
)

// EXPERIMENT RN5: The Galaxy-Star Hierarchy (All Modes)
// Scenario: 3 Galaxies, each containing 2 Stars.
// Task: Predict Galaxy ID (0, 1, or 2) from point coordinates.

const (
	RN5InputDim       = 2
	RN5NumGalaxies    = 3
	RN5StarsPerGalaxy = 2
	RN5TotalStars     = RN5NumGalaxies * RN5StarsPerGalaxy // 6
	RN5NumSamples     = 400
	RN5NumRuns        = 100
	RN5LearningRate   = float32(0.1)
	RN5Epochs         = 40
)

type RN5TrainingMode int

const (
	RN5ModeNormalBP RN5TrainingMode = iota
	RN5ModeStepBP
	RN5ModeTween
	RN5ModeTweenChain
	RN5ModeStepTween
	RN5ModeStepTweenChain
	RN5ModeStandardDense
)

var rn5ModeNames = map[RN5TrainingMode]string{
	RN5ModeNormalBP:       "NormalBP",
	RN5ModeStepBP:         "StepBP",
	RN5ModeTween:          "Tween",
	RN5ModeTweenChain:     "TweenChain",
	RN5ModeStepTween:      "StepTween",
	RN5ModeStepTweenChain: "StepTweenChain",
	RN5ModeStandardDense:  "StandardDense",
}

func main() {
	rand.Seed(time.Now().UnixNano())

	fmt.Println("╔════════════════════════════════════════════════════════════════╗")
	fmt.Println("║   EXPERIMENT RN5:  The Galaxy-Star Hierarchy (All Modes)      ║")
	fmt.Println("╚════════════════════════════════════════════════════════════════╝")
	fmt.Printf("Scenario: %d Galaxies, %d Stars each | %d Runs per mode\n\n", RN5NumGalaxies, RN5StarsPerGalaxy, RN5NumRuns)

	modes := []RN5TrainingMode{
		RN5ModeNormalBP, RN5ModeStepBP,
		RN5ModeTween, RN5ModeTweenChain,
		RN5ModeStepTween, RN5ModeStepTweenChain,
		RN5ModeStandardDense,
	}

	for _, m := range modes {
		name := rn5ModeNames[m]
		fmt.Printf("🏃 Testing %-16s ", name)

		accuracies := make([]float64, RN5NumRuns)
		start := time.Now()

		for run := 0; run < RN5NumRuns; run++ {
			// 1. Data Generation
			galaxies := [][]float32{{0.6, 0.6}, {-0.6, 0.6}, {0.0, -0.6}}
			stars := make([][]float32, RN5TotalStars)
			starLabels := make([]int, RN5TotalStars)
			for g := 0; g < RN5NumGalaxies; g++ {
				for s := 0; s < RN5StarsPerGalaxy; s++ {
					idx := g*RN5StarsPerGalaxy + s
					r := float32(0.2)
					theta := rand.Float64() * 2 * math.Pi
					stars[idx] = []float32{
						galaxies[g][0] + r*float32(math.Cos(theta)),
						galaxies[g][1] + r*float32(math.Sin(theta)),
					}
					starLabels[idx] = g
				}
			}

			trainData := make([][]float32, RN5NumSamples)
			trainLabels := make([]int, RN5NumSamples)
			for i := 0; i < RN5NumSamples; i++ {
				sIdx := rand.Intn(RN5TotalStars)
				noise := float32(0.05)
				trainData[i] = []float32{
					stars[sIdx][0] + (rand.Float32()*2-1)*noise,
					stars[sIdx][1] + (rand.Float32()*2-1)*noise,
				}
				trainLabels[i] = starLabels[sIdx]
			}

			// 2. Network: Hybrid Recursive or Standard Dense
			var net *nn.Network
			if m == RN5ModeStandardDense {
				// Competitor: Standard MLP
				net = nn.NewNetwork(2, 1, 1, 1)
				c1 := nn.InitDenseLayer(2, 32, nn.ActivationTanh)
				c2 := nn.InitDenseLayer(32, 32, nn.ActivationTanh)
				c3 := nn.InitDenseLayer(32, 3, nn.ActivationSigmoid)
				net.SetLayer(0, 0, 0, nn.InitSequentialLayer(c1, c2, c3))
			} else {
				// Loom: Hybrid Recursive
				// Layer 1: Stars (6)
				l1Sub := nn.InitDenseLayer(2, 4, nn.ActivationTanh)
				l1KMeans := nn.InitKMeansLayer(RN5TotalStars, l1Sub, "features")
				l1KMeans.KMeansTemperature = 0.1

				// Layer 2: Galaxies (3)
				l2KMeans := nn.InitKMeansLayer(RN5NumGalaxies, l1KMeans, "probabilities")
				l2KMeans.KMeansTemperature = 0.1

				// Head
				head := nn.InitDenseLayer(3, 3, nn.ActivationSigmoid)

				net = nn.NewNetwork(2, 1, 1, 2)
				net.SetLayer(0, 0, 0, l2KMeans)
				net.SetLayer(0, 0, 1, head)
			}
			net.InitializeWeights()

			var ts *nn.TweenState
			if m >= RN5ModeTween && m != RN5ModeStandardDense {
				ts = nn.NewTweenState(net, nil)
				if m == RN5ModeTweenChain || m == RN5ModeStepTweenChain {
					ts.Config.UseChainRule = true
				}
			}

			// 3. Train
			state := net.InitStepState(RN5InputDim)
			for epoch := 0; epoch < RN5Epochs; epoch++ {
				indices := rand.Perm(RN5NumSamples)
				for _, idx := range indices {
					input := trainData[idx]
					target := make([]float32, 3)
					target[trainLabels[idx]] = 1.0

					state.SetInput(input)
					net.StepForward(state)
					output := state.GetOutput()

					switch m {
					case RN5ModeNormalBP, RN5ModeStepBP, RN5ModeStandardDense:
						grad := make([]float32, 3)
						for j := range grad {
							grad[j] = output[j] - target[j]
						}
						net.StepBackward(state, grad)
						net.ApplyGradients(RN5LearningRate)
					case RN5ModeTween, RN5ModeTweenChain, RN5ModeStepTween, RN5ModeStepTweenChain:
						ts.TweenStep(net, input, trainLabels[idx], 3, RN5LearningRate)

						// Hybrid Update for Centers
						grad := make([]float32, 3)
						for j := range grad {
							grad[j] = output[j] - target[j]
						}
						net.StepBackward(state, grad)
					}
				}
			}

			// 4. Eval
			correct := 0
			for i := 0; i < RN5NumSamples; i++ {
				out, _ := net.ForwardCPU(trainData[i])
				if rn5argmax(out) == trainLabels[i] {
					correct++
				}
			}
			accuracies[run] = float64(correct) / float64(RN5NumSamples)
			if (run+1)%20 == 0 {
				fmt.Print(".")
			}
		}

		mean, std, best, perfect := rn5CalculateStats(accuracies)
		fmt.Printf(" DONE | Mean: %.2f%% (±%.2f%%) | Best: %.2f%% | Perfect: %d | Time: %v\n",
			mean*100, std*100, best*100, perfect, time.Since(start).Truncate(time.Second))
	}
	fmt.Println("\n🏁 Experiment Complete.")
}

func rn5CalculateStats(data []float64) (mean, stdDev, maxVal float64, countPerfect int) {
	sum := 0.0
	maxVal = 0.0
	for _, v := range data {
		sum += v
		if v > maxVal {
			maxVal = v
		}
		if v >= 0.9999 {
			countPerfect++
		}
	}
	mean = sum / float64(len(data))
	sumSq := 0.0
	for _, v := range data {
		sumSq += math.Pow(v-mean, 2)
	}
	stdDev = math.Sqrt(sumSq / float64(len(data)))
	return
}

func rn5argmax(v []float32) int {
	mi, mv := 0, v[0]
	for i, val := range v {
		if val > mv {
			mi, mv = i, val
		}
	}
	return mi
}
