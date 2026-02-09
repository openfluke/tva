package main

import (
	"fmt"
	"math"
	"math/rand"
	"strings"
	"time"

	"github.com/openfluke/loom/nn"
)

// CON1 FIXED v3: The "Unseen Threat" (Online Intrusion Detection)
//
// Fix: Used ActivationSigmoid for OutputDim=2 to avoid 'undefined' error.
// Logic: Neuron 0 = Normal Prob, Neuron 1 = Attack Prob.

const (
	InputDim     = 30
	HiddenDim    = 64
	NumClusters  = 32
	OutputDim    = 2
	LearningRate = 0.05
	TotalSamples = 5000
)

func main() {
	rand.Seed(time.Now().UnixNano())

	fmt.Println("╔════════════════════════════════════════════════════════════════╗")
	fmt.Println("║   EXPERIMENT CON1: The \"Unseen Threat\" (Cybersecurity)         ║")
	fmt.Println("║   Task: Detect Zero-Day Attacks without stopping to retrain.   ║")
	fmt.Println("╚════════════════════════════════════════════════════════════════╝")

	fmt.Println("⚠️  Using Synthetic KDD Stream (Guaranteed deterministic behavior)")

	// 2. Build The Online Brain
	fmt.Println("🧠 Initializing Self-Organizing Firewall...")
	brain, tweenState := buildOnlineBrain()

	// 3. Run The Stream
	fmt.Println("⚡ STREAM STARTED: Monitoring Packets...")
	runStream(brain, tweenState)
}

// --- The Brain ---

func buildOnlineBrain() (*nn.Network, *nn.TweenState) {
	// 1. Input -> Dense
	l1 := nn.InitDenseLayer(InputDim, HiddenDim, nn.ActivationTanh)
	l1.InputHeight = InputDim
	l1.OutputHeight = HiddenDim
	resetWeights(&l1)

	// 2. Self-Organizing Layer (KMeans)
	subNet := nn.NewNetwork(HiddenDim, 1, 1, 1)
	subNet.SetLayer(0, 0, 0, nn.InitDenseLayer(HiddenDim, HiddenDim, 0))

	km := nn.LayerConfig{
		Type:               nn.LayerKMeans,
		NumClusters:        NumClusters,
		ClusterDim:         HiddenDim,
		SubNetwork:         subNet,
		KMeansTemperature:  0.5,
		KMeansOutputMode:   "features",
		InputHeight:        HiddenDim,
		OutputHeight:       HiddenDim,
		ClusterCenters:     make([]float32, NumClusters*HiddenDim),
		KMeansLearningRate: 0.1,
	}
	for i := range km.ClusterCenters {
		km.ClusterCenters[i] = (rand.Float32()*2 - 1) * 0.1
	}

	// 3. Classifier (Readout) - FIXED: Using Sigmoid for compatibility
	lOut := nn.InitDenseLayer(HiddenDim, OutputDim, nn.ActivationSigmoid)
	lOut.InputHeight = HiddenDim
	lOut.OutputHeight = OutputDim
	resetWeights(&lOut)

	// Assemble
	net := nn.NewNetwork(InputDim, 1, 1, 3)
	net.SetLayer(0, 0, 0, l1)
	net.SetLayer(0, 0, 1, km)
	net.SetLayer(0, 0, 2, lOut)

	// Init StepTweenChain State
	ts := nn.NewTweenState(net, nil)
	ts.Config.UseChainRule = true

	return net, ts
}

// --- The Stream Simulation ---

func runStream(net *nn.Network, ts *nn.TweenState) {
	var (
		errors     []float32
		accuracies []float32
		windowSize = 100
		driftPoint = 2500
	)

	fmt.Printf("%-10s | %-15s | %-10s | %-10s | %-10s | %s\n", "Packet #", "Traffic Type", "Attack Prob", "Error", "Acc %", "Status")
	fmt.Println(strings.Repeat("-", 90))

	for t := 0; t < TotalSamples; t++ {
		var input []float32
		var targetClass int
		var label string

		if t < driftPoint {
			input = generateNormalPacket()
			targetClass = 0
			label = "Normal"
		} else {
			input = generateAttackPacket()
			targetClass = 1
			label = "ATTACK!"
		}

		// Predict
		predOut, _ := net.ForwardCPU(input)
		probAttack := predOut[1]
		probNormal := predOut[0]

		// Train (StepTweenChain)
		ts.TweenStep(net, input, targetClass, 2, LearningRate)

		// Stats
		var err float32
		if targetClass == 1 {
			err = 1.0 - probAttack
		} else {
			err = 1.0 - probNormal
		}
		errors = append(errors, err)

		predictedClass := 0
		if probAttack > probNormal {
			predictedClass = 1
		}

		acc := float32(0.0)
		if predictedClass == targetClass {
			acc = 1.0
		}
		accuracies = append(accuracies, acc)

		// Log
		if t%windowSize == 0 && t > 0 {
			sumErr := float32(0)
			for _, e := range errors[len(errors)-windowSize:] {
				sumErr += e
			}
			avgErr := sumErr / float32(windowSize)

			sumAcc := float32(0)
			for _, a := range accuracies[len(accuracies)-windowSize:] {
				sumAcc += a
			}
			avgAcc := (sumAcc / float32(windowSize)) * 100

			status := "🟢 Stable"
			if avgErr > 0.1 {
				status = "🟡 Learning"
			}
			if avgErr > 0.4 {
				status = "🔴 ADAPTING"
			}
			if t == driftPoint {
				status = "⚠️  INTRUSION STARTED!"
			}

			fmt.Printf("%-10d | %-15s | %-10.4f | %-10.4f | %-10.1f | %s\n",
				t, label, probAttack, avgErr, avgAcc, status)
		}
	}
}

// --- Synthetic Data ---

func generateNormalPacket() []float32 {
	v := make([]float32, InputDim)
	for i := range v {
		v[i] = float32(rand.NormFloat64() * 0.5)
	}
	return v
}

func generateAttackPacket() []float32 {
	v := make([]float32, InputDim)
	for i := range v {
		v[i] = float32(rand.NormFloat64()*0.1 + 2.0)
	}
	return v
}

// --- Utils ---

func resetWeights(l *nn.LayerConfig) {
	limit := float32(math.Sqrt(6.0 / float64(l.InputHeight+l.OutputHeight)))
	for i := range l.Kernel {
		l.Kernel[i] = (rand.Float32()*2 - 1) * limit
	}
}
