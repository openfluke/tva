package main

import (
	"fmt"
	"math"
	"math/rand"
	"time"

	"github.com/openfluke/loom/nn"
)

// RN9: Deep KMeans vs Deep Dense (Blind Hierarchy Discovery)
// Ground Truth Data: 12 Species -> 6 Phyla -> 2 Kingdoms
// Architecture 1 (Deep KMeans): Dense -> KMeans(24) -> KMeans(16) -> KMeans(8) -> Dense -> Dense
// Architecture 2 (Deep Dense): Dense -> Dense(24) -> Dense(16) -> Dense(8) -> Dense -> Dense
// Purpose: Can KMeans discover the structure even when given "too many" slots?

const (
	RN9InputDim     = 32
	RN9OutputDim    = 2
	RN9Samples      = 1000
	RN9Epochs       = 50
	RN9LearningRate = float32(0.05)
	RN9Runs         = 100 // Benchmarking runs

	// Hierarchy constants (Ground Truth)
	RN9Kingdoms    = 2
	RN9Phyla       = 3
	RN9Species     = 2
	RN9TotalLeaves = RN9Kingdoms * RN9Phyla * RN9Species // 12
)

func main() {
	rand.Seed(time.Now().UnixNano())

	fmt.Println("╔════════════════════════════════════════════════════════════════╗")
	fmt.Println("║   EXPERIMENT RN9: Blind Hierarchy Discovery Benchmark        ║")
	fmt.Println("║   Runs: 100 | Data: Hierarchical | Arch: Over-parameterized    ║")
	fmt.Println("╚════════════════════════════════════════════════════════════════╝")

	var kMeansResults []float64
	var denseResults []float64

	fmt.Printf("\nRunning %d iterations for Deep KMeans...\n", RN9Runs)
	for i := 0; i < RN9Runs; i++ {
		data, labels := generateHierarchicalData()
		net := buildDeepKMeansNet()
		acc := trainAndEvalRN9(net, data, labels, "DeepKMeansBlind", true)
		kMeansResults = append(kMeansResults, acc)
		if i%10 == 0 {
			fmt.Printf(".")
		}
	}
	fmt.Println(" Done.")

	fmt.Printf("\nRunning %d iterations for Deep Dense...\n", RN9Runs)
	for i := 0; i < RN9Runs; i++ {
		data, labels := generateHierarchicalData()
		net := buildDeepDenseNet()
		acc := trainAndEvalRN9(net, data, labels, "DeepDenseBlind", true)
		denseResults = append(denseResults, acc)
		if i%10 == 0 {
			fmt.Printf(".")
		}
	}
	fmt.Println(" Done.")

	printStats("Deep KMeans Blind", kMeansResults)
	printStats("Deep Dense Blind", denseResults)
}

func printStats(name string, results []float64) {
	var sum, minVal, maxVal float64
	minVal = 1.0
	maxVal = 0.0

	for _, v := range results {
		sum += v
		if v < minVal {
			minVal = v
		}
		if v > maxVal {
			maxVal = v
		}
	}
	mean := sum / float64(len(results))

	var variance float64
	for _, v := range results {
		variance += (v - mean) * (v - mean)
	}
	stdDev := float64(0)
	if len(results) > 1 {
		stdDev = math.Sqrt(variance / float64(len(results)-1))
	}

	fmt.Printf("\n=== RESULTS: %s ===\n", name)
	fmt.Printf("Mean Accuracy: %.2f%%\n", mean*100)
	fmt.Printf("Std Dev:       %.2f%%\n", stdDev*100)
	fmt.Printf("Min:           %.2f%%\n", minVal*100)
	fmt.Printf("Max:           %.2f%%\n", maxVal*100)
}

func buildDeepKMeansNet() *nn.Network {
	net := nn.NewNetwork(RN9InputDim, 1, 1, 3)

	// Block 1: The Stack
	// Dense 16 (Transform input)
	l1_conf := nn.InitDenseLayer(RN9InputDim, 16, nn.ActivationTanh)

	// KM 24 (Given 24 slots to find 12 real clusters)
	km1_conf := nn.InitKMeansLayer(24, l1_conf, "probabilities")
	km1_conf.KMeansLearningRate = RN9LearningRate
	km1_conf.KMeansTemperature = 0.1
	if km1_conf.SubNetwork != nil {
		if sn, ok := km1_conf.SubNetwork.(*nn.Network); ok {
			sn.BatchSize = 1
		}
	}

	// KM 16 (Given 16 slots to find 6 real clusters)
	km2_conf := nn.InitKMeansLayer(16, km1_conf, "probabilities")
	km2_conf.KMeansLearningRate = RN9LearningRate
	km2_conf.KMeansTemperature = 0.1
	if km2_conf.SubNetwork != nil {
		if sn, ok := km2_conf.SubNetwork.(*nn.Network); ok {
			sn.BatchSize = 1
		}
	}

	// KM 8 (Given 8 slots to find 2 real clusters - or maybe 2 Kingdoms are too simple, but finding the structure)
	km3_conf := nn.InitKMeansLayer(8, km2_conf, "probabilities")
	km3_conf.KMeansLearningRate = RN9LearningRate
	km3_conf.KMeansTemperature = 0.1
	if km3_conf.SubNetwork != nil {
		if sn, ok := km3_conf.SubNetwork.(*nn.Network); ok {
			sn.BatchSize = 1
		}
	}

	// Block 2: Dense
	l2_conf := nn.InitDenseLayer(8, 6, nn.ActivationTanh)

	// Block 3: Dense
	l3_conf := nn.InitDenseLayer(6, RN9OutputDim, nn.ActivationSigmoid)

	net.SetLayer(0, 0, 0, km3_conf)
	net.SetLayer(0, 0, 1, l2_conf)
	net.SetLayer(0, 0, 2, l3_conf)

	net.InitializeWeights()
	return net
}

func buildDeepDenseNet() *nn.Network {
	net := nn.NewNetwork(RN9InputDim, 1, 1, 6)

	// Dense -> Dense -> Dense -> Dense -> Dense -> Dense
	// Dims: 16 -> 24 -> 16 -> 8 -> 6 -> 2

	l1 := nn.InitDenseLayer(RN9InputDim, 16, nn.ActivationTanh)
	l2 := nn.InitDenseLayer(16, 24, nn.ActivationTanh)
	l3 := nn.InitDenseLayer(24, 16, nn.ActivationTanh)
	l4 := nn.InitDenseLayer(16, 8, nn.ActivationTanh)
	l5 := nn.InitDenseLayer(8, 6, nn.ActivationTanh)
	l6 := nn.InitDenseLayer(6, RN9OutputDim, nn.ActivationSigmoid)

	net.SetLayer(0, 0, 0, l1)
	net.SetLayer(0, 0, 1, l2)
	net.SetLayer(0, 0, 2, l3)
	net.SetLayer(0, 0, 3, l4)
	net.SetLayer(0, 0, 4, l5)
	net.SetLayer(0, 0, 5, l6)

	net.InitializeWeights()
	return net
}

// ... Wait, I can't easily add imports with `ReplaceFileContent` if they are far away.
// I'll use `multi_replace_file_content` to add import AND change main.
// Or just use `variance` for now? No, user asked for deviation.
// I'll add "math" to imports.

func trainAndEvalRN9(net *nn.Network, data [][]float32, labels []int, name string, quiet bool) float64 {
	// accPre := evaluate(net, data, labels)
	// if !quiet {
	// 	fmt.Printf("[%s] Pre-Train Acc: %.2f%%\n", name, accPre*100)
	// }

	net.BatchSize = 1

	config := &nn.TrainingConfig{
		Epochs:          RN9Epochs,
		LearningRate:    RN9LearningRate,
		UseGPU:          false,
		LossType:        "mse",
		PrintEveryBatch: 0,
		Verbose:         false,
	}

	net.TrainLabels(data, labels, config)

	accPost := evaluate(net, data, labels)
	if !quiet {
		fmt.Printf("[%s] Post-Train Acc: %.2f%%\n", name, accPost*100)
	}
	return accPost
}

func evaluate(net *nn.Network, data [][]float32, labels []int) float64 {
	correct := 0
	for i, input := range data {
		out, _ := net.ForwardCPU(input)
		if argmax(out) == labels[i] {
			correct++
		}
	}
	return float64(correct) / float64(len(data))
}

func argmax(v []float32) int {
	mi, mv := 0, v[0]
	for i, val := range v {
		if val > mv {
			mi, mv = i, val
		}
	}
	return mi
}

func generateHierarchicalData() ([][]float32, []int) {
	// Same generation as RN8 (12 leaves)
	kingdoms := make([][]float32, RN9Kingdoms)
	for k := 0; k < RN9Kingdoms; k++ {
		kingdoms[k] = randomVector(RN9InputDim, 1.0)
	}

	phyla := make([][][]float32, RN9Kingdoms)
	for k := 0; k < RN9Kingdoms; k++ {
		phyla[k] = make([][]float32, RN9Phyla)
		for p := 0; p < RN9Phyla; p++ {
			drift := randomVector(RN9InputDim, 0.5)
			phyla[k][p] = addVectors(kingdoms[k], drift)
		}
	}

	species := make([][][][]float32, RN9Kingdoms)
	for k := 0; k < RN9Kingdoms; k++ {
		species[k] = make([][][]float32, RN9Phyla)
		for p := 0; p < RN9Phyla; p++ {
			species[k][p] = make([][]float32, RN9Species)
			for s := 0; s < RN9Species; s++ {
				drift := randomVector(RN9InputDim, 0.2)
				species[k][p][s] = addVectors(phyla[k][p], drift)
			}
		}
	}

	data := make([][]float32, 0)
	labels := make([]int, 0)

	samplesPerSpecies := RN9Samples / RN9TotalLeaves

	for k := 0; k < RN9Kingdoms; k++ {
		for p := 0; p < RN9Phyla; p++ {
			for s := 0; s < RN9Species; s++ {
				proto := species[k][p][s]
				for i := 0; i < samplesPerSpecies; i++ {
					noise := randomVector(RN9InputDim, 0.1)
					vec := addVectors(proto, noise)
					data = append(data, vec)
					labels = append(labels, k)
				}
			}
		}
	}

	for len(data) < RN9Samples {
		data = append(data, randomVector(RN9InputDim, 1.0))
		labels = append(labels, 0)
	}

	perm := rand.Perm(len(data))
	shuffledData := make([][]float32, len(data))
	shuffledLabels := make([]int, len(labels))
	for i, idx := range perm {
		shuffledData[i] = data[idx]
		shuffledLabels[i] = labels[idx]
	}

	return shuffledData, shuffledLabels
}

func randomVector(dim int, scale float32) []float32 {
	v := make([]float32, dim)
	for i := 0; i < dim; i++ {
		v[i] = (rand.Float32()*2 - 1) * scale
	}
	return v
}

func addVectors(a, b []float32) []float32 {
	res := make([]float32, len(a))
	for i := range a {
		res[i] = a[i] + b[i]
	}
	return res
}
