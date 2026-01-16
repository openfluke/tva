package main

import (
	"fmt"
	"math"
	"math/rand"
	"time"

	"github.com/openfluke/loom/nn"
)

func main() {
	rand.Seed(time.Now().UnixNano())

	fmt.Println("╔════════════════════════════════════════════════════════════════╗")
	fmt.Println("║   EXPERIMENT RN4: Spurious Correlation Defense (100 Runs)     ║")
	fmt.Println("╚════════════════════════════════════════════════════════════════╝")
	fmt.Println("Scenario: Training features include a 'shortcut' (spurious) dimension.")
	fmt.Println("Event:    At test time, the shortcut is broken (randomized).")
	fmt.Println("Goal:     Resist overfitting to shortcuts via geometric prototypes.")

	numRuns := 100
	var loomAccuracies []float64
	var compAccuracies []float64

	// Ground Truth Centers (3D Spatial)
	safeCenter := []float32{0.2, 0.2, 0.2}
	attackCenter := []float32{0.8, 0.8, 0.8}

	startTotal := time.Now()

	for run := 0; run < numRuns; run++ {
		// 1. Data Generation
		numTrain := 800
		numTest := 300

		// TRAIN DATA (Spurious dim is perfectly correlated)
		trainData := make([][]float32, numTrain)
		trainLabels := make([]int, numTrain)
		for i := 0; i < numTrain; i++ {
			class := rand.Intn(2)
			center := safeCenter
			spurious := float32(0.1) // Safe
			if class == 1 {
				center = attackCenter
				spurious = 0.9 // Attack
			}

			// Sample 4D (3 Real + 1 Spurious)
			sample := make([]float32, 4)
			for d := 0; d < 3; d++ {
				sample[d] = center[d] + (rand.Float32()*2-1)*0.15
			}
			sample[3] = spurious + (rand.Float32()*2-1)*0.05 // Tiny noise on shortcut

			trainData[i] = sample
			trainLabels[i] = class
		}

		// TEST DATA (Spurious dim is randomized - Uniform 0.0 to 1.0)
		testData := make([][]float32, numTest)
		testLabels := make([]int, numTest)
		for i := 0; i < numTest; i++ {
			class := i % 2
			center := safeCenter
			if class == 1 {
				center = attackCenter
			}

			sample := make([]float32, 4)
			for d := 0; d < 3; d++ {
				sample[d] = center[d] + (rand.Float32()*2-1)*0.15
			}
			sample[3] = rand.Float32() // RANDOMIZED SHORTCUT

			testData[i] = sample
			testLabels[i] = class
		}

		// 2. Models
		// Loom: Prototype-based (Recursive candidate)
		proj := nn.InitDenseLayer(4, 4, nn.ActivationTanh)
		loomKMeans := nn.InitKMeansLayer(2, proj, "probabilities")
		loomKMeans.KMeansTemperature = 0.4
		loomNet := nn.NewNetwork(4, 1, 1, 1)
		loomNet.SetLayer(0, 0, 0, loomKMeans)
		loomNet.InitializeWeights()

		// Competitor: Standard Deep Dense (Shortcut-prone)
		compNet := nn.NewNetwork(4, 1, 1, 1)
		c1 := nn.InitDenseLayer(4, 24, nn.ActivationTanh)
		c2 := nn.InitDenseLayer(24, 24, nn.ActivationTanh)
		c3 := nn.InitDenseLayer(24, 2, nn.ActivationSigmoid)
		compNet.SetLayer(0, 0, 0, nn.InitSequentialLayer(c1, c2, c3))
		compNet.InitializeWeights()

		// 3. Train
		lr := float32(0.05)
		epochs := 40
		for epoch := 0; epoch < epochs; epoch++ {
			indices := rand.Perm(numTrain)
			for _, idx := range indices {
				target := make([]float32, 2)
				target[trainLabels[idx]] = 1.0

				// Loom
				outLoom, _ := loomNet.ForwardCPU(trainData[idx])
				gradLoom := make([]float32, 2)
				for j := range gradLoom {
					gradLoom[j] = outLoom[j] - target[j]
				}
				loomNet.BackwardCPU(gradLoom)
				loomNet.ApplyGradients(lr)

				// Competitor
				outComp, _ := compNet.ForwardCPU(trainData[idx])
				gradComp := make([]float32, 2)
				for j := range gradComp {
					gradComp[j] = outComp[j] - target[j]
				}
				compNet.BackwardCPU(gradComp)
				compNet.ApplyGradients(lr)
			}
		}

		// 4. Evaluate on Test Set (where spurious feature is broken)
		correctLoom := 0
		correctComp := 0
		for i := 0; i < numTest; i++ {
			outLoom, _ := loomNet.ForwardCPU(testData[i])
			if argmax(outLoom) == testLabels[i] {
				correctLoom++
			}

			outComp, _ := compNet.ForwardCPU(testData[i])
			if argmax(outComp) == testLabels[i] {
				correctComp++
			}
		}

		loomAccuracies = append(loomAccuracies, float64(correctLoom)/float64(numTest))
		compAccuracies = append(compAccuracies, float64(correctComp)/float64(numTest))

		if (run+1)%10 == 0 {
			fmt.Printf("Run %d/100 completed...\n", run+1)
		}
	}

	meanLoom, stdLoom, maxLoom, perfLoom := calculateStats(loomAccuracies)
	meanComp, stdComp, maxComp, perfComp := calculateStats(compAccuracies)

	fmt.Printf("\nRESULTS ON BROKEN SHORTCUT TEST (%d Runs):\n", numRuns)
	fmt.Println("--------------------------------------------------")
	fmt.Printf("Loom (Prototype) Net: Mean: %.2f%% (±%.2f%%) | Best: %.2f%% | Perfect: %d\n", meanLoom*100, stdLoom*100, maxLoom*100, perfLoom)
	fmt.Printf("Standard Dense Net:   Mean: %.2f%% (±%.2f%%) | Best: %.2f%% | Perfect: %d\n", meanComp*100, stdComp*100, maxComp*100, perfComp)
	fmt.Println("--------------------------------------------------")
	fmt.Printf("Total Experiment Time: %v\n", time.Since(startTotal))

	if meanLoom > meanComp+0.1 {
		fmt.Println("\n🏆 CONCLUSION: Loom architecture resists spurious shortcuts.")
		fmt.Println("   The geometric clustering of true features is more stable")
		fmt.Println("   than the black-box memorization of easy (but brittle) signals.")
	}
}

func calculateStats(data []float64) (mean, stdDev, maxVal float64, countPerfect int) {
	sum := 0.0
	maxVal = 0.0
	countPerfect = 0
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

func argmax(v []float32) int {
	maxI := 0
	maxV := v[0]
	for i, val := range v {
		if val > maxV {
			maxV = val
			maxI = i
		}
	}
	return maxI
}
