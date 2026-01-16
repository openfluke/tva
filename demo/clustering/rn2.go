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
	fmt.Println("║   EXPERIMENT RN2:  The Galaxy-Star Hierarchy (100 Runs)        ║")
	fmt.Println("╚════════════════════════════════════════════════════════════════╝")
	fmt.Println("Scenario: 3 Galaxies (Macro), each containing 5 Solar Systems (Micro).")
	fmt.Println("Task: Predict GALAXY ID (0, 1, or 2) from a point's coordinates.")
	fmt.Println("Logic: Hierarchical structure [Input -> Systems -> Galaxies].")

	numRuns := 100
	var recAccuracies []float64
	var compAccuracies []float64

	startTotal := time.Now()

	for run := 0; run < numRuns; run++ {
		// 1. Generate Hierarchical Data
		// Galaxies (Macro Centers)
		galaxies := [][]float32{
			{0.7, 0.7},  // Top Right
			{-0.7, 0.7}, // Top Left
			{0.0, -0.7}, // Bottom Center
		}

		// Solar Systems (Micro Centers)
		systems := make([][]float32, 15)
		systemLabels := make([]int, 15) // Galaxy ID for each system
		for g := 0; g < 3; g++ {
			for s := 0; s < 5; s++ {
				idx := g*5 + s
				// Micro centers scattered near the macro center
				r := 0.15 + rand.Float32()*0.15
				theta := rand.Float64() * 2 * math.Pi
				systems[idx] = []float32{
					galaxies[g][0] + r*float32(math.Cos(theta)),
					galaxies[g][1] + r*float32(math.Sin(theta)),
				}
				systemLabels[idx] = g
			}
		}

		// Generate Samples
		numSamples := 600
		trainData := make([][]float32, numSamples)
		trainLabels := make([]int, numSamples)
		for i := 0; i < numSamples; i++ {
			sysIdx := rand.Intn(15)
			center := systems[sysIdx]

			// Point noise within the solar system
			noise := float32(0.04)
			trainData[i] = []float32{
				center[0] + (rand.Float32()*2-1)*noise,
				center[1] + (rand.Float32()*2-1)*noise,
			}
			trainLabels[i] = systemLabels[sysIdx]
		}

		// 2. Recursive Hero Network
		// Input(2) -> [InnerKMeans(15, Output=Features)] -> [OuterKMeans(3, Output=Probs)] -> Head

		// Inner: Quantize to one of 15 system centers
		innerKMeans := nn.InitKMeansLayer(15, nn.InitDenseLayer(2, 2, nn.ActivationTanh), "features")
		innerKMeans.KMeansLearningRate = 0.05
		innerKMeans.KMeansTemperature = 0.01 // Very sharp

		// Outer: Quantize system centers to one of 3 galaxy centers
		outerKMeans := nn.InitKMeansLayer(3, innerKMeans, "probabilities")
		outerKMeans.KMeansLearningRate = 0.05
		outerKMeans.KMeansTemperature = 0.05

		head := nn.InitDenseLayer(3, 3, nn.ActivationSigmoid)

		recNet := nn.NewNetwork(2, 1, 1, 2)
		recNet.SetLayer(0, 0, 0, outerKMeans)
		recNet.SetLayer(0, 0, 1, head)
		recNet.InitializeWeights()

		// 3. Competitor Network: Standard Dense
		compNet := nn.NewNetwork(2, 1, 1, 1)
		c1 := nn.InitDenseLayer(2, 32, nn.ActivationTanh)
		c2 := nn.InitDenseLayer(32, 32, nn.ActivationTanh)
		c3 := nn.InitDenseLayer(32, 3, nn.ActivationSigmoid)
		compNet.SetLayer(0, 0, 0, nn.InitSequentialLayer(c1, c2, c3))
		compNet.InitializeWeights()

		// 4. Train
		for epoch := 0; epoch < 40; epoch++ {
			indices := rand.Perm(numSamples)
			for _, idx := range indices {
				target := make([]float32, 3)
				target[trainLabels[idx]] = 1.0

				// Train Recursive
				outRec, _ := recNet.ForwardCPU(trainData[idx])
				gradRec := make([]float32, 3)
				for j := range gradRec {
					gradRec[j] = outRec[j] - target[j]
				}
				recNet.BackwardCPU(gradRec)
				recNet.ApplyGradients(0.05)

				// Train Competitor
				outComp, _ := compNet.ForwardCPU(trainData[idx])
				gradComp := make([]float32, 3)
				for j := range gradComp {
					gradComp[j] = outComp[j] - target[j]
				}
				compNet.BackwardCPU(gradComp)
				compNet.ApplyGradients(0.05)
			}
		}

		// 5. Evaluate
		correctRec := 0
		correctComp := 0
		for i := 0; i < numSamples; i++ {
			outRec, _ := recNet.ForwardCPU(trainData[i])
			if argmax(outRec) == trainLabels[i] {
				correctRec++
			}

			outComp, _ := compNet.ForwardCPU(trainData[i])
			if argmax(outComp) == trainLabels[i] {
				correctComp++
			}
		}

		recAccuracies = append(recAccuracies, float64(correctRec)/float64(numSamples))
		compAccuracies = append(compAccuracies, float64(correctComp)/float64(numSamples))

		if (run+1)%10 == 0 {
			fmt.Printf("Run %d/100... Rec: %.1f%% | Comp: %.1f%%\n", run+1, recAccuracies[run]*100, compAccuracies[run]*100)
		}
	}

	meanRec, stdRec, maxRec, perfRec := calculateStats(recAccuracies)
	meanComp, stdComp, maxComp, perfComp := calculateStats(compAccuracies)

	fmt.Printf("\nFINAL RESULTS (%d Runs):\n", numRuns)
	fmt.Println("--------------------------------------------------")
	fmt.Printf("Recursive Neuro-Symbolic: Mean: %.2f%% (±%.2f%%) | Best: %.2f%% | Perfect: %d\n", meanRec*100, stdRec*100, maxRec*100, perfRec)
	fmt.Printf("Standard Deep Dense:      Mean: %.2f%% (±%.2f%%) | Best: %.2f%% | Perfect: %d\n", meanComp*100, stdComp*100, maxComp*100, perfComp)
	fmt.Println("--------------------------------------------------")
	fmt.Printf("Total Time: %v\n", time.Since(startTotal))
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
		if v >= 0.9999 { // 1.0 is 100%
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
