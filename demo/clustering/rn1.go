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

	fmt.Println("║   RECURSION TEST:  K-Means Inside K-Means (100 Runs)           ║")

	fmt.Println("╚════════════════════════════════════════════════════════════════╝")

	numRuns := 100

	var recAccuracies []float64

	var compAccuracies []float64

	startTotal := time.Now()

	for run := 0; run < numRuns; run++ {

		// 1. Data Generation (Fresh per run)

		trainData := make([][]float32, 400)

		trainLabels := make([]int, 400)

		for i := 0; i < 400; i++ {

			group := i % 4

			x, y := float32(0), float32(0)

			switch group {

			case 0: // TL

				x, y = randFloat(0.1, 0.4), randFloat(0.6, 0.9)

			case 1: // TR

				x, y = randFloat(0.6, 0.9), randFloat(0.6, 0.9)

			case 2: // BL

				x, y = randFloat(0.1, 0.4), randFloat(0.1, 0.4)

			case 3: // BR

				x, y = randFloat(0.6, 0.9), randFloat(0.1, 0.4)

			}

			trainData[i] = []float32{x, y}

			trainLabels[i] = group

		}

		// Labels for Top vs Bottom

		binaryLabels := make([]int, 400)

		for i, l := range trainLabels {

			if l == 0 || l == 1 {

				binaryLabels[i] = 0 // Top

			} else {

				binaryLabels[i] = 1 // Bottom

			}

		}

		// 2. Recursive Network

		rawInputProcessor := nn.InitDenseLayer(2, 2, nn.ActivationTanh)

		innerKMeans := nn.InitKMeansLayer(4, rawInputProcessor, "probabilities")

		innerKMeans.KMeansLearningRate = 0.05

		innerKMeans.KMeansTemperature = 0.1

		outerKMeans := nn.InitKMeansLayer(2, innerKMeans, "probabilities")

		outerKMeans.KMeansLearningRate = 0.05

		outerKMeans.KMeansTemperature = 0.1

		head := nn.InitDenseLayer(2, 2, nn.ActivationSigmoid)

		recNet := nn.NewNetwork(2, 1, 1, 2)

		recNet.SetLayer(0, 0, 0, outerKMeans)

		recNet.SetLayer(0, 0, 1, head)

		recNet.InitializeWeights()

		// 3. Competitor Network

		compNet := nn.NewNetwork(2, 1, 1, 1)

		l1 := nn.InitDenseLayer(2, 12, nn.ActivationTanh)

		l2 := nn.InitDenseLayer(12, 12, nn.ActivationTanh)

		l3 := nn.InitDenseLayer(12, 2, nn.ActivationSigmoid)

		seqConfig := nn.InitSequentialLayer(l1, l2, l3)

		compNet.SetLayer(0, 0, 0, seqConfig)

		compNet.InitializeWeights()

		// 4. Train

		for epoch := 0; epoch < 50; epoch++ {

			indices := rand.Perm(400)

			for _, idx := range indices {

				// Recursive Train

				outRec, _ := recNet.ForwardCPU(trainData[idx])

				target := make([]float32, 2)

				target[binaryLabels[idx]] = 1.0

				gradRec := make([]float32, 2)

				for i := range gradRec {

					gradRec[i] = outRec[i] - target[i]

				}

				recNet.BackwardCPU(gradRec)

				recNet.ApplyGradients(0.05)

				// Competitor Train

				outComp, _ := compNet.ForwardCPU(trainData[idx])

				gradComp := make([]float32, 2)

				for i := range gradComp {

					gradComp[i] = outComp[i] - target[i]

				}

				compNet.BackwardCPU(gradComp)

				compNet.ApplyGradients(0.05)

			}

		}

		// 5. Evaluate Final Accuracy

		correctRec := 0

		correctComp := 0

		for i := 0; i < 400; i++ {

			outRec, _ := recNet.ForwardCPU(trainData[i])

			if argmax(outRec) == binaryLabels[i] {

				correctRec++

			}

			outComp, _ := compNet.ForwardCPU(trainData[i])

			if argmax(outComp) == binaryLabels[i] {

				correctComp++

			}

		}

		accRec := float64(correctRec) / 4.0 // /400 * 100

		accComp := float64(correctComp) / 4.0

		recAccuracies = append(recAccuracies, accRec)

		compAccuracies = append(compAccuracies, accComp)

		if (run+1)%10 == 0 {

			fmt.Printf("Run %d/%d completed...\n", run+1, numRuns)

		}

	}

	// Stats

	meanRec, stdRec := calculateStats(recAccuracies)

	meanComp, stdComp := calculateStats(compAccuracies)

	fmt.Printf("\nRESULTS (%d Runs):\n", numRuns)

	fmt.Println("--------------------------------------------------")

	fmt.Printf("Recursive Neuro-Symbolic: %.2f%% (±%.2f%%)\n", meanRec, stdRec)

	fmt.Printf("Standard Dense Network:   %.2f%% (±%.2f%%)\n", meanComp, stdComp)

	fmt.Println("--------------------------------------------------")

	fmt.Printf("Total Experiment Time: %v\n", time.Since(startTotal))

}

func calculateStats(data []float64) (mean, stdDev float64) {

	sum := 0.0

	for _, v := range data {

		sum += v

	}

	mean = sum / float64(len(data))

	sumSq := 0.0

	for _, v := range data {

		sumSq += math.Pow(v-mean, 2)

	}

	stdDev = math.Sqrt(sumSq / float64(len(data)))

	return

}

func randFloat(min, max float32) float32 {

	return min + rand.Float32()*(max-min)

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
