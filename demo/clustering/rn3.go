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
	fmt.Println("║   EXPERIMENT RN3:  The 'Zero-Day' Hunter (100 Runs)            ║")
	fmt.Println("╚════════════════════════════════════════════════════════════════╝")
	fmt.Println("Scenario: Network trained on Safe traffic & DDoS.")
	fmt.Println("Event:    A brand new 'Zero-Day' attack appears (Out-of-Distribution).")

	numRuns := 100
	var compFailRates []float64 // Percent of zero-days misclassified (confident hallucination)
	var loomSaveRates []float64 // Percent of zero-days correctly flagged as anomaly

	// Stable centers for the experiment
	safeCenter := []float32{0.1, 0.2, 0.1}
	ddosCenter := []float32{0.8, 0.9, 0.9}
	zeroDayCenter := []float32{0.5, 0.1, 0.9}

	startTotal := time.Now()

	for run := 0; run < numRuns; run++ {
		// 1. Data Generation
		numTrain := 800
		numTest := 300 // 100 of each class

		trainData := make([][]float32, 0)
		trainLabels := make([]int, 0)
		for i := 0; i < numTrain; i++ {
			class := rand.Intn(2)
			center := safeCenter
			if class == 1 {
				center = ddosCenter
			}
			trainData = append(trainData, addNoise(center, 0.15))
			trainLabels = append(trainLabels, class)
		}

		testData := make([][]float32, 0)
		testLabels := make([]int, 0)
		for i := 0; i < numTest; i++ {
			class := i % 3
			var center []float32
			switch class {
			case 0:
				center = safeCenter
			case 1:
				center = ddosCenter
			case 2:
				center = zeroDayCenter
			}
			testData = append(testData, addNoise(center, 0.15))
			testLabels = append(testLabels, class)
		}

		// 2. Models
		// Competitor
		compNet := nn.NewNetwork(3, 1, 1, 1)
		c1 := nn.InitDenseLayer(3, 16, nn.ActivationTanh)
		c2 := nn.InitDenseLayer(16, 16, nn.ActivationTanh)
		c3 := nn.InitDenseLayer(16, 2, nn.ActivationSigmoid)
		compNet.SetLayer(0, 0, 0, nn.InitSequentialLayer(c1, c2, c3))
		compNet.InitializeWeights()

		// Loom
		proj := nn.InitDenseLayer(3, 3, nn.ActivationTanh)
		loomKMeans := nn.InitKMeansLayer(2, proj, "probabilities")
		loomKMeans.KMeansTemperature = 0.5
		loomNet := nn.NewNetwork(3, 1, 1, 1)
		loomNet.SetLayer(0, 0, 0, loomKMeans)
		loomNet.InitializeWeights()

		// 3. Train
		trainLoop(compNet, trainData, trainLabels, 40, 2)
		trainLoop(loomNet, trainData, trainLabels, 40, 2)

		// 4. Evaluate Zero-Day (Class 2)
		compFailures := 0
		loomSaves := 0
		zeroDayCount := 0
		anomalyThreshold := float32(0.2)

		for i := 0; i < len(testData); i++ {
			if testLabels[i] != 2 {
				continue
			}
			zeroDayCount++
			input := testData[i]

			// Competitor
			outComp, _ := compNet.ForwardCPU(input)
			predComp := argmax(outComp)
			if outComp[predComp] > 0.9 {
				compFailures++
			}

			// Loom (Manual check of distance)
			_, _ = loomNet.ForwardCPU(input)
			layer := loomNet.GetLayer(0, 0, 0)
			features := layer.PreActivations // sub-network output
			centers := layer.ClusterCenters

			minDist := float32(1000.0)
			for k := 0; k < 2; k++ {
				offset := k * 3
				dist := float32(0.0)
				for d := 0; d < 3; d++ {
					diff := features[d] - centers[offset+d]
					dist += diff * diff
				}
				if dist < minDist {
					minDist = dist
				}
			}
			if minDist > anomalyThreshold {
				loomSaves++
			}
		}

		compFailRates = append(compFailRates, float64(compFailures)/float64(zeroDayCount))
		loomSaveRates = append(loomSaveRates, float64(loomSaves)/float64(zeroDayCount))

		if (run+1)%10 == 0 {
			fmt.Printf("Run %d/100 completed...\n", run+1)
		}
	}

	meanFail, stdFail, _, _ := calculateStats(compFailRates)
	meanSave, stdSave, _, perfSave := calculateStats(loomSaveRates)

	fmt.Printf("\nRESULTS ON ZERO-DAY DETECTION (%d Runs):\n", numRuns)
	fmt.Println("--------------------------------------------------")
	fmt.Printf("Standard Net Hallucinations (Wrongly Confident): %.2f%% (±%.2f%%)\n", meanFail*100, stdFail*100)
	fmt.Printf("Loom Net Anomaly Detections (Correctly Skeptical): %.2f%% (±%.2f%%)\n", meanSave*100, stdSave*100)
	fmt.Printf("Loom 'Perfect' Detection Runs: %d\n", perfSave)
	fmt.Println("--------------------------------------------------")
	fmt.Printf("Total Time: %v\n", time.Since(startTotal))
}

func addNoise(center []float32, noise float32) []float32 {
	out := make([]float32, len(center))
	for i := range center {
		out[i] = center[i] + (rand.Float32()*2-1)*noise
	}
	return out
}

func trainLoop(net *nn.Network, data [][]float32, labels []int, epochs int, numClasses int) {
	for epoch := 0; epoch < epochs; epoch++ {
		indices := rand.Perm(len(data))
		for _, idx := range indices {
			out, _ := net.ForwardCPU(data[idx])
			target := make([]float32, numClasses)
			target[labels[idx]] = 1.0
			grad := make([]float32, numClasses)
			for i := range grad {
				grad[i] = out[i] - target[i]
			}
			net.BackwardCPU(grad)
			net.ApplyGradients(0.05)
		}
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
