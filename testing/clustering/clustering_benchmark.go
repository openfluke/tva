package main

import (
	"fmt"
	"math/rand"
	"sync"
	"time"

	"github.com/openfluke/loom/nn"
)

// Clustering Benchmark
//
// This benchmark tests the KMeansLayer's ability to discover clusters in synthetic data
// across different training modes.

const (
	InputDim       = 4
	NumClusters    = 4
	SamplesPerTask = 100
	NumTasks       = 10
	LearningRate   = float32(0.01)
	TestDuration   = 5 * time.Second
)

type TrainingMode int

const (
	ModeNormalBP TrainingMode = iota
	ModeStepBP
	ModeTween
	ModeTweenChain
	ModeStepTween
	ModeStepTweenChain
)

var modeNames = map[TrainingMode]string{
	ModeNormalBP:       "NormalBP",
	ModeStepBP:         "StepBP",
	ModeTween:          "Tween",
	ModeTweenChain:     "TweenChain",
	ModeStepTween:      "StepTween",
	ModeStepTweenChain: "StepTweenChain",
}

type ClusterSample struct {
	Input  []float32
	Target int // Cluster ID
}

func generateClusterData(numTasks int) [][]ClusterSample {
	tasks := make([][]ClusterSample, numTasks)
	for t := 0; t < numTasks; t++ {
		samples := make([]ClusterSample, SamplesPerTask)
		// Generate 4 clusters in 4D space
		centers := make([][]float32, 4)
		for i := 0; i < 4; i++ {
			centers[i] = make([]float32, InputDim)
			for d := 0; d < InputDim; d++ {
				centers[i][d] = rand.Float32()*2 - 1
			}
		}

		for i := 0; i < SamplesPerTask; i++ {
			cID := rand.Intn(4)
			input := make([]float32, InputDim)
			for d := 0; d < InputDim; d++ {
				input[d] = centers[cID][d] + (rand.Float32()*0.1 - 0.05)
			}
			samples[i] = ClusterSample{Input: input, Target: cID}
		}
		tasks[t] = samples
	}
	return tasks
}

func createKMeansNetwork() *nn.Network {
	net := nn.NewNetwork(InputDim, 1, 1, 2)
	net.BatchSize = 1

	// Feature extractor sub-network for KMeans
	featureLayer := nn.InitDenseLayer(InputDim, 8, nn.ActivationLeakyReLU)

	// KMeans Layer
	kmeansLayer := nn.InitKMeansLayer(NumClusters, featureLayer, "probabilities")
	net.SetLayer(0, 0, 0, kmeansLayer)

	// Output Classifier (KMeans Output -> Classes)
	outputLayer := nn.InitDenseLayer(NumClusters, NumClusters, nn.ActivationSigmoid)
	net.SetLayer(0, 0, 1, outputLayer)

	net.InitializeWeights()
	return net
}

type ModeResult struct {
	AverageAccuracy float32
	Throughput      float32
}

func runBenchmark(mode TrainingMode, tasks [][]ClusterSample) *ModeResult {
	net := createKMeansNetwork()
	opt := nn.NewSGDOptimizer()

	// Always initialize state for consistency in benchmark
	state := net.InitStepState(InputDim)

	var ts *nn.TweenState
	if mode >= ModeTween {
		ts = nn.NewTweenState(net, nil)
		if mode == ModeTweenChain || mode == ModeStepTweenChain {
			ts.Config.UseChainRule = true
		}
	}

	totalCorrect := 0
	totalSamples := 0
	start := time.Now()

	for time.Since(start) < TestDuration {
		taskIdx := rand.Intn(len(tasks))
		task := tasks[taskIdx]

		for _, sample := range task {
			var output []float32

			// Forward
			state.SetInput(sample.Input)
			net.StepForward(state)
			output = state.GetOutput()

			// Prediction
			pred := 0
			maxVal := float32(-1.0)
			for i, v := range output {
				if v > maxVal {
					maxVal = v
					pred = i
				}
			}
			if pred == sample.Target {
				totalCorrect++
			}
			totalSamples++

			// Train
			switch mode {
			case ModeNormalBP, ModeStepBP:
				// Manual gradient for StepBackward
				grad := make([]float32, NumClusters)
				for j := range grad {
					if j == sample.Target {
						grad[j] = 1.0 - output[j]
					} else {
						grad[j] = 0.0 - output[j]
					}
				}
				net.StepBackward(state, grad)
				opt.Step(net, LearningRate)
			case ModeTween, ModeTweenChain, ModeStepTween, ModeStepTweenChain:
				// Using float32 TweenState
				ts.TweenStep(net, sample.Input, sample.Target, NumClusters, LearningRate)
			}

			if time.Since(start) >= TestDuration {
				break
			}
		}
	}

	return &ModeResult{
		AverageAccuracy: float32(totalCorrect) / float32(totalSamples),
		Throughput:      float32(totalSamples) / float32(time.Since(start).Seconds()),
	}
}

func main() {
	fmt.Println("🚀 Starting Clustering Benchmark...")
	tasks := generateClusterData(NumTasks)

	modes := []TrainingMode{
		ModeNormalBP,
		ModeStepBP,
		ModeTween,
		ModeTweenChain,
		ModeStepTween,
		ModeStepTweenChain,
	}

	var wg sync.WaitGroup
	results := make(map[string]*ModeResult)
	var mu sync.Mutex

	for _, m := range modes {
		wg.Add(1)
		go func(mode TrainingMode) {
			defer wg.Done()
			name := modeNames[mode]
			res := runBenchmark(mode, tasks)
			mu.Lock()
			results[name] = res
			mu.Unlock()
			fmt.Printf("✅ [%s] Accuracy: %.2f%% | Throughput: %.0f samples/sec\n", name, res.AverageAccuracy*100, res.Throughput)
		}(m)
	}

	wg.Wait()
	fmt.Println("\n🏁 Benchmark Complete!")
}
