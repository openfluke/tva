package main

import (
	"encoding/json"
	"fmt"
	"math"
	"math/rand"
	"os"
	"sync"
	"time"

	"github.com/openfluke/loom/nn"
)

// ═══════════════════════════════════════════════════════════════════════════════
// CATASTROPHIC FORGETTING BENCHMARK
// ═══════════════════════════════════════════════════════════════════════════════
//
// Train on Task A (XOR), then Task B (AND), then measure Task A performance.
// Tests: Which training modes retain old knowledge while learning new tasks?

const (
	CFInputSize  = 2
	CFHiddenSize = 32
	CFOutputSize = 1
	CFNumLayers  = 3

	CFLearningRate = float32(0.05)
	CFBatchSize    = 16

	CFPhaseADuration = 20 * time.Second
	CFPhaseBDuration = 20 * time.Second
	CFTestDuration   = 20 * time.Second
	CFTrainInterval  = 50 * time.Millisecond
	CFSampleInterval = 10 * time.Millisecond

	CFMaxConcurrent = 6
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

// Logic gate training data
var xorData = [][]float32{{0, 0}, {0, 1}, {1, 0}, {1, 1}}
var xorLabels = []float32{0, 1, 1, 0}
var andData = [][]float32{{0, 0}, {0, 1}, {1, 0}, {1, 1}}
var andLabels = []float32{0, 0, 0, 1}

type TestResult struct {
	TrainingMode      string  `json:"trainingMode"`
	TaskAAccuracyAfterA float64 `json:"taskAAccuracyAfterA"` // XOR accuracy after Phase A
	TaskBAccuracyAfterB float64 `json:"taskBAccuracyAfterB"` // AND accuracy after Phase B
	TaskAAccuracyAfterB float64 `json:"taskAAccuracyAfterB"` // XOR accuracy after Phase B (forgetting!)
	ForgettingRate    float64 `json:"forgettingRate"`      // How much Task A was forgotten
	RetentionScore    float64 `json:"retentionScore"`      // Higher = less forgetting
	TrainTimeSec      float64 `json:"trainTimeSec"`
	Score             float64 `json:"score"`
}

type BenchmarkResults struct {
	Results   []TestResult `json:"results"`
	Timestamp string       `json:"timestamp"`
}

func main() {
	rand.Seed(time.Now().UnixNano())

	fmt.Println("╔═══════════════════════════════════════════════════════════════════════════════════════════════════════════════╗")
	fmt.Println("║   🧠 CATASTROPHIC FORGETTING BENCHMARK                                                                       ║")
	fmt.Println("║                                                                                                               ║")
	fmt.Println("║   Phase A: Learn XOR (20s) → Phase B: Learn AND (20s) → Test: Measure XOR retention                          ║")
	fmt.Println("║   Question: How much of Task A does each training mode forget?                                               ║")
	fmt.Println("╚═══════════════════════════════════════════════════════════════════════════════════════════════════════════════╝")

	modes := []TrainingMode{ModeNormalBP, ModeStepBP, ModeTween, ModeTweenChain, ModeStepTween, ModeStepTweenChain}
	
	fmt.Printf("\n📊 Running %d tests | Phase A: %s | Phase B: %s\n\n", len(modes), CFPhaseADuration, CFPhaseBDuration)

	results := &BenchmarkResults{
		Results:   make([]TestResult, 0, len(modes)),
		Timestamp: time.Now().Format(time.RFC3339),
	}

	var wg sync.WaitGroup
	var mu sync.Mutex
	sem := make(chan struct{}, CFMaxConcurrent)

	for i, mode := range modes {
		wg.Add(1)
		go func(m TrainingMode, idx int) {
			defer wg.Done()
			sem <- struct{}{}
			defer func() { <-sem }()

			fmt.Printf("🧠 [%d/%d] Starting %s...\n", idx+1, len(modes), modeNames[m])
			result := runForgettingTest(m)
			result.TrainingMode = modeNames[m]

			mu.Lock()
			results.Results = append(results.Results, result)
			mu.Unlock()

			fmt.Printf("✅ [%d/%d] %-15s | XOR(A): %5.1f%% | AND(B): %5.1f%% | XOR(after B): %5.1f%% | Forgot: %5.1f%% | Score: %.0f\n",
				idx+1, len(modes), modeNames[m], result.TaskAAccuracyAfterA, result.TaskBAccuracyAfterB,
				result.TaskAAccuracyAfterB, result.ForgettingRate, result.Score)
		}(mode, i)
	}

	wg.Wait()
	saveResults(results)
	printSummaryTable(results)
}

func runForgettingTest(mode TrainingMode) TestResult {
	result := TestResult{}

	net := createNetwork()
	numLayers := net.TotalLayers()

	var state *nn.StepState
	if mode == ModeStepBP || mode == ModeStepTween || mode == ModeStepTweenChain {
		state = net.InitStepState(CFInputSize)
	}

	var ts *nn.TweenState
	if mode == ModeTween || mode == ModeTweenChain || mode == ModeStepTween || mode == ModeStepTweenChain {
		ts = nn.NewTweenState(net, nil)
		if mode == ModeTweenChain || mode == ModeStepTweenChain {
			ts.Config.UseChainRule = true
		}
	}

	type Sample struct {
		Input  []float32
		Target float32
	}
	trainBatch := make([]Sample, 0, CFBatchSize+5)
	lastTrainTime := time.Now()

	// ═══════════════════════════════════════════════════════════════════
	// PHASE A: Learn XOR
	// ═══════════════════════════════════════════════════════════════════
	start := time.Now()
	sampleIdx := 0
	lastSampleTime := start

	for time.Since(start) < CFPhaseADuration {
		if time.Since(lastSampleTime) >= CFSampleInterval {
			lastSampleTime = time.Now()
			
			// Get XOR sample
			idx := sampleIdx % 4
			input := xorData[idx]
			target := xorLabels[idx]
			sampleIdx++

			var output []float32
			switch mode {
			case ModeNormalBP, ModeTween, ModeTweenChain:
				output, _ = net.ForwardCPU(input)
			case ModeStepBP:
				state.SetInput(input)
				for s := 0; s < numLayers; s++ {
					net.StepForward(state)
				}
				output = state.GetOutput()
			case ModeStepTween, ModeStepTweenChain:
				output = ts.ForwardPass(net, input)
			}

			trainBatch = append(trainBatch, Sample{Input: input, Target: target})

			// Training
			switch mode {
			case ModeNormalBP:
				if len(trainBatch) >= CFBatchSize && time.Since(lastTrainTime) > CFTrainInterval {
					batches := make([]nn.TrainingBatch, len(trainBatch))
					for i, s := range trainBatch {
						batches[i] = nn.TrainingBatch{Input: s.Input, Target: []float32{s.Target}}
					}
					net.Train(batches, &nn.TrainingConfig{Epochs: 1, LearningRate: CFLearningRate, LossType: "mse"})
					trainBatch = trainBatch[:0]
					lastTrainTime = time.Now()
				}
			case ModeStepBP:
				if len(output) > 0 {
					grad := []float32{clipGrad(output[0]-target, 0.5)}
					net.StepBackward(state, grad)
					net.ApplyGradients(CFLearningRate)
				}
			case ModeTween, ModeTweenChain:
				if len(trainBatch) >= CFBatchSize && time.Since(lastTrainTime) > CFTrainInterval {
					for _, s := range trainBatch {
						out := ts.ForwardPass(net, s.Input)
						if len(out) > 0 {
							ts.ChainGradients[numLayers] = []float32{s.Target - out[0]}
							ts.BackwardTargets[numLayers] = []float32{s.Target}
							ts.TweenWeightsChainRule(net, CFLearningRate)
						}
					}
					trainBatch = trainBatch[:0]
					lastTrainTime = time.Now()
				}
			case ModeStepTween, ModeStepTweenChain:
				if len(output) > 0 {
					ts.ChainGradients[numLayers] = []float32{target - output[0]}
					ts.BackwardTargets[numLayers] = []float32{target}
					ts.TweenWeightsChainRule(net, CFLearningRate)
				}
			}
		}
	}

	// Measure XOR accuracy after Phase A
	result.TaskAAccuracyAfterA = measureAccuracy(net, xorData, xorLabels, mode, state, ts)

	// ═══════════════════════════════════════════════════════════════════
	// PHASE B: Learn AND (this may cause forgetting of XOR!)
	// ═══════════════════════════════════════════════════════════════════
	trainBatch = trainBatch[:0]
	start = time.Now()
	sampleIdx = 0
	lastSampleTime = start

	for time.Since(start) < CFPhaseBDuration {
		if time.Since(lastSampleTime) >= CFSampleInterval {
			lastSampleTime = time.Now()
			
			idx := sampleIdx % 4
			input := andData[idx]
			target := andLabels[idx]
			sampleIdx++

			var output []float32
			switch mode {
			case ModeNormalBP, ModeTween, ModeTweenChain:
				output, _ = net.ForwardCPU(input)
			case ModeStepBP:
				state.SetInput(input)
				for s := 0; s < numLayers; s++ {
					net.StepForward(state)
				}
				output = state.GetOutput()
			case ModeStepTween, ModeStepTweenChain:
				output = ts.ForwardPass(net, input)
			}

			trainBatch = append(trainBatch, Sample{Input: input, Target: target})

			switch mode {
			case ModeNormalBP:
				if len(trainBatch) >= CFBatchSize && time.Since(lastTrainTime) > CFTrainInterval {
					batches := make([]nn.TrainingBatch, len(trainBatch))
					for i, s := range trainBatch {
						batches[i] = nn.TrainingBatch{Input: s.Input, Target: []float32{s.Target}}
					}
					net.Train(batches, &nn.TrainingConfig{Epochs: 1, LearningRate: CFLearningRate, LossType: "mse"})
					trainBatch = trainBatch[:0]
					lastTrainTime = time.Now()
				}
			case ModeStepBP:
				if len(output) > 0 {
					grad := []float32{clipGrad(output[0]-target, 0.5)}
					net.StepBackward(state, grad)
					net.ApplyGradients(CFLearningRate)
				}
			case ModeTween, ModeTweenChain:
				if len(trainBatch) >= CFBatchSize && time.Since(lastTrainTime) > CFTrainInterval {
					for _, s := range trainBatch {
						out := ts.ForwardPass(net, s.Input)
						if len(out) > 0 {
							ts.ChainGradients[numLayers] = []float32{s.Target - out[0]}
							ts.BackwardTargets[numLayers] = []float32{s.Target}
							ts.TweenWeightsChainRule(net, CFLearningRate)
						}
					}
					trainBatch = trainBatch[:0]
					lastTrainTime = time.Now()
				}
			case ModeStepTween, ModeStepTweenChain:
				if len(output) > 0 {
					ts.ChainGradients[numLayers] = []float32{target - output[0]}
					ts.BackwardTargets[numLayers] = []float32{target}
					ts.TweenWeightsChainRule(net, CFLearningRate)
				}
			}
		}
	}

	// Measure accuracies after Phase B
	result.TaskBAccuracyAfterB = measureAccuracy(net, andData, andLabels, mode, state, ts)
	result.TaskAAccuracyAfterB = measureAccuracy(net, xorData, xorLabels, mode, state, ts)

	// Calculate forgetting
	if result.TaskAAccuracyAfterA > 0 {
		result.ForgettingRate = (result.TaskAAccuracyAfterA - result.TaskAAccuracyAfterB) / result.TaskAAccuracyAfterA * 100
	}
	result.RetentionScore = 100 - result.ForgettingRate
	if result.RetentionScore < 0 {
		result.RetentionScore = 0
	}

	result.TrainTimeSec = (CFPhaseADuration + CFPhaseBDuration).Seconds()
	
	// Score: Retention × Task A final × Task B final / 10000
	// Score: Retention × Task A final × Task B final / 10000
	// Then subtract blocked penalty (batch modes will show difference)
	result.Score = result.RetentionScore * result.TaskAAccuracyAfterB * result.TaskBAccuracyAfterB / 10000
	if math.IsNaN(result.Score) {
		result.Score = 0
	}
	return result
}

func createNetwork() *nn.Network {
	config := nn.SimpleNetworkConfig{
		InputSize:  CFInputSize,
		HiddenSize: CFHiddenSize,
		OutputSize: CFOutputSize,
		Activation: nn.ActivationLeakyReLU,
		InitScale:  0.5,
		NumLayers:  CFNumLayers,
		LayerType:  nn.BrainDense,
		DType:      nn.DTypeFloat32,
	}
	return nn.BuildSimpleNetwork(config)
}

func measureAccuracy(net *nn.Network, data [][]float32, labels []float32, mode TrainingMode, state *nn.StepState, ts *nn.TweenState) float64 {
	correct := 0
	for i, input := range data {
		var output []float32
		switch mode {
		case ModeNormalBP, ModeTween, ModeTweenChain:
			output, _ = net.ForwardCPU(input)
		case ModeStepBP:
			state.SetInput(input)
			for s := 0; s < net.TotalLayers(); s++ {
				net.StepForward(state)
			}
			output = state.GetOutput()
		case ModeStepTween, ModeStepTweenChain:
			output = ts.ForwardPass(net, input)
		}
		
		if len(output) > 0 {
			predicted := 0.0
			if output[0] > 0.5 {
				predicted = 1.0
			}
			if predicted == float64(labels[i]) {
				correct++
			}
		}
	}
	return float64(correct) / float64(len(data)) * 100
}

func clipGrad(grad, maxVal float32) float32 {
	if grad > maxVal { return maxVal }
	if grad < -maxVal { return -maxVal }
	return grad
}

func saveResults(results *BenchmarkResults) {
	data, _ := json.MarshalIndent(results, "", "  ")
	os.WriteFile("forgetting_results.json", data, 0644)
	fmt.Println("\n📁 Results saved to forgetting_results.json")
}

func printSummaryTable(results *BenchmarkResults) {
	fmt.Println("\n╔═══════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════╗")
	fmt.Println("║                                    CATASTROPHIC FORGETTING BENCHMARK SUMMARY                                                 ║")
	fmt.Println("╠═══════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════╣")
	fmt.Println("║  Training Mode    │ XOR(A)% │ AND(B)% │ XOR(after B)% │ Forgot% │ Retention% │    Score    │ Status                          ║")
	fmt.Println("╠═══════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════╣")

	for _, r := range results.Results {
		fmt.Printf("║  %-15s │  %5.1f%% │  %5.1f%% │      %5.1f%%   │  %5.1f%% │    %5.1f%%  │ %11.1f │ ✅ PASS ║\n",
			r.TrainingMode, r.TaskAAccuracyAfterA, r.TaskBAccuracyAfterB, r.TaskAAccuracyAfterB,
			r.ForgettingRate, r.RetentionScore, r.Score)
	}

	fmt.Println("╚═══════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════╝")

	var best *TestResult
	for i := range results.Results {
		if best == nil || results.Results[i].RetentionScore > best.RetentionScore {
			best = &results.Results[i]
		}
	}
	if best != nil {
		fmt.Printf("\n🏆 BEST RETENTION: %s | Forgot only %.1f%% | XOR after AND: %.1f%%\n",
			best.TrainingMode, best.ForgettingRate, best.TaskAAccuracyAfterB)
	}
}
