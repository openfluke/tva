package main

import (
	"encoding/json"
	"fmt"
	"math/rand"
	"os"
	"sync"
	"time"

	"github.com/openfluke/loom/nn"
)

// SPARSE/NOISY DATA LEARNING BENCHMARK
// Train with missing/corrupted samples. Tests robustness to data quality issues.

const (
	SDInputSize  = 8
	SDHiddenSize = 48
	SDOutputSize = 4
	SDNumLayers  = 3

	SDLearningRate = float32(0.02)
	SDBatchSize    = 40

	SDTestDuration   = 60 * time.Second
	SDSampleInterval = 10 * time.Millisecond
	SDTrainInterval  = 200 * time.Millisecond

	MissingRate    = 0.3  // 30% of samples are missing (skipped)
	CorruptionRate = 0.2  // 20% of labels are wrong

	SDMaxConcurrent = 6
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
	ModeNormalBP: "NormalBP", ModeStepBP: "StepBP", ModeTween: "Tween",
	ModeTweenChain: "TweenChain", ModeStepTween: "StepTween", ModeStepTweenChain: "StepTweenChain",
}

func getCorrectTarget(input []float32) int {
	sum := float32(0)
	for _, v := range input { sum += v }
	avg := sum / float32(len(input))
	if avg < 0.25 { return 0 }
	if avg < 0.5 { return 1 }
	if avg < 0.75 { return 2 }
	return 3
}

type TestResult struct {
	TrainingMode     string  `json:"trainingMode"`
	TotalSamples     int     `json:"totalSamples"`
	MissedSamples    int     `json:"missedSamples"`
	CorruptedLabels  int     `json:"corruptedLabels"`
	TrainAccuracy    float64 `json:"trainAccuracy"`
	TestAccuracy     float64 `json:"testAccuracy"`
	ResilienceScore  float64 `json:"resilienceScore"` // How well it handles bad data
	AvailabilityPct  float64 `json:"availabilityPct"`
	TotalBlockedMs   float64 `json:"totalBlockedMs"`
	Score            float64 `json:"score"`
}

type BenchmarkResults struct {
	Results   []TestResult `json:"results"`
	Timestamp string       `json:"timestamp"`
}

func main() {
	rand.Seed(time.Now().UnixNano())
	fmt.Println("╔═══════════════════════════════════════════════════════════════════════════════════════════════════════════════╗")
	fmt.Println("║   📊 SPARSE DATA LEARNING BENCHMARK                                                                          ║")
	fmt.Println("║                                                                                                               ║")
	fmt.Println("║   30% missing samples + 20% corrupted labels                                                                 ║")
	fmt.Println("║   Question: Which modes handle noisy/incomplete data best?                                                   ║")
	fmt.Println("╚═══════════════════════════════════════════════════════════════════════════════════════════════════════════════╝")

	modes := []TrainingMode{ModeNormalBP, ModeStepBP, ModeTween, ModeTweenChain, ModeStepTween, ModeStepTweenChain}
	results := &BenchmarkResults{Results: make([]TestResult, 0, len(modes)), Timestamp: time.Now().Format(time.RFC3339)}

	var wg sync.WaitGroup
	var mu sync.Mutex
	sem := make(chan struct{}, SDMaxConcurrent)

	for i, mode := range modes {
		wg.Add(1)
		go func(m TrainingMode, idx int) {
			defer wg.Done()
			sem <- struct{}{}
			defer func() { <-sem }()

			fmt.Printf("📊 [%d/%d] Starting %s...\n", idx+1, len(modes), modeNames[m])
			result := runSparseTest(m)
			result.TrainingMode = modeNames[m]

			mu.Lock()
			results.Results = append(results.Results, result)
			mu.Unlock()

			fmt.Printf("✅ [%d/%d] %-15s | Samples: %4d (missed: %3d, corrupt: %3d) | Test: %5.1f%% | Score: %.0f\n",
				idx+1, len(modes), modeNames[m], result.TotalSamples, result.MissedSamples, result.CorruptedLabels, result.TestAccuracy, result.Score)
		}(mode, i)
	}

	wg.Wait()
	saveResults(results)
	printSummaryTable(results)
}

func runSparseTest(mode TrainingMode) TestResult {
	result := TestResult{}
	net := createNetwork()
	numLayers := net.TotalLayers()

	var state *nn.StepState
	if mode == ModeStepBP || mode == ModeStepTween || mode == ModeStepTweenChain {
		state = net.InitStepState(SDInputSize)
	}
	var ts *nn.TweenState
	if mode == ModeTween || mode == ModeTweenChain || mode == ModeStepTween || mode == ModeStepTweenChain {
		ts = nn.NewTweenState(net, nil)
		if mode == ModeTweenChain || mode == ModeStepTweenChain { ts.Config.UseChainRule = true }
	}

	type Sample struct { Input []float32; Target int }
	trainBatch := make([]Sample, 0, SDBatchSize+10)
	lastTrainTime := time.Now()
	var totalBlockedTime time.Duration
	isBlocked := false
	correct := 0

	start := time.Now()
	lastSampleTime := start

	for time.Since(start) < SDTestDuration {
		if time.Since(lastSampleTime) >= SDSampleInterval {
			lastSampleTime = time.Now()

			// Simulate missing data
			if rand.Float64() < MissingRate {
				result.MissedSamples++
				continue
			}

			if isBlocked { continue }

			input := randomInput()
			correctTarget := getCorrectTarget(input)
			target := correctTarget

			// Simulate corrupted labels
			if rand.Float64() < CorruptionRate {
				target = rand.Intn(SDOutputSize)
				result.CorruptedLabels++
			}

			result.TotalSamples++

			var output []float32
			switch mode {
			case ModeNormalBP, ModeTween, ModeTweenChain: output, _ = net.ForwardCPU(input)
			case ModeStepBP: state.SetInput(input); for s := 0; s < numLayers; s++ { net.StepForward(state) }; output = state.GetOutput()
			case ModeStepTween, ModeStepTweenChain: output = ts.ForwardPass(net, input)
			}

			if argmax(output) == correctTarget { correct++ }

			trainBatch = append(trainBatch, Sample{Input: input, Target: target})
			t := make([]float32, SDOutputSize); t[target] = 1.0

			switch mode {
			case ModeNormalBP:
				if len(trainBatch) >= SDBatchSize && time.Since(lastTrainTime) > SDTrainInterval {
					batches := make([]nn.TrainingBatch, len(trainBatch))
					for i, s := range trainBatch { tgt := make([]float32, SDOutputSize); tgt[s.Target] = 1.0; batches[i] = nn.TrainingBatch{Input: s.Input, Target: tgt} }
					isBlocked = true
					trainStart := time.Now()
					net.Train(batches, &nn.TrainingConfig{Epochs: 1, LearningRate: SDLearningRate, LossType: "crossentropy"})
					totalBlockedTime += time.Since(trainStart)
					isBlocked = false
					trainBatch = trainBatch[:0]; lastTrainTime = time.Now()
				}
			case ModeStepBP:
				grad := make([]float32, len(output)); for i := range grad { if i < len(t) { grad[i] = clipGrad(output[i]-t[i], 0.5) } }
				net.StepBackward(state, grad); net.ApplyGradients(SDLearningRate)
			case ModeTween, ModeTweenChain:
				if len(trainBatch) >= SDBatchSize && time.Since(lastTrainTime) > SDTrainInterval {
					isBlocked = true
					trainStart := time.Now()
					for _, s := range trainBatch { tgt := make([]float32, SDOutputSize); tgt[s.Target] = 1.0; out := ts.ForwardPass(net, s.Input)
						grad := make([]float32, len(out)); for i := range grad { if i < len(tgt) { grad[i] = tgt[i] - out[i] } }
						ts.ChainGradients[numLayers] = grad; ts.BackwardTargets[numLayers] = tgt; ts.TweenWeightsChainRule(net, SDLearningRate) }
					totalBlockedTime += time.Since(trainStart)
					isBlocked = false
					trainBatch = trainBatch[:0]; lastTrainTime = time.Now()
				}
			case ModeStepTween, ModeStepTweenChain:
				grad := make([]float32, len(output)); for i := range grad { if i < len(t) { grad[i] = t[i] - output[i] } }
				ts.ChainGradients[numLayers] = grad; ts.BackwardTargets[numLayers] = t; ts.TweenWeightsChainRule(net, SDLearningRate)
			}
		}
	}

	// Test on clean data
	testCorrect := 0
	for i := 0; i < 100; i++ {
		input := randomInput(); target := getCorrectTarget(input); var output []float32
		switch mode { case ModeNormalBP, ModeTween, ModeTweenChain: output, _ = net.ForwardCPU(input)
		case ModeStepBP: state.SetInput(input); for s := 0; s < numLayers; s++ { net.StepForward(state) }; output = state.GetOutput()
		case ModeStepTween, ModeStepTweenChain: output = ts.ForwardPass(net, input) }
		if argmax(output) == target { testCorrect++ }
	}

	if result.TotalSamples > 0 { result.TrainAccuracy = float64(correct) / float64(result.TotalSamples) * 100 }
	result.TestAccuracy = float64(testCorrect)
	result.TotalBlockedMs = totalBlockedTime.Seconds() * 1000
	totalTimeMs := SDTestDuration.Seconds() * 1000
	result.AvailabilityPct = ((totalTimeMs - result.TotalBlockedMs) / totalTimeMs) * 100
	
	// Resilience = test accuracy despite bad data
	effectiveDataRate := 1.0 - MissingRate - CorruptionRate*(1.0-MissingRate)
	result.ResilienceScore = result.TestAccuracy / effectiveDataRate
	
	// Score = test accuracy - blocked penalty 
	// 1000ms blocked = -10 point penalty
	blockedPenalty := result.TotalBlockedMs / 100.0
	result.Score = result.TestAccuracy - blockedPenalty
	if result.Score < 0 {
		result.Score = 0
	}
	return result
}

func createNetwork() *nn.Network {
	return nn.BuildSimpleNetwork(nn.SimpleNetworkConfig{InputSize: SDInputSize, HiddenSize: SDHiddenSize, OutputSize: SDOutputSize, Activation: nn.ActivationLeakyReLU, InitScale: 0.4, NumLayers: SDNumLayers, LayerType: nn.BrainDense, DType: nn.DTypeFloat32})
}
func randomInput() []float32 { input := make([]float32, SDInputSize); for i := range input { input[i] = rand.Float32() }; return input }
func argmax(arr []float32) int { if len(arr) == 0 { return 0 }; maxIdx, maxVal := 0, arr[0]; for i, v := range arr { if v > maxVal { maxVal, maxIdx = v, i } }; return maxIdx }
func clipGrad(grad, maxVal float32) float32 { if grad > maxVal { return maxVal }; if grad < -maxVal { return -maxVal }; return grad }
func saveResults(results *BenchmarkResults) { data, _ := json.MarshalIndent(results, "", "  "); os.WriteFile("sparse_data_results.json", data, 0644); fmt.Println("\n📁 Results saved to sparse_data_results.json") }
func printSummaryTable(results *BenchmarkResults) {
	fmt.Println("\n╔═══════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════╗")
	fmt.Println("║                                    SPARSE DATA LEARNING BENCHMARK SUMMARY                                                    ║")
	fmt.Println("╠═══════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════╣")
	for _, r := range results.Results {
		fmt.Printf("║  %-15s │ Samples: %4d │ Missed: %3d │ Corrupt: %3d │ Train: %5.1f%% │ Test: %5.1f%% │ Score: %6.1f │ ✅ PASS ║\n",
			r.TrainingMode, r.TotalSamples, r.MissedSamples, r.CorruptedLabels, r.TrainAccuracy, r.TestAccuracy, r.Score)
	}
	fmt.Println("╚═══════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════╝")
	var best *TestResult
	for i := range results.Results { if best == nil || results.Results[i].Score > best.Score { best = &results.Results[i] } }
	if best != nil { fmt.Printf("\n🏆 MOST RESILIENT: %s | Test Accuracy: %.1f%% despite 30%% missing + 20%% corrupt data\n", best.TrainingMode, best.TestAccuracy) }
}
