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

// TRANSFER LEARNING SPEED BENCHMARK - HARDCORE
// Pre-train on Task A, then fine-tune to Task B
// Deeper network, larger batch = more blocking = bigger differences

const (
	TLInputSize  = 16  // Bigger input
	TLHiddenSize = 96  // Wider
	TLOutputSize = 8   // More classes
	TLNumLayers  = 5   // Deeper

	TLLearningRate = float32(0.02)
	TLFineTuneLR   = float32(0.005)
	TLBatchSize    = 80  // Bigger batches = more blocking

	TLPretrainDuration = 30 * time.Second
	TLFineTuneDuration = 30 * time.Second
	TLSampleInterval   = 8 * time.Millisecond
	TLTrainInterval    = 300 * time.Millisecond

	TLMaxConcurrent = 6
)

type Sample struct {
	Input  []float32
	Target int
}

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

// Task A: Classify by pattern position (where is the peak?)
func taskATarget(input []float32) int {
	maxIdx := 0
	maxVal := input[0]
	for i, v := range input {
		if v > maxVal {
			maxVal = v
			maxIdx = i
		}
	}
	return maxIdx * TLOutputSize / len(input)
}

// Task B: Classify by pattern frequency (how many oscillations?)
func taskBTarget(input []float32) int {
	crossings := 0
	mean := float32(0)
	for _, v := range input {
		mean += v
	}
	mean /= float32(len(input))
	above := input[0] > mean
	for _, v := range input {
		if (v > mean) != above {
			crossings++
			above = !above
		}
	}
	return crossings % TLOutputSize
}

type TestResult struct {
	TrainingMode         string  `json:"trainingMode"`
	TaskAFinalAccuracy   float64 `json:"taskAFinalAccuracy"`
	TaskBInitialAccuracy float64 `json:"taskBInitialAccuracy"`
	TaskBFinalAccuracy   float64 `json:"taskBFinalAccuracy"`
	TransferBoost        float64 `json:"transferBoost"`
	AdaptationGain       float64 `json:"adaptationGain"`
	AvailabilityPct      float64 `json:"availabilityPct"`
	TotalBlockedMs       float64 `json:"totalBlockedMs"`
	Score                float64 `json:"score"`
}

type BenchmarkResults struct {
	Results   []TestResult `json:"results"`
	Timestamp string       `json:"timestamp"`
}

func main() {
	rand.Seed(time.Now().UnixNano())
	fmt.Println("╔═══════════════════════════════════════════════════════════════════════════════════════════════════════════════╗")
	fmt.Println("║   🔄 TRANSFER LEARNING SPEED BENCHMARK - HARDCORE                                                            ║")
	fmt.Println("║                                                                                                               ║")
	fmt.Println("║   5-layer network | Batch size 80 | Pre-train 30s → Fine-tune 30s                                           ║")
	fmt.Println("║   Task A: Peak position → Task B: Oscillation frequency                                                     ║")
	fmt.Println("╚═══════════════════════════════════════════════════════════════════════════════════════════════════════════════╝")

	modes := []TrainingMode{ModeNormalBP, ModeStepBP, ModeTween, ModeTweenChain, ModeStepTween, ModeStepTweenChain}
	results := &BenchmarkResults{Results: make([]TestResult, 0, len(modes)), Timestamp: time.Now().Format(time.RFC3339)}

	var wg sync.WaitGroup
	var mu sync.Mutex
	sem := make(chan struct{}, TLMaxConcurrent)

	for i, mode := range modes {
		wg.Add(1)
		go func(m TrainingMode, idx int) {
			defer wg.Done()
			sem <- struct{}{}
			defer func() { <-sem }()

			fmt.Printf("🔄 [%d/%d] Starting %s...\n", idx+1, len(modes), modeNames[m])
			result := runTransferTest(m)
			result.TrainingMode = modeNames[m]

			mu.Lock()
			results.Results = append(results.Results, result)
			mu.Unlock()

			fmt.Printf("✅ [%d/%d] %-15s | TaskA: %5.1f%% | TaskB: %5.1f%%→%5.1f%% | Blocked: %5.0fms | Score: %.0f\n",
				idx+1, len(modes), modeNames[m], result.TaskAFinalAccuracy,
				result.TaskBInitialAccuracy, result.TaskBFinalAccuracy, result.TotalBlockedMs, result.Score)
		}(mode, i)
	}

	wg.Wait()
	saveResults(results)
	printSummaryTable(results)
}

func runTransferTest(mode TrainingMode) TestResult {
	result := TestResult{}
	net := createNetwork()
	numLayers := net.TotalLayers()

	var state *nn.StepState
	if mode == ModeStepBP || mode == ModeStepTween || mode == ModeStepTweenChain {
		state = net.InitStepState(TLInputSize)
	}
	var ts *nn.TweenState
	if mode == ModeTween || mode == ModeTweenChain || mode == ModeStepTween || mode == ModeStepTweenChain {
		ts = nn.NewTweenState(net, nil)
		if mode == ModeTweenChain || mode == ModeStepTweenChain {
			ts.Config.UseChainRule = true
		}
	}

	trainBatch := make([]Sample, 0, TLBatchSize+10)
	lastTrainTime := time.Now()
	var totalBlockedTime time.Duration
	isBlocked := false

	// Phase 1: Pre-train Task A
	start := time.Now()
	lastSampleTime := start
	for time.Since(start) < TLPretrainDuration {
		if time.Since(lastSampleTime) >= TLSampleInterval {
			lastSampleTime = time.Now()
			if isBlocked {
				continue
			}

			input := randomInput()
			target := taskATarget(input)
			trainBatch = append(trainBatch, Sample{Input: input, Target: target})

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

			t := make([]float32, TLOutputSize)
			if target < TLOutputSize {
				t[target] = 1.0
			}

			switch mode {
			case ModeNormalBP:
				if len(trainBatch) >= TLBatchSize && time.Since(lastTrainTime) > TLTrainInterval {
					batches := make([]nn.TrainingBatch, len(trainBatch))
					for i, s := range trainBatch {
						tgt := make([]float32, TLOutputSize)
						if s.Target < TLOutputSize {
							tgt[s.Target] = 1.0
						}
						batches[i] = nn.TrainingBatch{Input: s.Input, Target: tgt}
					}
					isBlocked = true
					trainStart := time.Now()
					net.Train(batches, &nn.TrainingConfig{Epochs: 1, LearningRate: TLLearningRate, LossType: "crossentropy"})
					totalBlockedTime += time.Since(trainStart)
					isBlocked = false
					trainBatch = trainBatch[:0]
					lastTrainTime = time.Now()
				}
			case ModeStepBP:
				grad := make([]float32, len(output))
				for i := range grad {
					if i < len(t) {
						grad[i] = clipGrad(output[i]-t[i], 0.5)
					}
				}
				net.StepBackward(state, grad)
				net.ApplyGradients(TLLearningRate)
			case ModeTween, ModeTweenChain:
				if len(trainBatch) >= TLBatchSize && time.Since(lastTrainTime) > TLTrainInterval {
					isBlocked = true
					trainStart := time.Now()
					for _, s := range trainBatch {
						tgt := make([]float32, TLOutputSize)
						if s.Target < TLOutputSize {
							tgt[s.Target] = 1.0
						}
						out := ts.ForwardPass(net, s.Input)
						grad := make([]float32, len(out))
						for i := range grad {
							if i < len(tgt) {
								grad[i] = tgt[i] - out[i]
							}
						}
						ts.ChainGradients[numLayers] = grad
						ts.BackwardTargets[numLayers] = tgt
						ts.TweenWeightsChainRule(net, TLLearningRate)
					}
					totalBlockedTime += time.Since(trainStart)
					isBlocked = false
					trainBatch = trainBatch[:0]
					lastTrainTime = time.Now()
				}
			case ModeStepTween, ModeStepTweenChain:
				grad := make([]float32, len(output))
				for i := range grad {
					if i < len(t) {
						grad[i] = t[i] - output[i]
					}
				}
				ts.ChainGradients[numLayers] = grad
				ts.BackwardTargets[numLayers] = t
				ts.TweenWeightsChainRule(net, TLLearningRate)
			}
		}
	}
	result.TaskAFinalAccuracy = measureAccuracy(net, taskATarget, mode, state, ts, numLayers)
	result.TaskBInitialAccuracy = measureAccuracy(net, taskBTarget, mode, state, ts, numLayers)

	// Phase 2: Fine-tune Task B
	trainBatch = trainBatch[:0]
	start = time.Now()
	lastSampleTime = start
	for time.Since(start) < TLFineTuneDuration {
		if time.Since(lastSampleTime) >= TLSampleInterval {
			lastSampleTime = time.Now()
			if isBlocked {
				continue
			}

			input := randomInput()
			target := taskBTarget(input)
			trainBatch = append(trainBatch, Sample{Input: input, Target: target})

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

			t := make([]float32, TLOutputSize)
			if target < TLOutputSize {
				t[target] = 1.0
			}

			switch mode {
			case ModeNormalBP:
				if len(trainBatch) >= TLBatchSize && time.Since(lastTrainTime) > TLTrainInterval {
					batches := make([]nn.TrainingBatch, len(trainBatch))
					for i, s := range trainBatch {
						tgt := make([]float32, TLOutputSize)
						if s.Target < TLOutputSize {
							tgt[s.Target] = 1.0
						}
						batches[i] = nn.TrainingBatch{Input: s.Input, Target: tgt}
					}
					isBlocked = true
					trainStart := time.Now()
					net.Train(batches, &nn.TrainingConfig{Epochs: 1, LearningRate: TLFineTuneLR, LossType: "crossentropy"})
					totalBlockedTime += time.Since(trainStart)
					isBlocked = false
					trainBatch = trainBatch[:0]
					lastTrainTime = time.Now()
				}
			case ModeStepBP:
				grad := make([]float32, len(output))
				for i := range grad {
					if i < len(t) {
						grad[i] = clipGrad(output[i]-t[i], 0.5)
					}
				}
				net.StepBackward(state, grad)
				net.ApplyGradients(TLFineTuneLR)
			case ModeTween, ModeTweenChain:
				if len(trainBatch) >= TLBatchSize && time.Since(lastTrainTime) > TLTrainInterval {
					isBlocked = true
					trainStart := time.Now()
					for _, s := range trainBatch {
						tgt := make([]float32, TLOutputSize)
						if s.Target < TLOutputSize {
							tgt[s.Target] = 1.0
						}
						out := ts.ForwardPass(net, s.Input)
						grad := make([]float32, len(out))
						for i := range grad {
							if i < len(tgt) {
								grad[i] = tgt[i] - out[i]
							}
						}
						ts.ChainGradients[numLayers] = grad
						ts.BackwardTargets[numLayers] = tgt
						ts.TweenWeightsChainRule(net, TLFineTuneLR)
					}
					totalBlockedTime += time.Since(trainStart)
					isBlocked = false
					trainBatch = trainBatch[:0]
					lastTrainTime = time.Now()
				}
			case ModeStepTween, ModeStepTweenChain:
				grad := make([]float32, len(output))
				for i := range grad {
					if i < len(t) {
						grad[i] = t[i] - output[i]
					}
				}
				ts.ChainGradients[numLayers] = grad
				ts.BackwardTargets[numLayers] = t
				ts.TweenWeightsChainRule(net, TLFineTuneLR)
			}
		}
	}

	result.TaskBFinalAccuracy = measureAccuracy(net, taskBTarget, mode, state, ts, numLayers)
	result.TransferBoost = result.TaskBInitialAccuracy - (100.0 / float64(TLOutputSize)) // vs random
	result.AdaptationGain = result.TaskBFinalAccuracy - result.TaskBInitialAccuracy
	result.TotalBlockedMs = totalBlockedTime.Seconds() * 1000
	totalTimeMs := (TLPretrainDuration + TLFineTuneDuration).Seconds() * 1000
	result.AvailabilityPct = ((totalTimeMs - result.TotalBlockedMs) / totalTimeMs) * 100

	// Score: final accuracy × adaptation gain - blocked penalty
	// 1000ms blocked = -10 point penalty
	blockedPenalty := result.TotalBlockedMs / 100.0
	result.Score = result.TaskBFinalAccuracy * (1 + result.AdaptationGain/100) - blockedPenalty
	if result.Score < 0 {
		result.Score = 0
	}
	return result
}

func createNetwork() *nn.Network {
	return nn.BuildSimpleNetwork(nn.SimpleNetworkConfig{
		InputSize: TLInputSize, HiddenSize: TLHiddenSize, OutputSize: TLOutputSize,
		Activation: nn.ActivationLeakyReLU, InitScale: 0.3, NumLayers: TLNumLayers,
		LayerType: nn.BrainDense, DType: nn.DTypeFloat32,
	})
}

func randomInput() []float32 {
	input := make([]float32, TLInputSize)
	// Create structured patterns
	peak := rand.Intn(TLInputSize)
	for i := range input {
		dist := i - peak
		if dist < 0 {
			dist = -dist
		}
		input[i] = rand.Float32()*0.3 + float32(TLInputSize-dist)/float32(TLInputSize)*0.7
	}
	// Add oscillations
	freq := rand.Intn(5) + 1
	for i := range input {
		input[i] += float32(rand.Intn(freq)) * 0.1
	}
	return input
}

func measureAccuracy(net *nn.Network, targetFunc func([]float32) int, mode TrainingMode, state *nn.StepState, ts *nn.TweenState, numLayers int) float64 {
	correct := 0
	for i := 0; i < 200; i++ {
		input := randomInput()
		target := targetFunc(input)
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
		if argmax(output) == target {
			correct++
		}
	}
	return float64(correct) / 2.0 // Return as percentage
}

func argmax(arr []float32) int {
	if len(arr) == 0 {
		return 0
	}
	maxIdx, maxVal := 0, arr[0]
	for i, v := range arr {
		if v > maxVal {
			maxVal, maxIdx = v, i
		}
	}
	return maxIdx
}

func clipGrad(grad, maxVal float32) float32 {
	if grad > maxVal {
		return maxVal
	}
	if grad < -maxVal {
		return -maxVal
	}
	return grad
}

func saveResults(results *BenchmarkResults) {
	data, _ := json.MarshalIndent(results, "", "  ")
	os.WriteFile("transfer_results.json", data, 0644)
	fmt.Println("\n📁 Results saved to transfer_results.json")
}

func printSummaryTable(results *BenchmarkResults) {
	fmt.Println("\n╔═════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════╗")
	fmt.Println("║                                    TRANSFER LEARNING BENCHMARK SUMMARY                                                                     ║")
	fmt.Println("╠═════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════╣")
	fmt.Println("║  Training Mode    │ TaskA%  │ TaskB Init │ TaskB Final │ Adapt Gain │ Blocked(ms) │ Avail% │    Score    │ Status                          ║")
	fmt.Println("╠═════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════╣")
	for _, r := range results.Results {
		fmt.Printf("║  %-15s │  %5.1f%% │    %5.1f%%  │    %5.1f%%   │   %+5.1f%%  │   %8.0f  │ %5.1f%% │ %11.1f │ ✅ PASS ║\n",
			r.TrainingMode, r.TaskAFinalAccuracy, r.TaskBInitialAccuracy, r.TaskBFinalAccuracy,
			r.AdaptationGain, r.TotalBlockedMs, r.AvailabilityPct, r.Score)
	}
	fmt.Println("╚═════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════╝")

	var best *TestResult
	for i := range results.Results {
		if best == nil || results.Results[i].Score > best.Score {
			best = &results.Results[i]
		}
	}
	if best != nil {
		fmt.Printf("\n🏆 BEST: %s | Final: %.1f%% | Adaptation: %+.1f%% | Blocked: %.0fms\n",
			best.TrainingMode, best.TaskBFinalAccuracy, best.AdaptationGain, best.TotalBlockedMs)
	}
}
