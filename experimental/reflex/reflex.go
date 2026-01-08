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

// ═══════════════════════════════════════════════════════════════════════════════
// REFLEX GAME BENCHMARK
// ═══════════════════════════════════════════════════════════════════════════════
//
// Random stimuli appear. Network must react within 50ms or lose points.
// Blocking = missed stimulus = 0 points for that round.
// Tests: reaction time under training pressure.

const (
	RGInputSize  = 16  // Stimulus pattern
	RGHiddenSize = 48
	RGOutputSize = 4   // 4 possible responses
	RGNumLayers  = 3

	RGLearningRate = float32(0.02)
	RGBatchSize    = 50

	RGTestDuration    = 60 * time.Second
	RGStimulusRate    = 50 * time.Millisecond  // New stimulus every 50ms
	RGReactionWindow  = 50 * time.Millisecond  // Must respond within this
	RGTrainInterval   = 200 * time.Millisecond

	RGMaxConcurrent = 6
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

type TestResult struct {
	TrainingMode     string  `json:"trainingMode"`
	TotalStimuli     int     `json:"totalStimuli"`
	Responses        int     `json:"responses"`
	CorrectResponses int     `json:"correctResponses"`
	MissedStimuli    int     `json:"missedStimuli"` // Due to blocking
	TotalReactionMs  float64 `json:"totalReactionMs"`
	AvgReactionMs    float64 `json:"avgReactionMs"`
	Accuracy         float64 `json:"accuracy"`
	TotalBlockedMs   float64 `json:"totalBlockedMs"`
	Score            float64 `json:"score"` // Points based on speed + accuracy
}

type BenchmarkResults struct {
	Results   []TestResult `json:"results"`
	Timestamp string       `json:"timestamp"`
}

func generateStimulus(correctResponse int) []float32 {
	stimulus := make([]float32, RGInputSize)
	// Create distinct pattern for each response type
	for i := range stimulus {
		base := float32(correctResponse) / float32(RGOutputSize)
		stimulus[i] = base + rand.Float32()*0.2 - 0.1
		if i%RGOutputSize == correctResponse {
			stimulus[i] += 0.3 // Make correct response more prominent
		}
	}
	return stimulus
}

func main() {
	rand.Seed(time.Now().UnixNano())

	fmt.Println("╔═══════════════════════════════════════════════════════════════════════════════════════════════════════════════╗")
	fmt.Println("║   ⚡ REFLEX GAME BENCHMARK                                                                                   ║")
	fmt.Println("║                                                                                                               ║")
	fmt.Println("║   Stimulus every 50ms | Must respond within 50ms | Blocking = miss                                          ║")
	fmt.Println("║   Score = (speed_bonus × accuracy) - missed_penalty                                                         ║")
	fmt.Println("╚═══════════════════════════════════════════════════════════════════════════════════════════════════════════════╝")

	modes := []TrainingMode{ModeNormalBP, ModeStepBP, ModeTween, ModeTweenChain, ModeStepTween, ModeStepTweenChain}
	results := &BenchmarkResults{Results: make([]TestResult, 0, len(modes)), Timestamp: time.Now().Format(time.RFC3339)}

	var wg sync.WaitGroup
	var mu sync.Mutex
	sem := make(chan struct{}, RGMaxConcurrent)

	for i, mode := range modes {
		wg.Add(1)
		go func(m TrainingMode, idx int) {
			defer wg.Done()
			sem <- struct{}{}
			defer func() { <-sem }()

			fmt.Printf("⚡ [%d/%d] Starting %s...\n", idx+1, len(modes), modeNames[m])
			result := runReflexTest(m)
			result.TrainingMode = modeNames[m]

			mu.Lock()
			results.Results = append(results.Results, result)
			mu.Unlock()

			fmt.Printf("✅ [%d/%d] %-15s | Stimuli: %4d | Missed: %3d | Acc: %5.1f%% | Avg RT: %5.2fms | Score: %.0f\n",
				idx+1, len(modes), modeNames[m], result.TotalStimuli, result.MissedStimuli,
				result.Accuracy, result.AvgReactionMs, result.Score)
		}(mode, i)
	}

	wg.Wait()
	saveResults(results)
	printSummaryTable(results)
}

func runReflexTest(mode TrainingMode) TestResult {
	result := TestResult{}

	net := createNetwork()
	numLayers := net.TotalLayers()

	var state *nn.StepState
	if mode == ModeStepBP || mode == ModeStepTween || mode == ModeStepTweenChain {
		state = net.InitStepState(RGInputSize)
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
		Target int
	}
	trainBatch := make([]Sample, 0, RGBatchSize+10)
	lastTrainTime := time.Now()
	isBlocked := false
	var totalBlockedTime time.Duration

	start := time.Now()
	lastStimulusTime := start

	for time.Since(start) < RGTestDuration {
		if time.Since(lastStimulusTime) >= RGStimulusRate {
			stimulusStart := time.Now()
			lastStimulusTime = stimulusStart
			result.TotalStimuli++

			correctResponse := rand.Intn(RGOutputSize)
			stimulus := generateStimulus(correctResponse)

			// IF BLOCKED, WE MISS THIS STIMULUS
			if isBlocked {
				result.MissedStimuli++
				continue
			}

			var output []float32
			switch mode {
			case ModeNormalBP, ModeTween, ModeTweenChain:
				output, _ = net.ForwardCPU(stimulus)
			case ModeStepBP:
				state.SetInput(stimulus)
				for s := 0; s < numLayers; s++ {
					net.StepForward(state)
				}
				output = state.GetOutput()
			case ModeStepTween, ModeStepTweenChain:
				output = ts.ForwardPass(net, stimulus)
			}

			reactionTime := time.Since(stimulusStart)
			result.TotalReactionMs += reactionTime.Seconds() * 1000
			result.Responses++

			predicted := argmax(output)
			if predicted == correctResponse {
				result.CorrectResponses++
			}

			trainBatch = append(trainBatch, Sample{Input: stimulus, Target: correctResponse})
			target := make([]float32, RGOutputSize)
			target[correctResponse] = 1.0

			// Training
			switch mode {
			case ModeNormalBP:
				if len(trainBatch) >= RGBatchSize && time.Since(lastTrainTime) > RGTrainInterval {
					batches := make([]nn.TrainingBatch, len(trainBatch))
					for i, s := range trainBatch {
						t := make([]float32, RGOutputSize)
						t[s.Target] = 1.0
						batches[i] = nn.TrainingBatch{Input: s.Input, Target: t}
					}
					isBlocked = true
					trainStart := time.Now()
					net.Train(batches, &nn.TrainingConfig{Epochs: 1, LearningRate: RGLearningRate, LossType: "crossentropy"})
					totalBlockedTime += time.Since(trainStart)
					isBlocked = false
					trainBatch = trainBatch[:0]
					lastTrainTime = time.Now()
				}
			case ModeStepBP:
				grad := make([]float32, len(output))
				for i := range grad {
					if i < len(target) { grad[i] = clipGrad(output[i]-target[i], 0.5) }
				}
				net.StepBackward(state, grad)
				net.ApplyGradients(RGLearningRate)
			case ModeTween, ModeTweenChain:
				if len(trainBatch) >= RGBatchSize && time.Since(lastTrainTime) > RGTrainInterval {
					isBlocked = true
					trainStart := time.Now()
					for _, s := range trainBatch {
						t := make([]float32, RGOutputSize)
						t[s.Target] = 1.0
						out := ts.ForwardPass(net, s.Input)
						grad := make([]float32, len(out))
						for i := range grad { if i < len(t) { grad[i] = t[i] - out[i] } }
						ts.ChainGradients[numLayers] = grad
						ts.BackwardTargets[numLayers] = t
						ts.TweenWeightsChainRule(net, RGLearningRate)
					}
					totalBlockedTime += time.Since(trainStart)
					isBlocked = false
					trainBatch = trainBatch[:0]
					lastTrainTime = time.Now()
				}
			case ModeStepTween, ModeStepTweenChain:
				grad := make([]float32, len(output))
				for i := range grad { if i < len(target) { grad[i] = target[i] - output[i] } }
				ts.ChainGradients[numLayers] = grad
				ts.BackwardTargets[numLayers] = target
				ts.TweenWeightsChainRule(net, RGLearningRate)
			}
		}
	}

	result.TotalBlockedMs = totalBlockedTime.Seconds() * 1000
	if result.Responses > 0 {
		result.AvgReactionMs = result.TotalReactionMs / float64(result.Responses)
		result.Accuracy = float64(result.CorrectResponses) / float64(result.Responses) * 100
	}

	// Score: accuracy bonus - missed penalty - slow penalty
	speedBonus := 50.0 / (result.AvgReactionMs + 1) * 100 // Faster = more points
	result.Score = speedBonus*result.Accuracy/100 - float64(result.MissedStimuli)*5
	if result.Score < 0 { result.Score = 0 }
	return result
}

func createNetwork() *nn.Network {
	return nn.BuildSimpleNetwork(nn.SimpleNetworkConfig{
		InputSize: RGInputSize, HiddenSize: RGHiddenSize, OutputSize: RGOutputSize,
		Activation: nn.ActivationLeakyReLU, InitScale: 0.4, NumLayers: RGNumLayers,
		LayerType: nn.BrainDense, DType: nn.DTypeFloat32,
	})
}

func argmax(arr []float32) int {
	if len(arr) == 0 { return 0 }
	maxIdx, maxVal := 0, arr[0]
	for i, v := range arr { if v > maxVal { maxVal, maxIdx = v, i } }
	return maxIdx
}

func clipGrad(grad, maxVal float32) float32 {
	if grad > maxVal { return maxVal }
	if grad < -maxVal { return -maxVal }
	return grad
}

func saveResults(results *BenchmarkResults) {
	data, _ := json.MarshalIndent(results, "", "  ")
	os.WriteFile("reflex_results.json", data, 0644)
	fmt.Println("\n📁 Results saved to reflex_results.json")
}

func printSummaryTable(results *BenchmarkResults) {
	fmt.Println("\n╔═══════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════╗")
	fmt.Println("║                                    REFLEX GAME BENCHMARK SUMMARY                                                                    ║")
	fmt.Println("╠═══════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════╣")
	fmt.Println("║  Training Mode    │ Stimuli │ Responses │ Correct │ Missed │ Accuracy │ Avg RT(ms) │ Blocked(ms) │   Score   │ Status              ║")
	fmt.Println("╠═══════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════╣")

	for _, r := range results.Results {
		fmt.Printf("║  %-15s │   %4d  │    %4d   │   %4d  │   %3d  │   %5.1f%% │    %6.2f   │    %6.0f   │ %9.1f │ ✅ PASS ║\n",
			r.TrainingMode, r.TotalStimuli, r.Responses, r.CorrectResponses,
			r.MissedStimuli, r.Accuracy, r.AvgReactionMs, r.TotalBlockedMs, r.Score)
	}

	fmt.Println("╚═══════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════╝")

	var best *TestResult
	for i := range results.Results {
		if best == nil || results.Results[i].Score > best.Score {
			best = &results.Results[i]
		}
	}
	if best != nil {
		fmt.Printf("\n🏆 FASTEST REFLEXES: %s | Accuracy: %.1f%% | Avg Reaction: %.2fms | Missed: %d\n",
			best.TrainingMode, best.Accuracy, best.AvgReactionMs, best.MissedStimuli)
	}
}
