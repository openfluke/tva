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
// BALANCE BEAM v2 - Simplified physics, clear metrics
// ═══════════════════════════════════════════════════════════════════════════════

const (
	BBInputSize  = 2   // angle, angular velocity
	BBHiddenSize = 16
	BBOutputSize = 3   // left, stay, right
	BBNumLayers  = 2

	BBLearningRate = float32(0.05)
	BBBatchSize    = 80

	BBTestDuration  = 30 * time.Second
	BBControlRate   = 20 * time.Millisecond  // 50Hz
	BBTrainInterval = 400 * time.Millisecond

	BBMaxAngle = 30.0 // Degrees before fall
	BBMaxConcurrent = 6
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
	TrainingMode    string  `json:"trainingMode"`
	TotalSteps      int     `json:"totalSteps"`
	ControlledSteps int     `json:"controlledSteps"`
	SkippedDueBlock int     `json:"skippedDueBlock"`
	Falls           int     `json:"falls"`
	CorrectActions  int     `json:"correctActions"`
	Accuracy        float64 `json:"accuracy"`
	TotalBlockedMs  float64 `json:"totalBlockedMs"`
	Score           float64 `json:"score"`
}

type BenchmarkResults struct {
	Results   []TestResult `json:"results"`
	Timestamp string       `json:"timestamp"`
}

func main() {
	rand.Seed(time.Now().UnixNano())

	fmt.Println("╔═══════════════════════════════════════════════════════════════════════════════════════════════════════════════╗")
	fmt.Println("║   ⚖️ BALANCE BEAM v2                                                                                          ║")
	fmt.Println("║                                                                                                               ║")
	fmt.Println("║   50Hz control | Simple physics | Blocking = uncontrolled drift                                             ║")
	fmt.Println("║   Score = steps_balanced - falls*10 - skipped_steps                                                         ║")
	fmt.Println("╚═══════════════════════════════════════════════════════════════════════════════════════════════════════════════╝")

	modes := []TrainingMode{ModeNormalBP, ModeStepBP, ModeTween, ModeTweenChain, ModeStepTween, ModeStepTweenChain}
	results := &BenchmarkResults{Results: make([]TestResult, 0, len(modes)), Timestamp: time.Now().Format(time.RFC3339)}

	var wg sync.WaitGroup
	var mu sync.Mutex
	sem := make(chan struct{}, BBMaxConcurrent)

	for i, mode := range modes {
		wg.Add(1)
		go func(m TrainingMode, idx int) {
			defer wg.Done()
			sem <- struct{}{}
			defer func() { <-sem }()

			fmt.Printf("⚖️ [%d/%d] Starting %s...\n", idx+1, len(modes), modeNames[m])
			result := runTest(m)
			result.TrainingMode = modeNames[m]

			mu.Lock()
			results.Results = append(results.Results, result)
			mu.Unlock()

			fmt.Printf("✅ [%d/%d] %-15s | Steps: %4d | Controlled: %4d | Skipped: %3d | Falls: %2d | Score: %.0f\n",
				idx+1, len(modes), modeNames[m], result.TotalSteps, result.ControlledSteps,
				result.SkippedDueBlock, result.Falls, result.Score)
		}(mode, i)
	}

	wg.Wait()
	saveResults(results)
	printSummary(results)
}

func runTest(mode TrainingMode) TestResult {
	result := TestResult{}

	net := nn.BuildSimpleNetwork(nn.SimpleNetworkConfig{
		InputSize: BBInputSize, HiddenSize: BBHiddenSize, OutputSize: BBOutputSize,
		Activation: nn.ActivationLeakyReLU, InitScale: 0.5, NumLayers: BBNumLayers,
		LayerType: nn.BrainDense, DType: nn.DTypeFloat32,
	})
	numLayers := net.TotalLayers()

	var state *nn.StepState
	if mode == ModeStepBP || mode == ModeStepTween || mode == ModeStepTweenChain {
		state = net.InitStepState(BBInputSize)
	}
	var ts *nn.TweenState
	if mode == ModeTween || mode == ModeTweenChain || mode == ModeStepTween || mode == ModeStepTweenChain {
		ts = nn.NewTweenState(net, nil)
		if mode == ModeTweenChain || mode == ModeStepTweenChain { ts.Config.UseChainRule = true }
	}

	type Sample struct { Input []float32; Action int }
	trainBatch := make([]Sample, 0, BBBatchSize+10)
	lastTrainTime := time.Now()
	var totalBlockedTime time.Duration

	// Simple physics state
	angle := (rand.Float64() - 0.5) * 10 // Start near center
	angularVel := 0.0

	start := time.Now()
	lastControl := start

	for time.Since(start) < BBTestDuration {
		now := time.Now()
		if now.Sub(lastControl) >= BBControlRate {
			lastControl = now
			result.TotalSteps++

			// Physics step - gravity pulls toward extremes
			angularVel += angle * 0.02 // Gravity effect
			angle += angularVel

			// Check for fall
			if math.Abs(angle) > BBMaxAngle {
				result.Falls++
				angle = (rand.Float64() - 0.5) * 10 // Reset
				angularVel = 0
			}

			// What's the ideal action?
			idealAction := 1 // Stay
			if angle > 3 { idealAction = 0 }  // Push left
			if angle < -3 { idealAction = 2 } // Push right

			// Check if blocked
			shouldBlock := false
			if mode == ModeNormalBP || mode == ModeTween || mode == ModeTweenChain {
				if len(trainBatch) >= BBBatchSize && time.Since(lastTrainTime) > BBTrainInterval {
					shouldBlock = true
				}
			}

			if shouldBlock {
				result.SkippedDueBlock++
				
				// Training while blocked
				trainStart := time.Now()
				switch mode {
				case ModeNormalBP:
					batches := make([]nn.TrainingBatch, len(trainBatch))
					for i, s := range trainBatch {
						t := make([]float32, BBOutputSize); t[s.Action] = 1
						batches[i] = nn.TrainingBatch{Input: s.Input, Target: t}
					}
					net.Train(batches, &nn.TrainingConfig{Epochs: 1, LearningRate: BBLearningRate, LossType: "crossentropy"})
				case ModeTween, ModeTweenChain:
					for _, s := range trainBatch {
						t := make([]float32, BBOutputSize); t[s.Action] = 1
						out := ts.ForwardPass(net, s.Input)
						grad := make([]float32, len(out))
						for i := range grad { grad[i] = t[i] - out[i] }
						ts.ChainGradients[numLayers] = grad
						ts.BackwardTargets[numLayers] = t
						ts.TweenWeightsChainRule(net, BBLearningRate)
					}
				}
				totalBlockedTime += time.Since(trainStart)
				trainBatch = trainBatch[:0]
				lastTrainTime = time.Now()

				// No control applied - angle drifts!
				continue
			}

			result.ControlledSteps++

			input := []float32{float32(angle / BBMaxAngle), float32(angularVel / 10)}

			var output []float32
			switch mode {
			case ModeNormalBP, ModeTween, ModeTweenChain:
				output, _ = net.ForwardCPU(input)
			case ModeStepBP:
				state.SetInput(input)
				for s := 0; s < numLayers; s++ { net.StepForward(state) }
				output = state.GetOutput()
			case ModeStepTween, ModeStepTweenChain:
				output = ts.ForwardPass(net, input)
			}

			action := argmax(output)
			if action == idealAction { result.CorrectActions++ }

			// Apply control
			switch action {
			case 0: angularVel -= 0.5 // Push left
			case 2: angularVel += 0.5 // Push right
			}

			trainBatch = append(trainBatch, Sample{Input: input, Action: idealAction})

			// Step training
			target := make([]float32, BBOutputSize)
			target[idealAction] = 1
			switch mode {
			case ModeStepBP:
				grad := make([]float32, len(output))
				for i := range grad { if i < len(target) { grad[i] = output[i] - target[i] } }
				net.StepBackward(state, grad)
				net.ApplyGradients(BBLearningRate)
			case ModeStepTween, ModeStepTweenChain:
				grad := make([]float32, len(output))
				for i := range grad { if i < len(target) { grad[i] = target[i] - output[i] } }
				ts.ChainGradients[numLayers] = grad
				ts.BackwardTargets[numLayers] = target
				ts.TweenWeightsChainRule(net, BBLearningRate)
			}
		}
	}

	result.TotalBlockedMs = totalBlockedTime.Seconds() * 1000
	if result.ControlledSteps > 0 {
		result.Accuracy = float64(result.CorrectActions) / float64(result.ControlledSteps) * 100
	}

	result.Score = float64(result.ControlledSteps) - float64(result.Falls)*10 - float64(result.SkippedDueBlock)
	if result.Score < 0 { result.Score = 0 }
	return result
}

func argmax(arr []float32) int {
	if len(arr) == 0 { return 0 }
	maxIdx, maxVal := 0, arr[0]
	for i, v := range arr { if v > maxVal { maxVal, maxIdx = v, i } }
	return maxIdx
}

func saveResults(results *BenchmarkResults) {
	data, _ := json.MarshalIndent(results, "", "  ")
	os.WriteFile("balance_results.json", data, 0644)
	fmt.Println("\n📁 Results saved")
}

func printSummary(results *BenchmarkResults) {
	fmt.Println("\n╔══════════════════════════════════════════════════════════════════════════════════════════════════════════════════════╗")
	fmt.Println("║                                    BALANCE BEAM SUMMARY                                                             ║")
	fmt.Println("╠══════════════════════════════════════════════════════════════════════════════════════════════════════════════════════╣")
	fmt.Println("║  Mode            │ Steps │ Controlled │ Skipped │ Falls │ Accuracy │ Blocked(ms) │ Score                            ║")
	fmt.Println("╠══════════════════════════════════════════════════════════════════════════════════════════════════════════════════════╣")

	for _, r := range results.Results {
		fmt.Printf("║  %-15s │  %4d │      %4d  │    %3d  │   %2d  │   %5.1f%% │    %6.0f   │ %6.0f ✅ ║\n",
			r.TrainingMode, r.TotalSteps, r.ControlledSteps, r.SkippedDueBlock,
			r.Falls, r.Accuracy, r.TotalBlockedMs, r.Score)
	}
	fmt.Println("╚══════════════════════════════════════════════════════════════════════════════════════════════════════════════════════╝")
}
