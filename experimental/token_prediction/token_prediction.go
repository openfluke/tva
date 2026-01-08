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
// STREAMING TOKEN PREDICTION - HARDCORE EDITION
// ═══════════════════════════════════════════════════════════════════════════════
//
// HARDCORE: 60 second test, DEEP networks, LARGE batch training
// LLM generation cannot pause - blocking = terrible user experience

const (
	TPInputSize  = 32  // Longer context
	TPHiddenSize = 128
	TPOutputSize = 26
	TPNumLayers  = 5

	TPLearningRate = float32(0.008)
	TPInitScale    = float32(0.3)
	TPBatchSize    = 100

	TPTestDuration   = 60 * time.Second
	TPWindowDuration = 100 * time.Millisecond
	TPSwitchInterval = 5 * time.Second
	TPTrainInterval  = 500 * time.Millisecond
	TPTokenInterval  = 5 * time.Millisecond // Fast generation

	TPMaxConcurrent = 6
)

type WritingStyle int

const (
	StyleTechnical WritingStyle = iota
	StyleCasual
	StyleFormal
	StyleCode
	StylePoetry    // NEW
	StyleScientific // NEW
)

var stylePatterns = map[WritingStyle][]string{
	StyleTechnical:  {"algorithm", "function", "variable", "compute", "neural", "tensor", "gradient", "epoch", "layer", "model"},
	StyleCasual:     {"hey", "cool", "nice", "wow", "lol", "yeah", "thanks", "okay", "sure", "maybe"},
	StyleFormal:     {"pursuant", "therefore", "whereas", "hereby", "acknowledge", "consider", "regarding", "substantial", "implementation", "framework"},
	StyleCode:       {"func", "return", "if", "else", "for", "range", "var", "type", "struct", "import"},
	StylePoetry:     {"whisper", "moonlight", "shadow", "dream", "eternal", "flowing", "silence", "wander", "gentle", "azure"},
	StyleScientific: {"hypothesis", "experiment", "observation", "quantum", "molecular", "theorem", "equation", "coefficient", "derivative", "analysis"},
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
	ModeNormalBP:       "NormalBP",
	ModeStepBP:         "StepBP",
	ModeTween:          "Tween",
	ModeTweenChain:     "TweenChain",
	ModeStepTween:      "StepTween",
	ModeStepTweenChain: "StepTweenChain",
}

type TimeWindow struct {
	TimeMs    int     `json:"timeMs"`
	Tokens    int     `json:"tokens"`
	Correct   int     `json:"correct"`
	BlockedMs float64 `json:"blockedMs"`
}

type TestResult struct {
	TrainingMode       string       `json:"trainingMode"`
	Windows            []TimeWindow `json:"windows"`
	TotalTokens        int          `json:"totalTokens"`
	CorrectPredictions int          `json:"correctPredictions"`
	TokensGenerated    int          `json:"tokensGenerated"`
	TokensMissed       int          `json:"tokensMissed"` // Tokens we couldn't generate while blocked
	TrainTimeSec       float64      `json:"trainTimeSec"`
	Accuracy           float64      `json:"accuracy"`
	TokensPerSec       float64      `json:"tokensPerSec"`
	AvailabilityPct    float64      `json:"availabilityPct"`
	TotalBlockedMs     float64      `json:"totalBlockedMs"`
	Score              float64      `json:"score"`
	Passed             bool         `json:"passed"`
	Error              string       `json:"error,omitempty"`
}

type BenchmarkResults struct {
	Results    []TestResult `json:"results"`
	Timestamp  string       `json:"timestamp"`
	Duration   string       `json:"testDuration"`
	TotalTests int          `json:"totalTests"`
}

func main() {
	rand.Seed(time.Now().UnixNano())

	fmt.Println("╔═══════════════════════════════════════════════════════════════════════════════════════════════════════════════╗")
	fmt.Println("║   💬 STREAMING TOKEN PREDICTION: HARDCORE EDITION                                                            ║")
	fmt.Println("║                                                                                                               ║")
	fmt.Println("║   ⚠️  60 SECOND TEST | 5-LAYER DEEP NETWORK | BATCH SIZE 100                                                 ║")
	fmt.Println("║   LLM generation cannot pause - blocking = terrible latency                                                  ║")
	fmt.Println("╚═══════════════════════════════════════════════════════════════════════════════════════════════════════════════╝")

	modes := []TrainingMode{ModeNormalBP, ModeStepBP, ModeTween, ModeTweenChain, ModeStepTween, ModeStepTweenChain}
	
	fmt.Printf("\n📊 Running %d tests | 60s each | Batch size: %d\n\n", len(modes), TPBatchSize)

	results := &BenchmarkResults{
		Results:    make([]TestResult, 0, len(modes)),
		Timestamp:  time.Now().Format(time.RFC3339),
		Duration:   TPTestDuration.String(),
		TotalTests: len(modes),
	}

	var wg sync.WaitGroup
	var mu sync.Mutex
	sem := make(chan struct{}, TPMaxConcurrent)

	for i, mode := range modes {
		wg.Add(1)
		go func(m TrainingMode, idx int) {
			defer wg.Done()
			sem <- struct{}{}
			defer func() { <-sem }()

			modeName := modeNames[m]
			fmt.Printf("📝 [%d/%d] Starting %s...\n", idx+1, len(modes), modeName)

			result := runTokenTest(m)
			result.TrainingMode = modeName

			mu.Lock()
			results.Results = append(results.Results, result)
			mu.Unlock()

			fmt.Printf("✅ [%d/%d] %-15s | Accuracy: %5.1f%% | Tokens: %5d | Missed: %4d | Avail: %5.1f%% | Score: %.0f\n",
				idx+1, len(modes), modeName, result.Accuracy, result.TokensGenerated,
				result.TokensMissed, result.AvailabilityPct, result.Score)
		}(mode, i)
	}

	wg.Wait()
	saveResults(results)
	printSummaryTable(results)
}

func runTokenTest(mode TrainingMode) TestResult {
	result := TestResult{}
	defer func() {
		if r := recover(); r != nil {
			result.Error = fmt.Sprintf("panic: %v", r)
		}
	}()

	numWindows := int(TPTestDuration / TPWindowDuration)
	result.Windows = make([]TimeWindow, numWindows)
	for i := range result.Windows {
		result.Windows[i].TimeMs = (i + 1) * int(TPWindowDuration.Milliseconds())
	}

	net := createDeepNetwork()
	numLayers := net.TotalLayers()

	var state *nn.StepState
	if mode == ModeStepBP || mode == ModeStepTween || mode == ModeStepTweenChain {
		state = net.InitStepState(TPInputSize)
	}

	var ts *nn.TweenState
	if mode == ModeTween || mode == ModeTweenChain || mode == ModeStepTween || mode == ModeStepTweenChain {
		ts = nn.NewTweenState(net, nil)
		if mode == ModeTweenChain || mode == ModeStepTweenChain {
			ts.Config.UseChainRule = true
		}
	}

	currentStyle := StyleTechnical
	buffer := make([]byte, TPInputSize)
	for i := range buffer {
		buffer[i] = 'a' + byte(rand.Intn(26))
	}
	currentWord := getRandomWord(currentStyle)
	charIdx := 0

	type Sample struct {
		Input  []float32
		Target int
	}
	trainBatch := make([]Sample, 0, TPBatchSize+10)
	lastTrainTime := time.Now()
	isBlocked := false

	start := time.Now()
	currentWindow := 0
	lastSwitchTime := start
	lastTokenTime := start
	var totalBlockedTime time.Duration

	for time.Since(start) < TPTestDuration {
		elapsed := time.Since(start)

		newWindow := int(elapsed / TPWindowDuration)
		if newWindow > currentWindow && newWindow < numWindows {
			currentWindow = newWindow
		}

		if time.Since(lastSwitchTime) >= TPSwitchInterval {
			currentStyle = WritingStyle((int(currentStyle) + 1) % 6)
			lastSwitchTime = time.Now()
			currentWord = getRandomWord(currentStyle)
			charIdx = 0
		}

		if time.Since(lastTokenTime) >= TPTokenInterval {
			lastTokenTime = time.Now()
			result.TotalTokens++

			// If blocked, we MISS this token generation opportunity
			if isBlocked {
				result.TokensMissed++
				continue
			}

			if charIdx >= len(currentWord) {
				currentWord = getRandomWord(currentStyle)
				charIdx = 0
			}
			nextChar := currentWord[charIdx]
			charIdx++

			input := encodeBuffer(buffer)

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

			predicted := argmax(output)
			actual := int(nextChar - 'a')
			if actual < 0 || actual >= 26 {
				actual = 0
			}

			result.TokensGenerated++
			if currentWindow < numWindows {
				result.Windows[currentWindow].Tokens++
			}

			if predicted == actual {
				result.CorrectPredictions++
				if currentWindow < numWindows {
					result.Windows[currentWindow].Correct++
				}
			}

			copy(buffer[:TPInputSize-1], buffer[1:])
			if nextChar >= 'a' && nextChar <= 'z' {
				buffer[TPInputSize-1] = nextChar
			} else {
				buffer[TPInputSize-1] = 'a'
			}

			trainBatch = append(trainBatch, Sample{Input: input, Target: actual})

			switch mode {
			case ModeNormalBP:
				if len(trainBatch) >= TPBatchSize && time.Since(lastTrainTime) > TPTrainInterval {
					batches := make([]nn.TrainingBatch, len(trainBatch))
					for i, s := range trainBatch {
						target := make([]float32, TPOutputSize)
						if s.Target >= 0 && s.Target < TPOutputSize {
							target[s.Target] = 1.0
						}
						batches[i] = nn.TrainingBatch{Input: s.Input, Target: target}
					}
					isBlocked = true
					trainStart := time.Now()
					net.Train(batches, &nn.TrainingConfig{Epochs: 1, LearningRate: TPLearningRate, LossType: "crossentropy"})
					blockDuration := time.Since(trainStart)
					isBlocked = false
					totalBlockedTime += blockDuration
					trainBatch = trainBatch[:0]
					lastTrainTime = time.Now()
				}
			case ModeStepBP:
				targetVec := make([]float32, TPOutputSize)
				if actual >= 0 && actual < TPOutputSize {
					targetVec[actual] = 1.0
				}
				grad := make([]float32, len(output))
				for i := range grad {
					grad[i] = clipGrad(output[i]-targetVec[i], 0.5)
				}
				net.StepBackward(state, grad)
				net.ApplyGradients(TPLearningRate)
			case ModeTween, ModeTweenChain:
				if len(trainBatch) >= TPBatchSize && time.Since(lastTrainTime) > TPTrainInterval {
					isBlocked = true
					trainStart := time.Now()
					for _, s := range trainBatch {
						target := make([]float32, TPOutputSize)
						if s.Target >= 0 && s.Target < TPOutputSize {
							target[s.Target] = 1.0
						}
						out := ts.ForwardPass(net, s.Input)
						outputGrad := make([]float32, len(out))
						for i := range outputGrad {
							if i < len(target) {
								outputGrad[i] = target[i] - out[i]
							}
						}
						ts.ChainGradients[net.TotalLayers()] = outputGrad
						ts.BackwardTargets[net.TotalLayers()] = target
						ts.TweenWeightsChainRule(net, TPLearningRate)
					}
					blockDuration := time.Since(trainStart)
					isBlocked = false
					totalBlockedTime += blockDuration
					trainBatch = trainBatch[:0]
					lastTrainTime = time.Now()
				}
			case ModeStepTween, ModeStepTweenChain:
				targetVec := make([]float32, TPOutputSize)
				if actual >= 0 && actual < TPOutputSize {
					targetVec[actual] = 1.0
				}
				outputGrad := make([]float32, len(output))
				for i := range outputGrad {
					outputGrad[i] = targetVec[i] - output[i]
				}
				ts.ChainGradients[net.TotalLayers()] = outputGrad
				ts.BackwardTargets[net.TotalLayers()] = targetVec
				ts.TweenWeightsChainRule(net, TPLearningRate)
			}
		}
	}

	result.TrainTimeSec = time.Since(start).Seconds()
	result.TotalBlockedMs = totalBlockedTime.Seconds() * 1000

	if result.TokensGenerated > 0 {
		result.Accuracy = float64(result.CorrectPredictions) / float64(result.TokensGenerated) * 100
	}
	if result.TrainTimeSec > 0 {
		result.TokensPerSec = float64(result.TokensGenerated) / result.TrainTimeSec
		totalTimeMs := result.TrainTimeSec * 1000
		result.AvailabilityPct = ((totalTimeMs - result.TotalBlockedMs) / totalTimeMs) * 100
	}

	// Penalty for missed tokens
	missedPenalty := float64(result.TokensMissed) / 50.0
	result.Score = result.Accuracy * result.TokensPerSec * (result.AvailabilityPct / 100) / 100 - missedPenalty
	if result.Score < 0 {
		result.Score = 0
	}
	result.Passed = result.Score > 0
	return result
}

func createDeepNetwork() *nn.Network {
	config := nn.SimpleNetworkConfig{
		InputSize:  TPInputSize,
		HiddenSize: TPHiddenSize,
		OutputSize: TPOutputSize,
		Activation: nn.ActivationLeakyReLU,
		InitScale:  TPInitScale,
		NumLayers:  TPNumLayers,
		LayerType:  nn.BrainDense,
		DType:      nn.DTypeFloat32,
	}
	return nn.BuildSimpleNetwork(config)
}

func getRandomWord(style WritingStyle) string {
	words := stylePatterns[style]
	return words[rand.Intn(len(words))]
}

func encodeBuffer(buffer []byte) []float32 {
	input := make([]float32, TPInputSize)
	for i, b := range buffer {
		if b >= 'a' && b <= 'z' {
			input[i] = float32(b-'a') / 26.0
		} else {
			input[i] = 0.5
		}
	}
	return input
}

func argmax(arr []float32) int {
	if len(arr) == 0 {
		return 0
	}
	maxIdx := 0
	maxVal := arr[0]
	for i, v := range arr {
		if v > maxVal {
			maxVal = v
			maxIdx = i
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
	os.WriteFile("token_prediction_hardcore_results.json", data, 0644)
	fmt.Println("\n📁 Results saved to token_prediction_hardcore_results.json")
}

func printSummaryTable(results *BenchmarkResults) {
	fmt.Println("\n╔═══════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════╗")
	fmt.Println("║                                    TOKEN PREDICTION HARDCORE SUMMARY                                                                 ║")
	fmt.Println("╠═══════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════╣")
	fmt.Println("║  Training Mode    │ Accuracy │ Generated │ Missed │ Tokens/s │ Blocked(ms) │ Avail% │     Score     │ Status                         ║")
	fmt.Println("╠═══════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════╣")

	for _, r := range results.Results {
		status := "✅ PASS"
		if !r.Passed {
			status = "❌ FAIL"
		}
		fmt.Printf("║  %-15s │  %5.1f%%  │    %5d  │  %4d  │   %5.0f  │   %8.0f  │ %5.1f%% │ %13.1f │ %s ║\n",
			r.TrainingMode, r.Accuracy, r.TokensGenerated, r.TokensMissed,
			r.TokensPerSec, r.TotalBlockedMs, r.AvailabilityPct, r.Score, status)
	}

	fmt.Println("╚═══════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════╝")

	var best, worst *TestResult
	for i := range results.Results {
		if best == nil || results.Results[i].Score > best.Score {
			best = &results.Results[i]
		}
		if worst == nil || results.Results[i].Score < worst.Score {
			worst = &results.Results[i]
		}
	}
	if best != nil && worst != nil {
		fmt.Printf("\n🏆 BEST: %s | Score: %.1f | Accuracy: %.1f%% | Missed tokens: %d\n",
			best.TrainingMode, best.Score, best.Accuracy, best.TokensMissed)
		fmt.Printf("💀 WORST: %s | Score: %.1f | Missed %d tokens while training\n",
			worst.TrainingMode, worst.Score, worst.TokensMissed)
	}
}
