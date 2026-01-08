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
// ADVERSARIAL ROBUSTNESS BENCHMARK
// ═══════════════════════════════════════════════════════════════════════════════
//
// Train with noisy/perturbed inputs. Test on clean data.
// Question: Which training modes are most robust to input noise?

const (
	ARInputSize  = 16   // Bigger input
	ARHiddenSize = 96   // Wider
	AROutputSize = 6    // More classes
	ARNumLayers  = 5    // Deeper

	ARLearningRate = float32(0.015)
	ARBatchSize    = 80  // Larger batches

	ARTestDuration   = 60 * time.Second
	ARTrainInterval  = 250 * time.Millisecond
	ARSampleInterval = 8 * time.Millisecond

	NoiseLevel = 0.5  // 50% noise - MUCH harder

	ARMaxConcurrent = 6
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

type TestResult struct {
	TrainingMode      string  `json:"trainingMode"`
	TrainAccuracyNoisy float64 `json:"trainAccuracyNoisy"`
	TestAccuracyClean  float64 `json:"testAccuracyClean"`
	TestAccuracyNoisy  float64 `json:"testAccuracyNoisy"`
	RobustnessScore   float64 `json:"robustnessScore"` // How well it generalizes
	TotalSamples      int     `json:"totalSamples"`
	TrainTimeSec      float64 `json:"trainTimeSec"`
	AvailabilityPct   float64 `json:"availabilityPct"`
	TotalBlockedMs    float64 `json:"totalBlockedMs"`
	Score             float64 `json:"score"`
}

type BenchmarkResults struct {
	Results   []TestResult `json:"results"`
	Timestamp string       `json:"timestamp"`
}

// Generate clean class patterns - 6 distinct patterns
func generateCleanSample(classIdx int) []float32 {
	input := make([]float32, ARInputSize)
	for i := 0; i < ARInputSize; i++ {
		switch classIdx % AROutputSize {
		case 0: // Low ascending
			input[i] = float32(i) * 0.05
		case 1: // High ascending
			input[i] = 0.5 + float32(i)*0.03
		case 2: // Alternating high
			if i%2 == 0 { input[i] = 0.8 } else { input[i] = 0.2 }
		case 3: // Alternating low
			if i%2 == 0 { input[i] = 0.3 } else { input[i] = 0.7 }
		case 4: // Center peak
			dist := float32(i) - float32(ARInputSize/2)
			if dist < 0 { dist = -dist }
			input[i] = 1.0 - dist*0.1
		case 5: // Center valley
			dist := float32(i) - float32(ARInputSize/2)
			if dist < 0 { dist = -dist }
			input[i] = dist * 0.1
		}
	}
	return input
}

func addNoise(input []float32, noiseLevel float64) []float32 {
	noisy := make([]float32, len(input))
	for i, v := range input {
		noise := float32((rand.Float64() - 0.5) * 2 * noiseLevel)
		noisy[i] = v + noise
		if noisy[i] < 0 { noisy[i] = 0 }
		if noisy[i] > 1 { noisy[i] = 1 }
	}
	return noisy
}

func main() {
	rand.Seed(time.Now().UnixNano())

	fmt.Println("╔═══════════════════════════════════════════════════════════════════════════════════════════════════════════════╗")
	fmt.Println("║   🛡️ ADVERSARIAL ROBUSTNESS BENCHMARK                                                                        ║")
	fmt.Println("║                                                                                                               ║")
	fmt.Println("║   Train with 30% noise → Test on clean data                                                                  ║")
	fmt.Println("║   Question: Which modes generalize best despite noisy training?                                              ║")
	fmt.Println("╚═══════════════════════════════════════════════════════════════════════════════════════════════════════════════╝")

	modes := []TrainingMode{ModeNormalBP, ModeStepBP, ModeTween, ModeTweenChain, ModeStepTween, ModeStepTweenChain}
	
	fmt.Printf("\n📊 Running %d tests | 60s each | Noise level: %.0f%%\n\n", len(modes), NoiseLevel*100)

	results := &BenchmarkResults{
		Results:   make([]TestResult, 0, len(modes)),
		Timestamp: time.Now().Format(time.RFC3339),
	}

	var wg sync.WaitGroup
	var mu sync.Mutex
	sem := make(chan struct{}, ARMaxConcurrent)

	for i, mode := range modes {
		wg.Add(1)
		go func(m TrainingMode, idx int) {
			defer wg.Done()
			sem <- struct{}{}
			defer func() { <-sem }()

			fmt.Printf("🛡️ [%d/%d] Starting %s...\n", idx+1, len(modes), modeNames[m])
			result := runAdversarialTest(m)
			result.TrainingMode = modeNames[m]

			mu.Lock()
			results.Results = append(results.Results, result)
			mu.Unlock()

			fmt.Printf("✅ [%d/%d] %-15s | Train(noisy): %5.1f%% | Test(clean): %5.1f%% | Robust: %5.1f%% | Score: %.0f\n",
				idx+1, len(modes), modeNames[m], result.TrainAccuracyNoisy, result.TestAccuracyClean,
				result.RobustnessScore, result.Score)
		}(mode, i)
	}

	wg.Wait()
	saveResults(results)
	printSummaryTable(results)
}

func runAdversarialTest(mode TrainingMode) TestResult {
	result := TestResult{}

	net := createNetwork()
	numLayers := net.TotalLayers()

	var state *nn.StepState
	if mode == ModeStepBP || mode == ModeStepTween || mode == ModeStepTweenChain {
		state = net.InitStepState(ARInputSize)
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
	trainBatch := make([]Sample, 0, ARBatchSize+10)
	lastTrainTime := time.Now()
	isBlocked := false
	var totalBlockedTime time.Duration
	correctTrain := 0

	start := time.Now()
	lastSampleTime := start

	for time.Since(start) < ARTestDuration {
		if time.Since(lastSampleTime) >= ARSampleInterval {
			lastSampleTime = time.Now()

			if isBlocked {
				continue
			}

			// Generate noisy training sample
			classIdx := rand.Intn(AROutputSize)
			cleanInput := generateCleanSample(classIdx)
			noisyInput := addNoise(cleanInput, NoiseLevel)
			result.TotalSamples++

			var output []float32
			switch mode {
			case ModeNormalBP, ModeTween, ModeTweenChain:
				output, _ = net.ForwardCPU(noisyInput)
			case ModeStepBP:
				state.SetInput(noisyInput)
				for s := 0; s < numLayers; s++ {
					net.StepForward(state)
				}
				output = state.GetOutput()
			case ModeStepTween, ModeStepTweenChain:
				output = ts.ForwardPass(net, noisyInput)
			}

			if argmax(output) == classIdx {
				correctTrain++
			}

			trainBatch = append(trainBatch, Sample{Input: noisyInput, Target: classIdx})

			switch mode {
			case ModeNormalBP:
				if len(trainBatch) >= ARBatchSize && time.Since(lastTrainTime) > ARTrainInterval {
					batches := make([]nn.TrainingBatch, len(trainBatch))
					for i, s := range trainBatch {
						target := make([]float32, AROutputSize)
						target[s.Target] = 1.0
						batches[i] = nn.TrainingBatch{Input: s.Input, Target: target}
					}
					isBlocked = true
					trainStart := time.Now()
					net.Train(batches, &nn.TrainingConfig{Epochs: 1, LearningRate: ARLearningRate, LossType: "crossentropy"})
					totalBlockedTime += time.Since(trainStart)
					isBlocked = false
					trainBatch = trainBatch[:0]
					lastTrainTime = time.Now()
				}
			case ModeStepBP:
				target := make([]float32, AROutputSize)
				target[classIdx] = 1.0
				grad := make([]float32, len(output))
				for i := range grad {
					if i < len(target) {
						grad[i] = clipGrad(output[i]-target[i], 0.5)
					}
				}
				net.StepBackward(state, grad)
				net.ApplyGradients(ARLearningRate)
			case ModeTween, ModeTweenChain:
				if len(trainBatch) >= ARBatchSize && time.Since(lastTrainTime) > ARTrainInterval {
					isBlocked = true
					trainStart := time.Now()
					for _, s := range trainBatch {
						target := make([]float32, AROutputSize)
						target[s.Target] = 1.0
						out := ts.ForwardPass(net, s.Input)
						outputGrad := make([]float32, len(out))
						for i := range outputGrad {
							if i < len(target) {
								outputGrad[i] = target[i] - out[i]
							}
						}
						ts.ChainGradients[numLayers] = outputGrad
						ts.BackwardTargets[numLayers] = target
						ts.TweenWeightsChainRule(net, ARLearningRate)
					}
					totalBlockedTime += time.Since(trainStart)
					isBlocked = false
					trainBatch = trainBatch[:0]
					lastTrainTime = time.Now()
				}
			case ModeStepTween, ModeStepTweenChain:
				target := make([]float32, AROutputSize)
				target[classIdx] = 1.0
				outputGrad := make([]float32, len(output))
				for i := range outputGrad {
					if i < len(target) {
						outputGrad[i] = target[i] - output[i]
					}
				}
				ts.ChainGradients[numLayers] = outputGrad
				ts.BackwardTargets[numLayers] = target
				ts.TweenWeightsChainRule(net, ARLearningRate)
			}
		}
	}

	// Test on clean data
	cleanCorrect := 0
	noisyCorrect := 0
	testSamples := 100

	for i := 0; i < testSamples; i++ {
		classIdx := i % AROutputSize
		cleanInput := generateCleanSample(classIdx)
		noisyInput := addNoise(cleanInput, NoiseLevel)

		var cleanOut, noisyOut []float32
		switch mode {
		case ModeNormalBP, ModeTween, ModeTweenChain:
			cleanOut, _ = net.ForwardCPU(cleanInput)
			noisyOut, _ = net.ForwardCPU(noisyInput)
		case ModeStepBP:
			state.SetInput(cleanInput)
			for s := 0; s < numLayers; s++ { net.StepForward(state) }
			cleanOut = state.GetOutput()
			state.SetInput(noisyInput)
			for s := 0; s < numLayers; s++ { net.StepForward(state) }
			noisyOut = state.GetOutput()
		case ModeStepTween, ModeStepTweenChain:
			cleanOut = ts.ForwardPass(net, cleanInput)
			noisyOut = ts.ForwardPass(net, noisyInput)
		}

		if argmax(cleanOut) == classIdx { cleanCorrect++ }
		if argmax(noisyOut) == classIdx { noisyCorrect++ }
	}

	result.TrainTimeSec = time.Since(start).Seconds()
	result.TotalBlockedMs = totalBlockedTime.Seconds() * 1000
	if result.TotalSamples > 0 {
		result.TrainAccuracyNoisy = float64(correctTrain) / float64(result.TotalSamples) * 100
	}
	result.TestAccuracyClean = float64(cleanCorrect) / float64(testSamples) * 100
	result.TestAccuracyNoisy = float64(noisyCorrect) / float64(testSamples) * 100

	if result.TrainTimeSec > 0 {
		totalTimeMs := result.TrainTimeSec * 1000
		result.AvailabilityPct = ((totalTimeMs - result.TotalBlockedMs) / totalTimeMs) * 100
	}

	// Robustness = how well clean performance predicts noisy performance
	if result.TestAccuracyClean > 0 {
		result.RobustnessScore = result.TestAccuracyNoisy / result.TestAccuracyClean * 100
	}

	// Score = clean accuracy × robustness - blocked penalty
	// 1000ms blocked = -10 point penalty
	blockedPenalty := result.TotalBlockedMs / 100.0
	result.Score = result.TestAccuracyClean * result.RobustnessScore / 100 - blockedPenalty
	if math.IsNaN(result.Score) || result.Score < 0 {
		result.Score = 0
	}
	return result
}

func createNetwork() *nn.Network {
	config := nn.SimpleNetworkConfig{
		InputSize:  ARInputSize,
		HiddenSize: ARHiddenSize,
		OutputSize: AROutputSize,
		Activation: nn.ActivationLeakyReLU,
		InitScale:  0.4,
		NumLayers:  ARNumLayers,
		LayerType:  nn.BrainDense,
		DType:      nn.DTypeFloat32,
	}
	return nn.BuildSimpleNetwork(config)
}

func argmax(arr []float32) int {
	if len(arr) == 0 { return 0 }
	maxIdx, maxVal := 0, arr[0]
	for i, v := range arr {
		if v > maxVal { maxVal, maxIdx = v, i }
	}
	return maxIdx
}

func clipGrad(grad, maxVal float32) float32 {
	if grad > maxVal { return maxVal }
	if grad < -maxVal { return -maxVal }
	return grad
}

func saveResults(results *BenchmarkResults) {
	data, _ := json.MarshalIndent(results, "", "  ")
	os.WriteFile("adversarial_results.json", data, 0644)
	fmt.Println("\n📁 Results saved to adversarial_results.json")
}

func printSummaryTable(results *BenchmarkResults) {
	fmt.Println("\n╔═══════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════╗")
	fmt.Println("║                                    ADVERSARIAL ROBUSTNESS BENCHMARK SUMMARY                                                  ║")
	fmt.Println("╠═══════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════╣")
	fmt.Println("║  Training Mode    │ Train(noisy) │ Test(clean) │ Test(noisy) │ Robust% │ Blocked(ms) │ Avail% │    Score    │ Status        ║")
	fmt.Println("╠═══════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════╣")

	for _, r := range results.Results {
		fmt.Printf("║  %-15s │     %5.1f%%   │    %5.1f%%   │    %5.1f%%   │  %5.1f%% │   %8.0f  │ %5.1f%% │ %11.1f │ ✅ PASS ║\n",
			r.TrainingMode, r.TrainAccuracyNoisy, r.TestAccuracyClean, r.TestAccuracyNoisy,
			r.RobustnessScore, r.TotalBlockedMs, r.AvailabilityPct, r.Score)
	}

	fmt.Println("╚═══════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════╝")

	var best *TestResult
	for i := range results.Results {
		if best == nil || results.Results[i].Score > best.Score {
			best = &results.Results[i]
		}
	}
	if best != nil {
		fmt.Printf("\n🏆 MOST ROBUST: %s | Clean: %.1f%% | Noisy: %.1f%% | Robustness: %.1f%%\n",
			best.TrainingMode, best.TestAccuracyClean, best.TestAccuracyNoisy, best.RobustnessScore)
	}
}
