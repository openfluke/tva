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
// STREAMING ANOMALY DETECTION - HARDCORE EDITION
// ═══════════════════════════════════════════════════════════════════════════════
//
// HARDCORE: 60 second test, DEEP networks, LARGE batch training
// Missing anomalies during batch training = security vulnerabilities!

const (
	// Network architecture - DEEP
	ADInputSize  = 32  // Larger window
	ADHiddenSize = 128
	ADOutputSize = 1
	ADNumLayers  = 5

	// Training - HEAVY batching
	ADLearningRate = float32(0.01)
	ADInitScale    = float32(0.3)
	ADBatchSize    = 100

	AnomalyThreshold = 0.5

	// Timing - LONG
	ADTestDuration   = 60 * time.Second
	ADWindowDuration = 100 * time.Millisecond
	ADSwitchInterval = 5 * time.Second
	ADTrainInterval  = 500 * time.Millisecond
	ADSampleInterval = 5 * time.Millisecond // Fast sampling

	ADMaxConcurrent = 6
)

// TrafficPattern enum
type TrafficPattern int

const (
	PatternNormal TrafficPattern = iota
	PatternSpike
	PatternDrift
	PatternOscillation
	PatternBurst      // NEW
	PatternGradual    // NEW
)

var patternNames = []string{"NORMAL", "SPIKE", "DRIFT", "OSCILLATION", "BURST", "GRADUAL"}

// TrainingMode enum
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
	TimeMs        int     `json:"timeMs"`
	TruePositives int     `json:"truePositives"`
	FalseNegatives int    `json:"falseNegatives"`
	Samples       int     `json:"samples"`
	BlockedMs     float64 `json:"blockedMs"`
}

type TestResult struct {
	TrainingMode        string       `json:"trainingMode"`
	Windows             []TimeWindow `json:"windows"`
	TotalTruePositives  int          `json:"totalTruePositives"`
	TotalFalseNegatives int          `json:"totalFalseNegatives"`
	TotalAnomalies      int          `json:"totalAnomalies"`
	TotalSamples        int          `json:"totalSamples"`
	MissedDuringBlock   int          `json:"missedDuringBlock"`
	TrainTimeSec        float64      `json:"trainTimeSec"`
	DetectionRate       float64      `json:"detectionRate"`
	AvailabilityPct     float64      `json:"availabilityPct"`
	TotalBlockedMs      float64      `json:"totalBlockedMs"`
	Score               float64      `json:"score"`
	Passed              bool         `json:"passed"`
	Error               string       `json:"error,omitempty"`
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
	fmt.Println("║   🔍 STREAMING ANOMALY DETECTION: HARDCORE EDITION                                                           ║")
	fmt.Println("║                                                                                                               ║")
	fmt.Println("║   ⚠️  60 SECOND TEST | 5-LAYER DEEP NETWORK | BATCH SIZE 100                                                 ║")
	fmt.Println("║   Missing anomalies during batch training = SECURITY VULNERABILITY                                          ║")
	fmt.Println("╚═══════════════════════════════════════════════════════════════════════════════════════════════════════════════╝")

	modes := []TrainingMode{ModeNormalBP, ModeStepBP, ModeTween, ModeTweenChain, ModeStepTween, ModeStepTweenChain}
	
	fmt.Printf("\n📊 Running %d tests | 60s each | Batch size: %d\n\n", len(modes), ADBatchSize)

	results := &BenchmarkResults{
		Results:    make([]TestResult, 0, len(modes)),
		Timestamp:  time.Now().Format(time.RFC3339),
		Duration:   ADTestDuration.String(),
		TotalTests: len(modes),
	}

	var wg sync.WaitGroup
	var mu sync.Mutex
	sem := make(chan struct{}, ADMaxConcurrent)

	for i, mode := range modes {
		wg.Add(1)
		go func(m TrainingMode, idx int) {
			defer wg.Done()
			sem <- struct{}{}
			defer func() { <-sem }()

			modeName := modeNames[m]
			fmt.Printf("🔎 [%d/%d] Starting %s...\n", idx+1, len(modes), modeName)

			result := runAnomalyTest(m)
			result.TrainingMode = modeName

			mu.Lock()
			results.Results = append(results.Results, result)
			mu.Unlock()

			fmt.Printf("✅ [%d/%d] %-15s | Detected: %4d/%4d | Missed(blocked): %4d | Avail: %5.1f%% | Score: %.0f\n",
				idx+1, len(modes), modeName, result.TotalTruePositives, result.TotalAnomalies,
				result.MissedDuringBlock, result.AvailabilityPct, result.Score)
		}(mode, i)
	}

	wg.Wait()
	saveResults(results)
	printSummaryTable(results)
}

func runAnomalyTest(mode TrainingMode) TestResult {
	result := TestResult{}
	defer func() {
		if r := recover(); r != nil {
			result.Error = fmt.Sprintf("panic: %v", r)
		}
	}()

	numWindows := int(ADTestDuration / ADWindowDuration)
	result.Windows = make([]TimeWindow, numWindows)
	for i := range result.Windows {
		result.Windows[i].TimeMs = (i + 1) * int(ADWindowDuration.Milliseconds())
	}

	net := createDeepNetwork()
	numLayers := net.TotalLayers()

	var state *nn.StepState
	if mode == ModeStepBP || mode == ModeStepTween || mode == ModeStepTweenChain {
		state = net.InitStepState(ADInputSize)
	}

	var ts *nn.TweenState
	if mode == ModeTween || mode == ModeTweenChain || mode == ModeStepTween || mode == ModeStepTweenChain {
		ts = nn.NewTweenState(net, nil)
		if mode == ModeTweenChain || mode == ModeStepTweenChain {
			ts.Config.UseChainRule = true
		}
	}

	buffer := make([]float32, ADInputSize)
	for i := range buffer {
		buffer[i] = 0.5
	}
	currentPattern := PatternNormal
	sampleIdx := 0
	driftOffset := float32(0.0)

	type Sample struct {
		Input  []float32
		Target float32
	}
	trainBatch := make([]Sample, 0, ADBatchSize+10)
	lastTrainTime := time.Now()
	isBlocked := false

	start := time.Now()
	currentWindow := 0
	lastSwitchTime := start
	lastSampleTime := start
	var totalBlockedTime time.Duration

	for time.Since(start) < ADTestDuration {
		elapsed := time.Since(start)

		newWindow := int(elapsed / ADWindowDuration)
		if newWindow > currentWindow && newWindow < numWindows {
			currentWindow = newWindow
		}

		if time.Since(lastSwitchTime) >= ADSwitchInterval {
			currentPattern = TrafficPattern((int(currentPattern) + 1) % 6)
			lastSwitchTime = time.Now()
			driftOffset = 0.0
		}

		if time.Since(lastSampleTime) >= ADSampleInterval {
			lastSampleTime = time.Now()
			sampleIdx++

			newValue, isAnomaly := generateTrafficSample(currentPattern, sampleIdx, &driftOffset)
			copy(buffer[:ADInputSize-1], buffer[1:])
			buffer[ADInputSize-1] = newValue

			input := make([]float32, ADInputSize)
			copy(input, buffer)

			result.TotalSamples++
			if isAnomaly {
				result.TotalAnomalies++
			}

			// If blocked, we MISS this detection opportunity
			if isBlocked {
				if isAnomaly {
					result.MissedDuringBlock++
					result.TotalFalseNegatives++
				}
				continue
			}

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

			predicted := len(output) > 0 && output[0] > AnomalyThreshold

			if isAnomaly && predicted {
				result.TotalTruePositives++
				if currentWindow < numWindows {
					result.Windows[currentWindow].TruePositives++
				}
			} else if isAnomaly && !predicted {
				result.TotalFalseNegatives++
				if currentWindow < numWindows {
					result.Windows[currentWindow].FalseNegatives++
				}
			}

			targetVal := float32(0.0)
			if isAnomaly {
				targetVal = 1.0
			}
			trainBatch = append(trainBatch, Sample{Input: input, Target: targetVal})

			switch mode {
			case ModeNormalBP:
				if len(trainBatch) >= ADBatchSize && time.Since(lastTrainTime) > ADTrainInterval {
					batches := make([]nn.TrainingBatch, len(trainBatch))
					for i, s := range trainBatch {
						batches[i] = nn.TrainingBatch{Input: s.Input, Target: []float32{s.Target}}
					}
					isBlocked = true
					trainStart := time.Now()
					net.Train(batches, &nn.TrainingConfig{Epochs: 1, LearningRate: ADLearningRate, LossType: "bce"})
					blockDuration := time.Since(trainStart)
					isBlocked = false
					totalBlockedTime += blockDuration
					if currentWindow < numWindows {
						result.Windows[currentWindow].BlockedMs += blockDuration.Seconds() * 1000
					}
					trainBatch = trainBatch[:0]
					lastTrainTime = time.Now()
				}
			case ModeStepBP:
				grad := []float32{clipGrad(output[0]-targetVal, 0.5)}
				net.StepBackward(state, grad)
				net.ApplyGradients(ADLearningRate)
			case ModeTween, ModeTweenChain:
				if len(trainBatch) >= ADBatchSize && time.Since(lastTrainTime) > ADTrainInterval {
					isBlocked = true
					trainStart := time.Now()
					for _, s := range trainBatch {
						out := ts.ForwardPass(net, s.Input)
						ts.ChainGradients[net.TotalLayers()] = []float32{s.Target - out[0]}
						ts.BackwardTargets[net.TotalLayers()] = []float32{s.Target}
						ts.TweenWeightsChainRule(net, ADLearningRate)
					}
					blockDuration := time.Since(trainStart)
					isBlocked = false
					totalBlockedTime += blockDuration
					trainBatch = trainBatch[:0]
					lastTrainTime = time.Now()
				}
			case ModeStepTween, ModeStepTweenChain:
				ts.ChainGradients[net.TotalLayers()] = []float32{targetVal - output[0]}
				ts.BackwardTargets[net.TotalLayers()] = []float32{targetVal}
				ts.TweenWeightsChainRule(net, ADLearningRate)
			}
		}
	}

	result.TrainTimeSec = time.Since(start).Seconds()
	result.TotalBlockedMs = totalBlockedTime.Seconds() * 1000
	
	if result.TotalAnomalies > 0 {
		result.DetectionRate = float64(result.TotalTruePositives) / float64(result.TotalAnomalies) * 100
	}
	if result.TrainTimeSec > 0 {
		totalTimeMs := result.TrainTimeSec * 1000
		result.AvailabilityPct = ((totalTimeMs - result.TotalBlockedMs) / totalTimeMs) * 100
	}
	
	// Score heavily penalizes missed anomalies during blocking
	missedPenalty := float64(result.MissedDuringBlock) * 2.0
	result.Score = result.DetectionRate * (result.AvailabilityPct / 100) - missedPenalty
	if result.Score < 0 {
		result.Score = 0
	}
	result.Passed = result.Score > 0
	return result
}

func createDeepNetwork() *nn.Network {
	config := nn.SimpleNetworkConfig{
		InputSize:  ADInputSize,
		HiddenSize: ADHiddenSize,
		OutputSize: ADOutputSize,
		Activation: nn.ActivationLeakyReLU,
		InitScale:  ADInitScale,
		NumLayers:  ADNumLayers,
		LayerType:  nn.BrainDense,
		DType:      nn.DTypeFloat32,
	}
	return nn.BuildSimpleNetwork(config)
}

func generateTrafficSample(pattern TrafficPattern, idx int, driftOffset *float32) (float32, bool) {
	t := float64(idx) * 0.1
	noise := float32(rand.Float64()*0.1 - 0.05)

	switch pattern {
	case PatternNormal:
		return float32(0.5+0.1*math.Sin(t)) + noise, false
	case PatternSpike:
		if rand.Float32() < 0.2 {
			return 0.95 + float32(rand.Float64()*0.05), true
		}
		return 0.5 + noise, false
	case PatternDrift:
		*driftOffset += 0.003
		if *driftOffset > 0.25 {
			return 0.5 + *driftOffset + noise, true
		}
		return 0.5 + *driftOffset + noise, false
	case PatternOscillation:
		value := float32(0.5 + 0.4*math.Sin(t*5))
		return value + noise, math.Abs(float64(value)-0.5) > 0.3
	case PatternBurst:
		if idx%50 < 5 {
			return 0.9 + float32(rand.Float64()*0.1), true
		}
		return 0.5 + noise, false
	case PatternGradual:
		progress := float32(idx%100) / 100.0
		if progress > 0.8 {
			return 0.5 + progress + noise, true
		}
		return 0.5 + progress*0.3 + noise, false
	}
	return 0.5 + noise, false
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
	os.WriteFile("anomaly_detection_hardcore_results.json", data, 0644)
	fmt.Println("\n📁 Results saved to anomaly_detection_hardcore_results.json")
}

func printSummaryTable(results *BenchmarkResults) {
	fmt.Println("\n╔═══════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════╗")
	fmt.Println("║                                 ANOMALY DETECTION HARDCORE SUMMARY                                                           ║")
	fmt.Println("╠═══════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════╣")
	fmt.Println("║  Training Mode    │ Detected │ Anomalies │ Missed(Block) │ Detection% │ Blocked(ms) │ Avail% │    Score    │ Status         ║")
	fmt.Println("╠═══════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════╣")

	for _, r := range results.Results {
		status := "✅ PASS"
		if !r.Passed {
			status = "❌ FAIL"
		}
		fmt.Printf("║  %-15s │   %5d  │    %5d  │      %5d    │    %5.1f%%  │   %8.0f  │ %5.1f%% │ %11.1f │ %s ║\n",
			r.TrainingMode, r.TotalTruePositives, r.TotalAnomalies, r.MissedDuringBlock,
			r.DetectionRate, r.TotalBlockedMs, r.AvailabilityPct, r.Score, status)
	}

	fmt.Println("╚═══════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════╝")

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
		fmt.Printf("\n🏆 BEST: %s | Score: %.1f | Detection: %.1f%% | Missed during block: %d\n",
			best.TrainingMode, best.Score, best.DetectionRate, best.MissedDuringBlock)
		fmt.Printf("💀 WORST: %s | Score: %.1f | Missed %d anomalies while training\n",
			worst.TrainingMode, worst.Score, worst.MissedDuringBlock)
	}
}
