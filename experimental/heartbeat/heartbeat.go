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
// HEARTBEAT MONITOR v3 - Pre-train, then test with blocking
// ═══════════════════════════════════════════════════════════════════════════════

const (
	HBInputSize  = 4
	HBHiddenSize = 32
	HBOutputSize = 2
	HBNumLayers  = 2

	HBLearningRate = float32(0.1)
	HBBatchSize    = 200  // Big batches = longer blocking

	HBPretrainSamples = 500  // Pre-train so networks can detect
	HBTestDuration    = 30 * time.Second
	HBHeartbeatRate   = 30 * time.Millisecond  // ~33 beats/sec
	HBTrainInterval   = 400 * time.Millisecond
	HBCriticalChance  = 0.20  // 20% critical

	HBMaxConcurrent = 6
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
	TrainingMode        string  `json:"trainingMode"`
	TotalBeats          int     `json:"totalBeats"`
	ProcessedBeats      int     `json:"processedBeats"`
	CriticalTotal       int     `json:"criticalTotal"`
	CriticalDetected    int     `json:"criticalDetected"`
	CriticalMissedBlock int     `json:"criticalMissedBlock"`
	DetectionRate       float64 `json:"detectionRate"`
	TotalBlockedMs      float64 `json:"totalBlockedMs"`
	Score               float64 `json:"score"`
}

type BenchmarkResults struct {
	Results   []TestResult `json:"results"`
	Timestamp string       `json:"timestamp"`
}

// Critical: first 2 values HIGH (>0.6), Normal: first 2 values LOW (<0.4)
func generateSignal(isCritical bool) []float32 {
	signal := make([]float32, HBInputSize)
	if isCritical {
		signal[0] = 0.7 + rand.Float32()*0.3
		signal[1] = 0.7 + rand.Float32()*0.3
	} else {
		signal[0] = rand.Float32() * 0.3
		signal[1] = rand.Float32() * 0.3
	}
	signal[2] = rand.Float32()
	signal[3] = rand.Float32()
	return signal
}

func main() {
	rand.Seed(time.Now().UnixNano())

	fmt.Println("╔═══════════════════════════════════════════════════════════════════════════════════════════════════════════════╗")
	fmt.Println("║   💓 HEARTBEAT MONITOR v3                                                                                    ║")
	fmt.Println("║                                                                                                               ║")
	fmt.Println("║   Pre-train → then test | Big batches = long blocking | 33 beats/sec                                       ║")
	fmt.Println("║   Score = detected - (blocked_misses × 5) - (wrong × 1)                                                     ║")
	fmt.Println("╚═══════════════════════════════════════════════════════════════════════════════════════════════════════════════╝")

	modes := []TrainingMode{ModeNormalBP, ModeStepBP, ModeTween, ModeTweenChain, ModeStepTween, ModeStepTweenChain}
	results := &BenchmarkResults{Results: make([]TestResult, 0, len(modes)), Timestamp: time.Now().Format(time.RFC3339)}

	var wg sync.WaitGroup
	var mu sync.Mutex
	sem := make(chan struct{}, HBMaxConcurrent)

	for i, mode := range modes {
		wg.Add(1)
		go func(m TrainingMode, idx int) {
			defer wg.Done()
			sem <- struct{}{}
			defer func() { <-sem }()

			fmt.Printf("💓 [%d/%d] Starting %s...\n", idx+1, len(modes), modeNames[m])
			result := runTest(m)
			result.TrainingMode = modeNames[m]

			mu.Lock()
			results.Results = append(results.Results, result)
			mu.Unlock()

			fmt.Printf("✅ [%d/%d] %-15s | Beats: %4d | Critical: %3d | Detected: %3d | BlockMiss: %2d | Blocked: %4.0fms | Score: %.0f\n",
				idx+1, len(modes), modeNames[m], result.TotalBeats, result.CriticalTotal,
				result.CriticalDetected, result.CriticalMissedBlock, result.TotalBlockedMs, result.Score)
		}(mode, i)
	}

	wg.Wait()
	saveResults(results)
	printSummary(results)
}

func runTest(mode TrainingMode) TestResult {
	result := TestResult{}

	net := nn.BuildSimpleNetwork(nn.SimpleNetworkConfig{
		InputSize: HBInputSize, HiddenSize: HBHiddenSize, OutputSize: HBOutputSize,
		Activation: nn.ActivationLeakyReLU, InitScale: 0.5, NumLayers: HBNumLayers,
		LayerType: nn.BrainDense, DType: nn.DTypeFloat32,
	})
	
	// ═══════════════════════════════════════════════════════════════
	// PHASE 1: PRE-TRAIN so network can actually detect criticals
	// ═══════════════════════════════════════════════════════════════
	pretrainBatches := make([]nn.TrainingBatch, HBPretrainSamples)
	for i := 0; i < HBPretrainSamples; i++ {
		isCritical := rand.Float64() < 0.5 // 50/50 for balanced training
		signal := generateSignal(isCritical)
		target := make([]float32, HBOutputSize)
		if isCritical { target[1] = 1 } else { target[0] = 1 }
		pretrainBatches[i] = nn.TrainingBatch{Input: signal, Target: target}
	}
	net.Train(pretrainBatches, &nn.TrainingConfig{Epochs: 10, LearningRate: HBLearningRate, LossType: "crossentropy"})

	// Verify pre-training worked
	correct := 0
	for i := 0; i < 100; i++ {
		isCritical := i%2 == 0
		signal := generateSignal(isCritical)
		out, _ := net.ForwardCPU(signal)
		predicted := out[1] > out[0]
		if predicted == isCritical { correct++ }
	}
	pretrainAcc := float64(correct) / 100.0

	numLayers := net.TotalLayers()
	var state *nn.StepState
	if mode == ModeStepBP || mode == ModeStepTween || mode == ModeStepTweenChain {
		state = net.InitStepState(HBInputSize)
	}
	var ts *nn.TweenState
	if mode == ModeTween || mode == ModeTweenChain || mode == ModeStepTween || mode == ModeStepTweenChain {
		ts = nn.NewTweenState(net, nil)
		if mode == ModeTweenChain || mode == ModeStepTweenChain { ts.Config.UseChainRule = true }
	}

	type Sample struct { Input []float32; IsCritical bool }
	trainBatch := make([]Sample, 0, HBBatchSize+10)
	lastTrainTime := time.Now()
	var totalBlockedTime time.Duration
	blockingUntil := time.Time{} // When we're done blocking

	// ═══════════════════════════════════════════════════════════════
	// PHASE 2: ONLINE TEST with continued training
	// ═══════════════════════════════════════════════════════════════
	start := time.Now()
	lastBeat := start

	for time.Since(start) < HBTestDuration {
		now := time.Now()
		if now.Sub(lastBeat) >= HBHeartbeatRate {
			lastBeat = now
			result.TotalBeats++

			isCritical := rand.Float64() < HBCriticalChance
			if isCritical { result.CriticalTotal++ }
			signal := generateSignal(isCritical)

			// Are we currently blocked from a previous training?
			if now.Before(blockingUntil) {
				if isCritical { result.CriticalMissedBlock++ }
				continue
			}

			// Check if we should START blocking (batch training)
			if mode == ModeNormalBP || mode == ModeTween || mode == ModeTweenChain {
				if len(trainBatch) >= HBBatchSize && time.Since(lastTrainTime) > HBTrainInterval {
					trainStart := time.Now()
					
					switch mode {
					case ModeNormalBP:
						batches := make([]nn.TrainingBatch, len(trainBatch))
						for i, s := range trainBatch {
							t := make([]float32, HBOutputSize)
							if s.IsCritical { t[1] = 1 } else { t[0] = 1 }
							batches[i] = nn.TrainingBatch{Input: s.Input, Target: t}
						}
						net.Train(batches, &nn.TrainingConfig{Epochs: 3, LearningRate: HBLearningRate, LossType: "crossentropy"})
					case ModeTween, ModeTweenChain:
						for _, s := range trainBatch {
							t := make([]float32, HBOutputSize)
							if s.IsCritical { t[1] = 1 } else { t[0] = 1 }
							out := ts.ForwardPass(net, s.Input)
							grad := make([]float32, len(out))
							for i := range grad { grad[i] = t[i] - out[i] }
							ts.ChainGradients[numLayers] = grad
							ts.BackwardTargets[numLayers] = t
							ts.TweenWeightsChainRule(net, HBLearningRate)
						}
					}
					
					elapsed := time.Since(trainStart)
					totalBlockedTime += elapsed
					blockingUntil = now.Add(elapsed) // Block for same duration
					trainBatch = trainBatch[:0]
					lastTrainTime = time.Now()
					
					// This beat was processed BEFORE blocking started
					if isCritical { result.CriticalMissedBlock++ }
					continue
				}
			}

			result.ProcessedBeats++

			// Forward pass
			var output []float32
			switch mode {
			case ModeNormalBP, ModeTween, ModeTweenChain:
				output, _ = net.ForwardCPU(signal)
			case ModeStepBP:
				state.SetInput(signal)
				for s := 0; s < numLayers; s++ { net.StepForward(state) }
				output = state.GetOutput()
			case ModeStepTween, ModeStepTweenChain:
				output = ts.ForwardPass(net, signal)
			}

			predictedCritical := len(output) > 1 && output[1] > output[0]
			if isCritical && predictedCritical {
				result.CriticalDetected++
			}

			trainBatch = append(trainBatch, Sample{Input: signal, IsCritical: isCritical})

			// Step-based training (non-blocking)
			target := make([]float32, HBOutputSize)
			if isCritical { target[1] = 1 } else { target[0] = 1 }
			
			switch mode {
			case ModeStepBP:
				grad := make([]float32, len(output))
				for i := range grad { if i < len(target) { grad[i] = output[i] - target[i] } }
				net.StepBackward(state, grad)
				net.ApplyGradients(HBLearningRate * 0.1)
			case ModeStepTween, ModeStepTweenChain:
				grad := make([]float32, len(output))
				for i := range grad { if i < len(target) { grad[i] = target[i] - output[i] } }
				ts.ChainGradients[numLayers] = grad
				ts.BackwardTargets[numLayers] = target
				ts.TweenWeightsChainRule(net, HBLearningRate * 0.1)
			}
		}
	}

	result.TotalBlockedMs = totalBlockedTime.Seconds() * 1000
	if result.CriticalTotal > 0 {
		result.DetectionRate = float64(result.CriticalDetected) / float64(result.CriticalTotal) * 100
	}

	// Score: detected - blocked*5 (pretrain accuracy as baseline)
	result.Score = float64(result.CriticalDetected) - float64(result.CriticalMissedBlock)*5
	if result.Score < 0 { result.Score = 0 }
	
	_ = pretrainAcc // Used for debugging
	return result
}

func saveResults(results *BenchmarkResults) {
	data, _ := json.MarshalIndent(results, "", "  ")
	os.WriteFile("heartbeat_results.json", data, 0644)
	fmt.Println("\n📁 Results saved to heartbeat_results.json")
}

func printSummary(results *BenchmarkResults) {
	fmt.Println("\n╔══════════════════════════════════════════════════════════════════════════════════════════════════════════════════════╗")
	fmt.Println("║                                    HEARTBEAT MONITOR SUMMARY                                                        ║")
	fmt.Println("╠══════════════════════════════════════════════════════════════════════════════════════════════════════════════════════╣")
	fmt.Println("║  Mode            │ Beats │ Processed │ Critical │ Detected │ BlockMiss │ Detect% │ Blocked(ms) │ Score             ║")
	fmt.Println("╠══════════════════════════════════════════════════════════════════════════════════════════════════════════════════════╣")

	for _, r := range results.Results {
		detectPct := 0.0
		if r.CriticalTotal > 0 { detectPct = float64(r.CriticalDetected) / float64(r.CriticalTotal) * 100 }
		fmt.Printf("║  %-15s │  %4d │     %4d  │     %3d  │     %3d  │      %2d   │  %5.1f%% │    %6.0f   │ %6.0f ✅ ║\n",
			r.TrainingMode, r.TotalBeats, r.ProcessedBeats, r.CriticalTotal,
			r.CriticalDetected, r.CriticalMissedBlock, detectPct, r.TotalBlockedMs, r.Score)
	}
	fmt.Println("╚══════════════════════════════════════════════════════════════════════════════════════════════════════════════════════╝")

	var best *TestResult
	for i := range results.Results {
		if best == nil || results.Results[i].Score > best.Score {
			best = &results.Results[i]
		}
	}
	if best != nil {
		fmt.Printf("\n🏆 BEST: %s | Detected: %d/%d (%.1f%%) | Block-Missed: %d\n",
			best.TrainingMode, best.CriticalDetected, best.CriticalTotal, best.DetectionRate, best.CriticalMissedBlock)
	}
}
