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
// AUCTION BIDDER v2 - Simple profitable opportunities
// ═══════════════════════════════════════════════════════════════════════════════

const (
	ABInputSize  = 3   // price_change, momentum, opportunity_strength
	ABHiddenSize = 16
	ABOutputSize = 2   // pass=0, bid=1
	ABNumLayers  = 2

	ABLearningRate = float32(0.05)
	ABBatchSize    = 80

	ABTestDuration  = 30 * time.Second
	ABTickRate      = 50 * time.Millisecond  // 20 ticks/sec
	ABTrainInterval = 400 * time.Millisecond

	ABMaxConcurrent = 6
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
	TotalTicks      int     `json:"totalTicks"`
	ProcessedTicks  int     `json:"processedTicks"`
	Opportunities   int     `json:"opportunities"`
	OpsTaken        int     `json:"opsTaken"`
	OpsCorrect      int     `json:"opsCorrect"`
	OpsMissedBlock  int     `json:"opsMissedBlock"`
	Profit          float64 `json:"profit"`
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
	fmt.Println("║   💰 AUCTION BIDDER v2                                                                                       ║")
	fmt.Println("║                                                                                                               ║")
	fmt.Println("║   20 ticks/sec | Clear buy signals | Blocking = missed profits                                              ║")
	fmt.Println("║   Score = profit - missed_opportunities*$10                                                                 ║")
	fmt.Println("╚═══════════════════════════════════════════════════════════════════════════════════════════════════════════════╝")

	modes := []TrainingMode{ModeNormalBP, ModeStepBP, ModeTween, ModeTweenChain, ModeStepTween, ModeStepTweenChain}
	results := &BenchmarkResults{Results: make([]TestResult, 0, len(modes)), Timestamp: time.Now().Format(time.RFC3339)}

	var wg sync.WaitGroup
	var mu sync.Mutex
	sem := make(chan struct{}, ABMaxConcurrent)

	for i, mode := range modes {
		wg.Add(1)
		go func(m TrainingMode, idx int) {
			defer wg.Done()
			sem <- struct{}{}
			defer func() { <-sem }()

			fmt.Printf("💰 [%d/%d] Starting %s...\n", idx+1, len(modes), modeNames[m])
			result := runTest(m)
			result.TrainingMode = modeNames[m]

			mu.Lock()
			results.Results = append(results.Results, result)
			mu.Unlock()

			fmt.Printf("✅ [%d/%d] %-15s | Ticks: %4d | Opps: %3d | Taken: %3d | MissedBlock: %2d | Profit: $%.0f | Score: %.0f\n",
				idx+1, len(modes), modeNames[m], result.TotalTicks, result.Opportunities,
				result.OpsTaken, result.OpsMissedBlock, result.Profit, result.Score)
		}(mode, i)
	}

	wg.Wait()
	saveResults(results)
	printSummary(results)
}

func runTest(mode TrainingMode) TestResult {
	result := TestResult{}

	net := nn.BuildSimpleNetwork(nn.SimpleNetworkConfig{
		InputSize: ABInputSize, HiddenSize: ABHiddenSize, OutputSize: ABOutputSize,
		Activation: nn.ActivationLeakyReLU, InitScale: 0.5, NumLayers: ABNumLayers,
		LayerType: nn.BrainDense, DType: nn.DTypeFloat32,
	})
	numLayers := net.TotalLayers()

	var state *nn.StepState
	if mode == ModeStepBP || mode == ModeStepTween || mode == ModeStepTweenChain {
		state = net.InitStepState(ABInputSize)
	}
	var ts *nn.TweenState
	if mode == ModeTween || mode == ModeTweenChain || mode == ModeStepTween || mode == ModeStepTweenChain {
		ts = nn.NewTweenState(net, nil)
		if mode == ModeTweenChain || mode == ModeStepTweenChain { ts.Config.UseChainRule = true }
	}

	type Sample struct { Input []float32; ShouldBid bool }
	trainBatch := make([]Sample, 0, ABBatchSize+10)
	lastTrainTime := time.Now()
	var totalBlockedTime time.Duration

	start := time.Now()
	lastTick := start

	for time.Since(start) < ABTestDuration {
		now := time.Now()
		if now.Sub(lastTick) >= ABTickRate {
			lastTick = now
			result.TotalTicks++

			// Generate market signal - simple pattern
			priceChange := rand.Float32()*2 - 1  // -1 to 1
			momentum := rand.Float32()*2 - 1
			oppStrength := rand.Float32()
			
			// Clear rule: should bid when price is rising AND strong signal
			shouldBid := priceChange > 0.3 && oppStrength > 0.6
			isOpportunity := oppStrength > 0.6
			if isOpportunity { result.Opportunities++ }

			input := []float32{priceChange, momentum, oppStrength}

			// Check if blocked
			shouldBlock := false
			if mode == ModeNormalBP || mode == ModeTween || mode == ModeTweenChain {
				if len(trainBatch) >= ABBatchSize && time.Since(lastTrainTime) > ABTrainInterval {
					shouldBlock = true
				}
			}

			if shouldBlock {
				if isOpportunity && shouldBid {
					result.OpsMissedBlock++
				}
				
				trainStart := time.Now()
				switch mode {
				case ModeNormalBP:
					batches := make([]nn.TrainingBatch, len(trainBatch))
					for i, s := range trainBatch {
						t := make([]float32, ABOutputSize)
						if s.ShouldBid { t[1] = 1 } else { t[0] = 1 }
						batches[i] = nn.TrainingBatch{Input: s.Input, Target: t}
					}
					net.Train(batches, &nn.TrainingConfig{Epochs: 1, LearningRate: ABLearningRate, LossType: "crossentropy"})
				case ModeTween, ModeTweenChain:
					for _, s := range trainBatch {
						t := make([]float32, ABOutputSize)
						if s.ShouldBid { t[1] = 1 } else { t[0] = 1 }
						out := ts.ForwardPass(net, s.Input)
						grad := make([]float32, len(out))
						for i := range grad { grad[i] = t[i] - out[i] }
						ts.ChainGradients[numLayers] = grad
						ts.BackwardTargets[numLayers] = t
						ts.TweenWeightsChainRule(net, ABLearningRate)
					}
				}
				totalBlockedTime += time.Since(trainStart)
				trainBatch = trainBatch[:0]
				lastTrainTime = time.Now()
				continue
			}

			result.ProcessedTicks++

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

			predictBid := len(output) > 1 && output[1] > output[0]
			
			if isOpportunity {
				if predictBid {
					result.OpsTaken++
					if shouldBid {
						result.OpsCorrect++
						result.Profit += 15 // Good bid
					} else {
						result.Profit -= 10 // Bad bid
					}
				}
			}

			trainBatch = append(trainBatch, Sample{Input: input, ShouldBid: shouldBid})

			target := make([]float32, ABOutputSize)
			if shouldBid { target[1] = 1 } else { target[0] = 1 }
			
			switch mode {
			case ModeStepBP:
				grad := make([]float32, len(output))
				for i := range grad { if i < len(target) { grad[i] = output[i] - target[i] } }
				net.StepBackward(state, grad)
				net.ApplyGradients(ABLearningRate)
			case ModeStepTween, ModeStepTweenChain:
				grad := make([]float32, len(output))
				for i := range grad { if i < len(target) { grad[i] = target[i] - output[i] } }
				ts.ChainGradients[numLayers] = grad
				ts.BackwardTargets[numLayers] = target
				ts.TweenWeightsChainRule(net, ABLearningRate)
			}
		}
	}

	result.TotalBlockedMs = totalBlockedTime.Seconds() * 1000
	result.Score = result.Profit - float64(result.OpsMissedBlock)*10
	if result.Score < 0 { result.Score = 0 }
	return result
}

func saveResults(results *BenchmarkResults) {
	data, _ := json.MarshalIndent(results, "", "  ")
	os.WriteFile("auction_results.json", data, 0644)
	fmt.Println("\n📁 Results saved")
}

func printSummary(results *BenchmarkResults) {
	fmt.Println("\n╔══════════════════════════════════════════════════════════════════════════════════════════════════════════════════════╗")
	fmt.Println("║                                    AUCTION BIDDER SUMMARY                                                           ║")
	fmt.Println("╠══════════════════════════════════════════════════════════════════════════════════════════════════════════════════════╣")
	fmt.Println("║  Mode            │ Ticks │ Processed │ Opps │ Taken │ Correct │ BlockMiss │ Profit  │ Blocked(ms) │ Score          ║")
	fmt.Println("╠══════════════════════════════════════════════════════════════════════════════════════════════════════════════════════╣")

	for _, r := range results.Results {
		fmt.Printf("║  %-15s │  %4d │     %4d  │  %3d │  %3d  │    %3d  │      %2d   │ $%5.0f  │    %6.0f   │ %6.0f ✅ ║\n",
			r.TrainingMode, r.TotalTicks, r.ProcessedTicks, r.Opportunities,
			r.OpsTaken, r.OpsCorrect, r.OpsMissedBlock, r.Profit, r.TotalBlockedMs, r.Score)
	}
	fmt.Println("╚══════════════════════════════════════════════════════════════════════════════════════════════════════════════════════╝")
}
