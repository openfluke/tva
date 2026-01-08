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
// PREDATOR-PREY ADAPTIVE CHASE GAME - HARDCORE EDITION
// ═══════════════════════════════════════════════════════════════════════════════
//
// HARDCORE: 60 second test, DEEP networks, LARGE batch training
// This should create MASSIVE differences between blocking and non-blocking modes.
//
// Key changes from regular version:
//   - 60 second duration (6x longer)
//   - 5-layer deep network with 128 hidden (much slower training)
//   - Batch size 100 before training (long blocking periods)
//   - Faster prey (harder to catch)
//   - More behavior switches

const (
	// Grid parameters
	GridSize = 12 // Bigger grid, harder to catch

	// Network architecture - DEEP and WIDE
	PPInputSize  = 20  // More context
	PPHiddenSize = 128 // Much bigger
	PPOutputSize = 4   // [up, down, left, right]
	PPNumLayers  = 5   // DEEP network

	// Training parameters - HEAVY batching
	PPLearningRate = float32(0.01)
	PPInitScale    = float32(0.3)
	PPBatchSize    = 100 // Accumulate 100 samples before training!

	// Timing - LONG test
	PPTestDuration    = 60 * time.Second // 60 SECONDS!
	PPWindowDuration  = 100 * time.Millisecond
	PPSwitchInterval  = 5 * time.Second // Switch behavior every 5 seconds
	PPTrainInterval   = 500 * time.Millisecond // Train every 500ms if batch ready
	PPStepInterval    = 10 * time.Millisecond  // Faster game tick

	// Concurrency
	PPMaxConcurrent = 6
)

// PreyBehavior enum
type PreyBehavior int

const (
	BehaviorRandom PreyBehavior = iota
	BehaviorFlee
	BehaviorZigZag
	BehaviorFreeze
	BehaviorMirror   // NEW: mirrors predator moves
	BehaviorCircle   // NEW: moves in circles
)

var behaviorNames = []string{"RANDOM", "FLEE", "ZIGZAG", "FREEZE", "MIRROR", "CIRCLE"}

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

// Position in grid
type Position struct {
	X, Y int
}

// GameState tracks the chase
type GameState struct {
	Predator        Position
	Prey            Position
	PreyHistory     []int
	Behavior        PreyBehavior
	ZigZagCounter   int
	FreezeCounter   int
	CircleCounter   int
	FreezeBursting  bool
	LastPredMove    int
}

// TimeWindow for tracking
type TimeWindow struct {
	TimeMs          int     `json:"timeMs"`
	Catches         int     `json:"catches"`
	Decisions       int     `json:"decisions"`
	BehaviorChanges int     `json:"behaviorChanges"`
	BlockedMs       float64 `json:"blockedMs"`
	AvailableMs     float64 `json:"availableMs"`
}

// TestResult holds benchmark results  
type TestResult struct {
	TrainingMode       string       `json:"trainingMode"`
	Windows            []TimeWindow `json:"windows"`
	TotalCatches       int          `json:"totalCatches"`
	TotalDecisions     int          `json:"totalDecisions"`
	TotalBehaviorSwitches int       `json:"totalBehaviorSwitches"`
	TrainTimeSec       float64      `json:"trainTimeSec"`
	CatchRate          float64      `json:"catchRate"`
	DecisionsPerSec    float64      `json:"decisionsPerSec"`
	AvailabilityPct    float64      `json:"availabilityPct"`
	TotalBlockedMs     float64      `json:"totalBlockedMs"`
	AdaptationScore    float64      `json:"adaptationScore"`
	MissedDuringBlock  int          `json:"missedDuringBlock"` // Opportunities lost while blocked
	Score              float64      `json:"score"`
	Passed             bool         `json:"passed"`
	Error              string       `json:"error,omitempty"`
}

// BenchmarkResults is the full output
type BenchmarkResults struct {
	Results    []TestResult `json:"results"`
	Timestamp  string       `json:"timestamp"`
	Duration   string       `json:"testDuration"`
	TotalTests int          `json:"totalTests"`
	Passed     int          `json:"passed"`
	Failed     int          `json:"failed"`
}

func main() {
	rand.Seed(time.Now().UnixNano())

	fmt.Println("╔═══════════════════════════════════════════════════════════════════════════════════════════════════════════════╗")
	fmt.Println("║   🎮 PREDATOR-PREY ADAPTIVE CHASE: HARDCORE EDITION                                                          ║")
	fmt.Println("║                                                                                                               ║")
	fmt.Println("║   ⚠️  60 SECOND TEST | 5-LAYER DEEP NETWORK | BATCH SIZE 100                                                 ║")
	fmt.Println("║   Prey behaviors: RANDOM → FLEE → ZIGZAG → FREEZE → MIRROR → CIRCLE (every 5s)                              ║")
	fmt.Println("║                                                                                                               ║")
	fmt.Println("║   NormalBP will BLOCK for training → miss catches | StepTweenChain NEVER blocks                             ║")
	fmt.Println("╚═══════════════════════════════════════════════════════════════════════════════════════════════════════════════╝")

	modes := []TrainingMode{ModeNormalBP, ModeStepBP, ModeTween, ModeTweenChain, ModeStepTween, ModeStepTweenChain}
	totalTests := len(modes)
	numWindows := int(PPTestDuration / PPWindowDuration)

	fmt.Printf("\n📊 Running %d tests | %d windows at %dms each | %s per test | Batch size: %d\n\n", 
		totalTests, numWindows, PPWindowDuration.Milliseconds(), PPTestDuration, PPBatchSize)

	results := &BenchmarkResults{
		Results:    make([]TestResult, 0, totalTests),
		Timestamp:  time.Now().Format(time.RFC3339),
		Duration:   PPTestDuration.String(),
		TotalTests: totalTests,
	}

	var wg sync.WaitGroup
	var mu sync.Mutex
	sem := make(chan struct{}, PPMaxConcurrent)

	for i, mode := range modes {
		wg.Add(1)
		go func(m TrainingMode, idx int) {
			defer wg.Done()
			sem <- struct{}{}
			defer func() { <-sem }()

			modeName := modeNames[m]
			fmt.Printf("🏃 [%d/%d] Starting %s (60 second test)...\n", idx+1, totalTests, modeName)

			result := runGameTest(m)
			result.TrainingMode = modeName

			mu.Lock()
			results.Results = append(results.Results, result)
			if result.Passed {
				results.Passed++
			} else {
				results.Failed++
			}
			mu.Unlock()

			status := "✅"
			if !result.Passed {
				status = "❌"
			}
			fmt.Printf("%s [%d/%d] %-15s | Catches: %4d | Missed: %4d | Blocked: %6.0fms | Avail: %5.1f%% | Score: %.0f\n",
				status, idx+1, totalTests, modeName, result.TotalCatches, result.MissedDuringBlock, 
				result.TotalBlockedMs, result.AvailabilityPct, result.Score)
		}(mode, i)
	}

	wg.Wait()
	fmt.Println("\n✅ All tests complete!")

	saveResults(results)
	printSummaryTable(results)
}

func runGameTest(mode TrainingMode) TestResult {
	result := TestResult{}

	defer func() {
		if r := recover(); r != nil {
			result.Error = fmt.Sprintf("panic: %v", r)
			result.Passed = false
		}
	}()

	numWindows := int(PPTestDuration / PPWindowDuration)
	result.Windows = make([]TimeWindow, numWindows)
	for i := range result.Windows {
		result.Windows[i].TimeMs = (i + 1) * int(PPWindowDuration.Milliseconds())
	}

	// Create DEEP network
	net := createDeepGameNetwork()
	numLayers := net.TotalLayers()

	var state *nn.StepState
	if mode == ModeStepBP || mode == ModeStepTween || mode == ModeStepTweenChain {
		state = net.InitStepState(PPInputSize)
	}

	var ts *nn.TweenState
	if mode == ModeTween || mode == ModeTweenChain || mode == ModeStepTween || mode == ModeStepTweenChain {
		ts = nn.NewTweenState(net, nil)
		if mode == ModeTweenChain || mode == ModeStepTweenChain {
			ts.Config.UseChainRule = true
		}
		ts.Config.LinkBudgetScale = 0.8
	}

	// Initialize game
	game := &GameState{
		Predator:    Position{X: 0, Y: 0},
		Prey:        Position{X: GridSize - 1, Y: GridSize - 1},
		PreyHistory: make([]int, 8),
		Behavior:    BehaviorRandom,
	}

	type Sample struct {
		Input  []float32
		Target []float32
	}
	trainBatch := make([]Sample, 0, PPBatchSize+10)
	lastTrainTime := time.Now()
	isBlocked := false

	start := time.Now()
	currentWindow := 0
	lastSwitchTime := start
	lastGameTick := start
	var totalBlockedTime time.Duration
	catchesAfterSwitch := 0
	missedDuringBlock := 0

	for time.Since(start) < PPTestDuration {
		elapsed := time.Since(start)

		// Update window
		newWindow := int(elapsed / PPWindowDuration)
		if newWindow > currentWindow && newWindow < numWindows {
			windowDurationMs := PPWindowDuration.Seconds() * 1000
			result.Windows[currentWindow].AvailableMs = windowDurationMs - result.Windows[currentWindow].BlockedMs
			currentWindow = newWindow
		}

		// Check for behavior switch (6 behaviors, cycle through)
		if time.Since(lastSwitchTime) >= PPSwitchInterval {
			game.Behavior = PreyBehavior((int(game.Behavior) + 1) % 6)
			lastSwitchTime = time.Now()
			result.TotalBehaviorSwitches++
			catchesAfterSwitch = 0
			if currentWindow < numWindows {
				result.Windows[currentWindow].BehaviorChanges++
			}
		}

		// Game tick
		if time.Since(lastGameTick) >= PPStepInterval {
			lastGameTick = time.Now()

			// If blocked, we MISS this opportunity
			if isBlocked {
				missedDuringBlock++
				continue
			}

			// Get predator input
			input := getGameInput(game)

			// Forward pass
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

			// Execute move
			predatorMove := argmax(output)
			movePredator(game, predatorMove)
			game.LastPredMove = predatorMove
			result.TotalDecisions++
			if currentWindow < numWindows {
				result.Windows[currentWindow].Decisions++
			}

			// Move prey
			preyMove := getPreyMove(game)
			movePrey(game, preyMove)

			// Check for catch
			if game.Predator == game.Prey {
				result.TotalCatches++
				catchesAfterSwitch++
				if currentWindow < numWindows {
					result.Windows[currentWindow].Catches++
				}
				// Reset prey
				game.Prey = Position{X: rand.Intn(GridSize), Y: rand.Intn(GridSize)}
				for game.Prey == game.Predator {
					game.Prey.X = (game.Prey.X + 3) % GridSize
					game.Prey.Y = (game.Prey.Y + 2) % GridSize
				}
			}

			// Create training target
			target := getIdealMove(game)
			trainBatch = append(trainBatch, Sample{Input: input, Target: target})

			// Training based on mode
			switch mode {
			case ModeNormalBP:
				if len(trainBatch) >= PPBatchSize && time.Since(lastTrainTime) > PPTrainInterval {
					batches := make([]nn.TrainingBatch, len(trainBatch))
					for i, s := range trainBatch {
						batches[i] = nn.TrainingBatch{Input: s.Input, Target: s.Target}
					}
					isBlocked = true
					trainStart := time.Now()
					net.Train(batches, &nn.TrainingConfig{Epochs: 1, LearningRate: PPLearningRate, LossType: "mse"})
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
				grad := make([]float32, len(output))
				for i := range grad {
					if i < len(target) {
						grad[i] = clipGrad(output[i]-target[i], 0.5)
					}
				}
				net.StepBackward(state, grad)
				net.ApplyGradients(PPLearningRate)
			case ModeTween, ModeTweenChain:
				if len(trainBatch) >= PPBatchSize && time.Since(lastTrainTime) > PPTrainInterval {
					isBlocked = true
					trainStart := time.Now()
					for _, s := range trainBatch {
						out := ts.ForwardPass(net, s.Input)
						outputGrad := make([]float32, len(out))
						for i := range outputGrad {
							if i < len(s.Target) {
								outputGrad[i] = s.Target[i] - out[i]
							}
						}
						totalLayers := net.TotalLayers()
						ts.ChainGradients[totalLayers] = outputGrad
						ts.BackwardTargets[totalLayers] = s.Target
						ts.TweenWeightsChainRule(net, PPLearningRate)
					}
					blockDuration := time.Since(trainStart)
					isBlocked = false
					totalBlockedTime += blockDuration
					if currentWindow < numWindows {
						result.Windows[currentWindow].BlockedMs += blockDuration.Seconds() * 1000
					}
					trainBatch = trainBatch[:0]
					lastTrainTime = time.Now()
				}
			case ModeStepTween, ModeStepTweenChain:
				// Non-blocking: train every sample immediately
				outputGrad := make([]float32, len(output))
				for i := range outputGrad {
					if i < len(target) {
						outputGrad[i] = target[i] - output[i]
					}
				}
				totalLayers := net.TotalLayers()
				ts.ChainGradients[totalLayers] = outputGrad
				ts.BackwardTargets[totalLayers] = target
				ts.TweenWeightsChainRule(net, PPLearningRate)
			}
		}
	}

	// Finalize
	for i := range result.Windows {
		windowDurationMs := PPWindowDuration.Seconds() * 1000
		if result.Windows[i].AvailableMs == 0 {
			result.Windows[i].AvailableMs = windowDurationMs - result.Windows[i].BlockedMs
		}
	}

	result.TrainTimeSec = time.Since(start).Seconds()
	result.TotalBlockedMs = totalBlockedTime.Seconds() * 1000
	result.AdaptationScore = float64(catchesAfterSwitch)
	result.MissedDuringBlock = missedDuringBlock
	calculateMetrics(&result)
	return result
}

func createDeepGameNetwork() *nn.Network {
	config := nn.SimpleNetworkConfig{
		InputSize:  PPInputSize,
		HiddenSize: PPHiddenSize,
		OutputSize: PPOutputSize,
		Activation: nn.ActivationLeakyReLU,
		InitScale:  PPInitScale,
		NumLayers:  PPNumLayers, // 5 layers deep!
		LayerType:  nn.BrainDense,
		DType:      nn.DTypeFloat32,
	}
	return nn.BuildSimpleNetwork(config)
}

func getGameInput(game *GameState) []float32 {
	input := make([]float32, PPInputSize)
	// Normalized positions
	input[0] = float32(game.Predator.X) / float32(GridSize-1)
	input[1] = float32(game.Predator.Y) / float32(GridSize-1)
	input[2] = float32(game.Prey.X) / float32(GridSize-1)
	input[3] = float32(game.Prey.Y) / float32(GridSize-1)
	// Direction vectors
	input[4] = float32(game.Prey.X-game.Predator.X) / float32(GridSize)
	input[5] = float32(game.Prey.Y-game.Predator.Y) / float32(GridSize)
	// Distance
	dx := float64(game.Prey.X - game.Predator.X)
	dy := float64(game.Prey.Y - game.Predator.Y)
	input[6] = float32(math.Sqrt(dx*dx+dy*dy)) / float32(GridSize)
	// Behavior encoding
	input[7] = float32(game.Behavior) / 6.0
	// Prey history
	for i := 0; i < 8 && i < len(game.PreyHistory); i++ {
		input[8+i] = float32(game.PreyHistory[i]) / 4.0
	}
	// Padding
	for i := 16; i < PPInputSize; i++ {
		input[i] = 0.0
	}
	return input
}

func getIdealMove(game *GameState) []float32 {
	target := make([]float32, 4)
	dx := game.Prey.X - game.Predator.X
	dy := game.Prey.Y - game.Predator.Y

	if abs(dx) > abs(dy) {
		if dx > 0 {
			target[3] = 1.0
		} else {
			target[2] = 1.0
		}
	} else if dy != 0 {
		if dy > 0 {
			target[1] = 1.0
		} else {
			target[0] = 1.0
		}
	}
	return target
}

func movePredator(game *GameState, move int) {
	switch move {
	case 0:
		if game.Predator.Y > 0 {
			game.Predator.Y--
		}
	case 1:
		if game.Predator.Y < GridSize-1 {
			game.Predator.Y++
		}
	case 2:
		if game.Predator.X > 0 {
			game.Predator.X--
		}
	case 3:
		if game.Predator.X < GridSize-1 {
			game.Predator.X++
		}
	}
}

func movePrey(game *GameState, move int) {
	if move < 0 {
		return // Freeze
	}
	game.PreyHistory = append([]int{move}, game.PreyHistory[:7]...)
	switch move {
	case 0:
		if game.Prey.Y > 0 {
			game.Prey.Y--
		}
	case 1:
		if game.Prey.Y < GridSize-1 {
			game.Prey.Y++
		}
	case 2:
		if game.Prey.X > 0 {
			game.Prey.X--
		}
	case 3:
		if game.Prey.X < GridSize-1 {
			game.Prey.X++
		}
	}
}

func getPreyMove(game *GameState) int {
	switch game.Behavior {
	case BehaviorRandom:
		return rand.Intn(4)
	case BehaviorFlee:
		dx := game.Predator.X - game.Prey.X
		dy := game.Predator.Y - game.Prey.Y
		if abs(dx) > abs(dy) {
			if dx > 0 {
				return 2
			}
			return 3
		}
		if dy > 0 {
			return 0
		}
		return 1
	case BehaviorZigZag:
		game.ZigZagCounter++
		dirs := []int{0, 3, 1, 2}
		return dirs[game.ZigZagCounter%4]
	case BehaviorFreeze:
		game.FreezeCounter++
		if game.FreezeCounter%10 < 7 {
			return -1
		}
		return rand.Intn(4)
	case BehaviorMirror:
		// Mirror predator's last move
		return game.LastPredMove
	case BehaviorCircle:
		game.CircleCounter++
		return game.CircleCounter % 4
	}
	return rand.Intn(4)
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

func abs(x int) int {
	if x < 0 {
		return -x
	}
	return x
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

func calculateMetrics(result *TestResult) {
	if result.TotalDecisions > 0 {
		result.CatchRate = float64(result.TotalCatches) / float64(result.TotalDecisions) * 100
	}
	if result.TrainTimeSec > 0 {
		result.DecisionsPerSec = float64(result.TotalDecisions) / result.TrainTimeSec
		totalTimeMs := result.TrainTimeSec * 1000
		result.AvailabilityPct = ((totalTimeMs - result.TotalBlockedMs) / totalTimeMs) * 100
	}
	
	// Score penalizes missed opportunities heavily
	missedPenalty := float64(result.MissedDuringBlock) / 10.0
	adaptBonus := result.AdaptationScore / 5.0
	result.Score = float64(result.TotalCatches) * (result.AvailabilityPct / 100) * (1 + adaptBonus) - missedPenalty
	if result.Score < 0 {
		result.Score = 0
	}
	result.Passed = result.TotalCatches > 0 && result.Score > 0
}

func saveResults(results *BenchmarkResults) {
	data, err := json.MarshalIndent(results, "", "  ")
	if err != nil {
		fmt.Printf("Error marshaling results: %v\n", err)
		return
	}
	filename := "predator_prey_hardcore_results.json"
	if err := os.WriteFile(filename, data, 0644); err != nil {
		fmt.Printf("Error saving results: %v\n", err)
		return
	}
	fmt.Printf("\n📁 Results saved to %s\n", filename)
}

func printSummaryTable(results *BenchmarkResults) {
	fmt.Println("\n╔═══════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════╗")
	fmt.Println("║                                    PREDATOR-PREY HARDCORE BENCHMARK SUMMARY                                                  ║")
	fmt.Println("╠═══════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════╣")
	fmt.Println("║  Training Mode    │ Catches │ Missed │ Decisions │ Catch% │ Blocked(ms) │ Avail% │ Adapt │    Score    │ Status             ║")
	fmt.Println("╠═══════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════╣")

	for _, r := range results.Results {
		status := "✅ PASS"
		if !r.Passed {
			status = "❌ FAIL"
		}
		fmt.Printf("║  %-15s │  %5d  │  %5d │   %5d   │ %5.1f%% │   %8.0f  │ %5.1f%% │ %5.0f │ %11.1f │ %s ║\n",
			r.TrainingMode, r.TotalCatches, r.MissedDuringBlock, r.TotalDecisions, 
			r.CatchRate, r.TotalBlockedMs, r.AvailabilityPct, r.AdaptationScore, r.Score, status)
	}

	fmt.Println("╚═══════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════╝")

	var best *TestResult
	var worst *TestResult
	for i := range results.Results {
		if best == nil || results.Results[i].Score > best.Score {
			best = &results.Results[i]
		}
		if worst == nil || results.Results[i].Score < worst.Score {
			worst = &results.Results[i]
		}
	}
	if best != nil && worst != nil {
		diff := best.Score - worst.Score
		pctDiff := 0.0
		if worst.Score > 0 {
			pctDiff = (diff / worst.Score) * 100
		}
		fmt.Printf("\n🏆 BEST: %s with Score %.1f | Catches: %d | Availability: %.1f%%\n",
			best.TrainingMode, best.Score, best.TotalCatches, best.AvailabilityPct)
		fmt.Printf("💀 WORST: %s with Score %.1f | Missed: %d opportunities while blocked\n",
			worst.TrainingMode, worst.Score, worst.MissedDuringBlock)
		fmt.Printf("📊 DIFFERENCE: %.1f points (%.1f%% improvement from worst to best)\n", diff, pctDiff)
	}
}
