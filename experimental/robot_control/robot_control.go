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
// ROBOT CONTROL WITH DISTURBANCE - HARDCORE EDITION
// ═══════════════════════════════════════════════════════════════════════════════
//
// HARDCORE: 60 second test, DEEP networks, LARGE batch training
// Control systems CANNOT pause - blocking literally = crash

const (
	RCInputSize  = 8   // More state info
	RCHiddenSize = 128
	RCOutputSize = 1
	RCNumLayers  = 5

	RCLearningRate = float32(0.005)
	RCInitScale    = float32(0.2)
	RCBatchSize    = 100

	AngleLimit    = 0.2
	PositionLimit = 2.0
	BaseGravity   = 9.8
	DeltaT        = 0.02

	RCTestDuration    = 60 * time.Second
	RCWindowDuration  = 100 * time.Millisecond
	RCSwitchInterval  = 5 * time.Second
	RCTrainInterval   = 500 * time.Millisecond
	RCControlInterval = 10 * time.Millisecond

	RCMaxConcurrent = 6
)

type PhysicsMode int

const (
	PhysicsNormal PhysicsMode = iota
	PhysicsLowGrav
	PhysicsHighGrav
	PhysicsWindy
	PhysicsIcy      // NEW
	PhysicsHeavyPole // NEW
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

type CartPoleState struct {
	CartPos, CartVel, PoleAngle, PoleAngVel float64
	Physics                                  PhysicsMode
	Crashed                                  bool
}

type TimeWindow struct {
	TimeMs     int     `json:"timeMs"`
	Stable     int     `json:"stable"`
	Crashes    int     `json:"crashes"`
	BlockedMs  float64 `json:"blockedMs"`
}

type TestResult struct {
	TrainingMode        string       `json:"trainingMode"`
	Windows             []TimeWindow `json:"windows"`
	TotalControlSteps   int          `json:"totalControlSteps"`
	TotalStableSteps    int          `json:"totalStableSteps"`
	TotalCrashes        int          `json:"totalCrashes"`
	CrashesDuringBlock  int          `json:"crashesDuringBlock"`
	MissedControlSteps  int          `json:"missedControlSteps"`
	TrainTimeSec        float64      `json:"trainTimeSec"`
	StabilityRate       float64      `json:"stabilityRate"`
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
	fmt.Println("║   🤖 ROBOT CONTROL WITH DISTURBANCE: HARDCORE EDITION                                                        ║")
	fmt.Println("║                                                                                                               ║")
	fmt.Println("║   ⚠️  60 SECOND TEST | 5-LAYER DEEP NETWORK | BATCH SIZE 100                                                 ║")
	fmt.Println("║   Control systems CANNOT pause - blocking = CRASH                                                            ║")
	fmt.Println("╚═══════════════════════════════════════════════════════════════════════════════════════════════════════════════╝")

	modes := []TrainingMode{ModeNormalBP, ModeStepBP, ModeTween, ModeTweenChain, ModeStepTween, ModeStepTweenChain}
	
	fmt.Printf("\n📊 Running %d tests | 60s each | Batch size: %d\n\n", len(modes), RCBatchSize)

	results := &BenchmarkResults{
		Results:    make([]TestResult, 0, len(modes)),
		Timestamp:  time.Now().Format(time.RFC3339),
		Duration:   RCTestDuration.String(),
		TotalTests: len(modes),
	}

	var wg sync.WaitGroup
	var mu sync.Mutex
	sem := make(chan struct{}, RCMaxConcurrent)

	for i, mode := range modes {
		wg.Add(1)
		go func(m TrainingMode, idx int) {
			defer wg.Done()
			sem <- struct{}{}
			defer func() { <-sem }()

			modeName := modeNames[m]
			fmt.Printf("🎛️  [%d/%d] Starting %s...\n", idx+1, len(modes), modeName)

			result := runControlTest(m)
			result.TrainingMode = modeName

			mu.Lock()
			results.Results = append(results.Results, result)
			mu.Unlock()

			fmt.Printf("✅ [%d/%d] %-15s | Stable: %5.1f%% | Crashes: %3d (Block: %2d) | Missed: %4d | Avail: %5.1f%% | Score: %.0f\n",
				idx+1, len(modes), modeName, result.StabilityRate, result.TotalCrashes,
				result.CrashesDuringBlock, result.MissedControlSteps, result.AvailabilityPct, result.Score)
		}(mode, i)
	}

	wg.Wait()
	saveResults(results)
	printSummaryTable(results)
}

func runControlTest(mode TrainingMode) TestResult {
	result := TestResult{}
	defer func() {
		if r := recover(); r != nil {
			result.Error = fmt.Sprintf("panic: %v", r)
		}
	}()

	numWindows := int(RCTestDuration / RCWindowDuration)
	result.Windows = make([]TimeWindow, numWindows)
	for i := range result.Windows {
		result.Windows[i].TimeMs = (i + 1) * int(RCWindowDuration.Milliseconds())
	}

	net := createDeepNetwork()
	numLayers := net.TotalLayers()

	var state *nn.StepState
	if mode == ModeStepBP || mode == ModeStepTween || mode == ModeStepTweenChain {
		state = net.InitStepState(RCInputSize)
	}

	var ts *nn.TweenState
	if mode == ModeTween || mode == ModeTweenChain || mode == ModeStepTween || mode == ModeStepTweenChain {
		ts = nn.NewTweenState(net, nil)
		if mode == ModeTweenChain || mode == ModeStepTweenChain {
			ts.Config.UseChainRule = true
		}
	}

	cp := &CartPoleState{
		PoleAngle: 0.05 * (rand.Float64() - 0.5),
		Physics:   PhysicsNormal,
	}

	type Sample struct {
		Input  []float32
		Target float32
	}
	trainBatch := make([]Sample, 0, RCBatchSize+10)
	lastTrainTime := time.Now()
	isBlocked := false

	start := time.Now()
	currentWindow := 0
	lastSwitchTime := start
	lastControlTime := start
	var totalBlockedTime time.Duration

	for time.Since(start) < RCTestDuration {
		elapsed := time.Since(start)

		newWindow := int(elapsed / RCWindowDuration)
		if newWindow > currentWindow && newWindow < numWindows {
			currentWindow = newWindow
		}

		if time.Since(lastSwitchTime) >= RCSwitchInterval {
			cp.Physics = PhysicsMode((int(cp.Physics) + 1) % 6)
			lastSwitchTime = time.Now()
		}

		if time.Since(lastControlTime) >= RCControlInterval {
			lastControlTime = time.Now()

			if isBlocked {
				result.MissedControlSteps++
				// Simulate physics with no control while blocked!
				simulateStep(cp, 0)
				if cp.Crashed {
					result.TotalCrashes++
					result.CrashesDuringBlock++
					resetCartPole(cp)
				}
				continue
			}

			input := getStateInput(cp)
			
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

			force := 0.0
			if len(output) > 0 {
				force = float64(output[0]) * 10.0
			}

			simulateStep(cp, force)
			result.TotalControlSteps++

			isStable := !cp.Crashed && math.Abs(cp.PoleAngle) < AngleLimit && math.Abs(cp.CartPos) < PositionLimit
			if isStable {
				result.TotalStableSteps++
				if currentWindow < numWindows {
					result.Windows[currentWindow].Stable++
				}
			}

			if cp.Crashed {
				result.TotalCrashes++
				if currentWindow < numWindows {
					result.Windows[currentWindow].Crashes++
				}
				resetCartPole(cp)
			}

			target := float32(pdController(cp) / 10.0)
			trainBatch = append(trainBatch, Sample{Input: input, Target: target})

			switch mode {
			case ModeNormalBP:
				if len(trainBatch) >= RCBatchSize && time.Since(lastTrainTime) > RCTrainInterval {
					batches := make([]nn.TrainingBatch, len(trainBatch))
					for i, s := range trainBatch {
						batches[i] = nn.TrainingBatch{Input: s.Input, Target: []float32{s.Target}}
					}
					isBlocked = true
					trainStart := time.Now()
					net.Train(batches, &nn.TrainingConfig{Epochs: 1, LearningRate: RCLearningRate, LossType: "mse"})
					blockDuration := time.Since(trainStart)
					isBlocked = false
					totalBlockedTime += blockDuration
					trainBatch = trainBatch[:0]
					lastTrainTime = time.Now()
				}
			case ModeStepBP:
				grad := []float32{clipGrad(output[0]-target, 0.5)}
				net.StepBackward(state, grad)
				net.ApplyGradients(RCLearningRate)
			case ModeTween, ModeTweenChain:
				if len(trainBatch) >= RCBatchSize && time.Since(lastTrainTime) > RCTrainInterval {
					isBlocked = true
					trainStart := time.Now()
					for _, s := range trainBatch {
						out := ts.ForwardPass(net, s.Input)
						ts.ChainGradients[net.TotalLayers()] = []float32{s.Target - out[0]}
						ts.BackwardTargets[net.TotalLayers()] = []float32{s.Target}
						ts.TweenWeightsChainRule(net, RCLearningRate)
					}
					blockDuration := time.Since(trainStart)
					isBlocked = false
					totalBlockedTime += blockDuration
					trainBatch = trainBatch[:0]
					lastTrainTime = time.Now()
				}
			case ModeStepTween, ModeStepTweenChain:
				ts.ChainGradients[net.TotalLayers()] = []float32{target - output[0]}
				ts.BackwardTargets[net.TotalLayers()] = []float32{target}
				ts.TweenWeightsChainRule(net, RCLearningRate)
			}
		}
	}

	result.TrainTimeSec = time.Since(start).Seconds()
	result.TotalBlockedMs = totalBlockedTime.Seconds() * 1000

	if result.TotalControlSteps > 0 {
		result.StabilityRate = float64(result.TotalStableSteps) / float64(result.TotalControlSteps) * 100
	}
	if result.TrainTimeSec > 0 {
		totalTimeMs := result.TrainTimeSec * 1000
		result.AvailabilityPct = ((totalTimeMs - result.TotalBlockedMs) / totalTimeMs) * 100
	}

	// Heavy penalty for crashes during block and missed steps
	crashPenalty := float64(result.CrashesDuringBlock) * 10.0
	missedPenalty := float64(result.MissedControlSteps) / 10.0
	result.Score = result.StabilityRate * (result.AvailabilityPct / 100) - crashPenalty - missedPenalty
	if result.Score < 0 {
		result.Score = 0
	}
	result.Passed = result.Score > 0
	return result
}

func createDeepNetwork() *nn.Network {
	config := nn.SimpleNetworkConfig{
		InputSize:  RCInputSize,
		HiddenSize: RCHiddenSize,
		OutputSize: RCOutputSize,
		Activation: nn.ActivationTanh,
		InitScale:  RCInitScale,
		NumLayers:  RCNumLayers,
		LayerType:  nn.BrainDense,
		DType:      nn.DTypeFloat32,
	}
	return nn.BuildSimpleNetwork(config)
}

func getStateInput(cp *CartPoleState) []float32 {
	return []float32{
		float32(cp.CartPos / PositionLimit),
		float32(cp.CartVel / 2.0),
		float32(cp.PoleAngle / AngleLimit),
		float32(cp.PoleAngVel / 2.0),
		float32(cp.Physics) / 6.0,
		float32(math.Sin(cp.PoleAngle)),
		float32(math.Cos(cp.PoleAngle)),
		float32(cp.CartPos * cp.CartVel),
	}
}

func simulateStep(cp *CartPoleState, force float64) {
	gravity := BaseGravity
	poleMass := 0.1
	friction := 0.0
	windForce := 0.0

	switch cp.Physics {
	case PhysicsLowGrav:
		gravity = BaseGravity * 0.4
	case PhysicsHighGrav:
		gravity = BaseGravity * 1.8
	case PhysicsWindy:
		windForce = (rand.Float64() - 0.5) * 8.0
	case PhysicsIcy:
		friction = -0.1 * cp.CartVel // Slippery
	case PhysicsHeavyPole:
		poleMass = 0.3
	}

	cartMass := 1.0
	poleLength := 0.5
	totalMass := cartMass + poleMass
	poleMassLength := poleMass * poleLength

	sinAngle := math.Sin(cp.PoleAngle)
	cosAngle := math.Cos(cp.PoleAngle)

	temp := (force + windForce + friction + poleMassLength*cp.PoleAngVel*cp.PoleAngVel*sinAngle) / totalMass
	poleAngAcc := (gravity*sinAngle - cosAngle*temp) / (poleLength * (4.0/3.0 - poleMass*cosAngle*cosAngle/totalMass))
	cartAcc := temp - poleMassLength*poleAngAcc*cosAngle/totalMass

	cp.CartPos += cp.CartVel * DeltaT
	cp.CartVel += cartAcc * DeltaT
	cp.PoleAngle += cp.PoleAngVel * DeltaT
	cp.PoleAngVel += poleAngAcc * DeltaT

	if math.Abs(cp.PoleAngle) > AngleLimit*2.5 || math.Abs(cp.CartPos) > PositionLimit {
		cp.Crashed = true
	}
}

func resetCartPole(cp *CartPoleState) {
	cp.CartPos = 0.0
	cp.CartVel = 0.0
	cp.PoleAngle = 0.05 * (rand.Float64() - 0.5)
	cp.PoleAngVel = 0.0
	cp.Crashed = false
}

func pdController(cp *CartPoleState) float64 {
	return -50.0*cp.PoleAngle - 20.0*cp.PoleAngVel
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
	os.WriteFile("robot_control_hardcore_results.json", data, 0644)
	fmt.Println("\n📁 Results saved to robot_control_hardcore_results.json")
}

func printSummaryTable(results *BenchmarkResults) {
	fmt.Println("\n╔═══════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════╗")
	fmt.Println("║                                    ROBOT CONTROL HARDCORE SUMMARY                                                                    ║")
	fmt.Println("╠═══════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════╣")
	fmt.Println("║  Training Mode    │ Stable% │ Crashes │ BlockCrash │ MissedCtrl │ Blocked(ms) │ Avail% │     Score     │ Status                      ║")
	fmt.Println("╠═══════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════╣")

	for _, r := range results.Results {
		status := "✅ PASS"
		if !r.Passed {
			status = "❌ FAIL"
		}
		fmt.Printf("║  %-15s │ %6.1f%% │   %3d   │     %3d    │     %4d   │   %8.0f  │ %5.1f%% │ %13.1f │ %s ║\n",
			r.TrainingMode, r.StabilityRate, r.TotalCrashes, r.CrashesDuringBlock,
			r.MissedControlSteps, r.TotalBlockedMs, r.AvailabilityPct, r.Score, status)
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
		fmt.Printf("\n🏆 BEST: %s | Score: %.1f | Stability: %.1f%% | Crashes during block: %d\n",
			best.TrainingMode, best.Score, best.StabilityRate, best.CrashesDuringBlock)
		fmt.Printf("💀 WORST: %s | Score: %.1f | Crashed %d times while training, missed %d control steps\n",
			worst.TrainingMode, worst.Score, worst.CrashesDuringBlock, worst.MissedControlSteps)
	}
}
