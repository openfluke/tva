package main

import (
	"fmt"
	"math"
	"math/rand"
	"strings"
	"sync"
	"time"

	"github.com/openfluke/loom/nn"
)

// ═══════════════════════════════════════════════════════════════════════════════
// HARMONIC CHAMELEON: THE "IMPOSSIBLE" ADAPTATION BENCHMARK
// ═══════════════════════════════════════════════════════════════════════════════

const (
	InputWindow    = 16
	RuleDuration   = 5 * time.Second
	TotalSteps     = 3000 // 30 seconds
	StepInterval   = 10 * time.Millisecond
	AccuracyWindow = 50
)

type RuleType int

const (
	RulePower RuleType = iota
	RuleRectify
	RuleSineFold
	RuleBinarize
)

var ruleNames = map[RuleType]string{
	RulePower:    "Power",
	RuleRectify:  "Rectify",
	RuleSineFold: "Fold",
	RuleBinarize: "Binary",
}

// Modes
// Modes
type TrainingMode int

const (
	ModeNormalBP       TrainingMode = iota // Batch BP
	ModeStepBP                             // Step BP
	ModeTween                              // Pure Tween (No Chain Rule)
	ModeTweenChain                         // Batched Tween (Chain Rule)
	ModeStepTweenChain                     // Immediate Tween (Chain Rule)
)

var modeNames = map[TrainingMode]string{
	ModeNormalBP:       "BP(Batch)",
	ModeStepBP:         "BP(Step)",
	ModeTween:          "Tween(Pure)",
	ModeTweenChain:     "Tween(Batch)",
	ModeStepTweenChain: "Tween(Step)",
}

// Shared State for Table
type ModeStatus struct {
	Loss       float32
	Adaptation string // "OK", "FAIL"
	Completed  bool
}

var (
	statusMap = make(map[TrainingMode]*ModeStatus)
	statusMu  sync.Mutex
)

func main() {
	modes := []TrainingMode{
		ModeNormalBP,
		ModeStepBP,
		ModeTween,
		ModeTweenChain,
		ModeStepTweenChain,
	}

	for _, m := range modes {
		statusMap[m] = &ModeStatus{Adaptation: "INIT"}
	}

	fmt.Println("🦎 Harmonic Chameleon: Initializing Parallel Mode Test...")

	// Table Header
	fmt.Printf("%-8s | %-10s", "Time", "Rule")
	for _, m := range modes {
		// Variable width depending on name length? Fixed is better.
		fmt.Printf(" | %-12s", modeNames[m])
	}
	fmt.Println()
	fmt.Println(strings.Repeat("-", 100))

	var wg sync.WaitGroup
	startTime := time.Now()

	for _, mode := range modes {
		wg.Add(1)
		go func(m TrainingMode) {
			defer wg.Done()
			runMode(m, startTime)
		}(mode)
	}

	// Reporter Loop
	monitorTicker := time.NewTicker(200 * time.Millisecond)
	defer monitorTicker.Stop()

	done := make(chan bool)
	go func() {
		wg.Wait()
		done <- true
	}()

	running := true
	for running {
		select {
		case <-done:
			running = false
		case <-monitorTicker.C:
			elapsed := time.Since(startTime)
			ruleCycle := int(elapsed/RuleDuration) % 4
			activeRuleName := ruleNames[RuleType(ruleCycle)]

			statusMu.Lock()
			fmt.Printf("%-8s | %-10s", elapsed.Round(100*time.Millisecond).String(), activeRuleName)
			for _, m := range modes {
				s := statusMap[m]
				val := fmt.Sprintf("%.4f", s.Loss)
				if s.Completed {
					val = "DONE"
				}
				fmt.Printf(" | %-12s", val)
			}
			fmt.Println()
			statusMu.Unlock()
		}
	}
	fmt.Println("\n\n🏁 Benchmark Complete.")
}

func runMode(mode TrainingMode, startTime time.Time) {
	// 1. Build Network
	net := nn.NewNetwork(InputWindow, 1, 3, 1)
	net.SetLayer(0, 0, 0, nn.InitDenseLayer(InputWindow, 32, nn.ActivationScaledReLU))
	net.SetLayer(0, 1, 0, nn.InitDenseLayer(32, 32, nn.ActivationLeakyReLU))
	net.SetLayer(0, 2, 0, nn.InitDenseLayer(32, 1, nn.ActivationTanh))

	// Ensure standard optimizer config for BP modes
	// net.Optimizer is usually Adam by default in NewNetwork? Or nil?
	// NewNetwork sets default optimizer.

	net.InitializeWeights()

	// stepState for StepBP
	var stepState *nn.StepState
	if mode == ModeStepBP {
		stepState = net.InitStepState(10)
	}

	// 2. Setup Tween
	ts := nn.NewTweenState(net, nil)
	ts.Config.FrontierEnabled = true
	ts.Config.FrontierThreshold = 0.55
	ts.Config.IgnoreThreshold = 0.01
	ts.Config.DenseRate = 1.0

	if mode == ModeTween {
		ts.Config.UseChainRule = false // Pure Tween
	} else {
		ts.Config.UseChainRule = true
	}

	inputBuffer := make([]float32, InputWindow)
	rollingErrors := make([]float32, 0, AccuracyWindow)

	// Batched Logic
	type TrainingSample struct {
		Input  []float32
		Target []float32
	}
	// For TweenBatch we need slightly different struct or convert
	trainBatch := make([]TrainingSample, 0, 10)

	lastTrainTime := time.Now()
	trainInterval := 50 * time.Millisecond

	// Standard SGD Learning Rate
	learningRate := float32(0.02)
	// For Tween we use the Config rate? or override?
	// In test17: Tween uses rate param in function call.

	ticker := time.NewTicker(StepInterval)
	defer ticker.Stop()

	for step := 0; step < TotalSteps; step++ {
		<-ticker.C

		elapsed := time.Since(startTime)

		// Rule
		ruleCycle := int(elapsed/RuleDuration) % 4
		activeRule := RuleType(ruleCycle)

		// Signal
		t := float64(step) * 0.1
		rawSignal := float32(math.Sin(t)) + (rand.Float32() * 0.02)

		copy(inputBuffer, inputBuffer[1:])
		inputBuffer[InputWindow-1] = rawSignal

		// Target
		x := rawSignal
		var targetVal float32
		switch activeRule {
		case RulePower:
			targetVal = x * x
		case RuleRectify:
			targetVal = float32(math.Abs(float64(x)))
		case RuleSineFold:
			targetVal = float32(math.Sin(float64(x) * 5.0))
		case RuleBinarize:
			if x >= 0 {
				targetVal = 1.0
			} else {
				targetVal = -1.0
			}
		}

		// Forward
		var prediction float32

		switch mode {
		case ModeTween, ModeTweenChain, ModeNormalBP:
			// CPU Forward (Stateless/Batch Style)
			predArgs, _ := net.ForwardCPU(inputBuffer)
			prediction = predArgs[0]

		case ModeStepTweenChain, ModeStepBP:
			if mode == ModeStepBP {
				stepState.SetInput(inputBuffer)
				net.StepForward(stepState)
				out := stepState.GetOutput()
				if len(out) > 0 {
					prediction = out[0]
				}
			} else {
				// StepTweenChain uses generic forward pass logic internally or wrapper?
				// For now use stateless forward unless we want strict step state.
				// test17 uses ForwardCPU for StepTweenChain.
				predArgs, _ := net.ForwardCPU(inputBuffer)
				prediction = predArgs[0]
			}
		}

		loss := (targetVal - prediction) * (targetVal - prediction)

		// Training
		switch mode {
		case ModeNormalBP:
			// Accumulate Batch
			targetHist := []float32{targetVal}
			// Copy input
			inHist := make([]float32, InputWindow)
			copy(inHist, inputBuffer)
			trainBatch = append(trainBatch, TrainingSample{Input: inHist, Target: targetHist})

			if time.Since(lastTrainTime) > trainInterval && len(trainBatch) > 0 {
				// Convert to nn.TrainingBatch
				batches := make([]nn.TrainingBatch, len(trainBatch))
				for i, s := range trainBatch {
					batches[i] = nn.TrainingBatch{Input: s.Input, Target: s.Target}
				}
				net.Train(batches, &nn.TrainingConfig{Epochs: 1, LearningRate: learningRate, LossType: "mse"})
				trainBatch = trainBatch[:0]
				lastTrainTime = time.Now()
			}

		case ModeStepBP:
			// Immediate Backprop
			// Grad = Pred - Target
			grad := []float32{prediction - targetVal}
			net.StepBackward(stepState, grad)
			net.ApplyGradients(learningRate)

		case ModeStepTweenChain:
			if ts.Config.UseChainRule {
				ts.BackwardPassRegression(net, []float32{targetVal})
				ts.TweenWeightsChainRule(net, 0.05)
			} else {
				// Should not happen for StepTweenChain but good for safety
				ts.BackwardPassRegression(net, []float32{targetVal})
				ts.TweenWeights(net, 0.05)
			}

		case ModeTweenChain, ModeTween:
			// Batch Logic for Tween
			targetHist := []float32{targetVal}
			inHist := make([]float32, InputWindow)
			copy(inHist, inputBuffer)
			trainBatch = append(trainBatch, TrainingSample{Input: inHist, Target: targetHist})

			if time.Since(lastTrainTime) > trainInterval && len(trainBatch) > 0 {
				for _, s := range trainBatch {
					net.ForwardCPU(s.Input)                  // Prime
					ts.BackwardPassRegression(net, s.Target) // Prep gradients/targets

					if ts.Config.UseChainRule {
						ts.TweenWeightsChainRule(net, 0.05)
					} else {
						ts.TweenWeights(net, 0.05)
					}
				}
				trainBatch = trainBatch[:0]
				lastTrainTime = time.Now()
			}
		}

		// Metrics
		rollingErrors = append(rollingErrors, loss)
		if len(rollingErrors) > AccuracyWindow {
			rollingErrors = rollingErrors[1:]
		}
		avgLoss := float32(0)
		for _, e := range rollingErrors {
			avgLoss += e
		}
		avgLoss /= float32(len(rollingErrors))

		statusMu.Lock()
		statusMap[mode].Loss = avgLoss
		statusMu.Unlock()
	}

	statusMu.Lock()
	statusMap[mode].Completed = true
	statusMu.Unlock()
}
