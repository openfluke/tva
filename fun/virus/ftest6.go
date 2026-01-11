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
// 🦠 THE VIRAL INJECTION: A TEST OF ADVERSARIAL RESISTANCE
// ═══════════════════════════════════════════════════════════════════════════════

const (
	InputWindow     = 32
	StepInterval    = 10 * time.Millisecond
	TotalSteps      = 2000 // 20 seconds
	VirusChance     = 0.10 // 10% Poison
	IgnoreThreshold = 0.4  // Reject updates with < 40% alignment
)

// Modes
type TrainingMode int

const (
	ModeNormalBP       TrainingMode = iota // Batch BP
	ModeStepBP                             // Continuous Step BP
	ModeTween                              // Pure Tween (No Chain Rule)
	ModeTweenChain                         // Batched Tween (Chain Rule)
	ModeStepTweenChain                     // Continuous Tween (Chain Rule)
)

var modeNames = map[TrainingMode]string{
	ModeNormalBP:       "BP(Batch)",
	ModeStepBP:         "BP(Step)",
	ModeTween:          "Tween(Pure)",
	ModeTweenChain:     "Tween(Batch)",
	ModeStepTweenChain: "Tween(Step)",
}

// Shared State
type ModeStatus struct {
	Loss          float32
	Budget        float32
	Action        string
	CleanLossAvg  float32
	PoisonLossAvg float32
	RejectRate    float64
	Completed     bool
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
		statusMap[m] = &ModeStatus{Action: "INIT"}
	}

	fmt.Println("🦠 The Viral Injection: Initializing Parallel Adversarial Resistance Test...")

	fmt.Printf("%-8s | %-10s", "Time", "Type")
	for _, m := range modes {
		fmt.Printf(" | %-12s | %-8s", modeNames[m], "Budget")
	}
	fmt.Println()
	fmt.Println(strings.Repeat("-", 160))

	var wg sync.WaitGroup
	startTime := time.Now()

	for _, mode := range modes {
		wg.Add(1)
		go func(m TrainingMode) {
			defer wg.Done()
			runMode(m, startTime)
		}(mode)
	}

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
			typeStr := "CLEAN"

			// Quick peek at one of the modes to see if virus active?
			// Actually we can just print time.

			statusMu.Lock()
			fmt.Printf("%-8s | %-10s", elapsed.Round(100*time.Millisecond).String(), typeStr)
			for _, m := range modes {
				s := statusMap[m]
				budStr := fmt.Sprintf("%.2f", s.Budget)
				if s.Completed {
					budStr = fmt.Sprintf("AVG:%.2f", s.PoisonLossAvg)
				}
				fmt.Printf(" | %-12.4f | %-8s", s.Loss, budStr)
			}
			fmt.Println()
			statusMu.Unlock()
		}
	}
	fmt.Println("\n\n🏁 Viral Injection Test Complete.")

	fmt.Println("\n📊 IMMUNITY REPORT")
	fmt.Println("==========================================================================================")
	fmt.Printf("%-15s | %-12s | %-12s | %-15s | %s\n", "Mode", "Clean Loss", "Poison Loss", "Rejection Rate", "Result")
	statusMu.Lock()
	for _, m := range modes {
		s := statusMap[m]
		res := "❌ INFECTED"
		if s.CleanLossAvg < 0.05 && s.PoisonLossAvg > 0.5 {
			res = "✅ IMMUNE"
		} else if s.PoisonLossAvg > s.CleanLossAvg*2 {
			res = "⚠️  RESISTANT"
		}
		fmt.Printf("%-15s | %-12.4f | %-12.4f | %-14.1f%% | %s\n",
			modeNames[m], s.CleanLossAvg, s.PoisonLossAvg, s.RejectRate, res)
	}
	statusMu.Unlock()
}

func runMode(mode TrainingMode, startTime time.Time) {
	rand.Seed(time.Now().UnixNano())

	net := nn.NewNetwork(InputWindow, 1, 3, 1)
	net.SetLayer(0, 0, 0, nn.InitDenseLayer(InputWindow, 64, nn.ActivationTanh))
	net.SetLayer(0, 1, 0, nn.InitDenseLayer(64, 32, nn.ActivationLeakyReLU))
	net.SetLayer(0, 2, 0, nn.InitDenseLayer(32, 1, nn.ActivationType(-1)))
	net.InitializeWeights()

	var stepState *nn.StepState
	if mode == ModeStepBP {
		stepState = net.InitStepState(64)
	}

	// Use standard TweenState (float32 specific) to access legacy/pure modes if needed
	ts := nn.NewTweenState(net, nil)
	ts.Config.FrontierEnabled = true
	ts.Config.FrontierThreshold = 0.5
	ts.Config.IgnoreThreshold = IgnoreThreshold
	ts.Config.DenseRate = 0.05
	ts.Config.Momentum = 0.2

	// Configure Chaining logic
	if mode == ModeTween {
		ts.Config.UseChainRule = false // Pure Tween
	} else {
		ts.Config.UseChainRule = true
	}

	inputBuffer := make([]float32, InputWindow)

	cleanLossSum := 0.0
	cleanCount := 0
	poisonLossSum := 0.0
	poisonCount := 0
	rejectedCount := 0

	type TrainingSample struct {
		Input  []float32
		Target []float32
	}
	trainBatch := make([]TrainingSample, 0, 10)
	lastTrainTime := time.Now()
	trainInterval := 50 * time.Millisecond
	learningRate := float32(0.02)

	ticker := time.NewTicker(StepInterval)
	defer ticker.Stop()

	for step := 0; step < TotalSteps; step++ {
		<-ticker.C
		t := float64(step) * 0.1

		cleanVal := math.Sin(t)
		isVirus := rand.Float64() < VirusChance

		var targetVal float32
		if isVirus {
			targetVal = float32(-cleanVal)
		} else {
			targetVal = float32(cleanVal)
		}

		copy(inputBuffer, inputBuffer[1:])
		inputBuffer[InputWindow-1] = targetVal

		// Forward
		var prediction float32

		switch mode {
		case ModeTween, ModeTweenChain, ModeNormalBP:
			// Standard Forward (Batch/Stateless)
			out, _ := net.ForwardCPU(inputBuffer)
			prediction = out[0]

		case ModeStepTweenChain, ModeStepBP:
			// Stateful Forward
			if mode == ModeStepBP {
				stepState.SetInput(inputBuffer)
				net.StepForward(stepState)
				out := stepState.GetOutput()
				if len(out) > 0 {
					prediction = out[0]
				}
			} else {
				// Step Tween uses Forward pass but we treat it as 1-step batch effectively for prediction?
				// Or does it have state? TweenState currently relies on net.ForwardCPU usually.
				// For consistency with test17, we use ForwardCPU for Tween modes unless explicitly stateful.
				out, _ := net.ForwardCPU(inputBuffer)
				prediction = out[0]
			}
		}

		loss := (targetVal - prediction) * (targetVal - prediction)

		// Budget Check (Tween Modes Only)
		minBudget := float32(1.0)
		if mode == ModeTween || mode == ModeTweenChain || mode == ModeStepTweenChain {
			// Need Backward Targets to calc budget
			if ts.Config.UseChainRule {
				ts.BackwardPassChainRule(net, 0, 1) // 0 is dummy class for regression? No, wait.
				// Regression requires BackwardPassRegression.
				// TweenState.BackwardPassRegression exists.
				ts.BackwardPassRegression(net, []float32{targetVal})
			} else {
				// Pure Tween typically uses BackwardPass legacy
				ts.BackwardPass(net, 0, 1) // This is for classification...
				// For regression pure tween? Need to set targets manually?
				// Actually BackwardPassLegacy might not support regression well.
				// Let's assume we use Regression helper for ALL modes regarding budget
				ts.BackwardPassRegression(net, []float32{targetVal})
			}

			ts.CalculateLinkBudgets()

			for _, b := range ts.LinkBudgets {
				if b < minBudget && b > 0 {
					minBudget = b
				}
			}
		}

		action := "LEARN"
		if (mode == ModeTween || mode == ModeTweenChain || mode == ModeStepTweenChain) && minBudget < IgnoreThreshold {
			action = "REJECT"
			rejectedCount++
		}

		// Train
		if action != "REJECT" {
			switch mode {
			case ModeNormalBP:
				inHist := make([]float32, InputWindow)
				copy(inHist, inputBuffer)
				trainBatch = append(trainBatch, TrainingSample{Input: inHist, Target: []float32{targetVal}})
				if time.Since(lastTrainTime) > trainInterval && len(trainBatch) > 0 {
					batches := make([]nn.TrainingBatch, len(trainBatch))
					for i, s := range trainBatch {
						batches[i] = nn.TrainingBatch{Input: s.Input, Target: s.Target}
					}
					net.Train(batches, &nn.TrainingConfig{Epochs: 1, LearningRate: learningRate, LossType: "mse"})
					trainBatch = trainBatch[:0]
					lastTrainTime = time.Now()
				}

			case ModeStepBP:
				grad := []float32{prediction - targetVal}
				net.StepBackward(stepState, grad)
				net.ApplyGradients(learningRate)

			case ModeStepTweenChain:
				// Step Tween: Immediate update
				if ts.Config.UseChainRule {
					ts.TweenWeightsChainRule(net, 0.05)
				} else {
					ts.TweenWeights(net, 0.05)
				}

			case ModeTweenChain, ModeTween:
				// Batched Tween
				inHist := make([]float32, InputWindow)
				copy(inHist, inputBuffer)
				trainBatch = append(trainBatch, TrainingSample{Input: inHist, Target: []float32{targetVal}})

				if time.Since(lastTrainTime) > 50*time.Millisecond && len(trainBatch) > 0 {
					for _, s := range trainBatch {
						// For TweenBatch, we process samples
						net.ForwardCPU(s.Input) // Prime activations
						ts.BackwardPassRegression(net, s.Target)

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
		}

		if isVirus {
			poisonLossSum += float64(loss)
			poisonCount++
		} else {
			cleanLossSum += float64(loss)
			cleanCount++
		}

		statusMu.Lock()
		statusMap[mode].Loss = loss
		statusMap[mode].Budget = minBudget
		statusMap[mode].Action = action
		statusMu.Unlock()
	}

	statusMu.Lock()
	statusMap[mode].Completed = true
	if cleanCount > 0 {
		statusMap[mode].CleanLossAvg = float32(cleanLossSum / float64(cleanCount))
	}
	if poisonCount > 0 {
		statusMap[mode].PoisonLossAvg = float32(poisonLossSum / float64(poisonCount))
	}
	if TotalSteps > 0 {
		statusMap[mode].RejectRate = float64(rejectedCount) / float64(TotalSteps) * 100
	}
	statusMu.Unlock()
}
