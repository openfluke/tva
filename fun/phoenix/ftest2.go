package main

import (
	"fmt"
	"math"
	"strings"
	"sync"
	"time"

	"github.com/openfluke/loom/nn"
)

// ═══════════════════════════════════════════════════════════════════════════════
// THE PHOENIX RESILIENCE: A TEST OF NEURAL NEUROPLASTICITY
// ═══════════════════════════════════════════════════════════════════════════════

const (
	InputWindow    = 24
	InjuryTime     = 8 * time.Second
	TotalDuration  = 20 * time.Second
	StepInterval   = 10 * time.Millisecond
	AccuracyWindow = 50
)

// Modes
type TrainingMode int

const (
	ModeNormalBP       TrainingMode = iota // Batch BP
	ModeStepBP                             // Step BP
	ModeTween                              // Pure Tween
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

// Shared State
type ModeStatus struct {
	Loss      float32
	ExpertL2  float32
	Status    string
	Completed bool
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
		statusMap[m] = &ModeStatus{Status: "INIT"}
	}

	fmt.Println("🦅 The Phoenix Resilience: Initializing Parallel Neuroplasticity Test...")

	fmt.Printf("%-8s | %-12s", "Time", "Condition")
	for _, m := range modes {
		fmt.Printf(" | %-12s | %-10s", modeNames[m], "L2")
	}
	fmt.Println()
	fmt.Println(strings.Repeat("-", 140))

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
			condition := "HEALTHY"
			if elapsed > InjuryTime {
				condition = "INJURED 💥"
				if elapsed > InjuryTime+5*time.Second {
					condition = "HEALING"
				}
			}

			statusMu.Lock()
			fmt.Printf("%-8s | %-12s", elapsed.Round(100*time.Millisecond).String(), condition)
			for _, m := range modes {
				s := statusMap[m]
				lossVal := fmt.Sprintf("%.4f", s.Loss)
				l2Val := fmt.Sprintf("%.4f", s.ExpertL2)
				if s.Completed {
					lossVal = "DONE"
				}
				fmt.Printf(" | %-12s | %-10s", lossVal, l2Val)
			}
			fmt.Println()
			statusMu.Unlock()
		}
	}
	fmt.Println("\n\n🏁 Phoenix Test Complete.")
}

func runMode(mode TrainingMode, startTime time.Time) {
	// 1. Build Network
	branchA := nn.InitDenseLayer(InputWindow, 32, nn.ActivationScaledReLU)
	branchA.Kernel = make([]float32, InputWindow*32)
	branchB := nn.InitDenseLayer(InputWindow, 32, nn.ActivationScaledReLU)

	parallelLayer := nn.LayerConfig{
		Type:             nn.LayerParallel,
		CombineMode:      "add",
		ParallelBranches: []nn.LayerConfig{branchA, branchB},
	}

	net := nn.NewNetwork(InputWindow, 1, 3, 1)
	net.SetLayer(0, 0, 0, nn.InitDenseLayer(InputWindow, InputWindow, nn.ActivationScaledReLU))
	net.SetLayer(0, 1, 0, parallelLayer)
	net.SetLayer(0, 2, 0, nn.InitDenseLayer(32, 1, nn.ActivationTanh))
	net.InitializeWeights()

	// StepState for StepBP
	var stepState *nn.StepState
	if mode == ModeStepBP {
		stepState = net.InitStepState(10)
	}

	// 2. Twin Config
	ts := nn.NewTweenState(net, nil)
	ts.Config.FrontierEnabled = true
	ts.Config.FrontierThreshold = 0.5
	ts.Config.IgnoreThreshold = 0.005
	ts.Config.DenseRate = 1.0

	if mode == ModeTween {
		ts.Config.UseChainRule = false
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
	trainBatch := make([]TrainingSample, 0, 10)
	lastTrainTime := time.Now()
	trainInterval := 50 * time.Millisecond
	learningRate := float32(0.02)

	injured := false

	ticker := time.NewTicker(StepInterval)
	defer ticker.Stop()

	stepCount := 0
	for elapsed := time.Since(startTime); elapsed < TotalDuration; elapsed = time.Since(startTime) {
		<-ticker.C
		stepCount++

		// Signal
		t := float64(stepCount) * 0.1
		rawSignal := float32(math.Sin(t) + 0.5*math.Cos(t*2.0))
		copy(inputBuffer, inputBuffer[1:])
		inputBuffer[InputWindow-1] = rawSignal

		// Target
		targetVal := float32(math.Sin(t+0.1) + 0.5*math.Cos((t+0.1)*2.0))

		// 4. Simulate Injury
		if !injured && elapsed > InjuryTime {
			injured = true
			// WIPE Branch A -- Need to get from net
			parallelCfg := net.GetLayer(0, 1, 0)
			if len(parallelCfg.ParallelBranches) > 0 {
				branch := &parallelCfg.ParallelBranches[0]
				for i := range branch.Kernel {
					branch.Kernel[i] = 0
				}
				for i := range branch.Bias {
					branch.Bias[i] = 0
				}
			}
		}

		// Forward
		var prediction float32

		switch mode {
		case ModeTween, ModeTweenChain, ModeNormalBP:
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
				// StepTween uses ForwardCPU logic + Step training
				predArgs, _ := net.ForwardCPU(inputBuffer)
				prediction = predArgs[0]
			}
		}

		loss := (targetVal - prediction) * (targetVal - prediction)

		adaptRate := float32(0.05)
		if injured && elapsed < InjuryTime+2*time.Second {
			adaptRate = 0.15
		}

		// Training
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
			if ts.Config.UseChainRule {
				ts.BackwardPassRegression(net, []float32{targetVal})
				ts.TweenWeightsChainRule(net, adaptRate)
			} else {
				ts.BackwardPassRegression(net, []float32{targetVal})
				ts.TweenWeights(net, adaptRate)
			}
		case ModeTweenChain, ModeTween:
			inHist := make([]float32, InputWindow)
			copy(inHist, inputBuffer)
			trainBatch = append(trainBatch, TrainingSample{Input: inHist, Target: []float32{targetVal}})
			if time.Since(lastTrainTime) > 50*time.Millisecond && len(trainBatch) > 0 {
				for _, s := range trainBatch {
					net.ForwardCPU(s.Input)                  // Prime
					ts.BackwardPassRegression(net, s.Target) // Gradients

					if ts.Config.UseChainRule {
						ts.TweenWeightsChainRule(net, adaptRate)
					} else {
						ts.TweenWeights(net, adaptRate)
					}
				}
				trainBatch = trainBatch[:0]
				lastTrainTime = time.Now()
			}
		}

		// Metric
		rollingErrors = append(rollingErrors, loss)
		if len(rollingErrors) > AccuracyWindow {
			rollingErrors = rollingErrors[1:]
		}
		avgLoss := float32(0)
		for _, e := range rollingErrors {
			avgLoss += e
		}
		avgLoss /= float32(len(rollingErrors))

		// Introspect L2
		l2 := float32(0)
		pCfg := net.GetLayer(0, 1, 0)
		if len(pCfg.ParallelBranches) > 0 {
			for _, w := range pCfg.ParallelBranches[0].Kernel {
				l2 += w * w
			}
		}

		statusMu.Lock()
		statusMap[mode].Loss = avgLoss
		statusMap[mode].ExpertL2 = l2
		if injured {
			statusMap[mode].Status = "INJURED"
		} else {
			statusMap[mode].Status = "HEALTHY"
		}
		statusMu.Unlock()
	}
	statusMu.Lock()
	statusMap[mode].Completed = true
	statusMu.Unlock()
}
