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
// ⏳ THE CHRONOS PARADOX: A TEST OF IMPLICIT RECURRENCE
// ═══════════════════════════════════════════════════════════════════════════════

const (
	DeepLayers     = 12
	InputWindow    = 1
	StepInterval   = 20 * time.Millisecond
	TrainingSteps  = 300
	OcclusionSteps = 20
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
	Prediction float32
	Status     string
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
		statusMap[m] = &ModeStatus{Status: "INIT"}
	}

	fmt.Println("⏳ The Chronos Paradox: Initializing Parallel Object Permanence Test...")

	fmt.Printf("%-8s | %-12s | %-10s", "Step", "State", "Input")
	for _, m := range modes {
		fmt.Printf(" | %-12s | %-10s", fmt.Sprintf("Pred %s", modeNames[m]), "Status")
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

	monitorTicker := time.NewTicker(StepInterval)
	defer monitorTicker.Stop()

	done := make(chan bool)
	go func() {
		wg.Wait()
		done <- true
	}()

	running := true
	stepRel := 0

	for running {
		select {
		case <-done:
			running = false
		case <-monitorTicker.C:
			// We can't perfectly sync steps for display, so we just sample at interval
			step := stepRel // Rough approx
			_ = step

			elapsed := time.Since(startTime)
			// Approx step
			approxStep := int(elapsed / StepInterval)
			stateStr := "TRAINING"
			inputVal := "SIGNAL"

			if approxStep >= TrainingSteps {
				stateStr = "OCCLUDED"
				inputVal = "0.0000"
			}

			statusMu.Lock()
			fmt.Printf("%-8d | %-12s | %-10s", approxStep, stateStr, inputVal)
			for _, m := range modes {
				s := statusMap[m]
				val := fmt.Sprintf("%.4f", s.Prediction)
				if s.Completed {
					val = "DONE"
				}
				fmt.Printf(" | %-12s | %-10s", val, s.Status)
			}
			fmt.Println()
			statusMu.Unlock()

			stepRel++
			if stepRel > TrainingSteps+OcclusionSteps+10 {
				// Fallback exit
			}
		}
	}
	fmt.Println("\n\n🏁 Chronos Paradox Test Complete.")
}

func runMode(mode TrainingMode, startTime time.Time) {
	// 1. Build Deep Architecture (12 Layers)
	net := nn.NewNetwork(1, 1, 1, DeepLayers+2)
	net.SetLayer(0, 0, 0, nn.InitDenseLayer(1, 16, nn.ActivationLeakyReLU)) // Input
	for i := 1; i <= DeepLayers; i++ {
		net.SetLayer(0, 0, i, nn.InitDenseLayer(16, 16, nn.ActivationLeakyReLU))
	}
	net.SetLayer(0, 0, DeepLayers+1, nn.InitDenseLayer(16, 1, nn.ActivationType(-1))) // Output
	net.InitializeWeights()

	var stepState *nn.StepState
	if mode == ModeStepBP {
		stepState = net.InitStepState(10)
	}

	ts := nn.NewTweenState(net, nil)
	ts.Config.FrontierEnabled = true
	ts.Config.DenseRate = 0.5
	ts.Config.Momentum = 0.0

	if mode == ModeTween {
		ts.Config.UseChainRule = false
	} else {
		ts.Config.UseChainRule = true
	}

	inputBuffer := make([]float32, 1)

	type TrainingSample struct {
		Input  []float32
		Target []float32
	}
	trainBatch := make([]TrainingSample, 0, 10)
	lastTrainTime := time.Now()
	trainInterval := 50 * time.Millisecond
	learningRate := float32(0.5)

	occluded := false
	occlusionStepStart := 0

	ticker := time.NewTicker(StepInterval)
	defer ticker.Stop()

	for step := 0; step < TrainingSteps+OcclusionSteps; step++ {
		<-ticker.C
		t := float64(step) * 0.2
		realSignal := float32(0.5 + 0.4*math.Sin(t))

		var inputVal float32
		if step >= TrainingSteps {
			occluded = true
			if occlusionStepStart == 0 {
				occlusionStepStart = step
			}
			inputVal = 0.0
		} else {
			inputVal = realSignal
		}

		inputBuffer[0] = inputVal

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
				// StepTween uses ForwardCPU logic
				predArgs, _ := net.ForwardCPU(inputBuffer)
				prediction = predArgs[0]
			}
		}

		target := realSignal

		// Training (Only if not occluded)
		if !occluded {
			switch mode {
			case ModeNormalBP:
				inHist := make([]float32, InputWindow)
				copy(inHist, inputBuffer)
				trainBatch = append(trainBatch, TrainingSample{Input: inHist, Target: []float32{target}})
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
				grad := []float32{prediction - target}
				net.StepBackward(stepState, grad)
				net.ApplyGradients(learningRate)

			case ModeStepTweenChain:
				if ts.Config.UseChainRule {
					ts.BackwardPassRegression(net, []float32{target})
					ts.TweenWeightsChainRule(net, 0.5)
				} else {
					ts.BackwardPassRegression(net, []float32{target})
					ts.TweenWeights(net, 0.5)
				}

			case ModeTweenChain, ModeTween:
				inHist := make([]float32, InputWindow)
				copy(inHist, inputBuffer)
				trainBatch = append(trainBatch, TrainingSample{Input: inHist, Target: []float32{target}})
				if time.Since(lastTrainTime) > 50*time.Millisecond && len(trainBatch) > 0 {
					for _, s := range trainBatch {
						net.ForwardCPU(s.Input) // Prime
						ts.BackwardPassRegression(net, s.Target)

						if ts.Config.UseChainRule {
							ts.TweenWeightsChainRule(net, 0.5)
						} else {
							ts.TweenWeights(net, 0.5)
						}
					}
					trainBatch = trainBatch[:0]
					lastTrainTime = time.Now()
				}
			}
		}

		// Analysis
		status := "OK"
		if occluded {
			if math.Abs(float64(prediction)) < 0.01 {
				status = "COLLAPSED"
			} else {
				status = "GHOST"
			}
		} else {
			diff := prediction - realSignal
			if diff*diff > 0.1 {
				status = "LOST"
			}
		}

		statusMu.Lock()
		statusMap[mode].Prediction = prediction
		statusMap[mode].Status = status
		statusMu.Unlock()
	}

	statusMu.Lock()
	statusMap[mode].Completed = true
	statusMu.Unlock()
}
