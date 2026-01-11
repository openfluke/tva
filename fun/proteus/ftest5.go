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
// THE PROTEUS SIGNAL: A TEST OF CONTINUOUS ADAPTATION
// ═══════════════════════════════════════════════════════════════════════════════

const (
	InputWindow         = 32
	PhaseDuration       = 5 * time.Second
	StepInterval        = 10 * time.Millisecond
	TotalPhases         = 4
	AdaptationThreshold = 0.05
)

// Modes
type TrainingMode int

const (
	ModeNormalBP       TrainingMode = iota // Batch BP
	ModeStepBP                             // Step BP
	ModeTween                              // Pure Tween
	ModeTweenChain                         // Batched Tween (Chain Rule)
	ModeStepTween                          // Immediate Tween (No Chain Rule)
	ModeStepTweenChain                     // Immediate Tween (Chain Rule)
)

var modeNames = map[TrainingMode]string{
	ModeNormalBP:       "BP(Batch)",
	ModeStepBP:         "BP(Step)",
	ModeTween:          "Tween(Pure)",
	ModeTweenChain:     "Tween(Batch)",
	ModeStepTween:      "Tween(Step)",
	ModeStepTweenChain: "Tween(Step+)",
}

// Shared State
type ModeStatus struct {
	Loss      float32
	Status    string
	Latency   time.Duration
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
		ModeStepTween,
		ModeStepTweenChain,
	}

	for _, m := range modes {
		statusMap[m] = &ModeStatus{Status: "INIT"}
	}

	fmt.Println("🌊 The Proteus Signal: Initializing Parallel Continuous Adaptation Test...")

	fmt.Printf("%-8s | %-16s", "Time", "Signal")
	for _, m := range modes {
		fmt.Printf(" | %-8s | %-10s", fmt.Sprintf("Loss %s", modeNames[m]), "Status")
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

	phaseNames := []string{"Sine Wave", "High Freq Sine", "Square Wave", "Composite Wave"}

	running := true
	for running {
		select {
		case <-done:
			running = false
		case <-monitorTicker.C:
			elapsed := time.Since(startTime)

			// Approximate phase
			phaseIdx := int(elapsed / PhaseDuration)
			if phaseIdx >= len(phaseNames) {
				phaseIdx = len(phaseNames) - 1
			}
			phaseName := phaseNames[phaseIdx]

			statusMu.Lock()
			fmt.Printf("%-8s | %-16s", elapsed.Round(100*time.Millisecond).String(), phaseName)
			for _, m := range modes {
				s := statusMap[m]
				val := fmt.Sprintf("%.4f", s.Loss)
				if s.Completed {
					val = "DONE"
				}
				statusStr := s.Status
				if s.Status == "LOCKED" && s.Latency > 0 {
					statusStr = fmt.Sprintf("LOCK(%s)", s.Latency.Round(time.Millisecond))
				}
				fmt.Printf(" | %-8s | %-10s", val, statusStr)
			}
			fmt.Println()
			statusMu.Unlock()
		}
	}
	fmt.Println("\n\n🏁 Proteus Signal Test Complete.")
}

func runMode(mode TrainingMode, startTime time.Time) {
	// 1. Build Architecture
	net := nn.NewNetwork(InputWindow, 1, 3, 1)
	net.SetLayer(0, 0, 0, nn.InitDenseLayer(InputWindow, 64, nn.ActivationTanh))
	net.SetLayer(0, 1, 0, nn.InitDenseLayer(64, 32, nn.ActivationLeakyReLU))
	net.SetLayer(0, 2, 0, nn.InitDenseLayer(32, 1, nn.ActivationType(-1)))
	net.InitializeWeights()

	var stepState *nn.StepState
	if mode == ModeStepBP {
		stepState = net.InitStepState(64)
	}

	ts := nn.NewTweenState(net, nil)
	ts.Config.FrontierEnabled = true
	ts.Config.FrontierThreshold = 0.5
	ts.Config.DenseRate = 0.2
	ts.Config.Momentum = 0.4

	if mode == ModeTween || mode == ModeStepTween {
		ts.Config.UseChainRule = false
	} else {
		ts.Config.UseChainRule = true
	}

	inputBuffer := make([]float32, InputWindow)

	// Phase Info
	phaseNames := []string{"Sine Wave", "High Freq Sine", "Square Wave", "Composite Wave"}
	currentPhaseIdx := 0

	type TrainingSample struct {
		Input  []float32
		Target []float32
	}
	trainBatch := make([]TrainingSample, 0, 10)
	lastTrainTime := time.Now()
	trainInterval := 50 * time.Millisecond
	learningRate := float32(0.02)

	phaseStart := startTime
	lastTrainTime = startTime
	morphTime := startTime
	hasAdapted := false

	ticker := time.NewTicker(StepInterval)
	defer ticker.Stop()

	stepCount := 0

	for {
		now := time.Now()
		phaseElapsed := now.Sub(phaseStart)

		if phaseElapsed >= PhaseDuration {
			currentPhaseIdx++
			if currentPhaseIdx >= len(phaseNames) {
				break
			}
			phaseStart = now
			morphTime = now
			hasAdapted = false
		}

		<-ticker.C
		stepCount++
		t := float64(stepCount) * 0.1

		// 1. Generate Signal
		var targetVal float32
		var val float64

		switch currentPhaseIdx {
		case 0:
			val = math.Sin(t)
		case 1:
			val = math.Sin(t * 2.5)
		case 2:
			s := math.Sin(t)
			if s > 0 {
				val = 0.8
			} else {
				val = -0.8
			}
		case 3:
			val = (math.Sin(t) + (math.Sin(t*3) * 0.5)) / 1.5
		}
		targetVal = float32(val)

		copy(inputBuffer, inputBuffer[1:])
		inputBuffer[InputWindow-1] = targetVal

		// Forward
		var prediction float32

		switch mode {
		case ModeTween, ModeTweenChain, ModeNormalBP:
			predArgs, _ := net.ForwardCPU(inputBuffer)
			prediction = predArgs[0]

		case ModeStepTweenChain, ModeStepTween, ModeStepBP:
			if mode == ModeStepBP {
				stepState.SetInput(inputBuffer)
				net.StepForward(stepState)
				out := stepState.GetOutput()
				if len(out) > 0 {
					prediction = out[0]
				}
			} else {
				// StepTween variants use ForwardCPU logic
				predArgs, _ := net.ForwardCPU(inputBuffer)
				prediction = predArgs[0]
			}
		}

		loss := (targetVal - prediction) * (targetVal - prediction)

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

		case ModeStepTweenChain, ModeStepTween:
			if ts.Config.UseChainRule {
				ts.BackwardPassRegression(net, []float32{targetVal})
				ts.TweenWeightsChainRule(net, 0.2)
			} else {
				ts.BackwardPassRegression(net, []float32{targetVal})
				ts.TweenWeights(net, 0.2)
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
						ts.TweenWeightsChainRule(net, 0.2)
					} else {
						ts.TweenWeights(net, 0.2)
					}
				}
				trainBatch = trainBatch[:0]
				lastTrainTime = time.Now()
			}
		}

		status := "ADAPTING"
		var latency time.Duration
		if loss < AdaptationThreshold {
			status = "LOCKED"
			if !hasAdapted {
				hasAdapted = true
				latency = time.Since(morphTime)
			}
		}

		statusMu.Lock()
		statusMap[mode].Loss = loss
		statusMap[mode].Status = status
		if latency > 0 {
			statusMap[mode].Latency = latency
		}
		statusMu.Unlock()
	}

	statusMu.Lock()
	statusMap[mode].Completed = true
	statusMu.Unlock()
}
