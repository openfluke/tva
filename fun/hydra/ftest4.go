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
// THE HYDRA MEMORY: A TEST OF CATASTROPHIC FORGETTING (SIMPLIFIED)
// ═══════════════════════════════════════════════════════════════════════════════

const (
	InputWindow        = 16
	PhaseDuration      = 5 * time.Second
	StepInterval       = 10 * time.Millisecond
	TotalPhases        = 4
	StabilityThreshold = 0.05
	StabilityWindow    = 10
)

// Phase Info
type PhaseInfo struct {
	Name      string
	Converged bool
	ConvTime  time.Duration
}

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
	Status    string
	Phases    []PhaseInfo
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
		// Init phases copy for each mode
		initPhases := []PhaseInfo{
			{Name: "Phase A"},
			{Name: "Phase B"},
			{Name: "Phase C"},
			{Name: "Phase A'"},
		}
		statusMap[m] = &ModeStatus{Status: "INIT", Phases: initPhases}
	}

	fmt.Println("🐍 The Hydra Memory: Initializing Parallel Retention Test...")

	fmt.Printf("%-8s | %-12s", "Time", "Phase")
	for _, m := range modes {
		fmt.Printf(" | %-12s | %-12s", fmt.Sprintf("Loss %s", modeNames[m]), "Status")
	}
	fmt.Println()
	fmt.Println(strings.Repeat("-", 150))

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

Loop:
	for {
		select {
		case <-done:
			break Loop
		case <-monitorTicker.C:
			elapsed := time.Since(startTime)
			// Phase is mode-dependent roughly, but driven by wall clock in runMode
			// We can approximate global phase or read from one mode
			phaseName := "UNKNOWN"
			statusMu.Lock()
			// Just use first active mode's phase name
			for _, m := range modes {
				// The phases update in `Phases` struct.
				// Let's just calculate based on time
				if elapsed < PhaseDuration {
					phaseName = "Phase A"
				} else if elapsed < 2*PhaseDuration {
					phaseName = "Phase B"
				} else if elapsed < 3*PhaseDuration {
					phaseName = "Phase C"
				} else {
					phaseName = "Phase A'"
				}
				_ = m
				break
			}

			fmt.Printf("%-8s | %-12s", elapsed.Round(100*time.Millisecond).String(), phaseName)
			for _, m := range modes {
				s := statusMap[m]
				lossVal := fmt.Sprintf("%.4f", s.Loss)
				if s.Completed {
					lossVal = "DONE"
				}
				fmt.Printf(" | %-12s | %-12s", lossVal, s.Status)
			}
			fmt.Println()
			statusMu.Unlock()
		}
	}
	fmt.Println("\n\n🏁 Hydra Memory Test Complete.")

	// Print Summary
	fmt.Println("\n📊 CONVERGENCE SUMMARY")
	fmt.Println("==================================================")
	fmt.Printf("%-15s | %-10s | %-10s | %-10s | %-10s\n", "Mode", "Phase A", "Phase B", "Phase C", "Phase A'")
	statusMu.Lock()
	for _, m := range modes {
		s := statusMap[m]
		fmt.Printf("%-15s", modeNames[m])
		for _, p := range s.Phases {
			tStr := "FAIL"
			if p.Converged {
				tStr = p.ConvTime.Round(10 * time.Millisecond).String()
			}
			fmt.Printf(" | %-10s", tStr)
		}
		fmt.Println()
	}
	statusMu.Unlock()
}

func runMode(mode TrainingMode, startTime time.Time) {
	// Experts
	// Layer 0 is 16->32. Experts must be 32->?
	exp1 := nn.InitDenseLayer(32, 32, nn.ActivationTanh)
	exp2 := nn.InitDenseLayer(32, 32, nn.ActivationScaledReLU)
	exp3 := nn.InitDenseLayer(32, 32, nn.ActivationLeakyReLU)

	moeLayer := nn.InitFilteredParallelLayer(
		[]nn.LayerConfig{exp1, exp2, exp3},
		32,
		nn.SoftmaxStandard,
		0.2,
	)

	net := nn.NewNetwork(InputWindow, 1, 3, 1)
	net.SetLayer(0, 0, 0, nn.InitDenseLayer(InputWindow, 32, nn.ActivationLeakyReLU))
	net.SetLayer(0, 1, 0, moeLayer)
	net.SetLayer(0, 2, 0, nn.InitDenseLayer(32, 1, nn.ActivationType(-1)))
	net.InitializeWeights()

	var stepState *nn.StepState
	if mode == ModeStepBP {
		stepState = net.InitStepState(10)
	}

	ts := nn.NewTweenState(net, nil)
	ts.Config.DenseRate = 0.1 // Higher rate for SGD

	if mode == ModeTween {
		ts.Config.UseChainRule = false
	} else {
		ts.Config.UseChainRule = true
	}

	inputBuffer := make([]float32, InputWindow)

	// Local Phase Tracking
	phases := []PhaseInfo{
		{Name: "Phase A"},
		{Name: "Phase B"},
		{Name: "Phase C"},
		{Name: "Phase A'"},
	}
	currentPhaseIdx := 0

	type TrainingSample struct {
		Input  []float32
		Target []float32
	}
	trainBatch := make([]TrainingSample, 0, 10)
	lastTrainTime := time.Now()
	trainInterval := 50 * time.Millisecond
	learningRate := float32(0.1)

	stableSteps := 0
	phaseStart := startTime
	lastTrainTime = startTime

	ticker := time.NewTicker(StepInterval)
	defer ticker.Stop()

	stepCount := 0

	for {
		now := time.Now()
		phaseElapsed := now.Sub(phaseStart)

		if phaseElapsed >= PhaseDuration {
			if !phases[currentPhaseIdx].Converged {
				phases[currentPhaseIdx].ConvTime = PhaseDuration
			}
			currentPhaseIdx++
			if currentPhaseIdx >= len(phases) {
				break
			}
			phaseStart = now
			stableSteps = 0
		}

		<-ticker.C
		stepCount++
		t := float64(stepCount) * 0.1

		var targetVal float32
		val := math.Sin(t)

		switch currentPhaseIdx {
		case 0, 3: // Phase A
			targetVal = float32(val)
		case 1: // Phase B
			targetVal = float32(val * val)
		case 2: // Phase C
			saw := math.Mod(t, math.Pi) / math.Pi
			targetVal = float32(saw*2 - 1)
		}

		copy(inputBuffer, inputBuffer[1:])
		inputBuffer[InputWindow-1] = float32(val)

		// Manual Gating
		moeCfg := net.GetLayer(0, 1, 0)
		for i := range moeCfg.ParallelBranches {
			moeCfg.ParallelBranches[i].Frozen = true
		}
		switch currentPhaseIdx {
		case 0, 3:
			moeCfg.ParallelBranches[0].Frozen = false // Expert 1 for Phase A
		case 1:
			moeCfg.ParallelBranches[1].Frozen = false // Expert 2 for Phase B
		case 2:
			moeCfg.ParallelBranches[2].Frozen = false // Expert 3 for Phase C
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
				// StepTween uses ForwardCPU logic
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

		case ModeStepTweenChain:
			if ts.Config.UseChainRule {
				ts.BackwardPassRegression(net, []float32{targetVal})
				ts.TweenWeightsChainRule(net, 0.1)
			} else {
				ts.BackwardPassRegression(net, []float32{targetVal})
				ts.TweenWeights(net, 0.1)
			}

		case ModeTweenChain, ModeTween:
			inHist := make([]float32, InputWindow)
			copy(inHist, inputBuffer)
			trainBatch = append(trainBatch, TrainingSample{Input: inHist, Target: []float32{targetVal}})
			if time.Since(lastTrainTime) > 50*time.Millisecond && len(trainBatch) > 0 {
				for _, s := range trainBatch {
					net.ForwardCPU(s.Input)
					ts.BackwardPassRegression(net, s.Target)

					if ts.Config.UseChainRule {
						ts.TweenWeightsChainRule(net, 0.1)
					} else {
						ts.TweenWeights(net, 0.1)
					}
				}
				trainBatch = trainBatch[:0]
				lastTrainTime = time.Now()
			}
		}

		if loss < StabilityThreshold {
			stableSteps++
		} else {
			stableSteps = 0
		}

		if !phases[currentPhaseIdx].Converged && stableSteps >= StabilityWindow {
			phases[currentPhaseIdx].Converged = true
			phases[currentPhaseIdx].ConvTime = time.Since(phaseStart)
		}

		statusMu.Lock()
		statusMap[mode].Loss = loss

		status := "LEARNING"
		if phases[currentPhaseIdx].Converged {
			status = "STABLE"
		}
		statusMap[mode].Status = status

		// Update persistent phases copy for summary
		statusMap[mode].Phases = phases
		statusMu.Unlock()
	}

	statusMu.Lock()
	statusMap[mode].Completed = true
	statusMu.Unlock()
}
