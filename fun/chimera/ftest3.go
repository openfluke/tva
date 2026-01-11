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
// THE CHIMERA FUSION: A TEST OF DYNAMIC EXPERT SWITCHING
// ═══════════════════════════════════════════════════════════════════════════════

const (
	InputWindow    = 16
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
	GateProbs []float32
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
		statusMap[m] = &ModeStatus{Loss: 0, GateProbs: []float32{0, 0}}
	}

	fmt.Println("🦁 The Chimera Fusion: Initializing Parallel Multi-Modal Expert Test...")

	fmt.Printf("%-8s | %-12s", "Time", "Phase")
	for _, m := range modes {
		fmt.Printf(" | %-12s | %-12s", modeNames[m], "Gate[A, B]")
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

	running := true
	for running {
		select {
		case <-done:
			running = false
		case <-monitorTicker.C:
			elapsed := time.Since(startTime)
			phase := "UNKNOWN"
			if elapsed < 5*time.Second {
				phase = "SIN MODE"
			} else if elapsed < 10*time.Second {
				phase = "SQUARE MODE"
			} else if elapsed < 15*time.Second {
				phase = "FUSION"
			} else {
				phase = "CHAOS"
			}

			statusMu.Lock()
			fmt.Printf("%-8s | %-12s", elapsed.Round(100*time.Millisecond).String(), phase)
			for _, m := range modes {
				s := statusMap[m]
				gateStr := "[?.??, ?.??]"
				if len(s.GateProbs) >= 2 {
					gateStr = fmt.Sprintf("[%.2f, %.2f]", s.GateProbs[0], s.GateProbs[1])
				}
				lossVal := fmt.Sprintf("%.4f", s.Loss)
				if s.Completed {
					lossVal = "DONE"
				}
				fmt.Printf(" | %-12s | %-12s", lossVal, gateStr)
			}
			fmt.Println()
			statusMu.Unlock()
		}
	}
	fmt.Println("\n\n🏁 Chimera Fusion Test Complete.")
}

func runMode(mode TrainingMode, startTime time.Time) {
	// 1. Build Architecture
	// Layer 0 outputs 32, so Experts must take 32
	expertA := nn.InitDenseLayer(32, 32, nn.ActivationTanh)
	expertB := nn.InitDenseLayer(32, 32, nn.ActivationScaledReLU)

	moeLayer := nn.InitFilteredParallelLayer(
		[]nn.LayerConfig{expertA, expertB},
		32, // Input is 32
		nn.SoftmaxStandard,
		0.5,
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

	// 2. Setup
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

	ticker := time.NewTicker(StepInterval)
	defer ticker.Stop()

	stepCount := 0
	for elapsed := time.Since(startTime); elapsed < TotalDuration; elapsed = time.Since(startTime) {
		<-ticker.C
		stepCount++
		t := float64(stepCount) * 0.1

		// Phase
		var targetVal float32
		val := math.Sin(t)

		if elapsed < 5*time.Second {
			targetVal = float32(val)
		} else if elapsed < 10*time.Second {
			targetVal = float32(val * val)
		} else if elapsed < 15*time.Second {
			targetVal = float32(val + val*val)
		} else {
			targetVal = float32(val * val * val)
		}

		copy(inputBuffer, inputBuffer[1:])
		inputBuffer[InputWindow-1] = float32(val)

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
			// Uses Chain Rule by default for StepTweenChain
			// But for ModeTween (pure), we set useChainRule=false
			// Here we assume StepTweenChain means Chain Rule
			if ts.Config.UseChainRule {
				ts.BackwardPassRegression(net, []float32{targetVal})
				ts.TweenWeightsChainRule(net, 0.05)
			} else {
				// Safety fallback if config was somehow false (e.g. ModeTween logic sharing)
				ts.BackwardPassRegression(net, []float32{targetVal})
				ts.TweenWeights(net, 0.05)
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
						ts.TweenWeightsChainRule(net, 0.05)
					} else {
						ts.TweenWeights(net, 0.05)
					}
				}
				trainBatch = trainBatch[:0]
				lastTrainTime = time.Now()
			}
		}

		// Introspect Gate
		// For StepBP, we need to inspect the StepState internals or LayerConfig if using StepForward.
		// StepForward updates StepState.Activations.
		// But MoE Gate probs might not be exposed easily in StepState unless we drill down.
		// With Tween/NormalBP we use ts.ForwardActs or recreate forward.
		// Recreating forward for inspection is safest.

		moeCfg := net.GetLayer(0, 1, 0)
		var gateProbs []float32

		// Input to MoE is at Layer 1.
		// For visualization of Gate Probs, we need the input to the MoE layer (L1).
		// Since we might be using ForwardCPU or StepForward, we can't reliably depend on intermediate state
		// being exposed in a generic way. We will effectively "peek" by running a partial forward pass
		// of Layer 0 just for this visualization.
		var inputToMoE *nn.Tensor[float32]

		l0 := net.GetLayer(0, 0, 0)
		if l0 != nil {
			gw := nn.NewTensorFromSlice(l0.Kernel, len(l0.Kernel))
			gb := nn.NewTensorFromSlice(l0.Bias, len(l0.Bias))
			inT := nn.NewTensorFromSlice(inputBuffer, InputWindow)
			_, l0Out := nn.DenseForward(inT, gw, gb, l0.InputHeight, l0.OutputHeight, 1, l0.Activation)
			inputToMoE = l0Out
		}

		if moeCfg.FilterGateConfig != nil && inputToMoE != nil && len(inputToMoE.Data) > 0 {
			gw := nn.NewTensorFromSlice(moeCfg.FilterGateConfig.Kernel, len(moeCfg.FilterGateConfig.Kernel))
			gb := nn.NewTensorFromSlice(moeCfg.FilterGateConfig.Bias, len(moeCfg.FilterGateConfig.Bias))
			_, gateOut := nn.DenseForward(inputToMoE, gw, gb,
				moeCfg.FilterGateConfig.InputHeight,
				moeCfg.FilterGateConfig.OutputHeight,
				1,
				moeCfg.FilterGateConfig.Activation)
			probs, _ := nn.ForwardSoftmaxCPU(gateOut.Data, &nn.LayerConfig{
				SoftmaxVariant: moeCfg.FilterSoftmax,
				Temperature:    moeCfg.FilterTemperature,
			})
			gateProbs = probs
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
		statusMap[mode].GateProbs = gateProbs
		statusMu.Unlock()
	}
	statusMu.Lock()
	statusMap[mode].Completed = true
	statusMu.Unlock()
}
