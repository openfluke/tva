package main

import (
	"fmt"
	"math"
	"strings"
	"time"

	"github.com/openfluke/loom/nn"
)

// ═══════════════════════════════════════════════════════════════════════════════
// THE CHIMERA FUSION: A TEST OF DYNAMIC EXPERT SWITCHING
// ═══════════════════════════════════════════════════════════════════════════════
//
// CHALLENGE: A multi-modal task where the required operation shifts between
// contradictory modes. The network must use a "Gated Mixture of Experts"
// (FilteredParallelLayer) to route data to the correct specialist.
//
// Phases:
//   1. Sin Mode:    y = sin(x)       (Expert A)
//   2. Square Mode: y = x^2          (Expert B)
//   3. Fusion Mode: y = sin(x) + x^2 (Both)
//   4. Chaos Mode:  y = sin(x) * x^2 (Modulation)
//
// We monitor the internal "Gate Weights" to prove the network is making
// real-time decisions about which expert to trust.
// ═══════════════════════════════════════════════════════════════════════════════

const (
	InputWindow    = 16
	TotalDuration  = 20 * time.Second
	StepInterval   = 10 * time.Millisecond
	AccuracyWindow = 50
)

func main() {
	fmt.Println("🦁 The Chimera Fusion: Initializing Multi-Modal Expert Test...")

	// 1. Build Architecture: Gate -> [Expert A, Expert B] -> Output
	// Expert A: Sine Specialist candidate
	expertA := nn.InitDenseLayer(InputWindow, 32, nn.ActivationTanh)
	// Expert B: Square Specialist candidate (ReLU better for x^2)
	expertB := nn.InitDenseLayer(InputWindow, 32, nn.ActivationScaledReLU)

	// Filtered Parallel Layer (MoE)
	// Input to gate is same as input to branches (InputWindow)
	moeLayer := nn.InitFilteredParallelLayer(
		[]nn.LayerConfig{expertA, expertB},
		InputWindow,
		nn.SoftmaxStandard,
		0.5, // Temperature (lower = sharper decisions)
	)

	net := nn.NewNetwork(InputWindow, 1, 3, 1)

	// Layer 0: Feature extraction
	net.SetLayer(0, 0, 0, nn.InitDenseLayer(InputWindow, 32, nn.ActivationLeakyReLU))
	// Layer 1: The Chimera (MoE)
	net.SetLayer(0, 1, 0, moeLayer)
	// Layer 2: Output projection
	net.SetLayer(0, 2, 0, nn.InitDenseLayer(32, 1, nn.ActivationType(-1)))

	net.InitializeWeights()

	// 2. Setup Step Tween Chain State
	tweenConfig := &nn.TweenConfig{
		FrontierEnabled:   true,
		FrontierThreshold: 0.5,
		IgnoreThreshold:   0.005,
		DenseRate:         1.0,
		AttentionRate:     1.0, // Gate is dense but treated special? No, it's just weights.
	}
	ts := nn.NewGenericTweenState[float32](net, tweenConfig)

	// 3. Main Simulation Loop
	startTime := time.Now()
	inputBuffer := make([]float32, InputWindow)
	rollingErrors := make([]float32, 0, AccuracyWindow)

	fmt.Printf("\n%-10s | %-16s | %-10s | %-20s\n", "Time", "Phase", "Loss (MSE)", "Gate [ExpA, ExpB]")
	fmt.Println(strings.Repeat("-", 70))

	ticker := time.NewTicker(StepInterval)
	defer ticker.Stop()

	stepCount := 0
	for elapsed := time.Since(startTime); elapsed < TotalDuration; elapsed = time.Since(startTime) {
		<-ticker.C
		stepCount++
		t := float64(stepCount) * 0.1

		// Determine Phase
		phase := "UNKNOWN"
		var target float32
		val := math.Sin(t) // Base value

		if elapsed < 5*time.Second {
			phase = "SIN MODE"
			target = float32(val)
		} else if elapsed < 10*time.Second {
			phase = "SQUARE MODE"
			target = float32(val * val)
		} else if elapsed < 15*time.Second {
			phase = "FUSION (+)"
			target = float32(val + val*val)
		} else {
			phase = "CHAOS (*)"
			target = float32(val * val * val) // sin * square approx
		}

		// Update Input
		copy(inputBuffer, inputBuffer[1:])
		inputBuffer[InputWindow-1] = float32(val)

		// 4. Training Step
		inputT := nn.NewTensorFromSlice(inputBuffer, InputWindow)

		// -- Manual Introspection of Gate --
		// We do this BEFORE the step updates weights to see the "decision" for this step
		// Get the MoE layer config
		moeCfg := net.GetLayer(0, 1, 0)
		// We need the input to the MoE layer.
		// Since we are running StepTween, the input to Layer 1 is the output of Layer 0.
		// However, StepTween forward pass hasn't happened yet for this step.
		// We can look at the *previous* step's output of Layer 0 using ts.ForwardActs?
		// Or just wait until after ForwardPass. Let's do after ForwardPass.

		predT := ts.ForwardPass(net, inputT, nil)
		prediction := predT.Data[0]

		loss := (target - prediction) * (target - prediction)
		ts.BackwardPassRegression(net, []float32{target})
		ts.TweenWeightsChainRule(net, 0.05)

		// 5. Introspect Gate Weights (After forward pass)
		// Input to MoE (Layer 1) was stored in ts.ForwardActs[1] (Input to Layer 1 is Output of Layer 0? No, ForwardActs indices align with layers)
		// ts.ForwardActs[0] = Network Input
		// ts.ForwardActs[1] = Layer 0 Output / Layer 1 Input
		// ts.ForwardActs[2] = Layer 1 Output / Layer 2 Input
		inputToMoE := ts.ForwardActs[1]

		var gateProbs []float32
		if moeCfg.FilterGateConfig != nil && inputToMoE != nil {
			// Manually run the gate layer to see what it did
			// Recreate tensors for gate weights/bias (pointers share memory with config)
			gw := nn.NewTensorFromSlice(moeCfg.FilterGateConfig.Kernel, len(moeCfg.FilterGateConfig.Kernel))
			gb := nn.NewTensorFromSlice(moeCfg.FilterGateConfig.Bias, len(moeCfg.FilterGateConfig.Bias))

			// Dense Forward
			_, gateOut := nn.DenseForward(inputToMoE, gw, gb,
				moeCfg.FilterGateConfig.InputHeight,
				moeCfg.FilterGateConfig.OutputHeight,
				1,
				moeCfg.FilterGateConfig.Activation)

			// Softmax
			probs, _ := nn.ForwardSoftmaxCPU(gateOut.Data, &nn.LayerConfig{
				SoftmaxVariant: moeCfg.FilterSoftmax,
				Temperature:    moeCfg.FilterTemperature,
			})
			gateProbs = probs
		}

		// 6. Report
		rollingErrors = append(rollingErrors, loss)
		if len(rollingErrors) > AccuracyWindow {
			rollingErrors = rollingErrors[1:]
		}
		avgLoss := float32(0)
		for _, e := range rollingErrors {
			avgLoss += e
		}
		avgLoss /= float32(len(rollingErrors))

		if stepCount%50 == 0 {
			gateStr := "[?.??, ?.??]"
			if len(gateProbs) >= 2 {
				gateStr = fmt.Sprintf("[%.2f, %.2f]", gateProbs[0], gateProbs[1])
			}

			fmt.Printf("%-10s | %-16s | %-10.4f | %s\n",
				elapsed.Round(100*time.Millisecond).String(),
				phase,
				avgLoss,
				gateStr)
		}
	}

	fmt.Println("\n🏁 Chimera Fusion Test Complete.")
}
