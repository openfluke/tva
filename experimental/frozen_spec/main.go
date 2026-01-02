package main

import (
	"fmt"
	"math"
	"math/rand"
	"time"

	"github.com/openfluke/loom/nn"
)

// Global constants
const (
	InputSize        = 8
	CommonOutputSize = 1
	TrainingEpochs   = 1000 // Increased for better Gate convergence
)

func main() {
	rand.Seed(time.Now().UnixNano())

	fmt.Println("╔══════════════════════════════════════════════════════════════════╗")
	fmt.Println("║       🧊 Frozen Specialization Training Mode Benchmark 🧊          ║")
	fmt.Println("║       (Fixed Experts: Manual Gradient Descent Pre-training)      ║")
	fmt.Println("╚══════════════════════════════════════════════════════════════════╝")

	// 1. Define Training Modes
	modes := []string{
		"Standard Forward/Backward",
		"StepBack",
		"Step Tween",
		"Tween",
		"Tween Chain",
		"Step Tween Chain",
	}

	results := make([]string, 0, len(modes))
	for _, mode := range modes {
		res := runExperimentForMode(mode)
		results = append(results, res)
	}

	fmt.Printf("\n\n")
	fmt.Printf("%-25s | %-10s | %-10s | %-10s | %-10s | %s\n", "Mode", "Expert 1", "Expert 2", "Network", "Ideal", "% Off")
	fmt.Println("------------------------------------------------------------------------------------------------------------------")
	for _, res := range results {
		fmt.Println(res)
	}
}

func runExperimentForMode(mode string) string {
	// 2. Setup Network with Frozen Experts (Common Foundation)
	expert1, expert2 := createExperts(mode)
	
	// Create Filter Layer
	gateLayer := nn.InitDenseLayer(InputSize, 2, nn.ActivationScaledReLU)
	
	// Initialize Gate to random small weights
	for i := range gateLayer.Kernel {
		gateLayer.Kernel[i] = (rand.Float32() - 0.5) * 0.1
	}

	filterLayer := nn.LayerConfig{
		Type:              nn.LayerParallel,
		ParallelBranches:  []nn.LayerConfig{expert1, expert2},
		CombineMode:       "filter", 
		FilterGateConfig:  &gateLayer,
		FilterSoftmax:     nn.SoftmaxStandard,
		FilterTemperature: 0.5,
	}

	net := nn.NewNetwork(InputSize, 1, 1, 1)
	net.SetLayer(0, 0, 0, filterLayer)
	
	// 3. Train Gate with Specific Mode
	trainGate(net, mode)

	// 4. Verify & Compare
	// We check if it routes correctly for High input (Expert 1) AND Low input (Expert 2)
	
	// Evaluate High Case (Expect Exp1)
	val1H, _, netValH, idealH := evaluateNetwork(net, true)
	diffH := float64(netValH - idealH)
	
	// Evaluate Low Case (Expect Exp2)
	_, val2L, netValL, idealL := evaluateNetwork(net, false)
	diffL := float64(netValL - idealL)

	// Combined Error
	avgOff := (math.Abs(diffH)/float64(idealH+0.0001) + math.Abs(diffL)/float64(idealL+0.0001)) / 2.0 * 100.0

	// Format result
	return fmt.Sprintf("%-25s | %-10.4f | %-10.4f | %-10.4f | %-10.4f | %.2f%%", 
		mode, val1H, val2L, netValH, idealH, avgOff)
}

func createExperts(mode string) (nn.LayerConfig, nn.LayerConfig) {
	fmt.Printf("\n=== Preparing Experts for %s ===\n", mode)

	// Expert 1
	// CHANGE: Hidden layer uses LeakyReLU to prevent vanishing gradients
	e1 := nn.InitDenseLayer(InputSize, 8, nn.ActivationLeakyReLU)
	s1 := nn.InitDenseLayer(8, CommonOutputSize, nn.ActivationSigmoid)
	b1 := nn.InitSequentialLayer(e1, s1)
	
	fmt.Print("   🎓 Pre-training Expert 1... ")
	trainExpert(&b1, InputSize, true) // High -> 1.0
	freezeLayer(&b1)

	// Expert 2
	e2 := nn.InitDenseLayer(InputSize, 8, nn.ActivationLeakyReLU)
	s2 := nn.InitDenseLayer(8, CommonOutputSize, nn.ActivationSigmoid)
	b2 := nn.InitSequentialLayer(e2, s2)
	
	fmt.Print("   🎓 Pre-training Expert 2... ")
	trainExpert(&b2, InputSize, false) // Low -> 1.0
	freezeLayer(&b2)

	return b1, b2
}

// trainExpert uses aggressive training with a quality check
func trainExpert(layer *nn.LayerConfig, inputSize int, highDetect bool) {
	// 1. Create a temporary network to train this expert
	// The layer passed in is a Sequential Layer containing [Dense(Input->8), Dense(8->1)]
	tempNet := nn.NewNetwork(inputSize, 1, 1, 1)
	tempNet.SetLayer(0, 0, 0, *layer) // Pass by value, but slices are shared

	// 2. Generate Training Data (previously 10000 random samples)
	// We'll generate 2000 samples and run for 5 epochs = 10000 total steps
	// Batch size is implicitly 1 in standard Forward (unless using ForwardBatch which Train might do? 
	// nn.Train iterates batches. We'll make batches of size 1 for simplicity to match previous behavior 
	// or larger for speed. Let's use batch size 10.
	
	trainingData := make([]nn.TrainingBatch, 2000)
	
	for i := 0; i < 2000; i++ {
		input := make([]float32, inputSize)
		for j := range input { input[j] = rand.Float32() * 0.1 }
		
		targetVal := float32(0.0)
		if highDetect {
			if rand.Float32() > 0.5 {
				input[0] = 0.9 // Trigger
				targetVal = 1.0
			}
		} else {
			if rand.Float32() > 0.5 {
				input[0] = 0.1 // Trigger
				targetVal = 1.0
			}
		}
		
		trainingData[i] = nn.TrainingBatch{
			Input:  input,
			Target: []float32{targetVal},
		}
	}

	// 3. Train
	config := &nn.TrainingConfig{
		Epochs:          5,
		LearningRate:    0.05,
		UseGPU:          false, // Use CPU for small demo
		Verbose:         true,
		LossType:        "mse",
		PrintEveryBatch: 0,
	}
	
	fmt.Printf("   Standard Framework Training (%d samples, %d epochs)...\n", len(trainingData), config.Epochs)
	_, err := tempNet.Train(trainingData, config)
	if err != nil {
		fmt.Printf("Training failed: %v\n", err)
	}
}

func trainGate(net *nn.Network, mode string) {
	// Optimizers
	sgd := nn.NewSGDOptimizerWithMomentum(0.9, 0, false) // Standard SGD for backprop modes
	
	// Tween State
	ts := nn.NewTweenState(net, nil)
	ts.Config.UseChainRule = true // Default for chain modes
	ts.Config.Momentum = 0.5 

	for i := 0; i < TrainingEpochs; i++ {
		// Input
		input := make([]float32, InputSize)
		for j := range input { input[j] = rand.Float32() * 0.1 }
		
		if i%2 == 0 {
			input[0] = 0.9 // High -> Expert 1 -> Target 1.0
		} else {
			input[0] = 0.1 // Low -> Expert 2 -> Target 1.0
		}
		// Note: Both cases target 1.0 because the CORRECT expert outputs 1.0
		
		switch mode {
		case "Standard Forward/Backward":
			stepState := net.InitStepState(InputSize)
			stepState.SetInput(input)
			net.StepForward(stepState)
			output := stepState.GetOutput()
			
			gradOut := make([]float32, 1)
			gradOut[0] = output[0] - 1.0 // Minimize distance to 1.0
			
			net.StepBackward(stepState, gradOut)
			sgd.Step(net, 0.05)
			
		case "StepBack":
			stepState := net.InitStepState(InputSize)
			stepState.SetInput(input)
			net.StepForward(stepState)
			output := stepState.GetOutput()
			
			gradOut := make([]float32, 1)
			gradOut[0] = output[0] - 1.0
			net.StepBackward(stepState, gradOut)
			sgd.Step(net, 0.05)

		case "Step Tween":
			ts.Config.UseChainRule = false
			// Target index 0 (value 1.0) isn't right for regression logic in TweenStep
			// We need generic target support.
			// Hack: Since experts output sigmoid, we just push towards 1.
			// Pass targetClass=0 (index 0).
			ts.TweenStep(net, input, 0, 1, 0.05)

		case "Tween":
			ts.ForwardPass(net, input)
			ts.BackwardPass(net, 0, 1) // Target index 0
			ts.CalculateLinkBudgets()
			ts.TweenWeights(net, 0.05)

		case "Tween Chain":
			ts.Config.UseChainRule = true
			ts.ForwardPass(net, input)
			ts.BackwardPassChainRule(net, 0, 1)
			ts.CalculateLinkBudgets()
			ts.TweenWeightsChainRule(net, 0.05)

		case "Step Tween Chain":
			ts.Config.UseChainRule = true
			ts.TweenStep(net, input, 0, 1, 0.05)
		}
	}
}

func freezeLayer(cfg *nn.LayerConfig) {
	cfg.Frozen = true
	// Recurse for Parallel
	if len(cfg.ParallelBranches) > 0 {
		for i := range cfg.ParallelBranches {
			freezeLayer(&cfg.ParallelBranches[i])
		}
	}
}

// evaluateNetwork returns (expert1Val, expert2Val, networkVal, idealVal)
func evaluateNetwork(net *nn.Network, isHigh bool) (float32, float32, float32, float32) {
	input := make([]float32, InputSize)
	if isHigh {
		input[0] = 0.9
	} else {
		input[0] = 0.1
	}
	inputTensor := nn.NewTensorFromSlice(input, InputSize)
	
	// Network Output
	netOut, _, _, _ := nn.GenericForwardPass(net, inputTensor, nil)
	netVal := float32(netOut.Data[0])
	
	// Experts (Frozen Snapshot)
	layer := net.GetLayer(0,0,0)
	
	e1Net := nn.NewNetwork(InputSize, 1, 1, 1)
	e1Net.SetLayer(0,0,0, layer.ParallelBranches[0])
	e1Out, _, _, _ := nn.GenericForwardPass(e1Net, inputTensor, nil)
	val1 := float32(e1Out.Data[0])
	
	e2Net := nn.NewNetwork(InputSize, 1, 1, 1)
	e2Net.SetLayer(0,0,0, layer.ParallelBranches[1])
	e2Out, _, _, _ := nn.GenericForwardPass(e2Net, inputTensor, nil)
	val2 := float32(e2Out.Data[0])
	
	ideal := val1
	if !isHigh { ideal = val2 }
	
	return val1, val2, netVal, ideal
}

func getPercentOff(net *nn.Network, isHigh bool) float64 {
	_, _, netVal, ideal := evaluateNetwork(net, isHigh)
	diff := float32(math.Abs(float64(netVal - ideal)))
	if ideal < 0.0001 { ideal = 0.0001 } 
	return (float64(diff) / float64(ideal)) * 100.0
}