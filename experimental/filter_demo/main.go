// Filter CombineMode Demo
// Demonstrates the dynamic learned gating mechanism (Mixture of Experts style)
// where a gate layer learns to route information through different expert branches.

package main

import (
	"fmt"
	"math"
	"math/rand"
	"time"

	"github.com/openfluke/loom/nn"
)

func main() {
	rand.Seed(time.Now().UnixNano())

	fmt.Println("╔══════════════════════════════════════════════════════════════════╗")
	fmt.Println("║        🔬 Filter CombineMode Demo (Mixture of Experts)           ║")
	fmt.Println("╚══════════════════════════════════════════════════════════════════╝")

	// ===========================================================================
	// DEMO 1: Simple 2-branch filter with Dense experts
	// ===========================================================================
	fmt.Println("\n📌 Demo 1: Two Dense Expert Branches with Learned Gating")
	demo1TwoBranchFilter()

	// ===========================================================================
	// DEMO 2: Multi-branch filter with different expert types
	// ===========================================================================
	fmt.Println("\n📌 Demo 2: Multi-Branch Filter (4 experts)")
	demo2MultiBranchFilter()

	// ===========================================================================
	// DEMO 3: Training the gate to specialize
	// ===========================================================================
	fmt.Println("\n📌 Demo 3: Training Gate Specialization")
	demo3TrainGateSpecialization()
}

// demo1TwoBranchFilter creates a simple 2-expert filtered parallel layer
func demo1TwoBranchFilter() {
	inputSize := 16
	expertSize := 8

	// Create two Dense expert branches
	expert1 := nn.InitDenseLayer(inputSize, expertSize, nn.ActivationLeakyReLU)
	expert2 := nn.InitDenseLayer(inputSize, expertSize, nn.ActivationLeakyReLU)

	// Create gate layer: decides how much to use each expert
	gateLayer := nn.InitDenseLayer(inputSize, 2, nn.ActivationScaledReLU)

	// Build the filtered parallel layer
	filterLayer := nn.LayerConfig{
		Type:              nn.LayerParallel,
		ParallelBranches:  []nn.LayerConfig{expert1, expert2},
		CombineMode:       "filter",
		FilterGateConfig:  &gateLayer,
		FilterSoftmax:     nn.SoftmaxStandard,
		FilterTemperature: 1.0,
	}

	// Create network: Input -> FilteredExperts -> Output
	net := nn.NewNetwork(inputSize, 1, 1, 3)
	inputL := nn.InitDenseLayer(inputSize, inputSize, nn.ActivationLeakyReLU)
	net.SetLayer(0, 0, 0, inputL)
	net.SetLayer(0, 0, 1, filterLayer)
	outputL := nn.InitDenseLayer(expertSize, inputSize, nn.ActivationSigmoid)
	net.SetLayer(0, 0, 2, outputL)

	// Test forward pass
	input := make([]float32, inputSize)
	for i := range input {
		input[i] = rand.Float32()
	}

	output, _ := net.ForwardCPU(input)
	if len(output) == 0 {
		fmt.Printf("   ❌ Forward failed: empty output\n")
		return
	}

	fmt.Printf("   ✅ Forward pass successful!\n")
	fmt.Printf("   📊 Input size: %d, Output size: %d\n", len(input), len(output))
	fmt.Printf("   📈 Sample output values: [%.3f, %.3f, %.3f, ...]\n",
		output[0], output[1], output[2])
}

// demo2MultiBranchFilter creates a 4-expert filtered parallel layer
func demo2MultiBranchFilter() {
	inputSize := 32
	expertSize := 16
	numExperts := 4

	// Create multiple Dense expert branches
	experts := make([]nn.LayerConfig, numExperts)
	for i := 0; i < numExperts; i++ {
		experts[i] = nn.InitDenseLayer(inputSize, expertSize, nn.ActivationLeakyReLU)
		// Add some variety in weights
		for j := range experts[i].Kernel {
			experts[i].Kernel[j] *= float32(1.0 + 0.2*float64(i))
		}
	}

	// Create gate layer for 4 experts
	gateLayer := nn.InitDenseLayer(inputSize, numExperts, nn.ActivationScaledReLU)

	// Build with sparse softmax for sharper routing
	filterLayer := nn.LayerConfig{
		Type:              nn.LayerParallel,
		ParallelBranches:  experts,
		CombineMode:       "filter",
		FilterGateConfig:  &gateLayer,
		FilterSoftmax:     nn.SoftmaxEntmax, // Sparse routing
		FilterTemperature: 0.5,              // Sharper selections
	}

	// Create network
	net := nn.NewNetwork(inputSize, 1, 1, 3)
	inputL := nn.InitDenseLayer(inputSize, inputSize, nn.ActivationLeakyReLU)
	net.SetLayer(0, 0, 0, inputL)
	net.SetLayer(0, 0, 1, filterLayer)
	outputL := nn.InitDenseLayer(expertSize, inputSize, nn.ActivationSigmoid)
	net.SetLayer(0, 0, 2, outputL)

	// Test with multiple inputs
	fmt.Printf("   🧪 Testing with 5 different inputs:\n")
	for trial := 0; trial < 5; trial++ {
		input := make([]float32, inputSize)
		for i := range input {
			input[i] = rand.Float32()
		}

		output, _ := net.ForwardCPU(input)
		if len(output) == 0 {
			fmt.Printf("      Trial %d: ❌ empty output\n", trial+1)
			continue
		}

		// Calculate output statistics
		sum := float32(0)
		for _, v := range output {
			sum += v
		}
		avg := sum / float32(len(output))

		fmt.Printf("      Trial %d: ✅ avg=%.4f\n", trial+1, avg)
	}
}

// demo3TrainGateSpecialization - Pre-train experts then train gate to route
func demo3TrainGateSpecialization() {
	inputSize := 8
	expertSize := 8

	// -----------------------------------------------------------
	// STEP 1: Create two simple networks and pre-train them
	// Expert 1: trained to output HIGH when input[0] > 0.5
	// Expert 2: trained to output HIGH when input[0] <= 0.5
	// -----------------------------------------------------------
	fmt.Printf("   🎓 Pre-training Expert 1 (responds to HIGH first element)...\n")
	expert1 := nn.InitDenseLayer(inputSize, expertSize, nn.ActivationSigmoid)
	
	// Create temp net for training Expert 1
	e1Net := nn.NewNetwork(inputSize, 1, 1, 1)
	e1Net.SetLayer(0, 0, 0, expert1)

	// Train Expert 1: High Input -> High Output (1.0), Low Input -> Low Output (0.0)
	trainData1 := make([]nn.TrainingBatch, 2000)
	for i := range trainData1 {
		input := make([]float32, inputSize)
		for j := range input { input[j] = rand.Float32() }
		
		target := float32(0.0)
		if rand.Float32() > 0.5 {
			input[0] = 0.7 + rand.Float32()*0.3 // High
			target = 1.0
		} else {
			input[0] = rand.Float32()*0.3 // Low
			target = 0.0
		}
		trainData1[i] = nn.TrainingBatch{Input: input, Target: []float32{target}}
	}
	
	config1 := &nn.TrainingConfig{
		Epochs: 10,
		LearningRate: 0.1,
		Verbose: false,
		LossType: "mse",
	}
	e1Net.Train(trainData1, config1)
	expert1 = *e1Net.GetLayer(0, 0, 0) // Update config

	fmt.Printf("   🎓 Pre-training Expert 2 (responds to LOW first element)...\n")
	expert2 := nn.InitDenseLayer(inputSize, expertSize, nn.ActivationSigmoid)
	
	// Create temp net for training Expert 2
	e2Net := nn.NewNetwork(inputSize, 1, 1, 1)
	e2Net.SetLayer(0, 0, 0, expert2)

	// Train Expert 2: Low Input -> High Output (1.0), High Input -> Low Output (0.0)
	trainData2 := make([]nn.TrainingBatch, 2000)
	for i := range trainData2 {
		input := make([]float32, inputSize)
		for j := range input { input[j] = rand.Float32() }
		
		target := float32(0.0)
		if rand.Float32() > 0.5 {
			input[0] = rand.Float32()*0.3 // Low
			target = 1.0
		} else {
			input[0] = 0.7 + rand.Float32()*0.3 // High
			target = 0.0
		}
		trainData2[i] = nn.TrainingBatch{Input: input, Target: []float32{target}}
	}
	
	config2 := &nn.TrainingConfig{
		Epochs: 10,
		LearningRate: 0.1,
		Verbose: false,
		LossType: "mse",
	}
	e2Net.Train(trainData2, config2)
	expert2 = *e2Net.GetLayer(0, 0, 0) // Update config

	// -----------------------------------------------------------
	// STEP 2: Create filter layer combining both experts
	// -----------------------------------------------------------
	gateLayer := nn.InitDenseLayer(inputSize, 2, nn.ActivationScaledReLU)

	filterLayer := nn.LayerConfig{
		Type:              nn.LayerParallel,
		ParallelBranches:  []nn.LayerConfig{expert1, expert2},
		CombineMode:       "filter",
		FilterGateConfig:  &gateLayer,
		FilterSoftmax:     nn.SoftmaxStandard,
		FilterTemperature: 0.5,
	}

	// Simple network: just the filter layer
	net := nn.NewNetwork(inputSize, 1, 1, 2)
	net.SetLayer(0, 0, 0, filterLayer)
	outputL := nn.InitDenseLayer(expertSize, 1, nn.ActivationSigmoid)
	net.SetLayer(0, 0, 1, outputL)

	// -----------------------------------------------------------
	// STEP 3: Test without training gate - should give mixed results
	// -----------------------------------------------------------
	fmt.Printf("   📊 Testing BEFORE gate training:\n")
	
	// Test with high first element
	highInput := make([]float32, inputSize)
	for j := range highInput {
		highInput[j] = rand.Float32() * 0.5
	}
	highInput[0] = 0.9

	// Test with low first element  
	lowInput := make([]float32, inputSize)
	for j := range lowInput {
		lowInput[j] = rand.Float32() * 0.5
	}
	lowInput[0] = 0.1

	highOut, _ := net.ForwardCPU(highInput)
	lowOut, _ := net.ForwardCPU(lowInput)
	
	fmt.Printf("      High input[0]=0.9 → output=%.4f\n", highOut[0])
	fmt.Printf("      Low input[0]=0.1  → output=%.4f\n", lowOut[0])

	// -----------------------------------------------------------
	// STEP 4: Train only the gate layer
	// -----------------------------------------------------------
	fmt.Printf("   🏋️ Training GATE layer for 2000 steps...\n")
	
	ts := nn.NewTweenState(net, nil)
	ts.Config.LinkBudgetScale = 0.3
	ts.Config.UseChainRule = true

	for epoch := 0; epoch < 2000; epoch++ {
		input := make([]float32, inputSize)
		for j := range input {
			input[j] = rand.Float32() * 0.5
		}
		
		// Half the time: high first element
		// Half the time: low first element
		if epoch%2 == 0 {
			input[0] = 0.7 + rand.Float32()*0.3
		} else {
			input[0] = rand.Float32() * 0.3
		}
		
		// Use targetIdx=0 for single output, outputSize=1
		ts.TweenStep(net, input, 0, 1, 0.01)
	}

	// -----------------------------------------------------------
	// STEP 5: Test AFTER gate training
	// -----------------------------------------------------------
	fmt.Printf("   📊 Testing AFTER gate training:\n")

	highOut2, _ := net.ForwardCPU(highInput)
	lowOut2, _ := net.ForwardCPU(lowInput)
	
	fmt.Printf("      High input[0]=0.9 → output=%.4f (was %.4f)\n", highOut2[0], highOut[0])
	fmt.Printf("      Low input[0]=0.1  → output=%.4f (was %.4f)\n", lowOut2[0], lowOut[0])

	// Check if outputs changed (indicating gate learned something)
	highDiff := math.Abs(float64(highOut2[0] - highOut[0]))
	lowDiff := math.Abs(float64(lowOut2[0] - lowOut[0]))
	
	if highDiff > 0.01 || lowDiff > 0.01 {
		fmt.Printf("   ✅ Gate learned to differentiate! (changes: high=%.4f, low=%.4f)\n", highDiff, lowDiff)
	} else {
		fmt.Printf("   ⚠️ Gate didn't learn much (changes: high=%.4f, low=%.4f)\n", highDiff, lowDiff)
	}
}

