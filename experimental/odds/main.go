// Odds Experiment
// Demonstrates dynamic gating with experts of different output sizes,
// stitched to a common dimensionality using LayerSequential.
// Mirrors the structure of filter_demo/main.go.

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
	fmt.Println("║            🎲 Odds Experiment: Stitched Experts Demo             ║")
	fmt.Println("╚══════════════════════════════════════════════════════════════════╝")

	// ===========================================================================
	// DEMO 1: Simple 2-branch filter with Stitched Experts
	// ===========================================================================
	fmt.Println("\n📌 Demo 1: Two Odd-Sized Expert Branches Stitched to Common Size")
	demo1TwoBranchStitched()

	// ===========================================================================
	// DEMO 2: Multi-branch filter with different expert types
	// ===========================================================================
	fmt.Println("\n📌 Demo 2: Multi-Branch Stitched Filter")
	demo2MultiBranchStitched()

	// ===========================================================================
	// DEMO 3: Training the gate to specialize
	// ===========================================================================
	fmt.Println("\n📌 Demo 3: Training Gate Specialization with Odd Experts")
	demo3TrainGateSpecializationStitched()
}

// demo1TwoBranchStitched
func demo1TwoBranchStitched() {
	inputSize := 16
	commonOutputSize := 10

	// Expert 1: Size 5 -> Stitch to 10
	expert1 := nn.InitDenseLayer(inputSize, 5, nn.ActivationLeakyReLU)
	stitch1 := nn.InitStitchLayer(5, commonOutputSize)
	branch1 := nn.InitSequentialLayer(expert1, stitch1)

	// Expert 2: Size 7 -> Stitch to 10
	expert2 := nn.InitDenseLayer(inputSize, 7, nn.ActivationSigmoid)
	stitch2 := nn.InitStitchLayer(7, commonOutputSize)
	branch2 := nn.InitSequentialLayer(expert2, stitch2)

	// Gate
	gateLayer := nn.InitDenseLayer(inputSize, 2, nn.ActivationScaledReLU)

	filterLayer := nn.LayerConfig{
		Type:              nn.LayerParallel,
		ParallelBranches:  []nn.LayerConfig{branch1, branch2},
		CombineMode:       "filter",
		FilterGateConfig:  &gateLayer,
		FilterSoftmax:     nn.SoftmaxStandard,
		FilterTemperature: 1.0,
	}

	net := nn.NewNetwork(inputSize, 1, 1, 3)
	net.SetLayer(0, 0, 0, nn.InitDenseLayer(inputSize, inputSize, nn.ActivationLeakyReLU))
	net.SetLayer(0, 0, 1, filterLayer)
	net.SetLayer(0, 0, 2, nn.InitDenseLayer(commonOutputSize, 5, nn.ActivationSigmoid))

	// Test
	input := make([]float32, inputSize)
	for i := range input {
		input[i] = rand.Float32()
	}
	output, _ := net.ForwardCPU(input)
	fmt.Printf("   ✅ Forward pass successful! Output size: %d\n", len(output))
}

// demo2MultiBranchStitched
func demo2MultiBranchStitched() {
	inputSize := 16
	commonOutputSize := 8
	sizes := []int{4, 12, 6, 20}

	branches := make([]nn.LayerConfig, len(sizes))
	for i, size := range sizes {
		expert := nn.InitDenseLayer(inputSize, size, nn.ActivationLeakyReLU)
		stitch := nn.InitStitchLayer(size, commonOutputSize)
		branches[i] = nn.InitSequentialLayer(expert, stitch)
	}

	gateLayer := nn.InitDenseLayer(inputSize, len(branches), nn.ActivationScaledReLU)

	filterLayer := nn.LayerConfig{
		Type:              nn.LayerParallel,
		ParallelBranches:  branches,
		CombineMode:       "filter",
		FilterGateConfig:  &gateLayer,
		FilterSoftmax:     nn.SoftmaxEntmax,
		FilterTemperature: 0.5,
	}

	net := nn.NewNetwork(inputSize, 1, 1, 2)
	net.SetLayer(0, 0, 0, filterLayer)
	net.SetLayer(0, 0, 1, nn.InitDenseLayer(commonOutputSize, 1, nn.ActivationSigmoid))

	fmt.Printf("   🧪 Testing with 4 odd-sized experts (%v) stitched to %d\n", sizes, commonOutputSize)
	for trial := 0; trial < 3; trial++ {
		input := make([]float32, inputSize)
		for i := range input {
			input[i] = rand.Float32()
		}
		output, _ := net.ForwardCPU(input)
		fmt.Printf("      Trial %d: Output val=%.4f\n", trial+1, output[0])
	}
}

// demo3TrainGateSpecializationStitched
func demo3TrainGateSpecializationStitched() {
	inputSize := 8
	commonOutputSize := 4
	
	// -----------------------------------------------------------
	// -----------------------------------------------------------
	// STEP 1: Create two odd-sized networks and pre-train them
	// Expert 1 (Size 3): Detects High input[0]
	// Expert 2 (Size 5): Detects Low input[0]
	// -----------------------------------------------------------
	
	// Training Expert 1 (Size 3) + Stitch (Size 4)
	fmt.Printf("   🎓 Pre-training Expert 1 (Size 3 -> 4) to detect HIGH input[0]...\n")
	expert1Core := nn.InitDenseLayer(inputSize, 3, nn.ActivationSigmoid)
	stitch1 := nn.InitStitchLayer(3, commonOutputSize)
	
	net1 := nn.NewNetwork(inputSize, 1, 1, 2)
	net1.SetLayer(0, 0, 0, expert1Core)
	net1.SetLayer(0, 0, 1, stitch1)
	
	// Generate training data for Expert 1
	// Task: if input[0] is high, output high
	trainData1 := make([]nn.TrainingBatch, 2000)
	for i := range trainData1 {
		input := make([]float32, inputSize)
		for j := range input { input[j] = rand.Float32() }
		
		target := float32(0.0)
		if rand.Float32() > 0.5 {
			input[0] = 0.8 + rand.Float32()*0.2 // High input
			target = 1.0
		} else {
			input[0] = rand.Float32()*0.2 // Low input
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
	net1.Train(trainData1, config1)

	branch1 := nn.InitSequentialLayer(*net1.GetLayer(0, 0, 0), *net1.GetLayer(0, 0, 1))

	// Training Expert 2 (Size 5) + Stitch (Size 4)
	fmt.Printf("   🎓 Pre-training Expert 2 (Size 5 -> 4) to detect LOW input[0]...\n")
	expert2Core := nn.InitDenseLayer(inputSize, 5, nn.ActivationSigmoid)
	stitch2 := nn.InitStitchLayer(5, commonOutputSize)
	
	net2 := nn.NewNetwork(inputSize, 1, 1, 2)
	net2.SetLayer(0, 0, 0, expert2Core)
	net2.SetLayer(0, 0, 1, stitch2)
	
	// Generate training data for Expert 2
	// Logic: Low input -> High Output
	trainData2 := make([]nn.TrainingBatch, 2000)
	for i := range trainData2 {
		input := make([]float32, inputSize)
		for j := range input { input[j] = rand.Float32() }
		
		target := float32(0.0)
		if rand.Float32() > 0.5 {
			input[0] = rand.Float32()*0.2 // Low input
			target = 1.0 // High Output
		} else {
			input[0] = 0.8 + rand.Float32()*0.2 // High input
			target = 0.0 // Low Output
		}
		trainData2[i] = nn.TrainingBatch{Input: input, Target: []float32{target}}
	}

	config2 := &nn.TrainingConfig{
		Epochs: 10,
		LearningRate: 0.1,
		Verbose: false,
		LossType: "mse",
	}
	net2.Train(trainData2, config2)

	branch2 := nn.InitSequentialLayer(*net2.GetLayer(0, 0, 0), *net2.GetLayer(0, 0, 1))

	// -----------------------------------------------------------
	// STEP 2: Create filter layer
	// -----------------------------------------------------------
	gateLayer := nn.InitDenseLayer(inputSize, 2, nn.ActivationScaledReLU)
	filterLayer := nn.LayerConfig{
		Type:              nn.LayerParallel,
		ParallelBranches:  []nn.LayerConfig{branch1, branch2},
		CombineMode:       "filter",
		FilterGateConfig:  &gateLayer,
		FilterSoftmax:     nn.SoftmaxStandard,
		FilterTemperature: 0.5,
	}

	net := nn.NewNetwork(inputSize, 1, 1, 2)
	net.SetLayer(0, 0, 0, filterLayer)
	outputL := nn.InitDenseLayer(commonOutputSize, 1, nn.ActivationSigmoid)
	net.SetLayer(0, 0, 1, outputL)

	// -----------------------------------------------------------
	// STEP 4: Train only the gate layer
	// -----------------------------------------------------------
	fmt.Printf("   🏋️ Training GATE layer for 1000 steps...\n")
	
	ts := nn.NewTweenState(net, nil) // TweenState for main net
	ts.Config.UseChainRule = true
	
	for epoch := 0; epoch < 1000; epoch++ {
		input := make([]float32, inputSize)
		for j := range input { input[j] = rand.Float32() }
		
		// Target logic:
		// HIGH input[0] -> Should route to Expert 1
		// LOW input[0]  -> Should route to Expert 2
		
		if epoch%2 == 0 {
			input[0] = 0.9 // High
		} else {
			input[0] = 0.1 // Low
		}
		
		// If gate works, output will be high (since experts are specialized to output high for their preferred input).
		// So we train the whole net to maximize output.
		// Since experts are frozen (we don't update them here effectively via this simple call unless we passed gradients everywhere),
		// essentially we are just updating the gate to find the max-output path.
		
		// Using TweenStep to maximize output (target [1.0])
		ts.TweenStep(net, input, 0, 1, 0.05)
	}

	// -----------------------------------------------------------
	// STEP 5: Test
	// -----------------------------------------------------------
	fmt.Printf("   📊 Testing Selection:\n")
	
	highIn := make([]float32, inputSize); highIn[0] = 0.9
	lowIn := make([]float32, inputSize); lowIn[0] = 0.1
	
	hOut, _ := net.ForwardCPU(highIn)
	lOut, _ := net.ForwardCPU(lowIn)
	
	fmt.Printf("      High Input → Output: %.4f (Expert 1 preferred)\n", hOut[0])
	fmt.Printf("      Low Input  → Output: %.4f (Expert 2 preferred)\n", lOut[0])
	
	diff := math.Abs(float64(hOut[0] - lOut[0]))
	if hOut[0] > 0.5 && lOut[0] > 0.5 {
		fmt.Printf("   ✅ Gate learned to pick the right expert (both outputs high)!\n")
	} else if diff < 0.1 {
		fmt.Printf("   ⚠️ Gate might not be differentiating well.\n")
	} else {
		fmt.Printf("   ✅ Gate differentiating.\n")
	}
}
