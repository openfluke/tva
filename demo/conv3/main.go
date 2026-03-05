package main

import (
	"fmt"
	"math"
	"math/rand"
	"runtime"
	"time"

	"github.com/openfluke/loom/nn"
)

// generateMovingCube produces an 8x8x8 volume with a 3x3x3 cube at X position
func generateMovingCube(x int) []float32 {
	vol := make([]float32, 8*8*8)
	for d := 0; d < 8; d++ {
		for h := 0; h < 8; h++ {
			for w := 0; w < 8; w++ {
				idx := d*64 + h*8 + w
				vol[idx] = rand.Float32() * 0.1 // Base noise

				// 3x3x3 cube
				if d >= 2 && d <= 4 && h >= 2 && h <= 4 && w >= x && w <= x+2 {
					vol[idx] = 1.0 - (rand.Float32() * 0.2)
				}
			}
		}
	}
	return vol
}

// generateVolume produces a 3D volume [depth][height][width]
func generateVolume(isCube bool) []float32 {
	vol := make([]float32, 8*8*8)

	for d := 0; d < 8; d++ {
		for h := 0; h < 8; h++ {
			for w := 0; w < 8; w++ {
				idx := d*64 + h*8 + w
				vol[idx] = rand.Float32() * 0.1 // Base noise

				if isCube {
					if d >= 2 && d <= 5 && h >= 2 && h <= 5 && w >= 2 && w <= 5 {
						vol[idx] = 1.0 - (rand.Float32() * 0.2)
					}
				} else {
					dist := math.Sqrt(math.Pow(float64(d)-3.5, 2) + math.Pow(float64(h)-3.5, 2) + math.Pow(float64(w)-3.5, 2))
					if dist <= 2.9 {
						vol[idx] = 1.0 - (rand.Float32() * 0.2)
					}
				}
			}
		}
	}
	return vol
}

func createNetwork() *nn.Network {
	inputSize := 512 // 8x8x8
	numClasses := 2
	net := nn.NewNetwork(inputSize, 1, 2, 1)

	conv3DConfig := nn.InitConv3DLayer(
		8, 8, 8, 1, // input dims
		3, 1, 1, 16, // kernel 3, stride 1, padding 1, filters 16
		nn.ActivationScaledReLU,
	)
	net.SetLayer(0, 0, 0, conv3DConfig)

	convOutputSize := 8 * 8 * 8 * 16

	denseConfig := nn.LayerConfig{
		Type:         nn.LayerDense,
		Activation:   nn.ActivationSigmoid,
		InputHeight:  convOutputSize,
		OutputHeight: numClasses,
		Kernel:       make([]float32, convOutputSize*numClasses),
		Bias:         make([]float32, numClasses),
	}
	scale := float32(1.0) / float32(convOutputSize)
	for i := range denseConfig.Kernel {
		denseConfig.Kernel[i] = (rand.Float32()*2 - 1) * scale
	}
	net.SetLayer(0, 1, 0, denseConfig)

	return net
}

type Sample struct {
	Data  []float32
	Label int
}

func printMemUsage(msg string) {
	var m runtime.MemStats
	runtime.ReadMemStats(&m)
	fmt.Printf("[%s] Alloc = %v MiB\tTotalAlloc = %v MiB\tSys = %v MiB\tNumGC = %v\n",
		msg,
		m.Alloc/1024/1024,
		m.TotalAlloc/1024/1024,
		m.Sys/1024/1024,
		m.NumGC)
}

func main() {
	rand.Seed(time.Now().UnixNano())

	fmt.Println("=======================================================")
	fmt.Println("   Loom Conv3D Optimization Benchmark                  ")
	fmt.Println("   Detecting Cubes vs Spheres in an 8x8x8 Volume       ")
	fmt.Println("=======================================================")

	fmt.Println("[*] Generating 50 Training Samples...")
	trainData := make([]Sample, 50)
	for i := 0; i < 50; i++ {
		isCube := i%2 == 0
		label := 0
		if isCube {
			label = 1
		}
		trainData[i] = Sample{Data: generateVolume(isCube), Label: label}
	}

	epochs := 2
	inputSize := 512
	numClasses := 2

	// =========================================================
	// Benchmark 1: Standard GPU Forward/Backward
	// =========================================================
	fmt.Println("\n--- 1. Standard Forward & Backward (GPU) ---")
	runtime.GC()
	printMemUsage("Before")

	netGPU := createNetwork()
	netGPU.SetGPU(true)
	err := netGPU.WeightsToGPU()
	if err != nil {
		fmt.Printf("Failed to mount GPU: %v\n", err)
	}

	var fwdTimeGPU, bwdTimeGPU time.Duration

	startStepGPU := time.Now()
	for epoch := 1; epoch <= epochs; epoch++ {
		correct := 0
		for _, sample := range trainData {
			// Measure explicit GPU Forward
			out, fdur := netGPU.Forward(sample.Data)
			fwdTimeGPU += fdur

			pred := 0
			if out[1] > out[0] {
				pred = 1
			}
			if pred == sample.Label {
				correct++
			}

			// Calc grad
			gradOutput := make([]float32, numClasses)
			for i := range gradOutput {
				if i == sample.Label {
					gradOutput[i] = out[i] - 1.0
				} else {
					gradOutput[i] = out[i]
				}
			}

			// Measure explicit GPU Backward
			_, bdur := netGPU.Backward(gradOutput)
			bwdTimeGPU += bdur
		}
		if epoch == epochs {
			fmt.Printf(" Final Epoch Accuracy: %d/%d (%.0f%%)\n", correct, len(trainData), float32(correct)/float32(len(trainData))*100)
		}
	}
	timeGPU := time.Since(startStepGPU)
	fmt.Printf(" Training Time: %v (Fwd: %v, Bwd: %v)\n", timeGPU, fwdTimeGPU, bwdTimeGPU)
	printMemUsage("After")

	// =========================================================
	// Benchmark 2: Standard CPU Forward/Backward
	// =========================================================
	fmt.Println("\n--- 2. Standard Forward & Backward (CPU) ---")
	runtime.GC()
	printMemUsage("Before")

	netCPU := createNetwork()

	var fwdTimeCPU, bwdTimeCPU time.Duration

	startStepCPU := time.Now()
	for epoch := 1; epoch <= epochs; epoch++ {
		correct := 0
		for _, sample := range trainData {
			// Measure explicit Forward
			out, dur := netCPU.Forward(sample.Data)
			fwdTimeCPU += dur

			pred := 0
			if out[1] > out[0] {
				pred = 1
			}
			if pred == sample.Label {
				correct++
			}

			// Calc grad
			gradOutput := make([]float32, numClasses)
			for i := range gradOutput {
				if i == sample.Label {
					gradOutput[i] = out[i] - 1.0
				} else {
					gradOutput[i] = out[i]
				}
			}

			// Measure explicit Backward
			_, bdur := netCPU.Backward(gradOutput)
			bwdTimeCPU += bdur
		}
		if epoch == epochs {
			fmt.Printf(" Final Epoch Accuracy: %d/%d (%.0f%%)\n", correct, len(trainData), float32(correct)/float32(len(trainData))*100)
		}
	}
	timeCPU := time.Since(startStepCPU)
	fmt.Printf(" Training Time: %v (Fwd: %v, Bwd: %v)\n", timeCPU, fwdTimeCPU, bwdTimeCPU)
	printMemUsage("After")

	// =========================================================
	// Benchmark 3: Generic Tween
	// =========================================================
	fmt.Println("\n--- 3. Generic Tween (Type Agnostic) ---")
	runtime.GC()
	printMemUsage("Before")

	netGenTween := createNetwork()
	backend := nn.NewCPUBackend[float32]()
	genTween := nn.NewGenericTweenState[float32](netGenTween, nil)
	genTween.Config.Conv3DRate = 0.5
	genTween.Config.DenseRate = 0.5

	startGenTween := time.Now()
	for epoch := 1; epoch <= epochs; epoch++ {
		correct := 0
		for _, sample := range trainData {
			inputT := nn.NewTensorFromSlice(sample.Data, inputSize)
			genTween.TweenStep(netGenTween, inputT, sample.Label, numClasses, 0.1, backend)

			probs := genTween.ForwardActs[genTween.TotalLayers].Data
			pred := 0
			if probs[1] > probs[0] {
				pred = 1
			}
			if pred == sample.Label {
				correct++
			}
		}
		if epoch == epochs {
			fmt.Printf(" Final Epoch Accuracy: %d/%d (%.0f%%)\n", correct, len(trainData), float32(correct)/float32(len(trainData))*100)
		}
	}
	timeGenTween := time.Since(startGenTween)
	fmt.Printf(" Training Time: %v\n", timeGenTween)
	printMemUsage("After")

	// =========================================================
	// Benchmark 4: Float32 Legacy Tween
	// =========================================================
	fmt.Println("\n--- 4. Float32 Tween (Legacy Mode) ---")
	runtime.GC()
	printMemUsage("Before")

	netLegTween := createNetwork()
	legTween := nn.NewTweenState(netLegTween, nil)
	legTween.Config.UseChainRule = false // Force legacy
	legTween.Config.Conv3DRate = 0.5
	legTween.Config.DenseRate = 0.5

	startLegTween := time.Now()
	for epoch := 1; epoch <= epochs; epoch++ {
		correct := 0
		for _, sample := range trainData {
			legTween.TweenStep(netLegTween, sample.Data, sample.Label, numClasses, 0.1)

			probs := legTween.ForwardActs[legTween.TotalLayers]
			pred := 0
			if probs[1] > probs[0] {
				pred = 1
			}
			if pred == sample.Label {
				correct++
			}
		}
		if epoch == epochs {
			fmt.Printf(" Final Epoch Accuracy: %d/%d (%.0f%%)\n", correct, len(trainData), float32(correct)/float32(len(trainData))*100)
		}
	}
	timeLegTween := time.Since(startLegTween)
	fmt.Printf(" Training Time: %v\n", timeLegTween)
	printMemUsage("After")

	// =========================================================
	// Benchmark 5: Float32 Chain Rule Tween
	// =========================================================
	fmt.Println("\n--- 5. Float32 Tween (Chain Rule Mode) ---")
	runtime.GC()
	printMemUsage("Before")

	netChainTween := createNetwork()
	chainTween := nn.NewTweenState(netChainTween, nil)
	chainTween.Config.UseChainRule = true // Force Chain Rule
	chainTween.Config.Conv3DRate = 0.5
	chainTween.Config.DenseRate = 0.5

	startChainTween := time.Now()
	for epoch := 1; epoch <= epochs; epoch++ {
		correct := 0
		for _, sample := range trainData {
			chainTween.TweenStep(netChainTween, sample.Data, sample.Label, numClasses, 0.1)

			probs := chainTween.ForwardActs[chainTween.TotalLayers]
			pred := 0
			if probs[1] > probs[0] {
				pred = 1
			}
			if pred == sample.Label {
				correct++
			}
		}
		if epoch == epochs {
			fmt.Printf(" Final Epoch Accuracy: %d/%d (%.0f%%)\n", correct, len(trainData), float32(correct)/float32(len(trainData))*100)
		}
	}
	timeChainTween := time.Since(startChainTween)
	fmt.Printf(" Training Time: %v\n", timeChainTween)
	printMemUsage("After")

	fmt.Println("\n=======================================================")
	fmt.Println("   Benchmark Summary                                   ")
	fmt.Println("=======================================================")
	fmt.Printf(" 1. Standard Step (GPU)  : %v \n    * Forward: %v | Backward: %v\n", timeGPU, fwdTimeGPU, bwdTimeGPU)
	fmt.Printf(" 2. Standard Step (CPU)  : %v \n    * Forward: %v | Backward: %v\n", timeCPU, fwdTimeCPU, bwdTimeCPU)
	fmt.Printf(" 3. Tween (Generic)      : %v\n", timeGenTween)
	fmt.Printf(" 4. Tween (Legacy)       : %v\n", timeLegTween)
	fmt.Printf(" 5. Tween (Chain Rule)   : %v\n", timeChainTween)

	// Show practical examples of the different modes
	showcaseModes()
}

// showcaseModes demonstrates when and why you'd use each execution strategy
func showcaseModes() {
	fmt.Println("\n\n=======================================================")
	fmt.Println("   Execution Mode Showcase: Tracking a Moving Cube     ")
	fmt.Println("=======================================================")
	fmt.Println("Goal: Train the network to detect if a 3x3x3 cube is on")
	fmt.Println("the LEFT side (Class 0) or RIGHT side (Class 1) of the ")
	fmt.Println("8x8x8 spatial volume. No optimizers will be used!      ")

	inputSize := 512
	numClasses := 2

	// Create some test data
	leftCube := generateMovingCube(1)  // Cube on the left
	rightCube := generateMovingCube(4) // Cube on the right

	fmt.Println("\n[1] STANDARD STEP (net.Forward / net.Backward)")
	fmt.Println("Use Case: Full training runs and maximum performance batch inference.")
	fmt.Println("Pros: Uses WebGPU backend. Handles full backpropagation. Fastest raw throughput.")
	netStd := createNetwork()
	// NO OPTIMIZER - we use net.ApplyGradients directly!

	outLeftBefore, _ := netStd.Forward(leftCube)
	outRightBefore, _ := netStd.Forward(rightCube)
	fmt.Printf("  -> Before Training (Left Cube) : [%.4f, %.4f]\n", outLeftBefore[0], outLeftBefore[1])
	fmt.Printf("  -> Before Training (Right Cube): [%.4f, %.4f]\n", outRightBefore[0], outRightBefore[1])

	// Quick standard train
	for epoch := 0; epoch < 10; epoch++ {
		// Train Left
		outL, _ := netStd.Forward(leftCube)
		gradL := []float32{outL[0] - 1.0, outL[1]} // Target Left=0
		netStd.Backward(gradL)
		netStd.ApplyGradients(0.05)

		// Train Right
		outR, _ := netStd.Forward(rightCube)
		gradR := []float32{outR[0], outR[1] - 1.0} // Target Right=1
		netStd.Backward(gradR)
		netStd.ApplyGradients(0.05)
	}

	outLeftAfter, _ := netStd.Forward(leftCube)
	outRightAfter, _ := netStd.Forward(rightCube)
	fmt.Printf("  -> After Training (Left Cube)  : [%.4f, %.4f] (Target: 0 dominates)\n", outLeftAfter[0], outLeftAfter[1])
	fmt.Printf("  -> After Training (Right Cube) : [%.4f, %.4f] (Target: 1 dominates)\n", outRightAfter[0], outRightAfter[1])

	fmt.Println("\n[2] TWEEN STATE CHAIN RULE (TweenStep with UseChainRule=true)")
	fmt.Println("Use Case: Fast adaptation tracking (Online Learning on sparse labels).")
	fmt.Println("Pros: Calculates real gradients on-the-fly dynamically. No formal 'Backward' step needed. Easy integration.")
	netChain := createNetwork()
	chainTween := nn.NewTweenState(netChain, nil)
	chainTween.Config.UseChainRule = true
	chainTween.Config.Conv3DRate = 0.5 // Update Conv3D weights dynamically
	chainTween.Config.DenseRate = 0.5

	chainTween.TweenStep(netChain, leftCube, 0, numClasses, 0.0) // Preview (lr=0.0 ignores training)
	outChainLeftBefore := chainTween.ForwardActs[chainTween.TotalLayers]
	fmt.Printf("  -> Before Training (Left Cube) : [%.4f, %.4f]\n", outChainLeftBefore[0], outChainLeftBefore[1])

	// Tween update step
	for epoch := 0; epoch < 10; epoch++ {
		chainTween.TweenStep(netChain, leftCube, 0, numClasses, 0.05)  // Train Left = 0
		chainTween.TweenStep(netChain, rightCube, 1, numClasses, 0.05) // Train Right = 1
	}

	chainTween.TweenStep(netChain, leftCube, 0, numClasses, 0.0)
	outChainLeftAfter := chainTween.ForwardActs[chainTween.TotalLayers]
	chainTween.TweenStep(netChain, rightCube, 0, numClasses, 0.0)
	outChainRightAfter := chainTween.ForwardActs[chainTween.TotalLayers]

	fmt.Printf("  -> After Training (Left Cube)  : [%.4f, %.4f] (Target: 0 dominates)\n", outChainLeftAfter[0], outChainLeftAfter[1])
	fmt.Printf("  -> After Training (Right Cube) : [%.4f, %.4f] (Target: 1 dominates)\n", outChainRightAfter[0], outChainRightAfter[1])

	fmt.Println("\n[3] TWEEN STATE LEGACY (TweenStep with UseChainRule=false)")
	fmt.Println("Use Case: Highly chaotic environments (RL, synthetic evolution, unstable data).")
	fmt.Println("Pros: Math-free geometric pseudo-gradients. Never explodes safely perturbing weights. Great for generative tasks.")
	netLegacy := createNetwork()
	legacyTween := nn.NewTweenState(netLegacy, nil)
	legacyTween.Config.UseChainRule = false // Pure geometric morphing
	legacyTween.Config.Conv3DRate = 0.5
	legacyTween.Config.DenseRate = 0.5

	legacyTween.TweenStep(netLegacy, leftCube, 0, numClasses, 0.0)
	outLegacyLeftBefore := legacyTween.ForwardActs[legacyTween.TotalLayers]
	fmt.Printf("  -> Before Training (Left Cube) : [%.4f, %.4f]\n", outLegacyLeftBefore[0], outLegacyLeftBefore[1])

	for epoch := 0; epoch < 20; epoch++ { // Geometry takes slightly longer to converge
		legacyTween.TweenStep(netLegacy, leftCube, 0, numClasses, 0.5) // High learning rate safely bounds
		legacyTween.TweenStep(netLegacy, rightCube, 1, numClasses, 0.5)
	}

	legacyTween.TweenStep(netLegacy, leftCube, 0, numClasses, 0.0)
	outLegacyLeftAfter := legacyTween.ForwardActs[legacyTween.TotalLayers]
	legacyTween.TweenStep(netLegacy, rightCube, 0, numClasses, 0.0)
	outLegacyRightAfter := legacyTween.ForwardActs[legacyTween.TotalLayers]

	fmt.Printf("  -> After Training (Left Cube)  : [%.4f, %.4f] (Target: 0 dominates)\n", outLegacyLeftAfter[0], outLegacyLeftAfter[1])
	fmt.Printf("  -> After Training (Right Cube) : [%.4f, %.4f] (Target: 1 dominates)\n", outLegacyRightAfter[0], outLegacyRightAfter[1])

	fmt.Println("\n[4] GENERIC TWEEN STATE (GenericTweenStep[T numeric])")
	fmt.Println("Use Case: Precision-agnostic or quantized online learning (e.g., int8, float64).")
	fmt.Println("Pros: Same functionality as Tween but works on any generic numeric tensor implementation.")
	netGen := createNetwork()
	backend := nn.NewCPUBackend[float32]()
	genTween := nn.NewGenericTweenState[float32](netGen, nil)
	genTween.Config.Conv3DRate = 0.5
	genTween.Config.DenseRate = 0.5

	tensorLeft := nn.NewTensorFromSlice(leftCube, inputSize)
	tensorRight := nn.NewTensorFromSlice(rightCube, inputSize)

	genTween.TweenStep(netGen, tensorLeft, 0, numClasses, 0.0, backend)
	outGenLeftBefore := genTween.ForwardActs[genTween.TotalLayers].Data
	fmt.Printf("  -> Before Training (Left Cube) : [%.4f, %.4f]\n", outGenLeftBefore[0], outGenLeftBefore[1])

	for epoch := 0; epoch < 10; epoch++ {
		genTween.TweenStep(netGen, tensorLeft, 0, numClasses, 0.05, backend)
		genTween.TweenStep(netGen, tensorRight, 1, numClasses, 0.05, backend)
	}

	genTween.TweenStep(netGen, tensorLeft, 0, numClasses, 0.0, backend)
	outGenLeftAfter := genTween.ForwardActs[genTween.TotalLayers].Data
	genTween.TweenStep(netGen, tensorRight, 0, numClasses, 0.0, backend)
	outGenRightAfter := genTween.ForwardActs[genTween.TotalLayers].Data

	fmt.Printf("  -> After Training (Left Cube)  : [%.4f, %.4f] (Target: 0 dominates)\n", outGenLeftAfter[0], outGenLeftAfter[1])
	fmt.Printf("  -> After Training (Right Cube) : [%.4f, %.4f] (Target: 1 dominates)\n", outGenRightAfter[0], outGenRightAfter[1])
}
