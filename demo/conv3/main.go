package main

import (
	"fmt"
	"math"
	"math/rand"
	"runtime"
	"time"

	"github.com/openfluke/loom/nn"
)

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
}
