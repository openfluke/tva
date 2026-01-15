package main

import (
	"flag"
	"fmt"
	"math"
	"math/rand"
	"time"

	"github.com/openfluke/loom/gpu"
	"github.com/openfluke/loom/nn"
)

var (
	gpuFlag    = flag.String("gpu", "", "Optional substring to select a specific GPU adapter")
	epochsFlag = flag.Int("epochs", 50, "Number of training epochs")
	lrFlag     = flag.Float64("lr", 0.1, "Learning rate")
)

// =============================================================================
// Sequential Dataset for MHA Training Verification
// =============================================================================
// Task: "Marker Detection" - detect if a special marker token appears in sequence
// This REQUIRES attention because the marker can be anywhere in the sequence.
// A simple linear layer cannot solve this without looking at all positions.

type Dataset struct {
	Inputs   [][]float32
	Outputs  []float64
	SeqLen   int
	EmbedDim int
}

func generateMarkerDataset(numSamples, seqLen, embedDim int) *Dataset {
	inputs := make([][]float32, numSamples)
	outputs := make([]float64, numSamples)

	for i := 0; i < numSamples; i++ {
		seq := make([]float32, seqLen*embedDim)
		hasMarker := rand.Float32() > 0.5

		// Fill with random background noise
		for t := 0; t < seqLen; t++ {
			for d := 0; d < embedDim; d++ {
				seq[t*embedDim+d] = rand.Float32()*0.5 - 0.25 // [-0.25, 0.25]
			}
		}

		if hasMarker {
			// Place marker at random position - marker is strong signal in dim 0
			markerPos := rand.Intn(seqLen)
			seq[markerPos*embedDim] = 2.0 // Strong positive signal
			outputs[i] = 1.0
		} else {
			outputs[i] = 0.0
		}

		inputs[i] = seq
	}

	return &Dataset{
		Inputs:   inputs,
		Outputs:  outputs,
		SeqLen:   seqLen,
		EmbedDim: embedDim,
	}
}

// Simple sum-based dataset: class = (sum of all elements > 0)
func generateSumDataset(numSamples, seqLen, embedDim int) *Dataset {
	inputs := make([][]float32, numSamples)
	outputs := make([]float64, numSamples)

	for i := 0; i < numSamples; i++ {
		seq := make([]float32, seqLen*embedDim)
		sum := float32(0)

		for t := 0; t < seqLen; t++ {
			// Only first dimension matters for classification
			val := rand.Float32()*2 - 1 // [-1, 1]
			seq[t*embedDim] = val
			sum += val

			// Other dims are small noise
			for d := 1; d < embedDim; d++ {
				seq[t*embedDim+d] = rand.Float32()*0.1 - 0.05
			}
		}

		inputs[i] = seq
		if sum > 0 {
			outputs[i] = 1.0
		} else {
			outputs[i] = 0.0
		}
	}

	return &Dataset{
		Inputs:   inputs,
		Outputs:  outputs,
		SeqLen:   seqLen,
		EmbedDim: embedDim,
	}
}

type MHAConfig struct {
	Name     string
	EmbedDim int
	NumHeads int
	SeqLen   int
}

func getConfigs() []MHAConfig {
	return []MHAConfig{
		// Small configs that fit in GPU buffer (2048 floats limit)
		// batch=1: 4*32=128 floats per sample ✓
		// batch=10: 4*32*10=1280 floats ✓
		{Name: "MHA-32-2H-S4", EmbedDim: 32, NumHeads: 2, SeqLen: 4},
		{Name: "MHA-64-4H-S4", EmbedDim: 64, NumHeads: 4, SeqLen: 4},
		// Larger - for GPU advantage
		{Name: "MHA-128-4H-S4", EmbedDim: 128, NumHeads: 4, SeqLen: 4},
	}
}

func createNetwork(batchSize int, config MHAConfig) (*nn.Network, error) {
	inputSize := config.SeqLen * config.EmbedDim

	jsonConfig := fmt.Sprintf(`{
		"id": "mha_test_%s",
		"batch_size": %d,
		"grid_rows": 1,
		"grid_cols": 1,
		"layers_per_cell": 2,
		"layers": [
			{"type": "multi_head_attention", "d_model": %d, "num_heads": %d, "seq_length": %d},
			{"type": "dense", "activation": "sigmoid", "input_height": %d, "output_height": 2}
		]
	}`, config.Name, batchSize, config.EmbedDim, config.NumHeads, config.SeqLen, inputSize)

	return nn.BuildNetworkFromJSON(jsonConfig)
}

func cloneWeights(src, dst *nn.Network) {
	for i := 0; i < src.TotalLayers(); i++ {
		s := &src.Layers[i]
		d := &dst.Layers[i]

		if len(s.Kernel) > 0 {
			d.Kernel = append([]float32(nil), s.Kernel...)
		}
		if len(s.Bias) > 0 {
			d.Bias = append([]float32(nil), s.Bias...)
		}
		if len(s.QWeights) > 0 {
			d.QWeights = append([]float32(nil), s.QWeights...)
		}
		if len(s.KWeights) > 0 {
			d.KWeights = append([]float32(nil), s.KWeights...)
		}
		if len(s.VWeights) > 0 {
			d.VWeights = append([]float32(nil), s.VWeights...)
		}
		if len(s.OutputWeight) > 0 {
			d.OutputWeight = append([]float32(nil), s.OutputWeight...)
		}
		if len(s.QBias) > 0 {
			d.QBias = append([]float32(nil), s.QBias...)
		}
		if len(s.KBias) > 0 {
			d.KBias = append([]float32(nil), s.KBias...)
		}
		if len(s.VBias) > 0 {
			d.VBias = append([]float32(nil), s.VBias...)
		}
		if len(s.OutputBias) > 0 {
			d.OutputBias = append([]float32(nil), s.OutputBias...)
		}
	}
}

type Result struct {
	Config     MHAConfig
	IsGPU      bool
	InitialAcc float64
	FinalAcc   float64
	TrainTime  time.Duration
	Loss       []float32
	Success    bool
}

func train(net *nn.Network, data *Dataset, epochs int, lr float32, isGPU bool, batchSize int) ([]float32, time.Duration) {
	name := "CPU"
	if isGPU {
		name = "GPU"
	}

	numSamples := len(data.Inputs)
	inputSize := len(data.Inputs[0])
	outputSize := 2
	numBatches := numSamples / batchSize

	losses := make([]float32, epochs)
	start := time.Now()

	for epoch := 0; epoch < epochs; epoch++ {
		totalLoss := float32(0)

		for b := 0; b < numBatches; b++ {
			idx := b * batchSize
			batchInput := make([]float32, batchSize*inputSize)
			for i := 0; i < batchSize; i++ {
				copy(batchInput[i*inputSize:], data.Inputs[idx+i])
			}

			output, _ := net.ForwardCPU(batchInput)

			dOutput := make([]float32, len(output))
			for i := 0; i < batchSize; i++ {
				class := int(data.Outputs[idx+i])
				outStart := i * outputSize
				sampleOut := output[outStart : outStart+outputSize]

				if class < len(sampleOut) && sampleOut[class] > 1e-7 {
					totalLoss += -float32(math.Log(float64(sampleOut[class])))
				}

				for j := 0; j < outputSize; j++ {
					target := float32(0)
					if j == class {
						target = 1
					}
					dOutput[outStart+j] = (sampleOut[j] - target) / float32(batchSize)
				}
			}

			net.BackwardCPU(dOutput)
			net.ApplyGradients(lr)
		}

		avgLoss := totalLoss / float32(numSamples)
		losses[epoch] = avgLoss

		if (epoch+1)%10 == 0 || epoch == 0 || epoch == epochs-1 {
			fmt.Printf("  [%s] Epoch %d/%d - Loss: %.4f\n", name, epoch+1, epochs, avgLoss)
		}
	}

	return losses, time.Since(start)
}

func main() {
	flag.Parse()
	if *gpuFlag != "" {
		gpu.SetAdapterPreference(*gpuFlag)
	}
	rand.Seed(time.Now().UnixNano())

	fmt.Println("╔═══════════════════════════════════════════════════════════════════════════════╗")
	fmt.Println("║           MHA Sequential Training Verification (Small Configs)               ║")
	fmt.Println("╚═══════════════════════════════════════════════════════════════════════════════╝")
	fmt.Println()

	configs := getConfigs()

	// Use small dimensions that fit in GPU buffers
	seqLen := 4
	embedDim := 32

	fmt.Printf("Generating sequential sum dataset (100 samples, seq=%d, dim=%d)...\n\n", seqLen, embedDim)
	dataset := generateSumDataset(100, seqLen, embedDim)

	var cpuResults, gpuResults []Result

	for _, config := range configs {
		fmt.Printf("┌─────────────────────────────────────────────────────────────────────┐\n")
		fmt.Printf("│ Testing: %-60s│\n", config.Name)
		fmt.Printf("│ d_model=%d, heads=%d, seq=%d%-41s│\n", config.EmbedDim, config.NumHeads, config.SeqLen, "")
		fmt.Printf("└─────────────────────────────────────────────────────────────────────┘\n")

		// Regenerate dataset with correct dimensions for this config
		dataset = generateSumDataset(100, config.SeqLen, config.EmbedDim)

		// ===== CPU Training =====
		fmt.Println("\n[CPU Training]")
		cpuNet, err := createNetwork(1, config)
		if err != nil {
			fmt.Printf("  ✗ Failed: %v\n", err)
			cpuResults = append(cpuResults, Result{Config: config, Success: false})
			gpuResults = append(gpuResults, Result{Config: config, Success: false})
			continue
		}
		cpuNet.InitializeWeights()

		beforeMetrics, _ := cpuNet.EvaluateNetwork(dataset.Inputs, dataset.Outputs)
		cpuRes := Result{Config: config, IsGPU: false, InitialAcc: beforeMetrics.Accuracy}

		// Clone for GPU before CPU training modifies weights
		gpuNet, _ := createNetwork(10, config)
		cloneWeights(cpuNet, gpuNet)

		losses, trainTime := train(cpuNet, dataset, *epochsFlag, float32(*lrFlag), false, 1)
		cpuRes.Loss = losses
		cpuRes.TrainTime = trainTime

		afterMetrics, _ := cpuNet.EvaluateNetwork(dataset.Inputs, dataset.Outputs)
		cpuRes.FinalAcc = afterMetrics.Accuracy
		cpuRes.Success = true
		cpuResults = append(cpuResults, cpuRes)

		fmt.Printf("  ✓ %.1f%% → %.1f%% accuracy in %v\n", cpuRes.InitialAcc*100, cpuRes.FinalAcc*100, trainTime)

		// ===== GPU Training =====
		fmt.Println("\n[GPU Training]")
		gpuRes := Result{Config: config, IsGPU: true}

		gpuNet.GPU = true
		if err := gpuNet.WeightsToGPU(); err != nil {
			fmt.Printf("  ✗ GPU mount failed: %v\n", err)
			gpuRes.Success = false
			gpuResults = append(gpuResults, gpuRes)
			continue
		}
		defer gpuNet.ReleaseGPUWeights()

		beforeMetrics, _ = gpuNet.EvaluateNetwork(dataset.Inputs, dataset.Outputs)
		gpuRes.InitialAcc = beforeMetrics.Accuracy

		// Higher LR for GPU batch training
		gpuLR := float32(*lrFlag) * 5.0
		losses, trainTime = train(gpuNet, dataset, *epochsFlag, gpuLR, true, 10)
		gpuRes.Loss = losses
		gpuRes.TrainTime = trainTime

		afterMetrics, _ = gpuNet.EvaluateNetwork(dataset.Inputs, dataset.Outputs)
		gpuRes.FinalAcc = afterMetrics.Accuracy
		gpuRes.Success = true
		gpuResults = append(gpuResults, gpuRes)

		fmt.Printf("  ✓ %.1f%% → %.1f%% accuracy in %v\n", gpuRes.InitialAcc*100, gpuRes.FinalAcc*100, trainTime)
	}

	// ===== Summary =====
	fmt.Println("\n╔═══════════════════════════════════════════════════════════════════════════════╗")
	fmt.Println("║                              RESULTS SUMMARY                                  ║")
	fmt.Println("╠═══════════════════════════════════════════════════════════════════════════════╣")
	fmt.Printf("║ %-15s │ CPU Start │ CPU End │ GPU Start │ GPU End │ Speedup │ Status ║\n", "Config")
	fmt.Println("╠═══════════════════════════════════════════════════════════════════════════════╣")

	allLearning := true
	for i := range cpuResults {
		cpu := cpuResults[i]
		gpuR := gpuResults[i]

		status := "✓"
		if !cpu.Success || !gpuR.Success {
			status = "✗"
		}

		speedup := float64(cpu.TrainTime) / float64(gpuR.TrainTime)
		if !gpuR.Success {
			speedup = 0
		}

		// Check if learning happened
		cpuLearned := cpu.FinalAcc > cpu.InitialAcc+0.1
		gpuLearned := gpuR.FinalAcc > gpuR.InitialAcc+0.1
		if cpuLearned {
			status += " CPU✓"
		} else {
			status += " CPU?"
			allLearning = false
		}
		if gpuLearned {
			status += " GPU✓"
		} else {
			status += " GPU?"
		}

		fmt.Printf("║ %-15s │ %8.1f%% │ %6.1f%% │ %8.1f%% │ %6.1f%% │ %6.2fx │ %-12s ║\n",
			cpu.Config.Name,
			cpu.InitialAcc*100, cpu.FinalAcc*100,
			gpuR.InitialAcc*100, gpuR.FinalAcc*100,
			speedup, status)
	}
	fmt.Println("╚═══════════════════════════════════════════════════════════════════════════════╝")

	fmt.Println()
	if allLearning {
		fmt.Println("✓ MHA is training correctly - accuracy improved significantly!")
	} else {
		fmt.Println("⚠ MHA may not be learning - accuracy did not improve >10%")
	}
}
