package main

import (
	"fmt"
	"math"
	"math/rand"

	"github.com/openfluke/loom/nn"
)

func main() {
	rand.Seed(42)

	fmt.Println("╔════════════════════════════════════════════════════════════════╗")
	fmt.Println("║   K-Means Layer: Testing All Attached Layer Types             ║")
	fmt.Println("╚════════════════════════════════════════════════════════════════╝\n")

	// Create training data
	numSamplesPerCluster := 334 // Approx 1000 total samples
	totalSamples := numSamplesPerCluster * 3
	inputDim := 8

	trainData := make([][]float32, totalSamples)
	trainLabels := make([]int, totalSamples)

	fmt.Printf("📊 Training Data: %d samples, %d dimensions, 3 clusters\n\n", totalSamples, inputDim)

	// Cluster 0: Low values
	for i := 0; i < numSamplesPerCluster; i++ {
		trainData[i] = make([]float32, inputDim)
		for j := 0; j < inputDim; j++ {
			trainData[i][j] = rand.Float32()*0.25 + 0.1
		}
		trainLabels[i] = 0
	}

	// Cluster 1: Medium values
	for i := numSamplesPerCluster; i < 2*numSamplesPerCluster; i++ {
		trainData[i] = make([]float32, inputDim)
		for j := 0; j < inputDim; j++ {
			trainData[i][j] = rand.Float32()*0.25 + 0.4
		}
		trainLabels[i] = 1
	}

	// Cluster 2: High values
	for i := 2 * numSamplesPerCluster; i < totalSamples; i++ {
		trainData[i] = make([]float32, inputDim)
		for j := 0; j < inputDim; j++ {
			trainData[i][j] = rand.Float32()*0.25 + 0.7
		}
		trainLabels[i] = 2
	}

	// Test configurations - ONE network per attached layer type
	configs := []struct {
		name        string
		createLayer func() nn.LayerConfig
		description string
		epochs      int
	}{
		{
			name: "Dense(6→4, Tanh)",
			createLayer: func() nn.LayerConfig {
				return nn.InitDenseLayer(6, 4, nn.ActivationTanh)
			},
			description: "Simple fully-connected transformation",
			epochs:      20,
		},
		{
			name: "Dense(6→8, Tanh)",
			createLayer: func() nn.LayerConfig {
				return nn.InitDenseLayer(6, 8, nn.ActivationTanh)
			},
			description: "Larger dense with Tanh activation",
			epochs:      20,
		},
		{
			name: "Dense(6→3, Sigmoid)",
			createLayer: func() nn.LayerConfig {
				return nn.InitDenseLayer(6, 3, nn.ActivationSigmoid)
			},
			description: "Smaller dense with Sigmoid",
			epochs:      100,
		},
		{
			name: "Dense(6→6, Linear)",
			createLayer: func() nn.LayerConfig {
				return nn.InitDenseLayer(6, 6, 0) // Linear/no activation
			},
			description: "Same size, linear transformation",
			epochs:      100,
		},
		{
			name: "Conv1D(6ch→4)",
			createLayer: func() nn.LayerConfig {
				// Create Conv1D with proper dimensions
				cfg := nn.LayerConfig{
					Type:             nn.LayerConv1D,
					Conv1DInChannels: 1,
					Conv1DFilters:    4,
					Conv1DKernelSize: 3,
					Conv1DStride:     1,
					Conv1DPadding:    1,
					Activation:       nn.ActivationTanh,
					InputHeight:      6,
					OutputHeight:     4,
				}
				// Initialize weights
				kernelSize := cfg.Conv1DFilters * cfg.Conv1DInChannels * cfg.Conv1DKernelSize
				cfg.Kernel = make([]float32, kernelSize)
				for i := range cfg.Kernel {
					cfg.Kernel[i] = (rand.Float32()*2 - 1) * 0.1
				}
				cfg.Bias = make([]float32, cfg.Conv1DFilters)
				return cfg
			},
			description: "1D convolution for pattern detection",
			epochs:      50,
		},
		{
			name: "Seq(Dense->Relu->Dense)",
			createLayer: func() nn.LayerConfig {
				l1 := nn.InitDenseLayer(6, 12, nn.ActivationTanh)
				l2 := nn.InitDenseLayer(12, 6, nn.ActivationTanh)
				return nn.InitSequentialLayer(l1, l2)
			},
			description: "Multi-layer feature extraction (Deep Clustering)",
			epochs:      50,
		},
		{
			name: "RNN(6→5)",
			createLayer: func() nn.LayerConfig {
				cfg := nn.LayerConfig{
					Type:         nn.LayerRNN,
					RNNInputSize: 6,
					HiddenSize:   5,
					SeqLength:    1,
					Activation:   nn.ActivationTanh,
					OutputHeight: 5,
				}
				// Initialize weights
				cfg.WeightIH = make([]float32, 6*5)
				cfg.WeightHH = make([]float32, 5*5)
				cfg.BiasH = make([]float32, 5)
				for i := range cfg.WeightIH {
					cfg.WeightIH[i] = (rand.Float32()*2 - 1) * 0.1
				}
				for i := range cfg.WeightHH {
					cfg.WeightHH[i] = (rand.Float32()*2 - 1) * 0.1
				}
				return cfg
			},
			description: "Recurrent neural network layer",
			epochs:      50,
		},
		{
			name: "LSTM(6→4)",
			createLayer: func() nn.LayerConfig {
				cfg := nn.LayerConfig{
					Type:         nn.LayerLSTM,
					RNNInputSize: 6,
					HiddenSize:   4,
					SeqLength:    1,
					OutputHeight: 4,
				}
				// Initialize all LSTM weights
				inputSize := 6
				hiddenSize := 4
				cfg.WeightIH_i = make([]float32, inputSize*hiddenSize)
				cfg.WeightHH_i = make([]float32, hiddenSize*hiddenSize)
				cfg.BiasH_i = make([]float32, hiddenSize)
				cfg.WeightIH_f = make([]float32, inputSize*hiddenSize)
				cfg.WeightHH_f = make([]float32, hiddenSize*hiddenSize)
				cfg.BiasH_f = make([]float32, hiddenSize)
				cfg.WeightIH_g = make([]float32, inputSize*hiddenSize)
				cfg.WeightHH_g = make([]float32, hiddenSize*hiddenSize)
				cfg.BiasH_g = make([]float32, hiddenSize)
				cfg.WeightIH_o = make([]float32, inputSize*hiddenSize)
				cfg.WeightHH_o = make([]float32, hiddenSize*hiddenSize)
				cfg.BiasH_o = make([]float32, hiddenSize)

				// Random init
				for i := range cfg.WeightIH_i {
					cfg.WeightIH_i[i] = (rand.Float32()*2 - 1) * 0.1
					cfg.WeightIH_f[i] = (rand.Float32()*2 - 1) * 0.1
					cfg.WeightIH_g[i] = (rand.Float32()*2 - 1) * 0.1
					cfg.WeightIH_o[i] = (rand.Float32()*2 - 1) * 0.1
				}
				for i := range cfg.WeightHH_i {
					cfg.WeightHH_i[i] = (rand.Float32()*2 - 1) * 0.1
					cfg.WeightHH_f[i] = (rand.Float32()*2 - 1) * 0.1
					cfg.WeightHH_g[i] = (rand.Float32()*2 - 1) * 0.1
					cfg.WeightHH_o[i] = (rand.Float32()*2 - 1) * 0.1
				}
				return cfg
			},
			description: "Long Short-Term Memory layer",
			epochs:      1,
		},
		{
			name: "Conv2D(1x6->4x4)",
			createLayer: func() nn.LayerConfig {
				cfg := nn.LayerConfig{
					Type:          nn.LayerConv2D,
					InputHeight:   1,
					InputWidth:    6,
					InputChannels: 1,
					Filters:       4,
					Stride:        1,

					// FIX 1: Set Padding to 0 to keep Height at 1
					Padding: 0,

					Activation: nn.ActivationTanh,

					// FIX 2: Set 2D dimensions for Conv2D (Flattened size will be 4*4*1 = 16)
					OutputHeight: 1,
					OutputWidth:  4,
				}

				// Initialize weights (This part was correct)
				kernelHeight := 1
				kernelWidth := 3
				cfg.Kernel = make([]float32, cfg.Filters*cfg.InputChannels*kernelHeight*kernelWidth)
				for i := range cfg.Kernel {
					cfg.Kernel[i] = (rand.Float32()*2 - 1) * 0.1
				}
				cfg.Bias = make([]float32, cfg.Filters)
				return cfg
			},
			description: "2D Convolution (Valid Padding, Flattened)",
			epochs:      50,
		},
		{
			name: "MHA(6, 1head, SeqLen=2)",
			createLayer: func() nn.LayerConfig {
				// Split 6D input into 2 tokens of 3D
				inputSize := 3
				numHeads := 1
				dModel := inputSize

				cfg := nn.LayerConfig{
					Type:          nn.LayerMultiHeadAttention,
					DModel:        dModel,
					InputChannels: dModel,
					NumHeads:      numHeads,
					HeadDim:       dModel / numHeads,
					SeqLength:     2,
					OutputHeight:  6, // Flattened output
				}

				weightSize := numHeads * inputSize * (inputSize / numHeads) // 1*3*3 = 9

				cfg.QWeights = make([]float32, weightSize)
				cfg.KWeights = make([]float32, weightSize)
				cfg.VWeights = make([]float32, weightSize)
				cfg.OutputWeight = make([]float32, weightSize) // Singular OutputWeight

				cfg.QBias = make([]float32, dModel)
				cfg.KBias = make([]float32, dModel)
				cfg.VBias = make([]float32, dModel)
				cfg.OutputBias = make([]float32, dModel)

				for i := range cfg.QWeights {
					cfg.QWeights[i] = (rand.Float32()*2 - 1) * 0.1
					cfg.KWeights[i] = (rand.Float32()*2 - 1) * 0.1
					cfg.VWeights[i] = (rand.Float32()*2 - 1) * 0.1
					cfg.OutputWeight[i] = (rand.Float32()*2 - 1) * 0.1
				}
				return cfg
			},
			description: "Multi-Head Attention (SeqLen=2)",
			epochs:      1,
		},
		{
			name: "LayerNorm(6)",
			createLayer: func() nn.LayerConfig {
				size := 6
				cfg := nn.LayerConfig{
					Type:         nn.LayerNorm,
					InputHeight:  size,
					OutputHeight: size,
					NormSize:     size,
					Gamma:        make([]float32, size),
					Beta:         make([]float32, size),
				}
				for i := range cfg.Gamma {
					cfg.Gamma[i] = 1.0
				}
				return cfg
			},
			description: "Layer Normalization",
			epochs:      1,
		},
		{
			name: "RMSNorm(6)",
			createLayer: func() nn.LayerConfig {
				size := 6
				cfg := nn.LayerConfig{
					Type:         nn.LayerRMSNorm,
					InputHeight:  size,
					OutputHeight: size,
					NormSize:     size,
					Gamma:        make([]float32, size),
				}
				for i := range cfg.Gamma {
					cfg.Gamma[i] = 1.0
				}
				return cfg
			},
			description: "RMS Normalization",
			epochs:      1,
		},
	}

	results := make([]float32, len(configs))
	movements := make([]float32, len(configs))

	// Test each configuration
	for cfgIdx, config := range configs {
		fmt.Println("═══════════════════════════════════════════════════════════════")
		fmt.Printf("Testing: %s\n", config.name)
		fmt.Printf("Description: %s\n", config.description)
		fmt.Println("═══════════════════════════════════════════════════════════════")

		// Create network
		net := nn.NewNetwork(inputDim, 1, 1, 3)
		net.BatchSize = 1

		// Layer 0: Dense preprocessing
		layer0 := nn.InitDenseLayer(inputDim, 6, nn.ActivationTanh)
		net.SetLayer(0, 0, 0, layer0)

		// Layer 1: KMeans with attached layer
		attachedLayer := config.createLayer()
		kmeansLayer := nn.InitKMeansLayer(3, attachedLayer, "probabilities")
		net.SetLayer(0, 0, 1, kmeansLayer)

		// Layer 2: Output
		layer2 := nn.InitDenseLayer(3, 3, nn.ActivationSigmoid)
		net.SetLayer(0, 0, 2, layer2)

		net.InitializeWeights()

		// Do one forward pass to trigger lazy initialization of cluster centers
		_, _ = net.ForwardCPU(trainData[0])

		// Store initial centers (after lazy init)
		clusterDim := net.Layers[1].ClusterDim
		initialCenters := make([]float32, 3*clusterDim)
		if len(net.Layers[1].ClusterCenters) > 0 {
			copy(initialCenters, net.Layers[1].ClusterCenters)
		}

		// Train
		epochs := config.epochs
		if epochs == 0 {
			epochs = 100
		}
		learningRate := float32(0.05)

		bestAccuracy := float32(0)

		for epoch := 0; epoch < epochs; epoch++ {
			correct := 0
			indices := rand.Perm(totalSamples)

			for _, idx := range indices {
				input := trainData[idx]
				targetLabel := trainLabels[idx]

				// Forward pass
				output, _ := net.ForwardCPU(input)

				// Target (one-hot)
				target := make([]float32, 3)
				target[targetLabel] = 1.0

				// Check prediction
				predLabel := argmax(output)
				if predLabel == targetLabel {
					correct++
				}

				// Compute gradient of loss w.r.t output
				// For cross-entropy with softmax: grad = output - target
				gradOutput := make([]float32, 3)
				for i := 0; i < 3; i++ {
					gradOutput[i] = output[i] - target[i]
				}

				// Backward pass - propagates gradients through all layers including KMeans
				net.BackwardCPU(gradOutput)

				// Apply gradients to update weights in all layers
				net.ApplyGradients(learningRate)
			}

			accuracy := float32(correct) / float32(totalSamples) * 100.0
			if accuracy > bestAccuracy {
				bestAccuracy = accuracy
			}
		}

		// Calculate movement
		finalCenters := net.Layers[1].ClusterCenters
		totalMovement := float32(0)
		finalClusterDim := net.Layers[1].ClusterDim
		if len(finalCenters) > 0 && len(initialCenters) > 0 && finalClusterDim > 0 {
			for c := 0; c < 3; c++ {
				movement := float32(0)
				offset := c * finalClusterDim
				if offset+finalClusterDim <= len(finalCenters) && offset+finalClusterDim <= len(initialCenters) {
					for d := 0; d < finalClusterDim; d++ {
						diff := finalCenters[offset+d] - initialCenters[offset+d]
						movement += diff * diff
					}
				}
				totalMovement += float32(math.Sqrt(float64(movement)))
			}
		}

		results[cfgIdx] = bestAccuracy
		movements[cfgIdx] = totalMovement

		fmt.Printf("✓ Best Accuracy: %.1f%%\n", bestAccuracy)
		fmt.Printf("✓ Total Cluster Movement: %.4f units\n\n", totalMovement)
	}

	// Final comparison
	fmt.Println("\n╔════════════════════════════════════════════════════════════════╗")
	fmt.Println("║                      FINAL RESULTS                              ║")
	fmt.Println("╚════════════════════════════════════════════════════════════════╝\n")

	fmt.Println("Attached Layer Type      | Accuracy | Movement")
	fmt.Println("-------------------------|----------|----------")

	bestIdx := 0
	bestAcc := results[0]

	for i, config := range configs {
		fmt.Printf("%-24s | %6.1f%% | %.4f\n", config.name, results[i], movements[i])
		if results[i] > bestAcc {
			bestAcc = results[i]
			bestIdx = i
		}
	}

	fmt.Printf("\n🏆 WINNER: %s with %.1f%% accuracy!\n", configs[bestIdx].name, bestAcc)

	fmt.Println("\n💡 KEY INSIGHTS:")
	fmt.Println("   - Different layer types learn different feature representations")
	fmt.Println("   - Cluster centers adapt differently based on the attached layer")
	fmt.Println("   - The best architecture depends on your specific data!")
}

func argmax(values []float32) int {
	maxIdx := 0
	maxVal := values[0]
	for i, v := range values {
		if v > maxVal {
			maxVal = v
			maxIdx = i
		}
	}
	return maxIdx
}
