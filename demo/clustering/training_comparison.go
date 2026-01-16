package main

import (
	"fmt"
	"math"
	"math/rand"
	"time"

	"github.com/openfluke/loom/nn"
)

func main() {
	rand.Seed(42) // Fixed seed for reproducibility

	fmt.Println("╔════════════════════════════════════════════════════════════════╗")
	fmt.Println("║   Learnable K-Means vs Traditional K-Means: Training Demo     ║")
	fmt.Println("╚════════════════════════════════════════════════════════════════╝\n")

	// ================================================================
	// Create Training Data: 3 clusters with some overlap
	// ================================================================
	numSamplesPerCluster := 100
	totalSamples := numSamplesPerCluster * 3
	inputDim := 8

	trainData := make([][]float32, totalSamples)
	trainLabels := make([]int, totalSamples)

	fmt.Println("📊 Generating training data...")
	fmt.Printf("   - %d samples per cluster\n", numSamplesPerCluster)
	fmt.Printf("   - %d total samples\n", totalSamples)
	fmt.Printf("   - %d input dimensions\n\n", inputDim)

	// Cluster 0: Around [0.2, 0.2, ...]
	for i := 0; i < numSamplesPerCluster; i++ {
		trainData[i] = make([]float32, inputDim)
		for j := 0; j < inputDim; j++ {
			trainData[i][j] = rand.Float32()*0.25 + 0.1
		}
		trainLabels[i] = 0
	}

	// Cluster 1: Around [0.5, 0.5, ...]
	for i := numSamplesPerCluster; i < 2*numSamplesPerCluster; i++ {
		trainData[i] = make([]float32, inputDim)
		for j := 0; j < inputDim; j++ {
			trainData[i][j] = rand.Float32()*0.25 + 0.4
		}
		trainLabels[i] = 1
	}

	// Cluster 2: Around [0.8, 0.8, ...]
	for i := 2 * numSamplesPerCluster; i < totalSamples; i++ {
		trainData[i] = make([]float32, inputDim)
		for j := 0; j < inputDim; j++ {
			trainData[i][j] = rand.Float32()*0.25 + 0.7
		}
		trainLabels[i] = 2
	}

	// ================================================================
	// Baseline: Traditional K-Means (non-learnable)
	// ================================================================
	fmt.Println("═══════════════════════════════════════════════════════════════")
	fmt.Println("APPROACH 1: Traditional K-Means (Fixed)")
	fmt.Println("═══════════════════════════════════════════════════════════════\n")

	start := time.Now()
	centroids, assignments := nn.KMeansCluster(trainData, 3, 100, false)
	traditionalTime := time.Since(start)

	// Evaluate traditional K-Means
	traditionalAccuracy := evaluateClustering(assignments, trainLabels, 3)

	fmt.Printf("✓ Training time: %v\n", traditionalTime)
	fmt.Printf("✓ Clustering accuracy: %.1f%%\n", traditionalAccuracy)

	fmt.Println("\n📍 Traditional K-Means Cluster Centers:")
	for c := 0; c < 3; c++ {
		fmt.Printf("   Cluster %d: [%.3f, %.3f, %.3f, ...]\n",
			c, centroids[c][0], centroids[c][1], centroids[c][2])
	}

	// ================================================================
	// Learnable K-Means in a Network
	// ================================================================
	fmt.Println("\n═══════════════════════════════════════════════════════════════")
	fmt.Println("APPROACH 2: Learnable K-Means Layer (Trainable)")
	fmt.Println("═══════════════════════════════════════════════════════════════\n")

	// Build network: Input → Dense → KMeans → Dense → Output
	net := nn.NewNetwork(inputDim, 1, 1, 3)
	net.BatchSize = 1

	// Layer 0: Dense transformation
	layer0 := nn.InitDenseLayer(inputDim, 6, nn.ActivationTanh)
	net.SetLayer(0, 0, 0, layer0)

	// Layer 1: Learnable K-Means
	embedLayer := nn.InitDenseLayer(6, 4, nn.ActivationTanh)
	kmeansLayer := nn.InitKMeansLayer(3, embedLayer, "probabilities")
	net.SetLayer(0, 0, 1, kmeansLayer)

	// Layer 2: Output classification
	layer2 := nn.InitDenseLayer(3, 3, nn.ActivationSigmoid)
	net.SetLayer(0, 0, 2, layer2)

	net.InitializeWeights()

	fmt.Println("🏗️  Network Architecture:")
	fmt.Println("   Input(8) → Dense(6) → KMeans(3 clusters) → Dense(3) → Output(3)")
	fmt.Println()

	// Show initial cluster centers
	initialCenters := make([][]float32, 3)
	for c := 0; c < 3; c++ {
		initialCenters[c] = make([]float32, 3)
		offset := c * kmeansLayer.ClusterDim
		copy(initialCenters[c], net.Layers[1].ClusterCenters[offset:offset+3])
	}

	fmt.Println("📍 Initial Cluster Centers (before training):")
	for c := 0; c < 3; c++ {
		fmt.Printf("   Cluster %d: [%.3f, %.3f, %.3f, ...]\n",
			c, initialCenters[c][0], initialCenters[c][1], initialCenters[c][2])
	}

	// Train the network
	fmt.Println("\n🎯 Training network...")
	epochs := 200
	learningRate := float32(0.05)

	start = time.Now()
	lossHistory := make([]float32, epochs)
	accuracyHistory := make([]float32, epochs)

	for epoch := 0; epoch < epochs; epoch++ {
		totalLoss := float32(0)
		correct := 0

		// Shuffle data
		indices := rand.Perm(totalSamples)

		for _, idx := range indices {
			input := trainData[idx]
			targetLabel := trainLabels[idx]

			// Forward pass through first dense layer
			layer0Cfg := net.GetLayer(0, 0, 0)
			denseOut, _ := net.ForwardCPU(input) // This goes through all layers

			// Get the K-Means layer output (probabilities) by running just that layer
			// We need to get the intermediate activations
			kmeansLayerCfg := net.GetLayer(0, 0, 1)

			// Forward through layer 0 manually to get KMeans input
			layer0Out := make([]float32, layer0Cfg.OutputHeight)
			for i := 0; i < layer0Cfg.OutputHeight; i++ {
				sum := layer0Cfg.Bias[i]
				for j := 0; j < layer0Cfg.InputHeight; j++ {
					sum += input[j] * layer0Cfg.Kernel[j*layer0Cfg.OutputHeight+i]
				}
				layer0Out[i] = float32(math.Tanh(float64(sum))) // Activation
			}

			// Get K-Means cluster probabilities
			kmeansOut, _ := nn.ForwardKMeansCPU(layer0Out, kmeansLayerCfg)

			// Final network output
			output := denseOut

			// Create one-hot target
			target := make([]float32, 3)
			target[targetLabel] = 1.0

			// Compute loss (cross-entropy)
			loss := float32(0)
			for i := 0; i < 3; i++ {
				if target[i] > 0 {
					loss -= target[i] * float32(math.Log(float64(output[i]+1e-10)))
				}
			}
			totalLoss += loss

			// Check prediction
			predLabel := argmax(output)
			if predLabel == targetLabel {
				correct++
			}

			// UPDATE CLUSTER CENTERS based on prediction error
			// Move the most-activated cluster toward correct class representation
			maxClusterIdx := argmax(kmeansOut)
			clusterWeight := kmeansOut[maxClusterIdx]

			// Get features from the attached dense layer inside K-Means
			attachedLayerCfg := kmeansLayerCfg.AttachedLayer
			features := make([]float32, attachedLayerCfg.OutputHeight)
			for i := 0; i < attachedLayerCfg.OutputHeight; i++ {
				sum := attachedLayerCfg.Bias[i]
				for j := 0; j < attachedLayerCfg.InputHeight; j++ {
					sum += layer0Out[j] * attachedLayerCfg.Kernel[j*attachedLayerCfg.OutputHeight+i]
				}
				features[i] = float32(math.Tanh(float64(sum)))
			}

			// Update cluster center based on whether prediction matches target
			// If correct: reinforce (move toward feature)
			// If wrong: repel (move away from feature)
			updateStrength := float32(0.1)
			if predLabel == targetLabel {
				updateStrength = 0.2 // Stronger reinforcement when correct
			} else {
				updateStrength = -0.05 // Small repulsion when wrong
			}

			// Update the activated cluster center
			centerOffset := maxClusterIdx * kmeansLayerCfg.ClusterDim
			for d := 0; d < kmeansLayerCfg.ClusterDim; d++ {
				// Move cluster center toward/away from feature
				diff := features[d] - kmeansLayerCfg.ClusterCenters[centerOffset+d]
				kmeansLayerCfg.ClusterCenters[centerOffset+d] += learningRate * updateStrength * diff * clusterWeight
			}

			// Update output layer weights more aggressively
			layer2Cfg := net.GetLayer(0, 0, 2)
			for i := 0; i < 3; i++ {
				for j := 0; j < 3; j++ {
					gradOutput := output[i] - target[i]
					// Update weight connecting cluster j to output i
					weightIdx := j*3 + i
					if weightIdx < len(layer2Cfg.Kernel) {
						layer2Cfg.Kernel[weightIdx] -= learningRate * gradOutput * kmeansOut[j]
					}
				}
				layer2Cfg.Bias[i] -= learningRate * (output[i] - target[i])
			}

			// Also update first Dense layer to better separate inputs
			layer0Update := net.GetLayer(0, 0, 0)
			for i := 0; i < layer0Update.OutputHeight; i++ {
				for j := 0; j < layer0Update.InputHeight; j++ {
					// Simple gradient based on output error
					outputError := float32(0)
					for k := 0; k < 3; k++ {
						outputError += (output[k] - target[k]) * (output[k] - target[k])
					}
					weightIdx := j*layer0Update.OutputHeight + i
					if weightIdx < len(layer0Update.Kernel) {
						layer0Update.Kernel[weightIdx] -= learningRate * 0.001 * outputError * input[j]
					}
				}
			}
		}

		avgLoss := totalLoss / float32(totalSamples)
		accuracy := float32(correct) / float32(totalSamples) * 100.0

		lossHistory[epoch] = avgLoss
		accuracyHistory[epoch] = accuracy

		if epoch%20 == 0 || epoch == epochs-1 {
			fmt.Printf("   Epoch %3d: Loss=%.4f, Accuracy=%.1f%%\n", epoch+1, avgLoss, accuracy)
		}
	}

	learnableTime := time.Since(start)

	// Show final cluster centers
	finalCenters := make([][]float32, 3)
	for c := 0; c < 3; c++ {
		finalCenters[c] = make([]float32, 3)
		offset := c * kmeansLayer.ClusterDim
		copy(finalCenters[c], net.Layers[1].ClusterCenters[offset:offset+3])
	}

	fmt.Println("\n📍 Final Cluster Centers (after training):")
	for c := 0; c < 3; c++ {
		fmt.Printf("   Cluster %d: [%.3f, %.3f, %.3f, ...]\n",
			c, finalCenters[c][0], finalCenters[c][1], finalCenters[c][2])
	}

	// Show cluster center movement
	fmt.Println("\n📏 Cluster Center Movement:")
	for c := 0; c < 3; c++ {
		distance := float32(0)
		for d := 0; d < 3; d++ {
			diff := finalCenters[c][d] - initialCenters[c][d]
			distance += diff * diff
		}
		distance = float32(math.Sqrt(float64(distance)))
		fmt.Printf("   Cluster %d moved: %.4f units\n", c, distance)
	}

	finalAccuracy := accuracyHistory[epochs-1]

	// ================================================================
	// Comparison
	// ================================================================
	fmt.Println("\n╔════════════════════════════════════════════════════════════════╗")
	fmt.Println("║                      FINAL COMPARISON                          ║")
	fmt.Println("╚════════════════════════════════════════════════════════════════╝\n")

	fmt.Printf("Traditional K-Means:\n")
	fmt.Printf("   Accuracy: %.1f%%\n", traditionalAccuracy)
	fmt.Printf("   Time: %v\n", traditionalTime)
	fmt.Printf("   Cluster centers: FIXED (don't adapt)\n\n")

	fmt.Printf("Learnable K-Means Layer:\n")
	fmt.Printf("   Accuracy: %.1f%%\n", finalAccuracy)
	fmt.Printf("   Time: %v (includes %d epochs training)\n", learnableTime, epochs)
	fmt.Printf("   Cluster centers: LEARNED (adapted to task)\n\n")

	improvement := finalAccuracy - traditionalAccuracy
	if improvement > 5.0 {
		fmt.Printf("🎉 Learnable K-Means improved accuracy by %.1f%%!\n", improvement)
	} else if improvement > 0 {
		fmt.Printf("✓ Learnable K-Means slightly better by %.1f%%\n", improvement)
	} else {
		fmt.Printf("⚠️  Results similar (difference: %.1f%%)\n", improvement)
	}

	fmt.Println("\n💡 KEY INSIGHT:")
	fmt.Println("   Traditional K-Means finds clusters based ONLY on input distances.")
	fmt.Println("   Learnable K-Means optimizes clusters for the TASK (classification)!")
	fmt.Println("   The cluster centers move during training to minimize loss.")
}

// Evaluate clustering by matching clusters to labels
func evaluateClustering(assignments, labels []int, numClusters int) float32 {
	// Find best label mapping for each cluster
	clusterToLabel := make([]int, numClusters)

	for c := 0; c < numClusters; c++ {
		labelCounts := make(map[int]int)
		for i, cluster := range assignments {
			if cluster == c {
				labelCounts[labels[i]]++
			}
		}

		// Find most common label for this cluster
		maxCount := 0
		bestLabel := 0
		for label, count := range labelCounts {
			if count > maxCount {
				maxCount = count
				bestLabel = label
			}
		}
		clusterToLabel[c] = bestLabel
	}

	// Count correct assignments
	correct := 0
	for i, cluster := range assignments {
		if clusterToLabel[cluster] == labels[i] {
			correct++
		}
	}

	return float32(correct) / float32(len(labels)) * 100.0
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
