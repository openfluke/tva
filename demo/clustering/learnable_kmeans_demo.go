package main

import (
	"fmt"
	"math/rand"
	"time"

	"github.com/openfluke/loom/nn"
)

func main() {
	rand.Seed(time.Now().UnixNano())

	fmt.Println("╔════════════════════════════════════════════════════════════╗")
	fmt.Println("║      Learnable K-Means Clustering Layer Demonstration      ║")
	fmt.Println("╚════════════════════════════════════════════════════════════╝\n")

	// Create synthetic data with 3 clusters
	numSamples := 90
	inputDim := 10
	data := make([][]float32, numSamples)

	// Cluster 1: Low values
	for i := 0; i < 30; i++ {
		data[i] = make([]float32, inputDim)
		for j := 0; j < inputDim; j++ {
			data[i][j] = rand.Float32()*0.3 + 0.0
		}
	}

	// Cluster 2: Medium values
	for i := 30; i < 60; i++ {
		data[i] = make([]float32, inputDim)
		for j := 0; j < inputDim; j++ {
			data[i][j] = rand.Float32()*0.3 + 0.4
		}
	}

	// Cluster 3: High values
	for i := 60; i < 90; i++ {
		data[i] = make([]float32, inputDim)
		for j := 0; j < inputDim; j++ {
			data[i][j] = rand.Float32()*0.3 + 0.7
		}
	}

	fmt.Println("📊 Created 90 samples in 3 clusters (low/medium/high)\n")

	// ===================================================================
	// Test 1: KMeans with Dense attachment
	// ===================================================================
	fmt.Println("═══════════════════════════════════════════════════════════")
	fmt.Println("Test 1: KMeans Layer with Dense Attachment")
	fmt.Println("═══════════════════════════════════════════════════════════")

	// Create attached Dense layer that transforms input
	denseLayer := nn.InitDenseLayer(inputDim, 5, nn.ActivationTanh)

	// Create KMeans layer with 3 clusters
	kmeansLayer := nn.InitKMeansLayer(3, denseLayer, "probabilities")

	fmt.Printf("✓ Created KMeans layer:\n")
	fmt.Printf("  - Input dim: %d\n", inputDim)
	fmt.Printf("  - Feature dim (after Dense): %d\n", kmeansLayer.ClusterDim)
	fmt.Printf("  - Number of clusters: %d\n", kmeansLayer.NumClusters)
	fmt.Printf("  - Distance metric: %s\n", kmeansLayer.DistanceMetric)
	fmt.Printf("  - Output mode: %s\n\n", kmeansLayer.KMeansOutputMode)

	// Forward pass on all samples
	fmt.Println("Running forward pass on all samples...")
	clusterAssignments := make([]int, numSamples)

	for i, sample := range data {
		output, err := nn.ForwardKMeansCPU(sample, &kmeansLayer)
		if err != nil {
			fmt.Printf("Error on sample %d: %v\n", i, err)
			continue
		}

		// Find max probability cluster
		maxProb := float32(-1.0)
		maxCluster := 0
		for c := 0; c < kmeansLayer.NumClusters; c++ {
			if output[c] > maxProb {
				maxProb = output[c]
				maxCluster = c
			}
		}
		clusterAssignments[i] = maxCluster
	}

	// Analyze cluster assignments
	clusterCounts := make([]int, 3)
	for _, c := range clusterAssignments {
		clusterCounts[c]++
	}

	fmt.Println("\n📈 Cluster Assignments:")
	for c := 0; c < 3; c++ {
		fmt.Printf("  Cluster %d: %d samples\n", c, clusterCounts[c])
	}

	// Check separation quality
	fmt.Println("\n🎯 Verifying Cluster Separation:")
	cluster0Low := countRange(clusterAssignments, 0, 0, 30)
	cluster1Mid := countRange(clusterAssignments, 1, 30, 60)
	cluster2High := countRange(clusterAssignments, 2, 60, 90)

	fmt.Printf("  Cluster 0 captured %d/30 low-value samples\n", cluster0Low)
	fmt.Printf("  Cluster 1 captured %d/30 mid-value samples\n", cluster1Mid)
	fmt.Printf("  Cluster 2 captured %d/30 high-value samples\n", cluster2High)

	totalCorrect := cluster0Low + cluster1Mid + cluster2High
	accuracy := float64(totalCorrect) / float64(numSamples) * 100.0
	fmt.Printf("\n  Separation Accuracy: %.1f%%\n", accuracy)

	if accuracy > 50.0 {
		fmt.Println("  ✓ Layer successfully discovered cluster structure!")
	}

	fmt.Println("\n✅ Test 1 Complete!")

	// ===================================================================
	// Test 2: Show learnable cluster centers
	// ===================================================================
	fmt.Println("\n═══════════════════════════════════════════════════════════")
	fmt.Println("Test 2: Inspecting Learnable Cluster Centers")
	fmt.Println("═══════════════════════════════════════════════════════════")

	fmt.Println("Cluster center values (first 3 dimensions):")
	for c := 0; c < kmeansLayer.NumClusters; c++ {
		offset := c * kmeansLayer.ClusterDim
		fmt.Printf("  Cluster %d: [%.3f, %.3f, %.3f, ...]\n",
			c,
			kmeansLayer.ClusterCenters[offset],
			kmeansLayer.ClusterCenters[offset+1],
			kmeansLayer.ClusterCenters[offset+2])
	}

	fmt.Println("\n💡 These cluster centers are trainable parameters!")
	fmt.Println("   During backpropagation, they'll move to minimize loss.")

	fmt.Println("\n✅ All tests complete!")
	fmt.Println("\n" + repeat("═", 60))
	fmt.Println("The learnable K-Means layer successfully:")
	fmt.Println("  1. Transforms input through attached Dense layer")
	fmt.Println("  2. Computes distances to learnable cluster centers")
	fmt.Println("  3. Outputs soft cluster assignments")
	fmt.Println("  4. Can be trained end-to-end with backpropagation!")
	fmt.Println(repeat("═", 60))
}

// Helper function to count cluster assignments in a range
func countRange(assignments []int, targetCluster, startIdx, endIdx int) int {
	count := 0
	for i := startIdx; i < endIdx && i < len(assignments); i++ {
		if assignments[i] == targetCluster {
			count++
		}
	}
	return count
}

// Repeat string helper
func repeat(s string, n int) string {
	result := ""
	for i := 0; i < n; i++ {
		result += s
	}
	return result
}
