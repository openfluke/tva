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
	fmt.Println("║  Simple K-Means Example: See Sample-to-Cluster Mapping    ║")
	fmt.Println("╚════════════════════════════════════════════════════════════╝\n")

	// Create 12 simple data points
	data := [][]float32{
		// Group A: Low values
		{1.0, 1.5}, {1.2, 1.8}, {0.9, 1.3}, {1.1, 1.6},
		// Group B: Medium values
		{5.0, 5.5}, {5.2, 5.8}, {4.9, 5.3}, {5.1, 5.6},
		// Group C: High values
		{9.0, 9.5}, {9.2, 9.8}, {8.9, 9.3}, {9.1, 9.6},
	}

	fmt.Println("📊 Original Data (12 samples):")
	for i, point := range data {
		fmt.Printf("  Sample %2d: [%.1f, %.1f]\n", i, point[0], point[1])
	}

	// Cluster into 3 groups
	k := 3
	fmt.Printf("\n🔄 Running K-Means with K=%d...\n\n", k)

	centroids, assignments := nn.KMeansCluster(data, k, 100, false)

	// Show cluster centers
	fmt.Println("🎯 Cluster Centers (Centroids):")
	for i, centroid := range centroids {
		fmt.Printf("  Cluster %d center: [%.2f, %.2f]\n", i, centroid[0], centroid[1])
	}

	// THIS IS WHAT YOU WANTED TO SEE!
	// Show which sample belongs to which cluster
	fmt.Println("\n✨ Sample-to-Cluster Assignments:")
	fmt.Println("  (This tells you exactly which cluster each sample belongs to)\n")

	for i, clusterID := range assignments {
		fmt.Printf("  Sample %2d → Cluster %d  (data: [%.1f, %.1f])\n",
			i, clusterID, data[i][0], data[i][1])
	}

	// Group by cluster
	fmt.Println("\n📦 Grouped by Cluster:")
	for c := 0; c < k; c++ {
		fmt.Printf("\n  Cluster %d contains:\n", c)
		for i, clusterID := range assignments {
			if clusterID == c {
				fmt.Printf("    - Sample %2d: [%.1f, %.1f]\n", i, data[i][0], data[i][1])
			}
		}
	}

	// Calculate quality
	silhouette := nn.ComputeSilhouetteScore(data, assignments)
	fmt.Printf("\n🎯 Silhouette Score: %.3f ", silhouette)
	if silhouette > 0.7 {
		fmt.Println("(Excellent! Clusters are well-separated)")
	} else {
		fmt.Println("(Good clustering)")
	}

	fmt.Println("\n============================================================")
	fmt.Println("KEY TAKEAWAY:")
	fmt.Println("The 'assignments' array tells you which cluster each sample belongs to!")
	fmt.Printf("For example: assignments[0] = %d means Sample 0 is in Cluster %d\n",
		assignments[0], assignments[0])
	fmt.Println("============================================================")
}
