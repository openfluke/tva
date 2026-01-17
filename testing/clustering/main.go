package main

import (
	"fmt"
	"math/rand"
	"time"

	"github.com/openfluke/loom/nn"
)

const (
	// ANSI color codes for terminal output
	colorReset  = "\033[0m"
	colorRed    = "\033[31m"
	colorGreen  = "\033[32m"
	colorYellow = "\033[33m"
	colorBlue   = "\033[34m"
	colorPurple = "\033[35m"
	colorCyan   = "\033[36m"
	colorWhite  = "\033[37m"
	colorBold   = "\033[1m"
	colorDim    = "\033[2m"
)

var clusterColors = []string{
	colorRed, colorGreen, colorYellow, colorBlue, colorPurple, colorCyan,
}

func main() {
	rand.Seed(time.Now().UnixNano())

	fmt.Println(colorBold + "╔══════════════════════════════════════════════════════════════════════╗" + colorReset)
	fmt.Println(colorBold + "║              K-Means Clustering Demonstration - LOOM                 ║" + colorReset)
	fmt.Println(colorBold + "╚══════════════════════════════════════════════════════════════════════╝" + colorReset)
	fmt.Println()

	// Demo 1: Simple 2D Clustering
	demo1Simple2DClustering()

	// Demo 2: Customer Segmentation
	demo2CustomerSegmentation()

	// Demo 3: Finding Optimal K
	demo3OptimalK()

	// Demo 4: High-Dimensional Clustering
	demo4HighDimensional()

	// Demo 5: Parallel vs Sequential Performance
	demo5PerformanceComparison()

	fmt.Println(colorBold + "\n✅ All clustering demonstrations complete!" + colorReset)
}

// Demo 1: Simple 2D point clustering with visualization
func demo1Simple2DClustering() {
	fmt.Println(colorBold + "\n┌──────────────────────────────────────────────────────────────────────┐" + colorReset)
	fmt.Println(colorBold + "│ Demo 1: Simple 2D Point Clustering                                  │" + colorReset)
	fmt.Println(colorBold + "└──────────────────────────────────────────────────────────────────────┘" + colorReset)

	// Create 3 distinct clusters in 2D space
	data := make([][]float32, 90)

	// Cluster 1: Bottom-left quadrant
	for i := 0; i < 30; i++ {
		data[i] = []float32{
			rand.Float32()*3.0 + 1.0, // x: 1-4
			rand.Float32()*3.0 + 1.0, // y: 1-4
		}
	}

	// Cluster 2: Top-right quadrant
	for i := 30; i < 60; i++ {
		data[i] = []float32{
			rand.Float32()*3.0 + 7.0, // x: 7-10
			rand.Float32()*3.0 + 7.0, // y: 7-10
		}
	}

	// Cluster 3: Bottom-right quadrant
	for i := 60; i < 90; i++ {
		data[i] = []float32{
			rand.Float32()*3.0 + 7.0, // x: 7-10
			rand.Float32()*3.0 + 1.0, // y: 1-4
		}
	}

	// Perform clustering
	k := 3
	centroids, assignments := nn.KMeansCluster(data, k, 100, false)

	fmt.Printf("\n📊 Clustered %d points into %d groups:\n\n", len(data), k)

	// Show cluster assignments
	clusterCounts := make(map[int]int)
	for _, clusterID := range assignments {
		clusterCounts[clusterID]++
	}

	for i := 0; i < k; i++ {
		color := clusterColors[i%len(clusterColors)]
		fmt.Printf("%s● Cluster %d%s: %d points | Center: (%.2f, %.2f)\n",
			color, i, colorReset, clusterCounts[i], centroids[i][0], centroids[i][1])
	}

	// Visualize in ASCII grid
	fmt.Println("\n📈 Visualization (10x10 grid):")
	visualize2D(data, assignments, centroids)

	// Calculate quality
	silhouette := nn.ComputeSilhouetteScore(data, assignments)
	fmt.Printf("\n🎯 Silhouette Score: %.3f ", silhouette)
	if silhouette > 0.5 {
		fmt.Printf(colorGreen + "(Excellent separation!)" + colorReset)
	} else if silhouette > 0.25 {
		fmt.Printf(colorYellow + "(Good clustering)" + colorReset)
	} else {
		fmt.Printf(colorRed + "(Poor separation)" + colorReset)
	}
	fmt.Println()
}

// Demo 2: Customer segmentation example
func demo2CustomerSegmentation() {
	fmt.Println(colorBold + "\n┌──────────────────────────────────────────────────────────────────────┐" + colorReset)
	fmt.Println(colorBold + "│ Demo 2: Customer Segmentation (3 Features)                          │" + colorReset)
	fmt.Println(colorBold + "└──────────────────────────────────────────────────────────────────────┘" + colorReset)

	// Simulate customer data: [spending, frequency, recency]
	data := make([][]float32, 120)

	// Budget shoppers (low spending, low frequency, high recency)
	for i := 0; i < 40; i++ {
		data[i] = []float32{
			rand.Float32()*30.0 + 10.0, // spending: $10-40
			rand.Float32()*5.0 + 1.0,   // frequency: 1-6 visits/month
			rand.Float32()*15.0 + 15.0, // recency: 15-30 days
		}
	}

	// Regular customers (medium spending, medium frequency, medium recency)
	for i := 40; i < 80; i++ {
		data[i] = []float32{
			rand.Float32()*50.0 + 50.0, // spending: $50-100
			rand.Float32()*10.0 + 5.0,  // frequency: 5-15 visits/month
			rand.Float32()*10.0 + 5.0,  // recency: 5-15 days
		}
	}

	// VIP customers (high spending, high frequency, low recency)
	for i := 80; i < 120; i++ {
		data[i] = []float32{
			rand.Float32()*150.0 + 100.0, // spending: $100-250
			rand.Float32()*15.0 + 15.0,   // frequency: 15-30 visits/month
			rand.Float32() * 5.0,         // recency: 0-5 days
		}
	}

	// Perform clustering
	k := 3
	centroids, assignments := nn.KMeansCluster(data, k, 100, false)

	fmt.Printf("\n📊 Segmented %d customers into %d groups:\n\n", len(data), k)

	// Analyze each cluster
	for i := 0; i < k; i++ {
		color := clusterColors[i%len(clusterColors)]

		// Count customers in this cluster
		count := 0
		for _, clusterID := range assignments {
			if clusterID == i {
				count++
			}
		}

		fmt.Printf("%s■ Segment %d%s (%d customers):\n", color, i, colorReset, count)
		fmt.Printf("  Avg Spending:  $%.2f/month\n", centroids[i][0])
		fmt.Printf("  Avg Frequency: %.1f visits/month\n", centroids[i][1])
		fmt.Printf("  Avg Recency:   %.1f days\n", centroids[i][2])

		// Classify segment
		if centroids[i][0] > 100 {
			fmt.Printf(colorGreen + "  → VIP Customers (high value)" + colorReset + "\n")
		} else if centroids[i][0] > 50 {
			fmt.Printf(colorYellow + "  → Regular Customers (medium value)" + colorReset + "\n")
		} else {
			fmt.Printf(colorBlue + "  → Budget Shoppers (price-sensitive)" + colorReset + "\n")
		}
		fmt.Println()
	}

	silhouette := nn.ComputeSilhouetteScore(data, assignments)
	fmt.Printf("🎯 Segmentation Quality: %.3f\n", silhouette)
}

// Demo 3: Finding optimal number of clusters
func demo3OptimalK() {
	fmt.Println(colorBold + "\n┌──────────────────────────────────────────────────────────────────────┐" + colorReset)
	fmt.Println(colorBold + "│ Demo 3: Finding Optimal K (Silhouette Analysis)                     │" + colorReset)
	fmt.Println(colorBold + "└──────────────────────────────────────────────────────────────────────┘" + colorReset)

	// Create data with 4 natural clusters
	data := make([][]float32, 80)
	centers := [][]float32{{2, 2}, {8, 2}, {2, 8}, {8, 8}}

	for i := 0; i < 80; i++ {
		centerIdx := i / 20
		center := centers[centerIdx]
		data[i] = []float32{
			center[0] + rand.Float32()*1.5 - 0.75,
			center[1] + rand.Float32()*1.5 - 0.75,
		}
	}

	fmt.Println("\n📊 Testing different values of K:\n")

	bestK := 2
	bestScore := float32(-1.0)

	for k := 2; k <= 6; k++ {
		_, assignments := nn.KMeansCluster(data, k, 100, false)
		score := nn.ComputeSilhouetteScore(data, assignments)

		// Visual bar
		barLength := int(score * 50)
		bar := ""
		for i := 0; i < barLength; i++ {
			bar += "█"
		}

		fmt.Printf("K=%d: %.3f %s%s%s", k, score, colorCyan, bar, colorReset)

		if score > bestScore {
			bestScore = score
			bestK = k
			fmt.Printf(colorGreen + " ← Best!" + colorReset)
		}
		fmt.Println()

		if k == bestK && score > bestScore {
			bestScore = score
		}
	}

	fmt.Printf("\n✨ Optimal K: %d (Silhouette Score: %.3f)\n", bestK, bestScore)
}

// Demo 4: High-dimensional clustering
func demo4HighDimensional() {
	fmt.Println(colorBold + "\n┌──────────────────────────────────────────────────────────────────────┐" + colorReset)
	fmt.Println(colorBold + "│ Demo 4: High-Dimensional Data (10D Feature Space)                   │" + colorReset)
	fmt.Println(colorBold + "└──────────────────────────────────────────────────────────────────────┘" + colorReset)

	// Create 10-dimensional data with 3 clusters
	dims := 10
	data := make([][]float32, 150)

	for i := 0; i < 150; i++ {
		data[i] = make([]float32, dims)
		clusterID := i / 50

		for d := 0; d < dims; d++ {
			// Each cluster has different mean values
			mean := float32(clusterID * 5)
			data[i][d] = mean + rand.Float32()*2.0 - 1.0
		}
	}

	fmt.Printf("\n📊 Clustering %d samples in %dD space...\n", len(data), dims)

	k := 3
	start := time.Now()
	centroids, assignments := nn.KMeansCluster(data, k, 100, false)
	elapsed := time.Since(start)

	// Count assignments
	clusterCounts := make(map[int]int)
	for _, clusterID := range assignments {
		clusterCounts[clusterID]++
	}

	fmt.Printf("\n%s✓ Completed in %v%s\n\n", colorGreen, elapsed, colorReset)

	for i := 0; i < k; i++ {
		color := clusterColors[i%len(clusterColors)]
		fmt.Printf("%s● Cluster %d%s: %d samples\n", color, i, colorReset, clusterCounts[i])
		fmt.Printf("  Centroid preview: [%.2f, %.2f, %.2f, ...]\n",
			centroids[i][0], centroids[i][1], centroids[i][2])
	}

	silhouette := nn.ComputeSilhouetteScore(data, assignments)
	fmt.Printf("\n🎯 Silhouette Score: %.3f\n", silhouette)
}

// Demo 5: Performance comparison (parallel vs sequential)
func demo5PerformanceComparison() {
	fmt.Println(colorBold + "\n┌──────────────────────────────────────────────────────────────────────┐" + colorReset)
	fmt.Println(colorBold + "│ Demo 5: Parallel vs Sequential Performance                          │" + colorReset)
	fmt.Println(colorBold + "└──────────────────────────────────────────────────────────────────────┘" + colorReset)

	// Create larger dataset
	data := make([][]float32, 2000)
	for i := 0; i < 2000; i++ {
		data[i] = []float32{
			rand.Float32() * 100,
			rand.Float32() * 100,
			rand.Float32() * 100,
			rand.Float32() * 100,
		}
	}

	fmt.Printf("\n📊 Clustering %d samples with 4 features...\n\n", len(data))

	k := 5

	// Sequential
	start := time.Now()
	_, assignmentsSeq := nn.KMeansCluster(data, k, 50, false)
	seqTime := time.Since(start)

	// Parallel
	start = time.Now()
	_, assignmentsPar := nn.KMeansCluster(data, k, 50, true)
	parTime := time.Since(start)

	fmt.Printf("Sequential: %v\n", seqTime)
	fmt.Printf("Parallel:   %v\n", parTime)

	speedup := float64(seqTime) / float64(parTime)
	fmt.Printf("\n⚡ %sSpeedup: %.2fx%s", colorGreen, speedup, colorReset)

	if speedup > 1.5 {
		fmt.Printf(colorGreen + " (Excellent!)" + colorReset)
	} else if speedup > 1.0 {
		fmt.Printf(colorYellow + " (Good)" + colorReset)
	}
	fmt.Println()

	// Verify they produce similar results
	silSeq := nn.ComputeSilhouetteScore(data, assignmentsSeq)
	silPar := nn.ComputeSilhouetteScore(data, assignmentsPar)
	fmt.Printf("\n🎯 Quality - Sequential: %.3f | Parallel: %.3f\n", silSeq, silPar)
}

// ASCII visualization for 2D data
func visualize2D(data [][]float32, assignments []int, centroids [][]float32) {
	const gridSize = 20
	grid := make([][]int, gridSize)
	for i := range grid {
		grid[i] = make([]int, gridSize)
		for j := range grid[i] {
			grid[i][j] = -1 // empty
		}
	}

	// Find data bounds
	minX, maxX := data[0][0], data[0][0]
	minY, maxY := data[0][1], data[0][1]
	for _, point := range data {
		if point[0] < minX {
			minX = point[0]
		}
		if point[0] > maxX {
			maxX = point[0]
		}
		if point[1] < minY {
			minY = point[1]
		}
		if point[1] > maxY {
			maxY = point[1]
		}
	}

	// Map points to grid
	for i, point := range data {
		x := int((point[0] - minX) / (maxX - minX) * float32(gridSize-1))
		y := int((point[1] - minY) / (maxY - minY) * float32(gridSize-1))
		if x >= 0 && x < gridSize && y >= 0 && y < gridSize {
			grid[gridSize-1-y][x] = assignments[i]
		}
	}

	// Print grid
	fmt.Println("\n  " + colorDim + "┌────────────────────────────────────────┐" + colorReset)
	for i := 0; i < gridSize; i++ {
		fmt.Print("  " + colorDim + "│" + colorReset)
		for j := 0; j < gridSize; j++ {
			if grid[i][j] == -1 {
				fmt.Print(colorDim + "· " + colorReset)
			} else {
				color := clusterColors[grid[i][j]%len(clusterColors)]
				fmt.Print(color + "█ " + colorReset)
			}
		}
		fmt.Println(colorDim + "│" + colorReset)
	}
	fmt.Println("  " + colorDim + "└────────────────────────────────────────┘" + colorReset)

	// Show centroids
	fmt.Println("\n  " + colorBold + "Centroids:" + colorReset)
	for i, centroid := range centroids {
		color := clusterColors[i%len(clusterColors)]
		x := (centroid[0] - minX) / (maxX - minX) * float32(gridSize-1)
		y := (centroid[1] - minY) / (maxY - minY) * float32(gridSize-1)
		fmt.Printf("  %s● Cluster %d%s at grid position (%.1f, %.1f)\n", color, i, colorReset, x, gridSize-1-y)
	}
}
