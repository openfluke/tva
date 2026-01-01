package main

import (
	"fmt"
	"math/rand"

	"github.com/openfluke/loom/nn"
)

func main() {
	fmt.Println("╔════════════════════════════════════════════════════════════════╗")
	fmt.Println("║          LOOM Correlation Analysis Demo                        ║")
	fmt.Println("║  Compute Pearson/Spearman correlation matrices for datasets    ║")
	fmt.Println("╚════════════════════════════════════════════════════════════════╝")
	fmt.Println()

	// Create sample dataset with known correlations
	// Features: Age, Income, Experience, Education, Satisfaction
	numSamples := 100
	labels := []string{"Age", "Income", "Experience", "Education", "Satisfaction"}

	data := generateSampleData(numSamples)

	// Compute Pearson correlation matrix
	fmt.Println("📊 Computing Pearson Correlation Matrix...")
	result := nn.ComputeCorrelationMatrix(data, labels)

	if result == nil {
		fmt.Println("❌ Failed to compute correlation matrix")
		return
	}

	// Print the correlation matrix
	fmt.Println("\n┌─────────────────────────────────────────────────────────────────┐")
	fmt.Println("│                 CORRELATION MATRIX                              │")
	fmt.Println("└─────────────────────────────────────────────────────────────────┘")
	fmt.Println()

	// Header row
	fmt.Printf("%-15s", "")
	for _, label := range result.Correlation.Labels {
		fmt.Printf("%-12s", label)
	}
	fmt.Println()
	fmt.Println("─────────────────────────────────────────────────────────────────")

	// Matrix rows
	for i, rowLabel := range result.Correlation.Labels {
		fmt.Printf("%-15s", rowLabel)
		for j := range result.Correlation.Labels {
			corr := result.Correlation.Matrix[i][j]
			// Color coding (conceptual - using symbols)
			symbol := " "
			if corr > 0.7 {
				symbol = "🔴" // Strong positive
			} else if corr > 0.3 {
				symbol = "🟠" // Moderate positive
			} else if corr < -0.7 {
				symbol = "🔵" // Strong negative
			} else if corr < -0.3 {
				symbol = "🟣" // Moderate negative
			}
			fmt.Printf("%6.2f%s    ", corr, symbol)
		}
		fmt.Println()
	}

	// Print feature statistics
	fmt.Println("\n┌─────────────────────────────────────────────────────────────────┐")
	fmt.Println("│                 FEATURE STATISTICS                              │")
	fmt.Println("└─────────────────────────────────────────────────────────────────┘")
	fmt.Println()
	fmt.Printf("%-15s %-12s %-12s %-12s %-12s\n", "Feature", "Mean", "StdDev", "Min", "Max")
	fmt.Println("─────────────────────────────────────────────────────────────────")
	for i, label := range result.Correlation.Labels {
		fmt.Printf("%-15s %-12.2f %-12.2f %-12.2f %-12.2f\n",
			label, result.Means[i], result.StdDevs[i], result.Mins[i], result.Maxs[i])
	}

	// Find strong correlations
	fmt.Println("\n┌─────────────────────────────────────────────────────────────────┐")
	fmt.Println("│              STRONG CORRELATIONS (|r| ≥ 0.5)                    │")
	fmt.Println("└─────────────────────────────────────────────────────────────────┘")
	fmt.Println()

	strongCorrs := result.GetStrongCorrelations(0.5)
	if len(strongCorrs) == 0 {
		fmt.Println("No strong correlations found (threshold: 0.5)")
	} else {
		fmt.Printf("%-15s %-15s %-12s\n", "Feature 1", "Feature 2", "Correlation")
		fmt.Println("─────────────────────────────────────────────")
		for _, pair := range strongCorrs {
			direction := "↗️"
			if pair.Correlation < 0 {
				direction = "↘️"
			}
			fmt.Printf("%-15s %-15s %+.3f %s\n",
				pair.Feature1, pair.Feature2, pair.Correlation, direction)
		}
	}

	// Correlations with a specific feature
	fmt.Println("\n┌─────────────────────────────────────────────────────────────────┐")
	fmt.Println("│            CORRELATIONS WITH 'Income'                          │")
	fmt.Println("└─────────────────────────────────────────────────────────────────┘")
	fmt.Println()

	incomeCorrs := result.GetCorrelationsWithFeature("Income")
	fmt.Printf("%-15s %-12s\n", "Feature", "Correlation")
	fmt.Println("─────────────────────────────")
	for _, pair := range incomeCorrs {
		fmt.Printf("%-15s %+.3f\n", pair.Feature2, pair.Correlation)
	}

	// Export to JSON (WASM-compatible)
	fmt.Println("\n┌─────────────────────────────────────────────────────────────────┐")
	fmt.Println("│              JSON EXPORT (WASM-Compatible)                      │")
	fmt.Println("└─────────────────────────────────────────────────────────────────┘")
	fmt.Println()

	jsonStr, err := result.ToJSONCompact()
	if err != nil {
		fmt.Printf("❌ JSON export error: %v\n", err)
	} else {
		fmt.Printf("✅ JSON size: %d bytes\n", len(jsonStr))
		fmt.Println("   (Use result.ToJSON() for pretty-printed version)")
		fmt.Println()
		// Show first 200 chars
		if len(jsonStr) > 200 {
			fmt.Printf("   Preview: %s...\n", jsonStr[:200])
		} else {
			fmt.Printf("   Preview: %s\n", jsonStr)
		}
	}

	// Summary
	fmt.Println("\n╔════════════════════════════════════════════════════════════════╗")
	fmt.Println("║                        SUMMARY                                 ║")
	fmt.Println("╠════════════════════════════════════════════════════════════════╣")
	fmt.Printf("║  Samples:     %-48d ║\n", result.Correlation.Samples)
	fmt.Printf("║  Features:    %-48d ║\n", result.Correlation.N)
	fmt.Printf("║  Strong pairs: %-47d ║\n", len(strongCorrs))
	fmt.Println("╚════════════════════════════════════════════════════════════════╝")
}

// generateSampleData creates synthetic data with known correlations
func generateSampleData(n int) [][]float32 {
	data := make([][]float32, n)

	for i := 0; i < n; i++ {
		// Age: 22-65
		age := float32(22 + rand.Float64()*43)

		// Experience: highly correlated with Age (r ≈ 0.85)
		experience := (age-22)*0.8 + float32(rand.Float64()*5)

		// Income: correlated with Experience and Age (r ≈ 0.7)
		income := experience*3000 + age*500 + float32(rand.Float64()*20000)

		// Education: weakly correlated with Income (r ≈ 0.3)
		education := float32(12 + rand.Float64()*8 + float64(income)/100000)

		// Satisfaction: slightly negative correlation with Age (r ≈ -0.2)
		satisfaction := float32(80) - age*0.3 + float32(rand.Float64()*30)

		data[i] = []float32{age, income, experience, education, satisfaction}
	}

	return data
}
