package main

import (
	"fmt"
	"math"
	"os"
	"path/filepath"

	"github.com/openfluke/loom/nn"
)

// TestResult holds the result of a dtype test
type TestResult struct {
	DType   string
	Passed  bool
	MaxDiff float32
	Error   string
}

// AllSafetensorsDTypes returns all supported SafeTensors dtypes
func AllSafetensorsDTypes() []string {
	return []string{
		"F32", "F64", "F16", "BF16", "F4",
		"I8", "I16", "I32", "I64",
		"U8", "U16", "U32", "U64",
	}
}

// getToleranceForDType returns acceptable tolerance for each dtype
func getToleranceForDType(dtype string) float32 {
	switch dtype {
	case "F32":
		return 1e-6 // High precision
	case "F64":
		return 1e-6 // High precision (downcast to F32 so same tolerance)
	case "F16":
		return 1e-3 // Lower precision
	case "BF16":
		return 1e-2 // Even lower precision
	case "F4":
		return 0.5 // Very coarse quantization
	case "I8", "U8":
		return 1.0 // Integer rounding
	case "I16", "U16":
		return 1.0
	case "I32", "U32":
		return 1.0
	case "I64", "U64":
		return 1.0
	default:
		return 0.01
	}
}

// generateTestData creates test data with various patterns
func generateTestData() []float32 {
	data := make([]float32, 100)
	for i := range data {
		switch {
		case i < 10:
			// Small values
			data[i] = float32(i) * 0.1
		case i < 20:
			// Negative values
			data[i] = -float32(i-10) * 0.5
		case i < 30:
			// Values around 1.0
			data[i] = 1.0 + float32(i-20)*0.01
		case i < 40:
			// Larger values
			data[i] = float32(i-30) * 10.0
		case i < 50:
			// Very small values
			data[i] = float32(i-40) * 0.001
		case i < 60:
			// Powers of 2
			data[i] = float32(math.Pow(2, float64(i-50)))
		case i < 70:
			// Fractional values
			data[i] = 1.0 / float32(i-59)
		default:
			// Mixed
			data[i] = float32(math.Sin(float64(i))) * 10.0
		}
	}
	return data
}

// testDType tests save and load for a specific dtype
func testDType(dtype string) TestResult {
	result := TestResult{
		DType:  dtype,
		Passed: false,
	}

	// Generate test data
	originalData := generateTestData()

	// Create tensor
	tensor := nn.TensorWithShape{
		Values: originalData,
		Shape:  []int{10, 10}, // 2D tensor
		DType:  dtype,
	}

	// Create tensors map
	tensors := map[string]nn.TensorWithShape{
		"test_tensor": tensor,
	}

	// Save to temporary file
	tmpFile := filepath.Join(os.TempDir(), fmt.Sprintf("test_%s.safetensors", dtype))
	defer os.Remove(tmpFile)

	err := nn.SaveSafetensors(tmpFile, tensors)
	if err != nil {
		result.Error = fmt.Sprintf("Save failed: %v", err)
		return result
	}

	// Load back
	data, err := os.ReadFile(tmpFile)
	if err != nil {
		result.Error = fmt.Sprintf("Read file failed: %v", err)
		return result
	}

	loadedTensors, err := nn.LoadSafetensorsWithShapes(data)
	if err != nil {
		result.Error = fmt.Sprintf("Load failed: %v", err)
		return result
	}

	// Verify tensor exists
	loadedTensor, ok := loadedTensors["test_tensor"]
	if !ok {
		result.Error = "Tensor not found in loaded file"
		return result
	}

	// Verify shape
	if len(loadedTensor.Shape) != len(tensor.Shape) {
		result.Error = fmt.Sprintf("Shape mismatch: expected %v, got %v",
			tensor.Shape, loadedTensor.Shape)
		return result
	}

	for i := range tensor.Shape {
		if tensor.Shape[i] != loadedTensor.Shape[i] {
			result.Error = fmt.Sprintf("Shape mismatch at dim %d: expected %d, got %d",
				i, tensor.Shape[i], loadedTensor.Shape[i])
			return result
		}
	}

	// Verify dtype
	if loadedTensor.DType != dtype {
		result.Error = fmt.Sprintf("DType mismatch: expected %s, got %s",
			dtype, loadedTensor.DType)
		return result
	}

	// Verify values within tolerance
	if len(loadedTensor.Values) != len(originalData) {
		result.Error = fmt.Sprintf("Data length mismatch: expected %d, got %d",
			len(originalData), len(loadedTensor.Values))
		return result
	}

	tolerance := getToleranceForDType(dtype)
	var maxDiff float32

	for i := range originalData {
		diff := float32(math.Abs(float64(loadedTensor.Values[i] - originalData[i])))
		if diff > maxDiff {
			maxDiff = diff
		}

		if diff > tolerance {
			result.Error = fmt.Sprintf("Value mismatch at index %d: expected %.6f, got %.6f (diff %.6f > tolerance %.6f)",
				i, originalData[i], loadedTensor.Values[i], diff, tolerance)
			result.MaxDiff = maxDiff
			return result
		}
	}

	result.MaxDiff = maxDiff
	result.Passed = true
	return result
}

// runAllTests runs tests for all dtypes
func runAllTests() []TestResult {
	dtypes := AllSafetensorsDTypes()
	results := make([]TestResult, len(dtypes))

	for i, dtype := range dtypes {
		fmt.Printf("Testing %s... ", dtype)
		results[i] = testDType(dtype)
		if results[i].Passed {
			fmt.Printf("✅ PASS (max diff: %.6f)\n", results[i].MaxDiff)
		} else {
			fmt.Printf("❌ FAIL: %s\n", results[i].Error)
		}
	}

	return results
}

// printSummary prints a summary of all tests
func printSummary(results []TestResult) {
	fmt.Println("\n╔══════════════════════════════════════════════════════════════╗")
	fmt.Println("║          SafeTensors Multi-Precision Test Summary           ║")
	fmt.Println("╚══════════════════════════════════════════════════════════════╝")
	fmt.Println()

	passed := 0
	failed := 0

	fmt.Printf("%-10s | %-10s | %-12s | %s\n", "DType", "Status", "Max Diff", "Error")
	fmt.Println("───────────────────────────────────────────────────────────────")

	for _, result := range results {
		status := "✅ PASS"
		if !result.Passed {
			status = "❌ FAIL"
			failed++
		} else {
			passed++
		}

		errorMsg := ""
		if result.Error != "" {
			// Truncate long errors
			errorMsg = result.Error
			if len(errorMsg) > 40 {
				errorMsg = errorMsg[:37] + "..."
			}
		}

		fmt.Printf("%-10s | %-10s | %-12.6f | %s\n",
			result.DType, status, result.MaxDiff, errorMsg)
	}

	fmt.Println("───────────────────────────────────────────────────────────────")
	fmt.Printf("\nTotal: %d tests | ✅ Passed: %d | ❌ Failed: %d\n\n",
		len(results), passed, failed)

	if failed == 0 {
		fmt.Println("🎉 All tests passed!")
	} else {
		fmt.Printf("⚠️  %d test(s) failed\n", failed)
	}
}

func main() {
	fmt.Println("╔══════════════════════════════════════════════════════════════╗")
	fmt.Println("║     SafeTensors Multi-Precision Save/Load Test              ║")
	fmt.Println("╚══════════════════════════════════════════════════════════════╝")
	fmt.Println()

	results := runAllTests()
	printSummary(results)
}
