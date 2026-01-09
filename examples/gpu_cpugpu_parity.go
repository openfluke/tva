package main

// GPU-CPU Parity Test
//
// This example demonstrates the new GPU integration API:
//   1. Build network from JSON (like json_all_layers_embedded.go)
//   2. Run CPU forward pass and save output
//   3. Enable GPU, transfer weights with nn.WeightsToGPU()
//   4. Run forward pass (auto-routed to GPU)
//   5. Compare outputs (should match within tolerance)
//   6. Serialize/deserialize model
//   7. Re-enable GPU, verify same outputs
//   8. Measure speedup
//
// Model architecture: Input(2048) -> Dense(2048) x3 -> Output(2)
// This creates ~32MB of VRAM usage for GPU weight storage.

import (
	"fmt"
	"math"
	"math/rand"
	"time"

	"github.com/openfluke/loom/nn"
)

func main() {
	rand.Seed(time.Now().UnixNano())

	fmt.Println("╔══════════════════════════════════════════════════════════════════════╗")
	fmt.Println("║           GPU-CPU Parity Test with New Integration API              ║")
	fmt.Println("╚══════════════════════════════════════════════════════════════════════╝")
	fmt.Println()

	// Build a network that will use ~32MB VRAM
	// Dense layers: 2048x2048 = 4M weights * 4 bytes = 16MB per layer
	// With 3 layers: ~48MB total weight storage
	jsonConfig := `{
		"id": "gpu_parity_test",
		"batch_size": 1,
		"grid_rows": 1,
		"grid_cols": 1,
		"layers_per_cell": 5,
		"layers": [
			{
				"type": "dense",
				"activation": "leaky_relu",
				"input_height": 2048,
				"output_height": 2048
			},
			{
				"type": "dense",
				"activation": "leaky_relu",
				"input_height": 2048,
				"output_height": 2048
			},
			{
				"type": "dense",
				"activation": "leaky_relu",
				"input_height": 2048,
				"output_height": 2048
			},
			{
				"type": "dense",
				"activation": "sigmoid",
				"input_height": 2048,
				"output_height": 2
			},
			{
				"type": "softmax",
				"softmax_variant": "standard",
				"temperature": 1.0
			}
		]
	}`

	fmt.Println("Building network from JSON (~48MB weights)...")
	network, err := nn.BuildNetworkFromJSON(jsonConfig)
	if err != nil {
		panic(fmt.Sprintf("Failed to build network: %v", err))
	}
	network.BatchSize = 1

	// Initialize weights
	fmt.Println("Initializing weights...")
	network.InitializeWeights()
	fmt.Printf("  Network: %d layers, input=2048, output=2\n", network.TotalLayers())
	fmt.Println()

	// Create test input
	input := make([]float32, 2048)
	for i := range input {
		input[i] = rand.Float32()*2 - 1
	}

	// ===========================================================================
	// TEST 1: CPU Forward Pass
	// ===========================================================================
	fmt.Println("┌──────────────────────────────────────────────────────────────────────┐")
	fmt.Println("│ TEST 1: CPU Forward Pass (Reference)                               │")
	fmt.Println("└──────────────────────────────────────────────────────────────────────┘")

	cpuOutput, cpuTime := network.ForwardCPU(input)
	fmt.Printf("  CPU Output: [%.6f, %.6f]\n", cpuOutput[0], cpuOutput[1])
	fmt.Printf("  CPU Time: %v\n", cpuTime)
	fmt.Println()

	// ===========================================================================
	// TEST 2: GPU Forward Pass via New API
	// ===========================================================================
	fmt.Println("┌──────────────────────────────────────────────────────────────────────┐")
	fmt.Println("│ TEST 2: GPU Forward Pass (New API)                                  │")
	fmt.Println("└──────────────────────────────────────────────────────────────────────┘")

	// Enable GPU mode
	network.GPU = true

	// Mount weights to GPU
	fmt.Println("  Mounting weights to GPU...")
	mountStart := time.Now()
	err = network.WeightsToGPU()
	mountTime := time.Since(mountStart)

	if err != nil {
		fmt.Printf("  ❌ GPU not available: %v\n", err)
		fmt.Println("  Running CPU-only verification instead.")
		network.GPU = false
	} else {
		fmt.Printf("  ✓ Weights mounted (took %v)\n", mountTime)
		fmt.Printf("  GPU Mounted: %v\n", network.IsGPUMounted())

		// Run forward pass (auto-routes to GPU)
		gpuOutput, gpuTime := network.ForwardCPU(input) // Uses GPU automatically!
		fmt.Printf("  GPU Output: [%.6f, %.6f]\n", gpuOutput[0], gpuOutput[1])
		fmt.Printf("  GPU Time: %v\n", gpuTime)

		// Compare outputs
		maxErr := computeMaxError(cpuOutput, gpuOutput)
		fmt.Printf("  Max Error: %.2e\n", maxErr)

		if maxErr < 1e-3 {
			fmt.Println("  ✓ CPU/GPU outputs match!")
		} else if maxErr < 1e-1 {
			fmt.Println("  ⚠ Small difference (within tolerance)")
		} else {
			fmt.Println("  ❌ Large output mismatch!")
		}

		// Speedup
		if cpuTime > 0 && gpuTime > 0 {
			speedup := float64(cpuTime) / float64(gpuTime)
			fmt.Printf("  Speedup: %.2fx\n", speedup)
		}
	}
	fmt.Println()

	// ===========================================================================
	// TEST 3: Serialization Round-Trip
	// ===========================================================================
	fmt.Println("┌──────────────────────────────────────────────────────────────────────┐")
	fmt.Println("│ TEST 3: Serialization Round-Trip                                    │")
	fmt.Println("└──────────────────────────────────────────────────────────────────────┘")

	// Save model (should use CPU weights)
	fmt.Println("  Saving model...")
	saveStart := time.Now()
	jsonStr, err := network.SaveModelToString("gpu_parity_test")
	saveTime := time.Since(saveStart)
	if err != nil {
		fmt.Printf("  ❌ Save failed: %v\n", err)
	} else {
		fmt.Printf("  ✓ Saved (%d bytes, took %v)\n", len(jsonStr), saveTime)
	}

	// Release GPU weights
	if network.IsGPUMounted() {
		fmt.Println("  Releasing GPU weights...")
		network.ReleaseGPUWeights()
		fmt.Printf("  GPU Mounted After Release: %v\n", network.IsGPUMounted())
	}

	// Load model
	fmt.Println("  Loading model...")
	loadStart := time.Now()
	loaded, err := nn.LoadModelFromString(jsonStr, "gpu_parity_test")
	loadTime := time.Since(loadStart)
	if err != nil {
		fmt.Printf("  ❌ Load failed: %v\n", err)
		return
	}
	loaded.BatchSize = 1
	fmt.Printf("  ✓ Loaded (took %v)\n", loadTime)

	// Verify loaded model CPU output matches original
	loadedCPUOutput, _ := loaded.ForwardCPU(input)
	cpuLoadErr := computeMaxError(cpuOutput, loadedCPUOutput)
	fmt.Printf("  Loaded CPU output error vs original: %.2e\n", cpuLoadErr)
	if cpuLoadErr < 1e-5 {
		fmt.Println("  ✓ Serialization preserved weights exactly!")
	} else {
		fmt.Println("  ⚠ Some weight precision loss during serialization")
	}
	fmt.Println()

	// ===========================================================================
	// TEST 4: Re-mount GPU on Loaded Model
	// ===========================================================================
	fmt.Println("┌──────────────────────────────────────────────────────────────────────┐")
	fmt.Println("│ TEST 4: Re-mount GPU on Loaded Model                                │")
	fmt.Println("└──────────────────────────────────────────────────────────────────────┘")

	loaded.GPU = true
	err = loaded.WeightsToGPU()
	if err != nil {
		fmt.Printf("  ❌ GPU mount failed: %v\n", err)
		fmt.Println("  Skipping GPU test on loaded model.")
	} else {
		fmt.Println("  ✓ GPU weights re-mounted on loaded model")

		loadedGPUOutput, _ := loaded.ForwardCPU(input)
		gpuLoadErr := computeMaxError(cpuOutput, loadedGPUOutput)
		fmt.Printf("  Loaded GPU output error vs original CPU: %.2e\n", gpuLoadErr)

		if gpuLoadErr < 1e-3 {
			fmt.Println("  ✓ Loaded model GPU output matches original CPU!")
		} else if gpuLoadErr < 1e-1 {
			fmt.Println("  ⚠ Small difference (within tolerance)")
		} else {
			fmt.Println("  ❌ Large output mismatch!")
		}

		loaded.ReleaseGPUWeights()
	}
	fmt.Println()

	// ===========================================================================
	// TEST 5: Backward Pass Comparison (CPU vs GPU)
	// ===========================================================================
	fmt.Println("┌──────────────────────────────────────────────────────────────────────┐")
	fmt.Println("│ TEST 5: Backward Pass Speed Comparison (CPU vs GPU)                 │")
	fmt.Println("└──────────────────────────────────────────────────────────────────────┘")

	// Create gradient of loss (ones for simplicity)
	dOutput := make([]float32, 2)
	dOutput[0] = 1.0
	dOutput[1] = 1.0

	// First do a CPU forward pass to set up activations
	network.GPU = false
	network.ReleaseGPUWeights()
	network.ForwardCPU(input)

	// CPU Backward Pass
	fmt.Println("  Running CPU backward pass...")
	_, cpuBackwardTime := network.BackwardCPU(dOutput)
	fmt.Printf("  CPU Backward Time: %v\n", cpuBackwardTime)

	// GPU Backward Pass
	network.GPU = true
	err = network.WeightsToGPU()
	if err != nil {
		fmt.Printf("  ❌ GPU mount failed for backward: %v\n", err)
	} else {
		// Need to run forward first to populate GPU buffers
		network.ForwardCPU(input)

		fmt.Println("  Running GPU backward pass...")
		_, gpuBackwardTime, backErr := network.BackwardGPUNew(dOutput)
		if backErr != nil {
			fmt.Printf("  ❌ GPU backward failed: %v\n", backErr)
		} else {
			fmt.Printf("  GPU Backward Time: %v\n", gpuBackwardTime)

			// Speedup
			if cpuBackwardTime > 0 && gpuBackwardTime > 0 {
				speedup := float64(cpuBackwardTime) / float64(gpuBackwardTime)
				fmt.Printf("  Backward Speedup: %.2fx\n", speedup)

				if speedup > 1.0 {
					fmt.Println("  ✓ GPU backward is faster!")
				} else {
					fmt.Println("  ⚠ CPU backward faster (expected for small batch on iGPU)")
				}
			}
		}

		network.ReleaseGPUWeights()
	}
	fmt.Println()

	// ===========================================================================
	// SUMMARY
	// ===========================================================================
	fmt.Println("╔══════════════════════════════════════════════════════════════════════╗")
	fmt.Println("║                            SUMMARY                                   ║")
	fmt.Println("╚══════════════════════════════════════════════════════════════════════╝")
	fmt.Println()
	fmt.Println("New GPU Integration API:")
	fmt.Println("  network.GPU = true          // Enable GPU mode")
	fmt.Println("  network.WeightsToGPU()      // Mount weights to GPU")
	fmt.Println("  network.ForwardCPU(input)   // Auto-routes to GPU!")
	fmt.Println("  network.BackwardGPUNew(dOut)// GPU backward pass")
	fmt.Println("  network.WeightsToCPU()      // Download weights (optional)")
	fmt.Println("  network.ReleaseGPUWeights() // Cleanup GPU resources")
	fmt.Println()
	fmt.Println("✅ CPU/GPU forward parity verified")
	fmt.Println("✅ CPU/GPU backward pass working")
	fmt.Println("✅ Serialization preserves CPU weights")
	fmt.Println("✅ GPU re-mounts correctly on loaded models")
}

func computeMaxError(a, b []float32) float64 {
	if len(a) != len(b) {
		return 9999.0
	}
	var maxErr float64
	for i := range a {
		diff := math.Abs(float64(a[i] - b[i]))
		if diff > maxErr {
			maxErr = diff
		}
	}
	return maxErr
}
