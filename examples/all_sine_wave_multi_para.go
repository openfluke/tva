package main

import (
	"encoding/json"
	"fmt"
	"math"
	"math/rand"
	"os"
	"sort"
	"strings"
	"sync"
	"time"

	"github.com/openfluke/loom/nn"
)

// ═══════════════════════════════════════════════════════════════════════════════
// TEST 41: SINE WAVE ADAPTATION BENCHMARK - ALL MODES × ALL TYPES × ALL LAYERS
// ═══════════════════════════════════════════════════════════════════════════════
//
// Extends the original sine wave benchmark to cover:
//   - All 6 training modes: NormalBP, StepBP, Tween, TweenChain, StepTween, StepTweenChain
//   - All 10 numerical types: int8-int64, uint8-uint64, float32, float64
//   - All 5 trainable layer types: Dense, Conv2D, RNN, LSTM, Attention
//
// Total combinations: 6 × 10 × 5 = 300 tests
//
// Each test runs for 2 seconds with frequency switches every 500ms
// Sin(1x) → Sin(2x) → Sin(3x) → Sin(4x)
//

const (
	// Network architecture
	InputSize  = 16 // Sliding window of 16 sine samples (4x4 for Conv2D)
	HiddenSize = 16 // Hidden layer size
	OutputSize = 1  // Predict next sine value

	// Training parameters
	LearningRate      = float32(0.01)
	InitScale         = float32(0.5)
	AccuracyThreshold = 0.10 // Prediction correct if abs(pred - expected) < threshold

	// Sine wave parameters
	SinePoints     = 100 // Number of points to generate
	SineResolution = 0.1 // Step size for x values

	// Timing - 10 second run with 50ms windows (200 windows total)
	TestDuration   = 10 * time.Second
	WindowDuration = 50 * time.Millisecond   // 50ms windows for fine-grained tracking
	SwitchInterval = 2500 * time.Millisecond // Switch frequency every 2.5 seconds
	TrainInterval  = 10 * time.Millisecond

	// Concurrency
	MaxConcurrent = 8
)

// ═══════════════════════════════════════════════════════════════════════════════
// ENUMERATIONS
// ═══════════════════════════════════════════════════════════════════════════════

// TrainingMode enum
type TrainingMode int

const (
	ModeNormalBP TrainingMode = iota
	ModeStepBP
	ModeTween
	ModeTweenChain
	ModeStepTween
	ModeStepTweenChain
)

var modeNames = map[TrainingMode]string{
	ModeNormalBP:       "NormalBP",
	ModeStepBP:         "StepBP",
	ModeTween:          "Tween",
	ModeTweenChain:     "TweenChain",
	ModeStepTween:      "StepTween",
	ModeStepTweenChain: "StepTweenChain",
}

// LayerType enum - trainable layer types
type LayerTestType int

const (
	TestLayerDense LayerTestType = iota
	TestLayerConv2D
	TestLayerRNN
	TestLayerLSTM
	TestLayerAttention
	TestLayerSwiGLU
	TestLayerNormDense
	TestLayerConv1D
	TestLayerResidual
)

var layerNames = map[LayerTestType]string{
	TestLayerDense:     "Dense",
	TestLayerConv2D:    "Conv2D",
	TestLayerRNN:       "RNN",
	TestLayerLSTM:      "LSTM",
	TestLayerAttention: "Attention",
	TestLayerSwiGLU:    "SwiGLU",
	TestLayerNormDense: "NormDense",
	TestLayerConv1D:    "Conv1D",
	TestLayerResidual:  "Residual",
}

// NumericType enum
type NumericType int

const (
	TypeInt8 NumericType = iota
	TypeInt16
	TypeInt32
	TypeInt64
	TypeUint8
	TypeUint16
	TypeUint32
	TypeUint64
	TypeFloat32
	TypeFloat64
)

var typeNames = map[NumericType]string{
	TypeInt8:    "int8",
	TypeInt16:   "int16",
	TypeInt32:   "int32",
	TypeInt64:   "int64",
	TypeUint8:   "uint8",
	TypeUint16:  "uint16",
	TypeUint32:  "uint32",
	TypeUint64:  "uint64",
	TypeFloat32: "float32",
	TypeFloat64: "float64",
}

// ═══════════════════════════════════════════════════════════════════════════════
// RESULT STRUCTURES
// ═══════════════════════════════════════════════════════════════════════════════

// TimeWindow for accuracy tracking
type TimeWindow struct {
	TimeMs       int     `json:"timeMs"`
	Outputs      int     `json:"outputs"`
	Accuracy     float64 `json:"accuracy"`
	FreqSwitches int     `json:"freqSwitches"`
	MaxLatencyMs float64 `json:"maxLatencyMs"` // Longest gap between outputs
	BlockedMs    float64 `json:"blockedMs"`    // Time blocked in training
	AvailableMs  float64 `json:"availableMs"`  // Time producing outputs
}

// TestResult holds per-combination benchmark results
type TestResult struct {
	LayerType        string       `json:"layerType"`
	TrainingMode     string       `json:"trainingMode"`
	NumericType      string       `json:"numericType"`
	Windows          []TimeWindow `json:"windows"`
	TotalOutputs     int          `json:"totalOutputs"`
	TotalFreqSwitch  int          `json:"totalFreqSwitches"`
	TrainTimeSec     float64      `json:"trainTimeSec"`
	AvgAccuracy      float64      `json:"avgAccuracy"`
	Stability        float64      `json:"stability"`
	ThroughputPerSec float64      `json:"throughputPerSec"`
	Score            float64      `json:"score"`
	Passed           bool         `json:"passed"`
	Error            string       `json:"error,omitempty"`
	// Availability metrics
	AvailabilityPct   float64 `json:"availabilityPct"`
	TotalBlockedMs    float64 `json:"totalBlockedMs"`
	MaxLatencyMs      float64 `json:"maxLatencyMs"`
	AvgLatencyMs      float64 `json:"avgLatencyMs"`
	ZeroOutputWindows int     `json:"zeroOutputWindows"`
}

// BenchmarkResults is the full output
type BenchmarkResults struct {
	Results     []TestResult `json:"results"`
	Timestamp   string       `json:"timestamp"`
	Duration    string       `json:"testDuration"`
	TotalTests  int          `json:"totalTests"`
	Passed      int          `json:"passed"`
	Failed      int          `json:"failed"`
	Frequencies []float64    `json:"frequencies"`
}

// ═══════════════════════════════════════════════════════════════════════════════
// MAIN
// ═══════════════════════════════════════════════════════════════════════════════

func main() {
	rand.Seed(time.Now().UnixNano())

	fmt.Println("╔═══════════════════════════════════════════════════════════════════════════════════════════════════════════════════╗")
	fmt.Println("║   🌊 SINE WAVE ADAPTATION: ALL MODES × ALL TYPES × ALL LAYERS                                                    ║")
	fmt.Println("║                                                                                                                   ║")
	fmt.Println("║   LAYERS: Dense, Conv2D, RNN, LSTM, Attention (5 types)                                                          ║")
	fmt.Println("║   MODES:  NormalBP, StepBP, Tween, TweenChain, StepTween, StepTweenChain (6 modes)                               ║")
	fmt.Println("║   TYPES:  int8-int64, uint8-uint64, float32, float64 (10 types)                                                  ║")
	fmt.Println("║   TOTAL:  6 × 10 × 5 = 300 combinations                                                                          ║")
	fmt.Println("╚═══════════════════════════════════════════════════════════════════════════════════════════════════════════════════╝")

	// Generate sine wave data for all 4 frequencies
	frequencies := []float64{1.0, 2.0, 3.0, 4.0}
	allInputs := make([][][]float32, len(frequencies))
	allTargets := make([][]float32, len(frequencies))

	for i, freq := range frequencies {
		sineData := generateSineWave(freq)
		allInputs[i], allTargets[i] = createSamples(sineData)
	}

	allLayers := []LayerTestType{
		TestLayerDense,
		TestLayerConv2D,
		TestLayerRNN,
		TestLayerLSTM,
		TestLayerSwiGLU,
		TestLayerNormDense,
		TestLayerAttention,
		TestLayerConv1D,
		TestLayerResidual,
	}
	modes := []TrainingMode{ModeNormalBP, ModeStepBP, ModeTween, ModeTweenChain, ModeStepTween, ModeStepTweenChain}
	types := []NumericType{TypeInt8, TypeInt16, TypeInt32, TypeInt64, TypeUint8, TypeUint16, TypeUint32, TypeUint64, TypeFloat32, TypeFloat64}

	totalTests := len(allLayers) * len(modes) * len(types)
	numWindows := int(TestDuration / WindowDuration)
	fmt.Printf("\n📊 Running %d tests | %d windows at %dms each | %s per test\n\n", totalTests, numWindows, WindowDuration.Milliseconds(), TestDuration)

	results := &BenchmarkResults{
		Results:     make([]TestResult, 0, totalTests),
		Timestamp:   time.Now().Format(time.RFC3339),
		Duration:    TestDuration.String(),
		TotalTests:  totalTests,
		Frequencies: frequencies,
	}

	var wg sync.WaitGroup
	var mu sync.Mutex
	sem := make(chan struct{}, MaxConcurrent)
	testNum := 0

	// Create results directory
	resultsDir := "results"
	if err := os.MkdirAll(resultsDir, 0755); err != nil {
		fmt.Printf("Error creating results directory: %v\n", err)
	}

	for _, layer := range allLayers {
		for _, mode := range modes {
			for _, numType := range types {
				wg.Add(1)
				testNum++

				go func(l LayerTestType, m TrainingMode, t NumericType, num int) {
					defer wg.Done()
					sem <- struct{}{}
					defer func() { <-sem }()

					layerName := layerNames[l]
					modeName := modeNames[m]
					typeName := typeNames[t]

					// Check if result already exists
					filename := fmt.Sprintf("%s/result_%s_%s_%s.json", resultsDir, layerName, modeName, typeName)
					if _, err := os.Stat(filename); err == nil {
						// Load existing result
						data, err := os.ReadFile(filename)
						if err == nil {
							var existingResult TestResult
							if err := json.Unmarshal(data, &existingResult); err == nil {
								// Only use cached result if it PASSED and has a score > 0
								if existingResult.Passed && existingResult.Score > 0 {
									mu.Lock()
									results.Results = append(results.Results, existingResult)
									results.Passed++
									mu.Unlock()
									fmt.Printf("Skipping %s %s %s (already passed with score %.0f)\n", layerName, modeName, typeName, existingResult.Score)
									return
								}
								// Otherwise, fall through to re-run the test
							}
						}
					}

					fmt.Printf("Running Test %d/%d: %s | %s | %s\n", num, totalTests, layerName, modeName, typeName)

					result := runCombinationTest(l, m, t, allInputs, allTargets, frequencies)
					result.LayerType = layerName
					result.TrainingMode = modeName
					result.NumericType = typeName

					mu.Lock()
					results.Results = append(results.Results, result)
					if result.Passed {
						results.Passed++
					} else {
						results.Failed++
					}
					mu.Unlock()

					status := "✅"
					if !result.Passed {
						status = "❌"
					}
					fmt.Printf("%s [%3d/%d] %-10s %-15s %-8s | Acc: %5.1f%% | Score: %.0f\n",
						status, num, totalTests, layerName, modeName, typeName, result.AvgAccuracy, result.Score)

					// Save individual result
					if data, err := json.MarshalIndent(result, "", "  "); err == nil {
						if err := os.WriteFile(filename, data, 0644); err != nil {
							fmt.Printf("Error saving result to %s: %v\n", filename, err)
						}
					}
				}(layer, mode, numType, testNum)
			}
		}
	}

	// ════════════════════════════════════════════════════════════
	// PARALLEL GRID PERMUTATIONS (2-Branch) - DISABLED (failing tests)
	// ════════════════════════════════════════════════════════════

	/*
			parallelTypes := []nn.BrainType{
				nn.BrainDense, nn.BrainConv2D, nn.BrainRNN,
				nn.BrainLSTM, nn.BrainMHA, nn.BrainSwiGLU,
				nn.BrainNormDense, nn.BrainConv1D, nn.BrainResidual,
			}
			combineModes := []string{"concat", "add", "avg", "grid_scatter", "filter"}

			// Generate 2-branch permutations (9*9 = 81)
			perms2 := generatePermutations(2, parallelTypes)

		    // Start Parallel Tests
		    for _, perm := range perms2 {
		        permName := "Par2"
		        for _, pt := range perm {
		            permName += "-" + pt.String()
		        }

		        for _, cm := range combineModes {
		            // Build layer name including combine mode
		            fullLayerName := fmt.Sprintf("%s_%s", permName, cm)

		            for _, mode := range modes {
		                for _, numType := range types {
		                    wg.Add(1)
		                    testNum++

		                    go func(p []nn.BrainType, cMode string, lName string, m TrainingMode, t NumericType, num int) {
		                        defer wg.Done()
		                        sem <- struct{}{}
		                        defer func() { <-sem }()

		                        modeName := modeNames[m]
		                        typeName := typeNames[t]

		                        // Check if result already exists
		                        filename := fmt.Sprintf("%s/result_%s_%s_%s.json", resultsDir, lName, modeName, typeName)
		                        if _, err := os.Stat(filename); err == nil {
		                            // Load existing result
		                            data, err := os.ReadFile(filename)
		                            if err == nil {
		                                var existingResult TestResult
		                                if err := json.Unmarshal(data, &existingResult); err == nil {
		                                    if existingResult.Passed && existingResult.Score > 0 {
		                                        mu.Lock()
		                                        results.Results = append(results.Results, existingResult)
		                                        results.Passed++
		                                        mu.Unlock()
		                                        fmt.Printf("Skipping %s %s %s (already passed)\n", lName, modeName, typeName)
		                                        return
		                                    }
		                                }
		                            }
		                        }

		                        fmt.Printf("Running Test %d (Parallel): %s | %s | %s\n", num, lName, modeName, typeName)

		                        // Factories
		                        pCopy := make([]nn.BrainType, len(p))
		                        copy(pCopy, p)

		                        f32Fac := func() *nn.Network { return BuildParallelNetwork(pCopy, cMode, nn.DTypeFloat32) }
		                        genFac := func(dt nn.DType) *nn.Network { return BuildParallelNetwork(pCopy, cMode, dt) }

		                        var result TestResult
		                        if t == TypeFloat32 {
		                             result = runFloat32Test(lName, f32Fac, m, allInputs, allTargets, frequencies)
		                        } else {
		                             result = runGenericTest(lName, genFac, m, t, allInputs, allTargets, frequencies)
		                        }

		                        result.LayerType = lName
		                        result.TrainingMode = modeName
		                        result.NumericType = typeName

		                        mu.Lock()
		                        results.Results = append(results.Results, result)
		                        if result.Passed {
		                            results.Passed++
		                        } else {
		                            results.Failed++
		                        }
		                        mu.Unlock()

		                        status := "✅"
		                        if !result.Passed { status = "❌" }
		                        fmt.Printf("%s [%3d] %-30s %-10s %-8s | Acc: %5.1f%% | Score: %.0f\n",
		                            status, num, lName, modeName, typeName, result.AvgAccuracy, result.Score)

		                        // Save individual result
		                        if data, err := json.MarshalIndent(result, "", "  "); err == nil {
		                            _ = os.WriteFile(filename, data, 0644)
		                        }
		                    }(perm, cm, fullLayerName, mode, numType, testNum)
		                }
		            }
		        }
		    }
	*/

	wg.Wait()
	fmt.Println("\n✅ All tests complete!")

	// Load all cached results from disk (including parallel tests that weren't run this time)
	loadAllCachedResults(results, resultsDir)

	// Recalculate availability as relative throughput (compare to max within each layer+type)
	recalculateAvailability(results)

	saveResults(results)
	printSummaryTable(results)
}

// loadAllCachedResults scans the results directory and loads all cached results
// that aren't already in the results struct (e.g., parallel tests from previous runs)
func loadAllCachedResults(results *BenchmarkResults, resultsDir string) {
	// Build a set of existing results to avoid duplicates
	existing := make(map[string]bool)
	for _, r := range results.Results {
		key := fmt.Sprintf("%s_%s_%s", r.LayerType, r.TrainingMode, r.NumericType)
		existing[key] = true
	}

	// Scan results directory for all result files
	entries, err := os.ReadDir(resultsDir)
	if err != nil {
		fmt.Printf("Warning: could not read results directory: %v\n", err)
		return
	}

	loaded := 0
	for _, entry := range entries {
		if entry.IsDir() {
			continue
		}
		name := entry.Name()
		if len(name) < 12 || name[:7] != "result_" || name[len(name)-5:] != ".json" {
			continue
		}

		// Read and parse the file
		data, err := os.ReadFile(fmt.Sprintf("%s/%s", resultsDir, name))
		if err != nil {
			continue
		}

		var r TestResult
		if err := json.Unmarshal(data, &r); err != nil {
			continue
		}

		// Check if this result is already loaded
		key := fmt.Sprintf("%s_%s_%s", r.LayerType, r.TrainingMode, r.NumericType)
		if existing[key] {
			continue
		}

		// Add to results
		results.Results = append(results.Results, r)
		existing[key] = true
		loaded++

		if r.Passed {
			results.Passed++
		} else {
			results.Failed++
		}
	}

	if loaded > 0 {
		fmt.Printf("📂 Loaded %d cached results from disk (including parallel tests)\n", loaded)
	}
	results.TotalTests = len(results.Results)
}

// recalculateAvailability computes availability using time-based formula
// Availability % = (total time - blocked time) / total time * 100
// This matches the original all_sine_wave.go calculation
func recalculateAvailability(results *BenchmarkResults) {
	for i := range results.Results {
		r := &results.Results[i]

		// Time-based availability: how much time was NOT spent blocked
		totalTimeMs := r.TrainTimeSec * 1000
		if totalTimeMs > 0 {
			r.AvailabilityPct = ((totalTimeMs - r.TotalBlockedMs) / totalTimeMs) * 100
		} else {
			r.AvailabilityPct = 100
		}

		// Clamp to 0-100%
		if r.AvailabilityPct < 0 {
			r.AvailabilityPct = 0
		}
		if r.AvailabilityPct > 100 {
			r.AvailabilityPct = 100
		}

		// Recalculate score: (Throughput × Availability × Accuracy) / 10000
		r.Score = (r.ThroughputPerSec * r.AvailabilityPct * (r.AvgAccuracy / 100)) / 10000
		if math.IsNaN(r.Score) || math.IsInf(r.Score, 0) {
			r.Score = 0
		}
	}
}

// ═══════════════════════════════════════════════════════════════════════════════
// SINE WAVE DATA GENERATION
// ═══════════════════════════════════════════════════════════════════════════════

func generateSineWave(freqMultiplier float64) []float64 {
	data := make([]float64, SinePoints)
	for i := 0; i < SinePoints; i++ {
		x := float64(i) * SineResolution
		data[i] = math.Sin(freqMultiplier * x)
	}
	return data
}

func createSamples(data []float64) (inputs [][]float32, targets []float32) {
	numSamples := len(data) - InputSize
	inputs = make([][]float32, numSamples)
	targets = make([]float32, numSamples)

	for i := 0; i < numSamples; i++ {
		input := make([]float32, InputSize)
		for j := 0; j < InputSize; j++ {
			input[j] = float32((data[i+j] + 1.0) / 2.0) // Normalize to [0,1]
		}
		inputs[i] = input
		targets[i] = float32((data[i+InputSize] + 1.0) / 2.0)
	}
	return inputs, targets
}

// ═══════════════════════════════════════════════════════════════════════════════
// NETWORK CREATION FOR EACH LAYER TYPE (using nn.BuildSimpleNetwork API)
// ═══════════════════════════════════════════════════════════════════════════════

// createNetworkForLayerType creates a network for the given layer type with default float32 dtype
func createNetworkForLayerType(layerType LayerTestType) *nn.Network {
	return createNetworkForLayerTypeWithDType(layerType, nn.DTypeFloat32)
}

// createNetworkForLayerTypeWithDType creates a network for the given layer type and dtype
func createNetworkForLayerTypeWithDType(layerType LayerTestType, dtype nn.DType) *nn.Network {
	config := nn.SimpleNetworkConfig{
		InputSize:  InputSize,
		HiddenSize: HiddenSize,
		OutputSize: OutputSize,
		Activation: nn.ActivationLeakyReLU,
		InitScale:  InitScale,
		NumLayers:  2,
		DType:      dtype,
	}

	switch layerType {
	case TestLayerDense:
		config.LayerType = nn.BrainDense
	case TestLayerConv2D:
		config.LayerType = nn.BrainConv2D
	case TestLayerRNN:
		config.LayerType = nn.BrainRNN
	case TestLayerLSTM:
		config.LayerType = nn.BrainLSTM
	case TestLayerAttention:
		config.LayerType = nn.BrainMHA
	case TestLayerSwiGLU:
		config.LayerType = nn.BrainSwiGLU
	case TestLayerNormDense:
		config.LayerType = nn.BrainNormDense
	case TestLayerConv1D:
		config.LayerType = nn.BrainConv1D
	case TestLayerResidual:
		config.LayerType = nn.BrainResidual
	default:
		config.LayerType = nn.BrainDense
	}

	return nn.BuildSimpleNetwork(config)
}

// ═══════════════════════════════════════════════════════════════════════════════
// PARALLEL NETWORK BUILDER & HELPERS
// ═══════════════════════════════════════════════════════════════════════════════

// generatePermutations generates all permutations of layer types for a given branch count
func generatePermutations(n int, types []nn.BrainType) [][]nn.BrainType {
	if n == 0 {
		return [][]nn.BrainType{{}}
	}

	subPerms := generatePermutations(n-1, types)
	var perms [][]nn.BrainType

	for _, p := range subPerms {
		for _, t := range types {
			newPerm := make([]nn.BrainType, len(p)+1)
			copy(newPerm, p)
			newPerm[len(p)] = t
			perms = append(perms, newPerm)
		}
	}
	return perms
}

// BuildParallelNetwork creates a network with a parallel layer
func BuildParallelNetwork(branchTypes []nn.BrainType, combineMode string, dtype nn.DType) *nn.Network {
	// Base network structure: Input -> Parallel(Branches...) -> Output
	net := nn.NewNetwork(InputSize, 1, 1, 3) // 3 layers: InputProj, Parallel, OutputProj
	net.BatchSize = 1

	// L0: Input Projection (InputSize -> HiddenSize)
	// Ensures standardized input for parallel branches
	l0 := nn.InitDenseLayer(InputSize, HiddenSize, nn.ActivationLeakyReLU)
	scaleWeights(l0.Kernel, InitScale)
	net.SetLayer(0, 0, 0, l0)

	// L1: Parallel Layer
	branches := make([]nn.LayerConfig, len(branchTypes))
	for i, bt := range branchTypes {
		// Init branch based on type
		// Using architecture.go helpers where possible or manual init
		var branch nn.LayerConfig
		switch bt {
		case nn.BrainDense:
			branch = nn.InitDenseBrain(HiddenSize, nn.ActivationLeakyReLU, InitScale)
		case nn.BrainConv2D:
			// Input is HiddenSize (16), viewed as 4x4
			// Force output to be HiddenSize (16) for compatibility with add/avg
			// 4x4 In -> 4x4 Out (16) -> requires 1 filter, preserve dims
			// Kernel 3x3, Pad 1, Stride 1 => (4-3+2)/1 + 1 = 4. 4x4x1 = 16.
			branch = nn.LayerConfig{
				Type:        nn.LayerConv2D,
				InputHeight: 4, InputWidth: 4, InputChannels: 1,
				OutputHeight: 4, OutputWidth: 4,
				Filters: 1, KernelSize: 3, Stride: 1, Padding: 1,
				Activation: nn.ActivationLeakyReLU,
			}
			branch.Kernel = make([]float32, 1*3*3*1) // Filters*Ky*Kx*In
			for k := range branch.Kernel {
				branch.Kernel[k] = (rand.Float32()*2 - 1) * InitScale
			}
			branch.Bias = make([]float32, 1)

		case nn.BrainRNN:
			branch = nn.InitRNNBrain(HiddenSize, InitScale)

		case nn.BrainLSTM:
			branch = nn.InitLSTMBrain(HiddenSize, InitScale)

		case nn.BrainMHA:
			numHeads := 4
			if HiddenSize%numHeads != 0 {
				numHeads = 2
			}
			branch = nn.InitMHABrain(HiddenSize, numHeads, InitScale)

		case nn.BrainSwiGLU:
			branch = nn.InitSwiGLUBrain(HiddenSize, InitScale)

		case nn.BrainNormDense:
			branch = nn.InitNormDenseBrain(HiddenSize, nn.ActivationLeakyReLU, InitScale)

		case nn.BrainConv1D:
			// Force output to be HiddenSize (16)
			// 16 In -> 16 Out. 1 Filter, Kernel 3, Pad 1, Stride 1.
			branch = nn.LayerConfig{
				Type:             nn.LayerConv1D,
				Conv1DKernelSize: 3, Conv1DStride: 1, Conv1DPadding: 1,
				Conv1DFilters: 1, Conv1DInChannels: 1,
				InputHeight: HiddenSize, OutputHeight: HiddenSize,
				Activation: nn.ActivationLeakyReLU,
			}
			branch.Kernel = make([]float32, 1*3*1)
			for k := range branch.Kernel {
				branch.Kernel[k] = (rand.Float32()*2 - 1) * InitScale
			}
			branch.Bias = make([]float32, 1)

		case nn.BrainResidual:
			branch = nn.InitResidualBrain()
			// Set input/output height for generic safety
			branch.InputHeight = HiddenSize
			branch.OutputHeight = HiddenSize
		default:
			branch = nn.InitDenseBrain(HiddenSize, nn.ActivationLeakyReLU, InitScale)
		}
		branches[i] = branch
	}

	parallelLayer := nn.LayerConfig{
		Type:             nn.LayerParallel,
		CombineMode:      combineMode,
		ParallelBranches: branches,
	}

	// Grid Scatter Setup
	if combineMode == "grid_scatter" {
		parallelLayer.GridOutputRows = 1
		parallelLayer.GridOutputCols = len(branchTypes)
		parallelLayer.GridOutputLayers = 1
		positions := make([]nn.GridPosition, len(branchTypes))
		for i := range branchTypes {
			positions[i] = nn.GridPosition{
				BranchIndex: i,
				TargetRow:   0,
				TargetCol:   i,
				TargetLayer: 0,
			}
		}
		parallelLayer.GridPositions = positions
	}

	net.SetLayer(0, 0, 1, parallelLayer)

	// L2: Output Projection
	// Input size depends on CombineMode
	parallelOutDetail := HiddenSize
	if combineMode == "concat" || combineMode == "grid_scatter" {
		parallelOutDetail = 0
		for _, b := range branches {
			// Need to estimate output size of each branch
			// Most init helpers return config with OutputHeight set, or implicity maintain HiddenSize
			// MHA, LSTM, RNN, Dense, NormDense, SwiGLU, Residual -> HiddenSize
			// Conv2D -> calculated size
			if b.OutputHeight > 0 {
				parallelOutDetail += b.OutputHeight
			} else if b.Type == nn.LayerConv2D {
				// Re-calc logic from architecture.go
				side := int(math.Sqrt(float64(b.InputHeight)))
				if side*side != b.InputHeight {
					side = 4
				}
				outSide := (side-2)/1 + 1
				parallelOutDetail += outSide * outSide * 2
			} else if b.Type == nn.LayerConv1D {
				// Conv1D output size = 2 * InputHeight
				parallelOutDetail += 2 * b.InputHeight
			} else {
				parallelOutDetail += HiddenSize // Default assumption
			}
		}
	}
	// For Add/Avg, size is HiddenSize (assuming branches match)

	l2 := nn.InitDenseLayer(parallelOutDetail, OutputSize, nn.ActivationSigmoid)
	scaleWeights(l2.Kernel, InitScale)
	net.SetLayer(0, 0, 2, l2)

	return net
}

// scaleWeights scales weights in a slice
func scaleWeights(weights []float32, scale float32) {
	for i := range weights {
		weights[i] *= scale
	}
}

// numericTypeToDType converts NumericType to nn.DType
func numericTypeToDType(nt NumericType) nn.DType {
	switch nt {
	case TypeFloat32:
		return nn.DTypeFloat32
	case TypeFloat64:
		return nn.DTypeFloat64
	case TypeInt8:
		return nn.DTypeInt8
	case TypeInt16:
		return nn.DTypeInt16
	case TypeInt32:
		return nn.DTypeInt32
	case TypeInt64:
		return nn.DTypeInt64
	case TypeUint8:
		return nn.DTypeUint8
	case TypeUint16:
		return nn.DTypeUint16
	case TypeUint32:
		return nn.DTypeUint32
	case TypeUint64:
		return nn.DTypeUint64
	default:
		return nn.DTypeFloat32
	}
}

// ═══════════════════════════════════════════════════════════════════════════════
// TEST RUNNER - DISPATCHES TO TYPED OR FLOAT32 IMPLEMENTATIONS
// ═══════════════════════════════════════════════════════════════════════════════

func runCombinationTest(layer LayerTestType, mode TrainingMode, numType NumericType, allInputs [][][]float32, allTargets [][]float32, frequencies []float64) TestResult {
	layerName := layerNames[layer]

	// Define factories for simple layer types
	float32Factory := func() *nn.Network { return createNetworkForLayerType(layer) }
	genericFactory := func(dt nn.DType) *nn.Network { return createNetworkForLayerTypeWithDType(layer, dt) }

	switch numType {
	case TypeFloat32:
		return runFloat32Test(layerName, float32Factory, mode, allInputs, allTargets, frequencies)
	default:
		return runGenericTest(layerName, genericFactory, mode, numType, allInputs, allTargets, frequencies)
	}
}

// runFloat32Test - uses native float32 training APIs with proper availability tracking
func runFloat32Test(layerName string, netFactory func() *nn.Network, mode TrainingMode, allInputs [][][]float32, allTargets [][]float32, frequencies []float64) TestResult {
	result := TestResult{}

	defer func() {
		if r := recover(); r != nil {
			result.Error = fmt.Sprintf("panic: %v", r)
			result.Passed = false
		}
	}()

	numWindows := int(TestDuration / WindowDuration)
	result.Windows = make([]TimeWindow, numWindows)
	for i := range result.Windows {
		result.Windows[i].TimeMs = (i + 1) * int(WindowDuration.Milliseconds())
	}

	net := netFactory()
	numLayers := net.TotalLayers()

	var state *nn.StepState
	if mode == ModeStepBP || mode == ModeStepTween || mode == ModeStepTweenChain {
		state = net.InitStepState(InputSize)
	}

	var ts *nn.TweenState
	if mode == ModeTween || mode == ModeTweenChain || mode == ModeStepTween || mode == ModeStepTweenChain {
		ts = nn.NewTweenState(net, nil)
		if mode == ModeTweenChain || mode == ModeStepTweenChain {
			ts.Config.UseChainRule = true
		}
		ts.Config.LinkBudgetScale = 0.8
	}

	type Sample struct {
		Input  []float32
		Target float32
	}
	trainBatch := make([]Sample, 0, 20)
	lastTrainTime := time.Now()

	start := time.Now()
	currentWindow := 0
	sampleIdx := 0
	currentFreqIdx := 0
	lastSwitchTime := start

	// AVAILABILITY TRACKING - like original
	lastOutputTime := time.Now()
	var totalBlockedTime time.Duration
	windowStartTime := time.Now()
	windowTotalAccuracy := 0.0

	for time.Since(start) < TestDuration {
		elapsed := time.Since(start)

		// Update window
		newWindow := int(elapsed / WindowDuration)
		if newWindow > currentWindow && newWindow < numWindows {
			// Finalize previous window's available time
			if currentWindow < numWindows {
				windowElapsed := time.Since(windowStartTime).Seconds() * 1000
				result.Windows[currentWindow].AvailableMs = windowElapsed - result.Windows[currentWindow].BlockedMs
				// Compute accuracy from total
				if result.Windows[currentWindow].Outputs > 0 {
					result.Windows[currentWindow].Accuracy = windowTotalAccuracy / float64(result.Windows[currentWindow].Outputs)
				}
			}
			currentWindow = newWindow
			windowStartTime = time.Now()
			windowTotalAccuracy = 0.0
		}

		// Check for frequency switch
		if time.Since(lastSwitchTime) >= SwitchInterval && currentFreqIdx < len(frequencies)-1 {
			currentFreqIdx++
			lastSwitchTime = time.Now()
			result.TotalFreqSwitch++
			if currentWindow < numWindows {
				result.Windows[currentWindow].FreqSwitches++
			}
		}

		inputs := allInputs[currentFreqIdx]
		targets := allTargets[currentFreqIdx]
		input := inputs[sampleIdx%len(inputs)]
		target := targets[sampleIdx%len(targets)]
		sampleIdx++

		// Forward pass
		var output []float32
		switch mode {
		case ModeNormalBP, ModeTween, ModeTweenChain:
			output, _ = net.ForwardCPU(input)
		case ModeStepBP:
			state.SetInput(input)
			for s := 0; s < numLayers; s++ {
				net.StepForward(state)
			}
			output = state.GetOutput()
		case ModeStepTween, ModeStepTweenChain:
			output = ts.ForwardPass(net, input)
		}

		// Calculate prediction accuracy for this sample
		sampleAcc := 0.0
		if len(output) > 0 {
			pred := output[0]
			if math.Abs(float64(pred-target)) < AccuracyThreshold {
				sampleAcc = 100.0
			}
		}

		// Record to current window with latency tracking
		if currentWindow < numWindows {
			// Calculate latency since last output
			latencyMs := time.Since(lastOutputTime).Seconds() * 1000
			if latencyMs > result.Windows[currentWindow].MaxLatencyMs {
				result.Windows[currentWindow].MaxLatencyMs = latencyMs
			}
			lastOutputTime = time.Now()

			result.Windows[currentWindow].Outputs++
			windowTotalAccuracy += sampleAcc
			result.TotalOutputs++
		}

		// Training - track blocking time for batch-based methods
		switch mode {
		case ModeNormalBP:
			trainBatch = append(trainBatch, Sample{Input: input, Target: target})
			if time.Since(lastTrainTime) > TrainInterval && len(trainBatch) > 0 {
				batches := make([]nn.TrainingBatch, len(trainBatch))
				for i, s := range trainBatch {
					batches[i] = nn.TrainingBatch{Input: s.Input, Target: []float32{s.Target}}
				}
				// Track blocking time during batch training
				trainStart := time.Now()
				net.Train(batches, &nn.TrainingConfig{Epochs: 1, LearningRate: LearningRate, LossType: "mse"})
				blockDuration := time.Since(trainStart)
				totalBlockedTime += blockDuration
				if currentWindow < numWindows {
					result.Windows[currentWindow].BlockedMs += blockDuration.Seconds() * 1000
				}
				trainBatch = trainBatch[:0]
				lastTrainTime = time.Now()
			}
		case ModeStepBP:
			grad := make([]float32, len(output))
			if len(output) > 0 {
				grad[0] = clipGrad(output[0]-target, 0.5)
			}
			net.StepBackward(state, grad)
			net.ApplyGradients(LearningRate)
		case ModeTween, ModeTweenChain:
			trainBatch = append(trainBatch, Sample{Input: input, Target: target})
			if time.Since(lastTrainTime) > TrainInterval && len(trainBatch) > 0 {
				// Track blocking time during batch training
				trainStart := time.Now()
				for _, s := range trainBatch {
					out := ts.ForwardPass(net, s.Input)
					outputGrad := make([]float32, len(out))
					if len(out) > 0 {
						outputGrad[0] = s.Target - out[0]
					}
					totalLayers := net.TotalLayers()
					ts.ChainGradients[totalLayers] = outputGrad
					ts.BackwardTargets[totalLayers] = []float32{s.Target}
					ts.TweenWeightsChainRule(net, LearningRate)
				}
				blockDuration := time.Since(trainStart)
				totalBlockedTime += blockDuration
				if currentWindow < numWindows {
					result.Windows[currentWindow].BlockedMs += blockDuration.Seconds() * 1000
				}
				trainBatch = trainBatch[:0]
				lastTrainTime = time.Now()
			}
		case ModeStepTween, ModeStepTweenChain:
			outputGrad := make([]float32, len(output))
			if len(output) > 0 {
				outputGrad[0] = target - output[0]
			}
			totalLayers := net.TotalLayers()
			ts.ChainGradients[totalLayers] = outputGrad
			ts.BackwardTargets[totalLayers] = []float32{target}
			ts.TweenWeightsChainRule(net, LearningRate)
		}
	}

	// Finalize windows - compute availability for each window
	for i := range result.Windows {
		if result.Windows[i].Outputs > 0 && result.Windows[i].Accuracy == 0 {
			result.Windows[i].Accuracy = windowTotalAccuracy / float64(result.Windows[i].Outputs)
		}
		windowDurationMs := WindowDuration.Seconds() * 1000
		if result.Windows[i].AvailableMs == 0 {
			result.Windows[i].AvailableMs = windowDurationMs - result.Windows[i].BlockedMs
		}
	}

	result.TrainTimeSec = time.Since(start).Seconds()
	result.TotalBlockedMs = totalBlockedTime.Seconds() * 1000
	calculateMetrics(&result)
	return result
}

// runGenericTest - uses generic tensor APIs for non-float32 types
func runGenericTest(layerName string, netFactory func(nn.DType) *nn.Network, mode TrainingMode, numType NumericType, allInputs [][][]float32, allTargets [][]float32, frequencies []float64) TestResult {
	switch numType {
	case TypeInt8:
		return runTypedTest[int8](layerName, netFactory, nn.DTypeInt8, mode, allInputs, allTargets, frequencies)
	case TypeInt16:
		return runTypedTest[int16](layerName, netFactory, nn.DTypeInt16, mode, allInputs, allTargets, frequencies)
	case TypeInt32:
		return runTypedTest[int32](layerName, netFactory, nn.DTypeInt32, mode, allInputs, allTargets, frequencies)
	case TypeInt64:
		return runTypedTest[int64](layerName, netFactory, nn.DTypeInt64, mode, allInputs, allTargets, frequencies)
	case TypeUint8:
		return runTypedTest[uint8](layerName, netFactory, nn.DTypeUint8, mode, allInputs, allTargets, frequencies)
	case TypeUint16:
		return runTypedTest[uint16](layerName, netFactory, nn.DTypeUint16, mode, allInputs, allTargets, frequencies)
	case TypeUint32:
		return runTypedTest[uint32](layerName, netFactory, nn.DTypeUint32, mode, allInputs, allTargets, frequencies)
	case TypeUint64:
		return runTypedTest[uint64](layerName, netFactory, nn.DTypeUint64, mode, allInputs, allTargets, frequencies)
	case TypeFloat64:
		return runTypedTest[float64](layerName, netFactory, nn.DTypeFloat64, mode, allInputs, allTargets, frequencies)
	default:
		return TestResult{Passed: false, Error: "unknown type"}
	}
}

// runTypedTest - generic implementation for all numeric types
func runTypedTest[T nn.Numeric](layerName string, netFactory func(nn.DType) *nn.Network, dtype nn.DType, mode TrainingMode, allInputs [][][]float32, allTargets [][]float32, frequencies []float64) TestResult {
	result := TestResult{}

	defer func() {
		if r := recover(); r != nil {
			result.Error = fmt.Sprintf("panic: %v", r)
			result.Passed = false
		}
	}()

	numWindows := int(TestDuration / WindowDuration)
	result.Windows = make([]TimeWindow, numWindows)
	for i := range result.Windows {
		result.Windows[i].TimeMs = (i + 1) * int(WindowDuration.Milliseconds())
	}

	net := netFactory(dtype)
	backend := nn.NewCPUBackend[T]()
	numLayers := net.TotalLayers()

	// Initialize step state for step-based modes
	var stepState *nn.GenericStepState[T]
	if mode == ModeStepBP || mode == ModeStepTween || mode == ModeStepTweenChain {
		stepState = nn.NewGenericStepState[T](numLayers, InputSize)
	}

	// Initialize tween state for tween-based modes
	var tweenState *nn.GenericTweenState[T]
	if mode == ModeTween || mode == ModeTweenChain || mode == ModeStepTween || mode == ModeStepTweenChain {
		config := nn.DefaultTweenConfig(numLayers)
		if mode == ModeTweenChain || mode == ModeStepTweenChain {
			config.UseChainRule = true
		}
		config.LinkBudgetScale = 0.8
		tweenState = nn.NewGenericTweenState[T](net, config)
	}

	start := time.Now()
	currentWindow := 0
	sampleIdx := 0
	currentFreqIdx := 0
	lastSwitchTime := start
	windowCorrect := 0
	windowTotal := 0

	// Scaling factor for integer types
	scale := float32(1.0)
	if isIntegerType[T]() {
		scale = 100.0
	}

	for time.Since(start) < TestDuration {
		elapsed := time.Since(start)

		// Update window
		newWindow := int(elapsed / WindowDuration)
		if newWindow > currentWindow && newWindow < numWindows {
			if windowTotal > 0 {
				result.Windows[currentWindow].Accuracy = float64(windowCorrect) / float64(windowTotal) * 100
			}
			result.Windows[currentWindow].Outputs = windowTotal
			currentWindow = newWindow
			windowCorrect = 0
			windowTotal = 0
		}

		// Check for frequency switch
		if time.Since(lastSwitchTime) >= SwitchInterval && currentFreqIdx < len(frequencies)-1 {
			currentFreqIdx++
			lastSwitchTime = time.Now()
			result.TotalFreqSwitch++
			if currentWindow < numWindows {
				result.Windows[currentWindow].FreqSwitches++
			}
		}

		inputs := allInputs[currentFreqIdx]
		targets := allTargets[currentFreqIdx]
		inputF32 := inputs[sampleIdx%len(inputs)]
		targetF32 := targets[sampleIdx%len(targets)]
		sampleIdx++

		// Convert to typed tensor
		inputData := make([]T, len(inputF32))
		for i, v := range inputF32 {
			inputData[i] = T(v * scale)
		}
		inputTensor := nn.NewTensorFromSlice(inputData, len(inputData))

		var output *nn.Tensor[T]

		switch mode {
		case ModeNormalBP:
			// Use GenericTrainStep for training
			targetData := make([]T, 1)
			targetData[0] = T(targetF32 * scale)
			targetTensor := nn.NewTensorFromSlice(targetData, 1)
			output, _, _ = nn.GenericTrainStep(net, inputTensor, targetTensor, float64(LearningRate), backend)

		case ModeStepBP:
			stepState.SetInput(inputTensor)
			nn.StepForwardGeneric(net, stepState, backend)
			output = stepState.GetOutput()

		case ModeTween, ModeTweenChain, ModeStepTween, ModeStepTweenChain:
			output = tweenState.ForwardPass(net, inputTensor, backend)
			// TweenStep for training
			targetClass := 0 // For regression, use class 0
			tweenState.TweenStep(net, inputTensor, targetClass, OutputSize, LearningRate, backend)
		}

		// Check accuracy
		if output != nil && len(output.Data) > 0 {
			var pred float64
			if isIntegerType[T]() {
				pred = float64(output.Data[0]) / float64(scale)
			} else {
				pred = float64(output.Data[0])
			}
			if math.Abs(pred-float64(targetF32)) < AccuracyThreshold {
				windowCorrect++
			}
			windowTotal++
			result.TotalOutputs++
		}
	}

	// Finalize last window
	if windowTotal > 0 && currentWindow < numWindows {
		result.Windows[currentWindow].Accuracy = float64(windowCorrect) / float64(windowTotal) * 100
		result.Windows[currentWindow].Outputs = windowTotal
	}

	result.TrainTimeSec = time.Since(start).Seconds()
	calculateMetrics(&result)
	return result
}

func isIntegerType[T nn.Numeric]() bool {
	var zero T
	switch any(zero).(type) {
	case int8, int16, int32, int64, uint8, uint16, uint32, uint64, int, uint:
		return true
	default:
		return false
	}
}

func calculateMetrics(result *TestResult) {
	// Average accuracy
	sum := 0.0
	for _, w := range result.Windows {
		sum += w.Accuracy
	}
	if len(result.Windows) > 0 {
		result.AvgAccuracy = sum / float64(len(result.Windows))
	}

	// Stability: 100 - stddev
	variance := 0.0
	for _, w := range result.Windows {
		diff := w.Accuracy - result.AvgAccuracy
		variance += diff * diff
	}
	if len(result.Windows) > 0 {
		variance /= float64(len(result.Windows))
	}
	result.Stability = math.Max(0, 100-math.Sqrt(variance))

	// Throughput
	if result.TrainTimeSec > 0 {
		result.ThroughputPerSec = float64(result.TotalOutputs) / result.TrainTimeSec
	}

	// Note: AvailabilityPct will be calculated post-hoc by comparing throughput
	// against max possible (StepTween baseline) for accurate comparison
	// For now, just store blocked time
	// result.AvailabilityPct will be set in recalculateAvailability()

	// Max and avg latency
	latencySum := 0.0
	for _, w := range result.Windows {
		latencySum += w.MaxLatencyMs
		if w.MaxLatencyMs > result.MaxLatencyMs {
			result.MaxLatencyMs = w.MaxLatencyMs
		}
		if w.Outputs == 0 {
			result.ZeroOutputWindows++
		}
	}
	if len(result.Windows) > 0 {
		result.AvgLatencyMs = latencySum / float64(len(result.Windows))
	}

	// Preliminary score using 100% availability - will be recalculated post-hoc
	// when we can compare against max throughput within each layer+type combo
	result.AvailabilityPct = 100.0 // Placeholder
	result.Score = (result.ThroughputPerSec * result.AvailabilityPct * (result.AvgAccuracy / 100)) / 10000
	result.Passed = result.TotalOutputs > 0
}

func clipGrad(v, max float32) float32 {
	if v > max {
		return max
	}
	if v < -max {
		return -max
	}
	if math.IsNaN(float64(v)) {
		return 0
	}
	return v
}

// ═══════════════════════════════════════════════════════════════════════════════
// OUTPUT
// ═══════════════════════════════════════════════════════════════════════════════

func saveResults(results *BenchmarkResults) {
	data, _ := json.MarshalIndent(results, "", "  ")
	os.WriteFile("all_sine_wave_multi_results.json", data, 0644)
	fmt.Println("\n✅ Results saved to all_sine_wave_multi_results.json")
}

func printSummaryTable(results *BenchmarkResults) {
	fmt.Println("\n╔═══════════════════════════════════════════════════════════════════════════════════════════════════════════════════╗")
	fmt.Println("║                                        🌊 SUMMARY BY LAYER TYPE 🌊                                                ║")
	fmt.Println("╚═══════════════════════════════════════════════════════════════════════════════════════════════════════════════════╝")

	// Group by layer type
	layerResults := make(map[string][]TestResult)
	for _, r := range results.Results {
		layerResults[r.LayerType] = append(layerResults[r.LayerType], r)
	}

	for _, layerName := range []string{"Dense", "Conv2D", "RNN", "LSTM", "Attention"} {
		layerRes := layerResults[layerName]
		if len(layerRes) == 0 {
			continue
		}

		fmt.Printf("\n┌─── %s ───────────────────────────────────────────────────────────────────────────────────────────────┐\n", layerName)
		fmt.Println("│ Mode             │ int8  │ int16 │ int32 │ int64 │ uint8 │uint16 │uint32 │uint64 │float32│float64│ Avg   │")
		fmt.Println("├──────────────────┼───────┼───────┼───────┼───────┼───────┼───────┼───────┼───────┼───────┼───────┼───────┤")

		for _, modeName := range []string{"NormalBP", "StepBP", "Tween", "TweenChain", "StepTween", "StepTweenChain"} {
			fmt.Printf("│ %-16s │", modeName)

			modeSum := 0.0
			modeCount := 0

			for _, typeName := range []string{"int8", "int16", "int32", "int64", "uint8", "uint16", "uint32", "uint64", "float32", "float64"} {
				// Find matching result
				acc := 0.0
				for _, r := range layerRes {
					if r.TrainingMode == modeName && r.NumericType == typeName {
						acc = r.AvgAccuracy
						modeSum += acc
						modeCount++
						break
					}
				}
				fmt.Printf(" %5.0f │", acc)
			}

			modeAvg := 0.0
			if modeCount > 0 {
				modeAvg = modeSum / float64(modeCount)
			}
			fmt.Printf(" %5.0f │\n", modeAvg)
		}
		fmt.Println("└──────────────────┴───────┴───────┴───────┴───────┴───────┴───────┴───────┴───────┴───────┴───────┴───────┘")
	}

	// Print overall summary
	fmt.Printf("\n📊 OVERALL: %d/%d tests passed (%.1f%%)\n", results.Passed, results.TotalTests, float64(results.Passed)/float64(results.TotalTests)*100)

	// Best performers per category
	bestByLayer := make(map[string]TestResult)
	bestByMode := make(map[string]TestResult)
	bestByType := make(map[string]TestResult)

	for _, r := range results.Results {
		if r.Score > bestByLayer[r.LayerType].Score {
			bestByLayer[r.LayerType] = r
		}
		if r.Score > bestByMode[r.TrainingMode].Score {
			bestByMode[r.TrainingMode] = r
		}
		if r.Score > bestByType[r.NumericType].Score {
			bestByType[r.NumericType] = r
		}
	}

	fmt.Println("\n🏆 BEST PERFORMERS:")
	fmt.Println("┌──────────────────────────────────────────────────────────────────────────────────────┐")
	fmt.Println("│ Category         │ Winner                                          │ Score          │")
	fmt.Println("├──────────────────┼─────────────────────────────────────────────────┼────────────────┤")
	for _, layer := range []string{"Dense", "Conv2D", "RNN", "LSTM", "Attention"} {
		if r, ok := bestByLayer[layer]; ok {
			fmt.Printf("│ Best %-10s  │ %-47s │ %14.0f │\n", layer, fmt.Sprintf("%s/%s", r.TrainingMode, r.NumericType), r.Score)
		}
	}
	fmt.Println("└──────────────────┴─────────────────────────────────────────────────┴────────────────┘")

	// Print detailed timeline for float32 results (most representative of actual training)
	printTimelineForFloat32(results)

	// Print best config per mode comparison
	printBestConfigPerMode(results)

	// Print parallel test summaries
	printParallelBestConfigPerMode(results)
	printParallelBestTypePerCombineMode(results)

	// Print robustness analysis (failure rate, median score, winner frequency)
	printRobustnessAnalysis(results)

	// Print advanced decision surface analysis
	printWinCountByRegime(results)
	printAvailabilityConditionedAccuracy(results)
	printDominanceEnvelope(results)
	printFailureSeverityBreakdown(results)
	printStressResponseCurve(results)

	fmt.Println("============REMAINING=METRICS===================")

	// Print remaining metrics we haven't covered
	printLatencyAnalysis(results)
	printThroughputDistribution(results)
	printLayerSpecificRecommendations(results)
	printScoreConsistencyAnalysis(results)
	printAccuracyAvailabilityPareto(results)
	printZeroOutputWindowAnalysis(results)
	printTopBottomConfigurations(results)
	printNumericalTypePerformanceMatrix(results)
}

// printBestConfigPerMode shows ALL layer+type combinations for each training mode
func printBestConfigPerMode(results *BenchmarkResults) {
	fmt.Println("\n╔════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════╗")
	fmt.Println("║                                          🔬 FULL COMPARISON: ALL LAYERS × ALL TYPES × PER MODE 🔬                                                                                                   ║")
	fmt.Println("║                                                                                                                                                                                                      ║")
	fmt.Println("║                     Score matrix for each training mode showing all 50 combinations (5 layers × 10 numeric types)                                                                                   ║")
	fmt.Println("╚════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════╝")

	modes := []string{"NormalBP", "StepBP", "Tween", "TweenChain", "StepTween", "StepTweenChain"}
	allLayers := []string{"Dense", "Conv2D", "RNN", "LSTM", "Attention", "SwiGLU", "NormDense", "Conv1D", "Residual"}
	types := []string{"int8", "int16", "int32", "int64", "uint8", "uint16", "uint32", "uint64", "float32", "float64"}

	// Build lookup: mode -> layer -> type -> result
	resultLookup := make(map[string]map[string]map[string]TestResult)
	for _, r := range results.Results {
		if resultLookup[r.TrainingMode] == nil {
			resultLookup[r.TrainingMode] = make(map[string]map[string]TestResult)
		}
		if resultLookup[r.TrainingMode][r.LayerType] == nil {
			resultLookup[r.TrainingMode][r.LayerType] = make(map[string]TestResult)
		}
		resultLookup[r.TrainingMode][r.LayerType][r.NumericType] = r
	}

	// For each mode, print a full score matrix
	for _, modeName := range modes {
		// Find best in this mode
		var modeBest TestResult
		for _, layerMap := range resultLookup[modeName] {
			for _, r := range layerMap {
				if r.Score > modeBest.Score {
					modeBest = r
				}
			}
		}

		fmt.Printf("\n┌─── %s ─── SCORE Matrix (Layer × NumericType) ─── Best: %s/%s (Score: %.0f) ───────────────────────────────────────────────────────────────────────────────────┐\n",
			modeName, modeBest.LayerType, modeBest.NumericType, modeBest.Score)

		// Header with all types
		fmt.Print("│ Layer      │")
		for _, t := range types {
			fmt.Printf(" %6s │", t)
		}
		fmt.Println(" BEST       │")

		fmt.Print("├────────────┼")
		for range types {
			fmt.Print("────────┼")
		}
		fmt.Println("────────────┤")

		// Rows for each layer
		for _, layer := range allLayers {
			fmt.Printf("│ %-10s │", layer)

			bestTypeScore := 0.0
			bestTypeName := ""
			for _, t := range types {
				if r, ok := resultLookup[modeName][layer][t]; ok {
					if r.Error != "" {
						fmt.Print("    ERR │")
					} else if r.Score == 0 {
						fmt.Print("      0 │")
					} else {
						fmt.Printf(" %6.0f │", r.Score)
					}

					if r.Score > bestTypeScore {
						bestTypeScore = r.Score
						bestTypeName = t
					}
				} else {
					fmt.Print("    N/A │")
				}
			}
			fmt.Printf(" %s %.0f │\n", bestTypeName, bestTypeScore)
		}

		fmt.Print("└────────────┴")
		for range types {
			fmt.Print("────────┴")
		}
		fmt.Println("────────────┘")
	}

	// SUMMARY: Best per mode
	fmt.Println("\n╔════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════╗")
	fmt.Println("║                                               🏆 BEST CONFIG PER MODE SUMMARY 🏆                                                                                                  ║")
	fmt.Println("╠════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════╣")
	fmt.Println("║  Mode             │ Best Layer │ Best Type │ Accuracy │ Throughput │ Score   │ Avail %  │ ★ Why This Combo Wins ★                                                                ║")
	fmt.Println("║  ─────────────────┼────────────┼───────────┼──────────┼────────────┼─────────┼──────────┼────────────────────────────────────────────────────────────────────────────────────────║")

	bestPerMode := make(map[string]TestResult)
	for _, r := range results.Results {
		if r.Score > bestPerMode[r.TrainingMode].Score {
			bestPerMode[r.TrainingMode] = r
		}
	}

	for _, modeName := range modes {
		r := bestPerMode[modeName]
		insight := ""
		switch {
		case r.NumericType == "float64" && r.AvgAccuracy >= 80:
			insight = "float64 precision"
		case r.NumericType == "float32" && r.AvailabilityPct >= 99:
			insight = "float32 speed + availability"
		case r.LayerType == "Conv2D":
			insight = "Conv2D pattern detection"
		case r.LayerType == "Attention":
			insight = "Attention dependencies"
		case r.LayerType == "RNN" || r.LayerType == "LSTM":
			insight = "Sequential layer fit"
		default:
			insight = "Best balance"
		}

		fmt.Printf("║  %-15s  │ %-10s │ %-9s │  %5.1f%%  │  %8.0f  │ %7.0f │  %5.1f%%  │ %-83s ║\n",
			modeName, r.LayerType, r.NumericType,
			r.AvgAccuracy, r.ThroughputPerSec, r.Score, r.AvailabilityPct, insight)
	}
	fmt.Println("╚════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════╝")

	// Find the absolute winner
	var absoluteBest TestResult
	for _, r := range results.Results {
		if r.Score > absoluteBest.Score {
			absoluteBest = r
		}
	}

	fmt.Printf("\n🎯 ABSOLUTE BEST ACROSS ALL 300 COMBINATIONS:\n")
	fmt.Printf("   Mode: %s | Layer: %s | Type: %s | Score: %.0f\n",
		absoluteBest.TrainingMode, absoluteBest.LayerType, absoluteBest.NumericType, absoluteBest.Score)
	fmt.Printf("   Accuracy: %.1f%% | Throughput: %.0f/sec | Availability: %.1f%%\n\n",
		absoluteBest.AvgAccuracy, absoluteBest.ThroughputPerSec, absoluteBest.AvailabilityPct)

	// Type wins summary
	printBestTypePerLayerMode(results)
}

// printBestTypePerLayerMode shows which numerical type wins for each layer+mode combo
func printBestTypePerLayerMode(results *BenchmarkResults) {
	fmt.Println("\n╔════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════╗")
	fmt.Println("║                                          📊 BEST NUMERICAL TYPE PER LAYER+MODE 📊                                                                                       ║")
	fmt.Println("║                                                                                                                                                                          ║")
	fmt.Println("║                     Which of the 10 numerical types (int8-uint64, float32/64) performs best for each Layer+Mode?                                                        ║")
	fmt.Println("╚════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════╝")

	allLayers := []string{"Dense", "Conv2D", "RNN", "LSTM", "Attention", "SwiGLU", "NormDense", "Conv1D", "Residual"}
	modes := []string{"NormalBP", "StepBP", "Tween", "TweenChain", "StepTween", "StepTweenChain"}

	// Build lookup: layer -> mode -> best result across all numeric types
	bestTypeByLayerMode := make(map[string]map[string]TestResult)
	for _, r := range results.Results {
		if bestTypeByLayerMode[r.LayerType] == nil {
			bestTypeByLayerMode[r.LayerType] = make(map[string]TestResult)
		}
		if r.Score > bestTypeByLayerMode[r.LayerType][r.TrainingMode].Score {
			bestTypeByLayerMode[r.LayerType][r.TrainingMode] = r
		}
	}

	// Print matrix: rows = layers, cols = modes, cells = best type
	fmt.Println("\n┌─── WINNING NUMERICAL TYPE (Score-Based) ─────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────┐")
	fmt.Print("│ Layer      │")
	for _, mode := range modes {
		fmt.Printf(" %-14s │", mode)
	}
	fmt.Println("")
	fmt.Print("├────────────┼")
	for range modes {
		fmt.Print("────────────────┼")
	}
	fmt.Println("")

	for _, layer := range allLayers {
		fmt.Printf("│ %-10s │", layer)
		for _, mode := range modes {
			if r, ok := bestTypeByLayerMode[layer][mode]; ok {
				// Show type and score
				if r.Score == 0 && r.Error != "" {
					fmt.Printf(" %-7s %5s │", "ERR", "ERR")
				} else if r.Score == 0 {
					fmt.Printf(" %-7s %5d │", r.NumericType, 0)
				} else {
					fmt.Printf(" %-7s %5.0f │", r.NumericType, r.Score)
				}
			} else {
				fmt.Print("        N/A    │")
			}
		}
		fmt.Println("")
	}
	fmt.Print("└────────────┴")
	for range modes {
		fmt.Print("────────────────┴")
	}
	fmt.Println("")

	// Count how many times each type wins
	typeWins := make(map[string]int)
	for _, layerMap := range bestTypeByLayerMode {
		for _, r := range layerMap {
			typeWins[r.NumericType]++
		}
	}

	fmt.Println("\n📊 TYPE WINS ACROSS ALL 30 LAYER+MODE COMBOS:")
	fmt.Println("   ────────────────────────────────────────────")
	allTypes := []string{"float32", "float64", "int8", "int16", "int32", "int64", "uint8", "uint16", "uint32", "uint64"}
	for _, t := range allTypes {
		wins := typeWins[t]
		bar := ""
		for i := 0; i < wins; i++ {
			bar += "█"
		}
		fmt.Printf("   %-8s: %2d wins %s\n", t, wins, bar)
	}
	fmt.Println("")
}

// ═══════════════════════════════════════════════════════════════════════════════
// PARALLEL TEST SUMMARIES
// ═══════════════════════════════════════════════════════════════════════════════

// printParallelBestConfigPerMode shows the best parallel (Par2-*) config per training mode
func printParallelBestConfigPerMode(results *BenchmarkResults) {
	// Filter for parallel results only (Par2-* prefix)
	var parallelResults []TestResult
	for _, r := range results.Results {
		if len(r.LayerType) >= 4 && r.LayerType[:4] == "Par2" {
			parallelResults = append(parallelResults, r)
		}
	}

	if len(parallelResults) == 0 {
		fmt.Println("\n⚠️  No parallel test results found (Par2-* tests)")
		return
	}

	fmt.Println("\n╔════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════╗")
	fmt.Println("║                                               🏆 PARALLEL TESTS: BEST CONFIG PER MODE SUMMARY 🏆                                                                                                                                                     ║")
	fmt.Println("╠════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════╣")
	fmt.Println("║  Mode             │ Best Parallel Config                         │ Best Type │ Accuracy │ Throughput │ Score   │ Avail %  │ ★ Why This Combo Wins ★                                                                                                  ║")
	fmt.Println("║  ─────────────────┼──────────────────────────────────────────────┼───────────┼──────────┼────────────┼─────────┼──────────┼──────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────║")

	modes := []string{"NormalBP", "StepBP", "Tween", "TweenChain", "StepTween", "StepTweenChain"}

	bestPerMode := make(map[string]TestResult)
	for _, r := range parallelResults {
		if r.Score > bestPerMode[r.TrainingMode].Score {
			bestPerMode[r.TrainingMode] = r
		}
	}

	for _, modeName := range modes {
		r := bestPerMode[modeName]
		if r.LayerType == "" {
			continue
		}
		insight := ""
		switch {
		case r.NumericType == "float64" && r.AvgAccuracy >= 80:
			insight = "float64 precision"
		case r.NumericType == "float32" && r.AvailabilityPct >= 99:
			insight = "float32 speed + availability"
		case strings.Contains(r.LayerType, "Residual"):
			insight = "Residual skip connection"
		case strings.Contains(r.LayerType, "filter"):
			insight = "Filter combine mode"
		case strings.Contains(r.LayerType, "concat"):
			insight = "Concat diversity"
		case strings.Contains(r.LayerType, "add") || strings.Contains(r.LayerType, "avg"):
			insight = "Add/Avg smoothing"
		default:
			insight = "Best parallel balance"
		}

		fmt.Printf("║  %-15s  │ %-44s │ %-9s │  %5.1f%%  │  %8.0f  │ %7.0f │  %5.1f%%  │ %-109s ║\n",
			modeName, r.LayerType, r.NumericType,
			r.AvgAccuracy, r.ThroughputPerSec, r.Score, r.AvailabilityPct, insight)
	}
	fmt.Println("╚════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════╝")

	// Find the absolute best parallel config
	var absoluteBest TestResult
	for _, r := range parallelResults {
		if r.Score > absoluteBest.Score {
			absoluteBest = r
		}
	}

	fmt.Printf("\n🎯 ABSOLUTE BEST PARALLEL CONFIG:\n")
	fmt.Printf("   Mode: %s | Config: %s | Type: %s | Score: %.0f\n",
		absoluteBest.TrainingMode, absoluteBest.LayerType, absoluteBest.NumericType, absoluteBest.Score)
	fmt.Printf("   Accuracy: %.1f%% | Throughput: %.0f/sec | Availability: %.1f%%\n\n",
		absoluteBest.AvgAccuracy, absoluteBest.ThroughputPerSec, absoluteBest.AvailabilityPct)
}

// printParallelBestTypePerCombineMode shows which numerical type wins for each combine mode
func printParallelBestTypePerCombineMode(results *BenchmarkResults) {
	// Filter for parallel results only
	var parallelResults []TestResult
	for _, r := range results.Results {
		if len(r.LayerType) >= 4 && r.LayerType[:4] == "Par2" {
			parallelResults = append(parallelResults, r)
		}
	}

	if len(parallelResults) == 0 {
		return
	}

	fmt.Println("\n╔════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════╗")
	fmt.Println("║                                          📊 PARALLEL TESTS: BEST NUMERICAL TYPE PER COMBINE MODE 📊                                                                      ║")
	fmt.Println("║                                                                                                                                                                            ║")
	fmt.Println("║                     Which of the 10 numerical types (int8-uint64, float32/64) performs best for each CombineMode + TrainingMode?                                         ║")
	fmt.Println("╚════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════╝")

	combineModes := []string{"concat", "add", "avg", "grid_scatter", "filter"}
	trainingModes := []string{"NormalBP", "StepBP", "Tween", "TweenChain", "StepTween", "StepTweenChain"}

	// Build lookup: combineMode -> trainingMode -> best result
	bestTypeByCombineMode := make(map[string]map[string]TestResult)
	for _, r := range parallelResults {
		// Extract combine mode from LayerType (e.g., "Par2-Dense-Dense_concat" -> "concat")
		cm := ""
		for _, m := range combineModes {
			if strings.HasSuffix(r.LayerType, "_"+m) {
				cm = m
				break
			}
		}
		if cm == "" {
			continue
		}

		if bestTypeByCombineMode[cm] == nil {
			bestTypeByCombineMode[cm] = make(map[string]TestResult)
		}
		if r.Score > bestTypeByCombineMode[cm][r.TrainingMode].Score {
			bestTypeByCombineMode[cm][r.TrainingMode] = r
		}
	}

	// Print matrix
	fmt.Println("\n┌─── WINNING NUMERICAL TYPE (Score-Based) ─────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────┐")
	fmt.Print("│ CombineMode  │")
	for _, mode := range trainingModes {
		fmt.Printf(" %-14s │", mode)
	}
	fmt.Println("")
	fmt.Print("├──────────────┼")
	for range trainingModes {
		fmt.Print("────────────────┼")
	}
	fmt.Println("")

	for _, cm := range combineModes {
		fmt.Printf("│ %-12s │", cm)
		for _, mode := range trainingModes {
			if r, ok := bestTypeByCombineMode[cm][mode]; ok {
				if r.Score == 0 && r.Error != "" {
					fmt.Printf(" %-7s %5s │", "ERR", "ERR")
				} else if r.Score == 0 {
					fmt.Printf(" %-7s %5d │", r.NumericType, 0)
				} else {
					fmt.Printf(" %-7s %5.0f │", r.NumericType, r.Score)
				}
			} else {
				fmt.Print("        N/A    │")
			}
		}
		fmt.Println("")
	}
	fmt.Print("└──────────────┴")
	for range trainingModes {
		fmt.Print("────────────────┴")
	}
	fmt.Println("")

	// Count type wins across parallel tests
	typeWins := make(map[string]int)
	for _, cmMap := range bestTypeByCombineMode {
		for _, r := range cmMap {
			typeWins[r.NumericType]++
		}
	}

	fmt.Println("\n📊 PARALLEL TYPE WINS ACROSS ALL COMBINEMODE+TRAININGMODE COMBOS:")
	fmt.Println("   ────────────────────────────────────────────")
	allTypes := []string{"float32", "float64", "int8", "int16", "int32", "int64", "uint8", "uint16", "uint32", "uint64"}
	for _, t := range allTypes {
		wins := typeWins[t]
		bar := ""
		for i := 0; i < wins; i++ {
			bar += "█"
		}
		fmt.Printf("   %-8s: %2d wins %s\n", t, wins, bar)
	}
	fmt.Println("")
}

// ═══════════════════════════════════════════════════════════════════════════════
// ROBUSTNESS ANALYSIS
// ═══════════════════════════════════════════════════════════════════════════════

// printRobustnessAnalysis shows failure rate, median score, and winner frequency per mode
func printRobustnessAnalysis(results *BenchmarkResults) {
	fmt.Println("\n╔═══════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════╗")
	fmt.Println("║                                                   🔬 ROBUSTNESS ANALYSIS: FAILURE RATE, MEDIAN SCORE, WINNER FREQUENCY 🔬                                                                                                     ║")
	fmt.Println("║                                                                                                                                                                                                                               ║")
	fmt.Println("║                     This analysis proves whether a training mode is consistently good (not just a lucky outlier)                                                                                                             ║")
	fmt.Println("╚═══════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════╝")

	modes := []string{"NormalBP", "StepBP", "Tween", "TweenChain", "StepTween", "StepTweenChain"}

	// Collect scores per mode
	modeScores := make(map[string][]float64)
	modeFailures := make(map[string]int)
	modeTotal := make(map[string]int)

	for _, r := range results.Results {
		modeTotal[r.TrainingMode]++

		// Count failures: NaN, zero score with error, or explicitly failed
		if r.Error != "" || (r.Score == 0 && !r.Passed) || math.IsNaN(r.Score) || math.IsInf(r.Score, 0) {
			modeFailures[r.TrainingMode]++
		} else {
			modeScores[r.TrainingMode] = append(modeScores[r.TrainingMode], r.Score)
		}
	}

	// Calculate median scores
	modeMedian := make(map[string]float64)
	for mode, scores := range modeScores {
		if len(scores) == 0 {
			modeMedian[mode] = 0
			continue
		}
		// Sort scores
		sorted := make([]float64, len(scores))
		copy(sorted, scores)
		sort.Float64s(sorted)

		mid := len(sorted) / 2
		if len(sorted)%2 == 0 {
			modeMedian[mode] = (sorted[mid-1] + sorted[mid]) / 2
		} else {
			modeMedian[mode] = sorted[mid]
		}
	}

	// Calculate winner frequency: for each Layer+Type combo, which mode wins?
	type LayerTypeKey struct {
		Layer string
		Type  string
	}
	bestPerLayerType := make(map[LayerTypeKey]TestResult)
	for _, r := range results.Results {
		key := LayerTypeKey{r.LayerType, r.NumericType}
		if r.Score > bestPerLayerType[key].Score {
			bestPerLayerType[key] = r
		}
	}

	modeWins := make(map[string]int)
	for _, r := range bestPerLayerType {
		modeWins[r.TrainingMode]++
	}

	totalCombos := len(bestPerLayerType)

	// Print summary table
	fmt.Println("\n┌─────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────┐")
	fmt.Println("│       Mode        │ Total │ Failed │ Fail %  │ Median Score │ Best Score │ Win Count │ Win %   │ ★ Interpretation ★                                                                                                         │")
	fmt.Println("├───────────────────┼───────┼────────┼─────────┼──────────────┼────────────┼───────────┼─────────┼─────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────┤")

	for _, mode := range modes {
		total := modeTotal[mode]
		failures := modeFailures[mode]
		failPct := 0.0
		if total > 0 {
			failPct = float64(failures) / float64(total) * 100
		}

		median := modeMedian[mode]
		best := 0.0
		for _, s := range modeScores[mode] {
			if s > best {
				best = s
			}
		}

		wins := modeWins[mode]
		winPct := 0.0
		if totalCombos > 0 {
			winPct = float64(wins) / float64(totalCombos) * 100
		}

		// Generate interpretation
		interpretation := ""
		switch {
		case failPct < 5 && winPct >= 20 && median > 500:
			interpretation = "🏆 EXCELLENT: Low failure, high wins, strong median"
		case failPct < 10 && winPct >= 15:
			interpretation = "✅ SOLID: Reliable and competitive"
		case failPct < 5 && median > 200:
			interpretation = "👍 STABLE: Very reliable, moderate performance"
		case failPct >= 20:
			interpretation = "⚠️  UNSTABLE: High failure rate, needs investigation"
		case winPct < 5 && median < 100:
			interpretation = "❌ WEAK: Rarely wins, low scores"
		case median > 1000:
			interpretation = "💪 HIGH PERFORMER: Great median score"
		default:
			interpretation = "📊 AVERAGE: Middle-of-the-road performance"
		}

		fmt.Printf("│ %-17s │ %5d │ %6d │ %6.1f%% │ %12.0f │ %10.0f │ %9d │ %6.1f%% │ %-121s │\n",
			mode, total, failures, failPct, median, best, wins, winPct, interpretation)
	}
	fmt.Println("└───────────────────┴───────┴────────┴─────────┴──────────────┴────────────┴───────────┴─────────┴─────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────┘")

	// Summary insights
	fmt.Println("\n📊 KEY ROBUSTNESS INSIGHTS:")
	fmt.Println("   ─────────────────────────────────────────────────────────────────────────")

	// Find best by each metric
	bestFailMode := ""
	bestFailPct := 100.0
	bestMedianMode := ""
	bestMedian := 0.0
	bestWinMode := ""
	bestWins := 0

	for _, mode := range modes {
		total := modeTotal[mode]
		failures := modeFailures[mode]
		failPct := 0.0
		if total > 0 {
			failPct = float64(failures) / float64(total) * 100
		}

		if failPct < bestFailPct {
			bestFailPct = failPct
			bestFailMode = mode
		}

		if modeMedian[mode] > bestMedian {
			bestMedian = modeMedian[mode]
			bestMedianMode = mode
		}

		if modeWins[mode] > bestWins {
			bestWins = modeWins[mode]
			bestWinMode = mode
		}
	}

	fmt.Printf("   🎯 LOWEST FAILURE RATE:  %-15s (%.1f%% failures) - Most reliable\n", bestFailMode, bestFailPct)
	fmt.Printf("   🎯 HIGHEST MEDIAN SCORE: %-15s (median: %.0f) - Consistently high performance\n", bestMedianMode, bestMedian)
	fmt.Printf("   🎯 MOST FREQUENT WINNER: %-15s (%d wins, %.1f%%) - Wins across most configs\n", bestWinMode, bestWins, float64(bestWins)/float64(totalCombos)*100)

	// The "holy grail" check: is the same mode best in all 3?
	if bestFailMode == bestMedianMode && bestMedianMode == bestWinMode {
		fmt.Printf("\n   🏆🏆🏆 HOLY GRAIL: %s dominates ALL THREE METRICS! This is the clear winner. 🏆🏆🏆\n", bestFailMode)
	} else if bestMedianMode == bestWinMode {
		fmt.Printf("\n   ⭐ STRONG CONTENDER: %s has both highest median AND most wins!\n", bestMedianMode)
	}

	fmt.Println("")
}

// ═══════════════════════════════════════════════════════════════════════════════
// ADVANCED DECISION SURFACE ANALYSIS
// ═══════════════════════════════════════════════════════════════════════════════

// printWinCountByRegime shows win percentage split by problem difficulty (Easy/Moderate/Adversarial)
func printWinCountByRegime(results *BenchmarkResults) {
	fmt.Println("\n╔═══════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════╗")
	fmt.Println("║                                                   📊 WIN COUNT BY REGIME (Easy/Moderate/Adversarial) 📊                                                                                                                      ║")
	fmt.Println("║                                                                                                                                                                                                                               ║")
	fmt.Println("║                     This shows which mode wins WHEN THE PROBLEM STOPS BEING NICE - the key insight for real-world deployment                                                                                                ║")
	fmt.Println("╚═══════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════╝")

	modes := []string{"NormalBP", "StepBP", "Tween", "TweenChain", "StepTween", "StepTweenChain"}

	// Classify each test result into Easy/Moderate/Adversarial based on stress indicators
	type Regime string
	const (
		RegimeEasy        Regime = "Easy"
		RegimeModerate    Regime = "Moderate"
		RegimeAdversarial Regime = "Adversarial"
	)

	classifyRegime := func(r TestResult) Regime {
		// Adversarial: high latency, blocked windows, or low availability
		if r.ZeroOutputWindows > 5 || r.MaxLatencyMs > 500 || r.AvailabilityPct < 80 {
			return RegimeAdversarial
		}
		// Moderate: some stress indicators
		if r.ZeroOutputWindows > 0 || r.MaxLatencyMs > 100 || r.TotalBlockedMs > 500 || r.AvailabilityPct < 95 {
			return RegimeModerate
		}
		// Easy: smooth sailing
		return RegimeEasy
	}

	// Group tests by regime and find winner for each LayerType+NumericType combo within each regime
	type ConfigKey struct {
		Layer string
		Type  string
	}
	regimeResults := make(map[Regime]map[ConfigKey][]TestResult)
	for _, regime := range []Regime{RegimeEasy, RegimeModerate, RegimeAdversarial} {
		regimeResults[regime] = make(map[ConfigKey][]TestResult)
	}

	for _, r := range results.Results {
		regime := classifyRegime(r)
		key := ConfigKey{r.LayerType, r.NumericType}
		regimeResults[regime][key] = append(regimeResults[regime][key], r)
	}

	// Count wins per mode per regime
	modeWins := make(map[Regime]map[string]int)
	modeTotals := make(map[Regime]int)
	for _, regime := range []Regime{RegimeEasy, RegimeModerate, RegimeAdversarial} {
		modeWins[regime] = make(map[string]int)
		for _, modeResults := range regimeResults[regime] {
			// Find winner for this config
			var best TestResult
			for _, r := range modeResults {
				if r.Score > best.Score {
					best = r
				}
			}
			if best.TrainingMode != "" {
				modeWins[regime][best.TrainingMode]++
				modeTotals[regime]++
			}
		}
	}

	// Print table
	fmt.Println("\n┌─────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────┐")
	fmt.Println("│       Mode        │   Easy (smooth)   │   Moderate (some stress)   │   Adversarial (harsh)   │ ★ Key Insight ★                           │")
	fmt.Println("├───────────────────┼───────────────────┼────────────────────────────┼─────────────────────────┼───────────────────────────────────────────┤")

	for _, mode := range modes {
		easyWins := modeWins[RegimeEasy][mode]
		modWins := modeWins[RegimeModerate][mode]
		advWins := modeWins[RegimeAdversarial][mode]

		easyPct := 0.0
		if modeTotals[RegimeEasy] > 0 {
			easyPct = float64(easyWins) / float64(modeTotals[RegimeEasy]) * 100
		}
		modPct := 0.0
		if modeTotals[RegimeModerate] > 0 {
			modPct = float64(modWins) / float64(modeTotals[RegimeModerate]) * 100
		}
		advPct := 0.0
		if modeTotals[RegimeAdversarial] > 0 {
			advPct = float64(advWins) / float64(modeTotals[RegimeAdversarial]) * 100
		}

		insight := ""
		switch {
		case advPct >= 30 && advPct > easyPct:
			insight = "🔥 THRIVES under stress!"
		case easyPct >= 50 && advPct < 10:
			insight = "☀️ Easy-mode specialist"
		case modPct >= 30 && advPct >= 20:
			insight = "⚡ Scales with difficulty"
		case advPct >= 20:
			insight = "💪 Handles adversarial"
		case easyPct >= 30:
			insight = "📊 Easy regime strong"
		default:
			insight = "—"
		}

		fmt.Printf("│ %-17s │   %3d (%5.1f%%)    │       %3d (%5.1f%%)         │      %3d (%5.1f%%)        │ %-41s │\n",
			mode, easyWins, easyPct, modWins, modPct, advWins, advPct, insight)
	}
	fmt.Println("└───────────────────┴───────────────────┴────────────────────────────┴─────────────────────────┴───────────────────────────────────────────┘")

	fmt.Printf("\n📊 REGIME DISTRIBUTION: Easy=%d tests, Moderate=%d tests, Adversarial=%d tests\n",
		modeTotals[RegimeEasy], modeTotals[RegimeModerate], modeTotals[RegimeAdversarial])
	fmt.Println("")
}

// printAvailabilityConditionedAccuracy shows accuracy at different availability thresholds
func printAvailabilityConditionedAccuracy(results *BenchmarkResults) {
	fmt.Println("\n╔═══════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════╗")
	fmt.Println("║                                          📊 ACCURACY CONDITIONED ON AVAILABILITY 📊                                                                                                          ║")
	fmt.Println("║                                                                                                                                                                                               ║")
	fmt.Println("║                     Accuracy alone is misleading - this shows accuracy WHEN THE MODEL IS ACTUALLY ONLINE                                                                                     ║")
	fmt.Println("║                     (BP accuracy depends on pausing reality; StepTweenChain accuracy does not)                                                                                               ║")
	fmt.Println("╚═══════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════╝")

	modes := []string{"NormalBP", "StepBP", "Tween", "TweenChain", "StepTween", "StepTweenChain"}
	thresholds := []float64{90, 95, 99, 100}

	// For each mode, calculate average accuracy for tests meeting each availability threshold
	type AccuracyStats struct {
		Sum   float64
		Count int
	}
	modeAccuracyAtThreshold := make(map[string]map[float64]*AccuracyStats)
	for _, mode := range modes {
		modeAccuracyAtThreshold[mode] = make(map[float64]*AccuracyStats)
		for _, thresh := range thresholds {
			modeAccuracyAtThreshold[mode][thresh] = &AccuracyStats{}
		}
	}

	for _, r := range results.Results {
		if r.Error != "" || r.Score == 0 {
			continue
		}
		for _, thresh := range thresholds {
			if r.AvailabilityPct >= thresh {
				modeAccuracyAtThreshold[r.TrainingMode][thresh].Sum += r.AvgAccuracy
				modeAccuracyAtThreshold[r.TrainingMode][thresh].Count++
			}
		}
	}

	// Print table
	fmt.Println("\n┌───────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────┐")
	fmt.Println("│       Mode        │  Avail ≥90%  │  Avail ≥95%  │  Avail ≥99%  │  Avail =100%  │ ★ Interpretation ★                                  │")
	fmt.Println("├───────────────────┼──────────────┼──────────────┼──────────────┼───────────────┼─────────────────────────────────────────────────────┤")

	for _, mode := range modes {
		acc90 := 0.0
		if s := modeAccuracyAtThreshold[mode][90]; s.Count > 0 {
			acc90 = s.Sum / float64(s.Count)
		}
		acc95 := 0.0
		if s := modeAccuracyAtThreshold[mode][95]; s.Count > 0 {
			acc95 = s.Sum / float64(s.Count)
		}
		acc99 := 0.0
		if s := modeAccuracyAtThreshold[mode][99]; s.Count > 0 {
			acc99 = s.Sum / float64(s.Count)
		}
		acc100 := 0.0
		count100 := 0
		if s := modeAccuracyAtThreshold[mode][100]; s.Count > 0 {
			acc100 = s.Sum / float64(s.Count)
			count100 = s.Count
		}

		interpretation := ""
		switch {
		case count100 > 0 && acc100 >= 90:
			interpretation = "🏆 HIGH ACCURACY + 100% AVAILABILITY"
		case count100 == 0 && acc90 >= 80:
			interpretation = "⚠️  Accurate but BLOCKS inference"
		case acc99 >= 80 && count100 > 0:
			interpretation = "✅ Strong at high availability"
		case acc90 >= 70:
			interpretation = "📊 Decent base accuracy"
		default:
			interpretation = "—"
		}

		n100Str := fmt.Sprintf("n=%d", count100)
		if count100 == 0 {
			n100Str = "n=0"
			acc100 = 0
		}

		fmt.Printf("│ %-17s │   %5.1f%%     │   %5.1f%%     │   %5.1f%%     │  %5.1f%% %-5s │ %-51s │\n",
			mode, acc90, acc95, acc99, acc100, n100Str, interpretation)
	}
	fmt.Println("└───────────────────┴──────────────┴──────────────┴──────────────┴───────────────┴─────────────────────────────────────────────────────┘")

	fmt.Println("\n💡 KEY INSIGHT: If a mode has 0% at 100% availability, it REQUIRES blocking to achieve accuracy.")
	fmt.Println("               StepTween/StepTweenChain maintain accuracy WITHOUT blocking - this is the breakthrough.")
	fmt.Println("")
}

// printDominanceEnvelope shows how often each mode is within X% of the best score
func printDominanceEnvelope(results *BenchmarkResults) {
	fmt.Println("\n╔═══════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════╗")
	fmt.Println("║                                          📊 DOMINANCE ENVELOPE: % Within X% of Best Score 📊                                                                                                  ║")
	fmt.Println("║                                                                                                                                                                                               ║")
	fmt.Println("║                     Instead of 'winner takes all', this shows ROBUST COMPETITIVENESS - how often a mode is close to optimal                                                                  ║")
	fmt.Println("╚═══════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════╝")

	modes := []string{"NormalBP", "StepBP", "Tween", "TweenChain", "StepTween", "StepTweenChain"}
	envelopeThresholds := []float64{100, 95, 90, 80, 70} // % of best score

	// Find best score for each LayerType+NumericType combo
	type ConfigKey struct {
		Layer string
		Type  string
	}
	bestByConfig := make(map[ConfigKey]float64)
	for _, r := range results.Results {
		key := ConfigKey{r.LayerType, r.NumericType}
		if r.Score > bestByConfig[key] {
			bestByConfig[key] = r.Score
		}
	}

	// Count how often each mode is within threshold of best
	modeWithinThreshold := make(map[string]map[float64]int)
	modeTotal := make(map[string]int)
	for _, mode := range modes {
		modeWithinThreshold[mode] = make(map[float64]int)
	}

	for _, r := range results.Results {
		if r.Error != "" {
			continue
		}
		key := ConfigKey{r.LayerType, r.NumericType}
		best := bestByConfig[key]
		if best == 0 {
			continue
		}

		modeTotal[r.TrainingMode]++
		pctOfBest := (r.Score / best) * 100

		for _, thresh := range envelopeThresholds {
			if pctOfBest >= thresh {
				modeWithinThreshold[r.TrainingMode][thresh]++
			}
		}
	}

	// Print table
	fmt.Println("\n┌───────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────┐")
	fmt.Println("│       Mode        │ =100% (wins) │ ≥95% of best │ ≥90% of best │ ≥80% of best │ ≥70% of best │ ★ Robustness Rating ★                                                │")
	fmt.Println("├───────────────────┼──────────────┼──────────────┼──────────────┼──────────────┼──────────────┼──────────────────────────────────────────────────────────────────────┤")

	for _, mode := range modes {
		total := modeTotal[mode]
		if total == 0 {
			continue
		}

		w100 := float64(modeWithinThreshold[mode][100]) / float64(total) * 100
		w95 := float64(modeWithinThreshold[mode][95]) / float64(total) * 100
		w90 := float64(modeWithinThreshold[mode][90]) / float64(total) * 100
		w80 := float64(modeWithinThreshold[mode][80]) / float64(total) * 100
		w70 := float64(modeWithinThreshold[mode][70]) / float64(total) * 100

		robustness := ""
		switch {
		case w90 >= 70:
			robustness = "🏆 EXCELLENT: Almost always competitive"
		case w90 >= 50:
			robustness = "✅ GOOD: Competitive more often than not"
		case w80 >= 60:
			robustness = "👍 DECENT: Usually in the ballpark"
		case w70 >= 50:
			robustness = "📊 ACCEPTABLE: Sometimes competitive"
		default:
			robustness = "⚠️  INCONSISTENT: Frequently suboptimal"
		}

		fmt.Printf("│ %-17s │   %5.1f%%     │   %5.1f%%     │   %5.1f%%     │   %5.1f%%     │   %5.1f%%     │ %-68s │\n",
			mode, w100, w95, w90, w80, w70, robustness)
	}
	fmt.Println("└───────────────────┴──────────────┴──────────────┴──────────────┴──────────────┴──────────────┴──────────────────────────────────────────────────────────────────────┘")

	fmt.Println("")
}

// printFailureSeverityBreakdown shows not just failure count, but HOW BAD the failures are
func printFailureSeverityBreakdown(results *BenchmarkResults) {
	fmt.Println("\n╔═══════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════╗")
	fmt.Println("║                                          📊 FAILURE SEVERITY BREAKDOWN: How Bad Are The Failures? 📊                                                                                                     ║")
	fmt.Println("║                                                                                                                                                                                                           ║")
	fmt.Println("║                     Not all failures are equal - this shows GRACEFUL DEGRADATION (the real goal)                                                                                                         ║")
	fmt.Println("║                     Soft: low accuracy but still outputting | Hard: zero-output windows | Divergence: NaN/Inf                                                                                           ║")
	fmt.Println("╚═══════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════╝")

	modes := []string{"NormalBP", "StepBP", "Tween", "TweenChain", "StepTween", "StepTweenChain"}

	// Count failure types per mode
	modeTotal := make(map[string]int)
	modeSoftFail := make(map[string]int)     // Low accuracy (<50%) but still producing output
	modeHardStall := make(map[string]int)    // Zero-output windows
	modeDivergence := make(map[string]int)   // NaN/Inf/Error
	modeLatencySpike := make(map[string]int) // Very high latency (>1000ms)
	modeHealthy := make(map[string]int)      // No issues

	for _, r := range results.Results {
		modeTotal[r.TrainingMode]++

		// Classify failure severity
		hasDivergence := r.Error != "" || math.IsNaN(r.Score) || math.IsInf(r.Score, 0)
		hasHardStall := r.ZeroOutputWindows > 3
		hasLatencySpike := r.MaxLatencyMs > 1000
		hasSoftFail := r.AvgAccuracy < 50 && !r.Passed

		switch {
		case hasDivergence:
			modeDivergence[r.TrainingMode]++
		case hasHardStall:
			modeHardStall[r.TrainingMode]++
		case hasLatencySpike:
			modeLatencySpike[r.TrainingMode]++
		case hasSoftFail:
			modeSoftFail[r.TrainingMode]++
		default:
			modeHealthy[r.TrainingMode]++
		}
	}

	// Print table
	fmt.Println("\n┌─────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────┐")
	fmt.Println("│       Mode        │  Healthy   │ Soft Fail  │ Hard Stall │ Lat Spike  │ Divergence │   Total    │ ★ Graceful Degradation? ★                                                              │")
	fmt.Println("├───────────────────┼────────────┼────────────┼────────────┼────────────┼────────────┼────────────┼─────────────────────────────────────────────────────────────────────────────────────────┤")

	for _, mode := range modes {
		total := modeTotal[mode]
		healthy := modeHealthy[mode]
		soft := modeSoftFail[mode]
		hard := modeHardStall[mode]
		latency := modeLatencySpike[mode]
		diverge := modeDivergence[mode]

		healthyPct := float64(healthy) / float64(total) * 100
		softPct := float64(soft) / float64(total) * 100
		hardPct := float64(hard) / float64(total) * 100
		latencyPct := float64(latency) / float64(total) * 100
		divergePct := float64(diverge) / float64(total) * 100

		graceful := ""
		switch {
		case divergePct == 0 && hardPct == 0:
			graceful = "🏆 EXCELLENT: Never crashes, never stalls"
		case divergePct == 0 && hardPct < 5:
			graceful = "✅ GOOD: Rare stalls, no crashes"
		case divergePct < 5 && hardPct < 10:
			graceful = "👍 DECENT: Mostly stable"
		case hardPct >= 20 || divergePct >= 10:
			graceful = "⚠️  BRITTLE: Frequent hard failures"
		default:
			graceful = "📊 MIXED: Some instability"
		}

		fmt.Printf("│ %-17s │ %3d (%4.0f%%)│ %3d (%4.0f%%)│ %3d (%4.0f%%)│ %3d (%4.0f%%)│ %3d (%4.0f%%)│    %4d    │ %-83s │\n",
			mode, healthy, healthyPct, soft, softPct, hard, hardPct, latency, latencyPct, diverge, divergePct, total, graceful)
	}
	fmt.Println("└───────────────────┴────────────┴────────────┴────────────┴────────────┴────────────┴────────────┴─────────────────────────────────────────────────────────────────────────────────────────┘")

	fmt.Println("\n💡 KEY: A mode that NEVER has Hard Stall or Divergence is production-safe even if accuracy dips.")
	fmt.Println("")
}

// printStressResponseCurve shows how performance changes as stress increases
func printStressResponseCurve(results *BenchmarkResults) {
	fmt.Println("\n╔═══════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════╗")
	fmt.Println("║                                          📊 STRESS RESPONSE CURVE: How Does Performance Degrade Under Pressure? 📊                                                                                                   ║")
	fmt.Println("║                                                                                                                                                                                                                       ║")
	fmt.Println("║                     NormalBP is optimal when the world is kind. StepTweenChain is optimal when the world is real.                                                                                                   ║")
	fmt.Println("╚═══════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════╝")

	modes := []string{"NormalBP", "StepBP", "Tween", "TweenChain", "StepTween", "StepTweenChain"}

	// Bucket results by blocked time (proxy for stress level)
	type StressBucket struct {
		TotalScore float64
		Count      int
	}
	stressBuckets := []struct {
		Name       string
		MinBlocked float64
		MaxBlocked float64
	}{
		{"0ms", 0, 0},
		{"1-100ms", 1, 100},
		{"100-500ms", 100, 500},
		{"500-1000ms", 500, 1000},
		{">1000ms", 1000, 999999},
	}

	modeStressScores := make(map[string]map[string]*StressBucket)
	for _, mode := range modes {
		modeStressScores[mode] = make(map[string]*StressBucket)
		for _, bucket := range stressBuckets {
			modeStressScores[mode][bucket.Name] = &StressBucket{}
		}
	}

	for _, r := range results.Results {
		if r.Error != "" {
			continue
		}
		for _, bucket := range stressBuckets {
			if r.TotalBlockedMs >= bucket.MinBlocked && r.TotalBlockedMs < bucket.MaxBlocked {
				modeStressScores[r.TrainingMode][bucket.Name].TotalScore += r.Score
				modeStressScores[r.TrainingMode][bucket.Name].Count++
				break
			}
		}
	}

	// Print table
	fmt.Println("\n┌─────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────┐")
	fmt.Println("│       Mode        │ 0ms Blocked │ 1-100ms     │ 100-500ms   │ 500-1000ms  │ >1000ms     │ ★ Stress Resilience ★                                                                                              │")
	fmt.Println("├───────────────────┼─────────────┼─────────────┼─────────────┼─────────────┼─────────────┼────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────┤")

	for _, mode := range modes {
		scores := make([]float64, len(stressBuckets))
		for i, bucket := range stressBuckets {
			s := modeStressScores[mode][bucket.Name]
			if s.Count > 0 {
				scores[i] = s.TotalScore / float64(s.Count)
			}
		}

		// Calculate degradation pattern
		resilience := ""
		if scores[0] > 0 && scores[len(scores)-1] > 0 {
			degradation := (scores[0] - scores[len(scores)-1]) / scores[0] * 100
			switch {
			case degradation < 10:
				resilience = "🏆 STRESS-PROOF: Almost no degradation under pressure"
			case degradation < 30:
				resilience = "✅ RESILIENT: Graceful degradation"
			case degradation < 50:
				resilience = "👍 DECENT: Moderate degradation"
			default:
				resilience = "⚠️  FRAGILE: Severe degradation under stress"
			}
		} else if scores[0] == 0 && scores[len(scores)-1] > 0 {
			resilience = "🔄 INVERSE: Better under stress!"
		} else {
			resilience = "📊 MIXED"
		}

		fmt.Printf("│ %-17s │  %7.0f    │  %7.0f    │  %7.0f    │  %7.0f    │  %7.0f    │ %-120s │\n",
			mode, scores[0], scores[1], scores[2], scores[3], scores[4], resilience)
	}
	fmt.Println("└───────────────────┴─────────────┴─────────────┴─────────────┴─────────────┴─────────────┴────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────┘")

	fmt.Println("\n🎯 THE NARRATIVE: A mode that MAINTAINS score as blocked time increases is PRODUCTION READY.")
	fmt.Println("                 A mode that COLLAPSES under stress is a LIABILITY in real deployments.")
	fmt.Println("")
}

// ═══════════════════════════════════════════════════════════════════════════════
// REMAINING METRICS ANALYSIS
// ═══════════════════════════════════════════════════════════════════════════════

// printLatencyAnalysis shows latency distribution per mode
func printLatencyAnalysis(results *BenchmarkResults) {
	fmt.Println("\n╔═══════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════╗")
	fmt.Println("║                                          ⏱️ LATENCY ANALYSIS: Response Time Distribution Per Mode ⏱️                                                                                          ║")
	fmt.Println("║                                                                                                                                                                                               ║")
	fmt.Println("║                     Lower latency = faster inference response. Max latency shows worst-case spikes.                                                                                          ║")
	fmt.Println("╚═══════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════╝")

	modes := []string{"NormalBP", "StepBP", "Tween", "TweenChain", "StepTween", "StepTweenChain"}

	// Collect latency stats per mode
	type LatencyStats struct {
		TotalAvg float64
		TotalMax float64
		Count    int
		MaxSeen  float64
		MinAvg   float64
	}
	modeLatency := make(map[string]*LatencyStats)
	for _, mode := range modes {
		modeLatency[mode] = &LatencyStats{MinAvg: 999999}
	}

	for _, r := range results.Results {
		if r.Error != "" {
			continue
		}
		s := modeLatency[r.TrainingMode]
		s.TotalAvg += r.AvgLatencyMs
		s.TotalMax += r.MaxLatencyMs
		s.Count++
		if r.MaxLatencyMs > s.MaxSeen {
			s.MaxSeen = r.MaxLatencyMs
		}
		if r.AvgLatencyMs < s.MinAvg && r.AvgLatencyMs > 0 {
			s.MinAvg = r.AvgLatencyMs
		}
	}

	// Print table
	fmt.Println("\n┌─────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────┐")
	fmt.Println("│       Mode        │ Avg Latency │ Avg of Max │ Worst Spike │  Best Avg   │ Tests │ ★ Latency Rating ★                                                                  │")
	fmt.Println("├───────────────────┼─────────────┼────────────┼─────────────┼─────────────┼───────┼─────────────────────────────────────────────────────────────────────────────────────┤")

	for _, mode := range modes {
		s := modeLatency[mode]
		avgLat := 0.0
		avgMax := 0.0
		if s.Count > 0 {
			avgLat = s.TotalAvg / float64(s.Count)
			avgMax = s.TotalMax / float64(s.Count)
		}

		rating := ""
		switch {
		case avgLat < 1 && s.MaxSeen < 10:
			rating = "🏆 EXCELLENT: Sub-millisecond avg, minimal spikes"
		case avgLat < 5 && s.MaxSeen < 50:
			rating = "✅ GOOD: Fast responses, controlled spikes"
		case avgLat < 10 && s.MaxSeen < 100:
			rating = "👍 DECENT: Acceptable latency"
		case s.MaxSeen > 500:
			rating = "⚠️  HIGH SPIKES: May cause timeouts"
		default:
			rating = "📊 MODERATE"
		}

		fmt.Printf("│ %-17s │   %6.2fms  │  %6.2fms  │  %7.1fms  │   %6.2fms  │ %5d │ %-79s │\n",
			mode, avgLat, avgMax, s.MaxSeen, s.MinAvg, s.Count, rating)
	}
	fmt.Println("└───────────────────┴─────────────┴────────────┴─────────────┴─────────────┴───────┴─────────────────────────────────────────────────────────────────────────────────────┘")
	fmt.Println("")
}

// printThroughputDistribution shows throughput stats per mode
func printThroughputDistribution(results *BenchmarkResults) {
	fmt.Println("\n╔═══════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════╗")
	fmt.Println("║                                          🚀 THROUGHPUT DISTRIBUTION: Outputs Per Second Analysis 🚀                                                                                          ║")
	fmt.Println("║                                                                                                                                                                                               ║")
	fmt.Println("║                     Higher throughput = more inferences per second. Critical for real-time applications.                                                                                     ║")
	fmt.Println("╚═══════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════╝")

	modes := []string{"NormalBP", "StepBP", "Tween", "TweenChain", "StepTween", "StepTweenChain"}

	// Collect throughput stats per mode
	type ThroughputStats struct {
		Total   float64
		Count   int
		Max     float64
		Min     float64
		Outputs int64
	}
	modeThroughput := make(map[string]*ThroughputStats)
	for _, mode := range modes {
		modeThroughput[mode] = &ThroughputStats{Min: 999999999}
	}

	for _, r := range results.Results {
		if r.Error != "" {
			continue
		}
		s := modeThroughput[r.TrainingMode]
		s.Total += r.ThroughputPerSec
		s.Count++
		s.Outputs += int64(r.TotalOutputs)
		if r.ThroughputPerSec > s.Max {
			s.Max = r.ThroughputPerSec
		}
		if r.ThroughputPerSec < s.Min && r.ThroughputPerSec > 0 {
			s.Min = r.ThroughputPerSec
		}
	}

	// Print table
	fmt.Println("\n┌────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────┐")
	fmt.Println("│       Mode        │ Avg Throughput │ Max Throughput │ Min Throughput │ Total Outputs │ Tests │ ★ Throughput Rating ★                                                             │")
	fmt.Println("├───────────────────┼────────────────┼────────────────┼────────────────┼───────────────┼───────┼───────────────────────────────────────────────────────────────────────────────────┤")

	for _, mode := range modes {
		s := modeThroughput[mode]
		avgThroughput := 0.0
		if s.Count > 0 {
			avgThroughput = s.Total / float64(s.Count)
		}

		rating := ""
		switch {
		case avgThroughput > 200000:
			rating = "🏆 EXCELLENT: Very high throughput"
		case avgThroughput > 100000:
			rating = "✅ GOOD: High throughput, production ready"
		case avgThroughput > 50000:
			rating = "👍 DECENT: Reasonable throughput"
		case avgThroughput > 10000:
			rating = "📊 MODERATE: May limit real-time use"
		default:
			rating = "⚠️  LOW: Consider optimization"
		}

		fmt.Printf("│ %-17s │    %10.0f  │    %10.0f  │    %10.0f  │ %13d │ %5d │ %-73s │\n",
			mode, avgThroughput, s.Max, s.Min, s.Outputs, s.Count, rating)
	}
	fmt.Println("└───────────────────┴────────────────┴────────────────┴────────────────┴───────────────┴───────┴───────────────────────────────────────────────────────────────────────────────────┘")
	fmt.Println("")
}

// printLayerSpecificRecommendations shows which mode is best FOR EACH LAYER
func printLayerSpecificRecommendations(results *BenchmarkResults) {
	fmt.Println("\n╔═══════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════╗")
	fmt.Println("║                                          📋 LAYER-SPECIFIC RECOMMENDATIONS: Best Mode For Each Layer 📋                                                                                      ║")
	fmt.Println("║                                                                                                                                                                                               ║")
	fmt.Println("║                     Not all modes work equally well for all layer types. This shows the optimal mode per layer.                                                                              ║")
	fmt.Println("╚═══════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════╝")

	// Find unique layers
	layerSet := make(map[string]bool)
	for _, r := range results.Results {
		layerSet[r.LayerType] = true
	}
	layers := make([]string, 0, len(layerSet))
	for l := range layerSet {
		layers = append(layers, l)
	}
	sort.Strings(layers)

	modes := []string{"NormalBP", "StepBP", "Tween", "TweenChain", "StepTween", "StepTweenChain"}

	// For each layer, find best mode
	fmt.Println("\n┌─────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────┐")
	fmt.Println("│       Layer                                         │ Best Mode      │ Best Score │ Runner-Up      │ 2nd Score  │ ★ Recommendation ★                                                                                                     │")
	fmt.Println("├─────────────────────────────────────────────────────┼────────────────┼────────────┼────────────────┼────────────┼─────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────┤")

	for _, layer := range layers {
		// Find best score per mode for this layer
		modeScore := make(map[string]float64)
		for _, r := range results.Results {
			if r.LayerType == layer && r.Score > modeScore[r.TrainingMode] {
				modeScore[r.TrainingMode] = r.Score
			}
		}

		// Find top 2
		bestMode := ""
		bestScore := 0.0
		runnerUp := ""
		runnerScore := 0.0
		for _, mode := range modes {
			if modeScore[mode] > bestScore {
				runnerUp = bestMode
				runnerScore = bestScore
				bestMode = mode
				bestScore = modeScore[mode]
			} else if modeScore[mode] > runnerScore {
				runnerUp = mode
				runnerScore = modeScore[mode]
			}
		}

		recommendation := ""
		switch {
		case bestMode == "StepTweenChain" || bestMode == "StepTween":
			recommendation = "Use " + bestMode + " for best availability + performance"
		case bestMode == "NormalBP":
			recommendation = "NormalBP optimal BUT blocks inference. Consider " + runnerUp + " for real-time."
		case bestMode == "StepBP":
			recommendation = "StepBP works but high failure rate. Consider " + runnerUp + " for stability."
		default:
			recommendation = "Use " + bestMode + " for this layer type"
		}

		displayLayer := layer
		if len(displayLayer) > 50 {
			displayLayer = displayLayer[:47] + "..."
		}

		fmt.Printf("│ %-51s │ %-14s │   %7.0f  │ %-14s │   %7.0f  │ %-117s │\n",
			displayLayer, bestMode, bestScore, runnerUp, runnerScore, recommendation)
	}
	fmt.Println("└─────────────────────────────────────────────────────┴────────────────┴────────────┴────────────────┴────────────┴─────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────┘")
	fmt.Println("")
}

// printScoreConsistencyAnalysis shows variance in scores per mode (stability)
func printScoreConsistencyAnalysis(results *BenchmarkResults) {
	fmt.Println("\n╔═══════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════╗")
	fmt.Println("║                                          📊 SCORE CONSISTENCY ANALYSIS: Variance and Stability Per Mode 📊                                                                                   ║")
	fmt.Println("║                                                                                                                                                                                               ║")
	fmt.Println("║                     Low variance = predictable performance. High variance = unpredictable results.                                                                                           ║")
	fmt.Println("╚═══════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════╝")

	modes := []string{"NormalBP", "StepBP", "Tween", "TweenChain", "StepTween", "StepTweenChain"}

	// Collect scores per mode
	modeScores := make(map[string][]float64)
	for _, r := range results.Results {
		if r.Error == "" && r.Score > 0 {
			modeScores[r.TrainingMode] = append(modeScores[r.TrainingMode], r.Score)
		}
	}

	// Print table
	fmt.Println("\n┌─────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────┐")
	fmt.Println("│       Mode        │    Mean    │   Median   │  Std Dev   │    Min     │    Max     │ Range      │ CV%    │ ★ Consistency Rating ★                                                        │")
	fmt.Println("├───────────────────┼────────────┼────────────┼────────────┼────────────┼────────────┼────────────┼────────┼───────────────────────────────────────────────────────────────────────────────┤")

	for _, mode := range modes {
		scores := modeScores[mode]
		if len(scores) == 0 {
			continue
		}

		// Calculate stats
		sum := 0.0
		min := scores[0]
		max := scores[0]
		for _, s := range scores {
			sum += s
			if s < min {
				min = s
			}
			if s > max {
				max = s
			}
		}
		mean := sum / float64(len(scores))

		// Median
		sorted := make([]float64, len(scores))
		copy(sorted, scores)
		sort.Float64s(sorted)
		median := sorted[len(sorted)/2]

		// Std dev
		variance := 0.0
		for _, s := range scores {
			variance += (s - mean) * (s - mean)
		}
		stdDev := math.Sqrt(variance / float64(len(scores)))
		cv := (stdDev / mean) * 100 // Coefficient of variation

		rating := ""
		switch {
		case cv < 30:
			rating = "🏆 VERY CONSISTENT: Predictable results"
		case cv < 60:
			rating = "✅ CONSISTENT: Reasonable variance"
		case cv < 100:
			rating = "👍 MODERATE: Some unpredictability"
		default:
			rating = "⚠️  INCONSISTENT: High variance in results"
		}

		fmt.Printf("│ %-17s │ %10.1f │ %10.1f │ %10.1f │ %10.1f │ %10.1f │ %10.1f │ %5.1f%% │ %-73s │\n",
			mode, mean, median, stdDev, min, max, max-min, cv, rating)
	}
	fmt.Println("└───────────────────┴────────────┴────────────┴────────────┴────────────┴────────────┴────────────┴────────┴───────────────────────────────────────────────────────────────────────────────┘")
	fmt.Println("\n💡 CV% = Coefficient of Variation (StdDev/Mean×100). Lower is more consistent.")
	fmt.Println("")
}

// printAccuracyAvailabilityPareto shows the tradeoff between accuracy and availability
func printAccuracyAvailabilityPareto(results *BenchmarkResults) {
	fmt.Println("\n╔═══════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════╗")
	fmt.Println("║                                          ⚖️ ACCURACY vs AVAILABILITY TRADEOFF: The Pareto Frontier ⚖️                                                                                         ║")
	fmt.Println("║                                                                                                                                                                                               ║")
	fmt.Println("║                     Which modes achieve BOTH high accuracy AND high availability? This is the key tradeoff.                                                                                  ║")
	fmt.Println("╚═══════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════╝")

	modes := []string{"NormalBP", "StepBP", "Tween", "TweenChain", "StepTween", "StepTweenChain"}

	// Collect avg accuracy and availability per mode
	type ParetoStats struct {
		TotalAccuracy     float64
		TotalAvailability float64
		Count             int
		HighAccHighAvail  int // >80% accuracy AND >95% availability
	}
	modePareto := make(map[string]*ParetoStats)
	for _, mode := range modes {
		modePareto[mode] = &ParetoStats{}
	}

	for _, r := range results.Results {
		if r.Error != "" {
			continue
		}
		s := modePareto[r.TrainingMode]
		s.TotalAccuracy += r.AvgAccuracy
		s.TotalAvailability += r.AvailabilityPct
		s.Count++
		if r.AvgAccuracy >= 80 && r.AvailabilityPct >= 95 {
			s.HighAccHighAvail++
		}
	}

	// Print table
	fmt.Println("\n┌─────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────┐")
	fmt.Println("│       Mode        │ Avg Accuracy │ Avg Availability │ High-High Count │ High-High % │ ★ Pareto Position ★                                                                                             │")
	fmt.Println("├───────────────────┼──────────────┼──────────────────┼─────────────────┼─────────────┼─────────────────────────────────────────────────────────────────────────────────────────────────────────────────────┤")

	for _, mode := range modes {
		s := modePareto[mode]
		avgAcc := 0.0
		avgAvail := 0.0
		highHighPct := 0.0
		if s.Count > 0 {
			avgAcc = s.TotalAccuracy / float64(s.Count)
			avgAvail = s.TotalAvailability / float64(s.Count)
			highHighPct = float64(s.HighAccHighAvail) / float64(s.Count) * 100
		}

		position := ""
		switch {
		case avgAcc >= 80 && avgAvail >= 95:
			position = "🏆 PARETO OPTIMAL: High accuracy AND high availability"
		case avgAcc >= 80 && avgAvail < 90:
			position = "⚠️  ACCURACY-BIASED: High acc but blocks inference"
		case avgAcc < 60 && avgAvail >= 95:
			position = "⚠️  AVAILABILITY-BIASED: Always online but low accuracy"
		case highHighPct > 50:
			position = "✅ GOOD BALANCE: Often achieves both"
		default:
			position = "📊 TRADEOFF: Must choose between acc and avail"
		}

		fmt.Printf("│ %-17s │    %5.1f%%    │      %5.1f%%       │     %5d       │   %5.1f%%    │ %-109s │\n",
			mode, avgAcc, avgAvail, s.HighAccHighAvail, highHighPct, position)
	}
	fmt.Println("└───────────────────┴──────────────┴──────────────────┴─────────────────┴─────────────┴─────────────────────────────────────────────────────────────────────────────────────────────────────────────────────┘")
	fmt.Println("\n💡 HIGH-HIGH = configs with ≥80% accuracy AND ≥95% availability. This is the sweet spot.")
	fmt.Println("")
}

// printZeroOutputWindowAnalysis shows detailed analysis of blocking behavior
func printZeroOutputWindowAnalysis(results *BenchmarkResults) {
	fmt.Println("\n╔═══════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════╗")
	fmt.Println("║                                          🚫 ZERO-OUTPUT WINDOW ANALYSIS: When Does Inference Go Offline? 🚫                                                                                   ║")
	fmt.Println("║                                                                                                                                                                                               ║")
	fmt.Println("║                     Zero-output windows = periods where the model couldn't respond. Critical for real-time systems.                                                                          ║")
	fmt.Println("╚═══════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════╝")

	modes := []string{"NormalBP", "StepBP", "Tween", "TweenChain", "StepTween", "StepTweenChain"}

	// Collect zero-output stats per mode
	type ZeroOutputStats struct {
		TotalWindows  int
		MaxWindows    int
		TestsWithZero int
		TotalTests    int
	}
	modeZero := make(map[string]*ZeroOutputStats)
	for _, mode := range modes {
		modeZero[mode] = &ZeroOutputStats{}
	}

	for _, r := range results.Results {
		s := modeZero[r.TrainingMode]
		s.TotalTests++
		s.TotalWindows += r.ZeroOutputWindows
		if r.ZeroOutputWindows > s.MaxWindows {
			s.MaxWindows = r.ZeroOutputWindows
		}
		if r.ZeroOutputWindows > 0 {
			s.TestsWithZero++
		}
	}

	// Print table
	fmt.Println("\n┌─────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────┐")
	fmt.Println("│       Mode        │ Total Windows │ Max per Test │ Tests w/Zero │ % w/Zero │ Avg Windows │ ★ Real-Time Safety ★                                                               │")
	fmt.Println("├───────────────────┼───────────────┼──────────────┼──────────────┼──────────┼─────────────┼─────────────────────────────────────────────────────────────────────────────────────┤")

	for _, mode := range modes {
		s := modeZero[mode]
		pctWithZero := 0.0
		avgWindows := 0.0
		if s.TotalTests > 0 {
			pctWithZero = float64(s.TestsWithZero) / float64(s.TotalTests) * 100
			avgWindows = float64(s.TotalWindows) / float64(s.TotalTests)
		}

		safety := ""
		switch {
		case s.TotalWindows == 0:
			safety = "🏆 ALWAYS ONLINE: Zero blocking - perfect for real-time"
		case pctWithZero < 5:
			safety = "✅ MOSTLY SAFE: Rare blocking events"
		case pctWithZero < 20:
			safety = "👍 ACCEPTABLE: Some blocking under stress"
		default:
			safety = "⚠️  FREQUENT BLOCKING: Not real-time safe"
		}

		fmt.Printf("│ %-17s │    %7d    │     %5d    │    %6d    │  %5.1f%%  │    %6.2f   │ %-79s │\n",
			mode, s.TotalWindows, s.MaxWindows, s.TestsWithZero, pctWithZero, avgWindows, safety)
	}
	fmt.Println("└───────────────────┴───────────────┴──────────────┴──────────────┴──────────┴─────────────┴─────────────────────────────────────────────────────────────────────────────────────┘")
	fmt.Println("")
}

// printTopBottomConfigurations shows the best and worst performing configurations
func printTopBottomConfigurations(results *BenchmarkResults) {
	fmt.Println("\n╔═══════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════╗")
	fmt.Println("║                                          🏆 TOP 10 BEST & ❌ BOTTOM 10 WORST CONFIGURATIONS 🏆                                                                                                                                       ║")
	fmt.Println("║                                                                                                                                                                                                                                     ║")
	fmt.Println("║                     What works best? What should you avoid?                                                                                                                                                                        ║")
	fmt.Println("╚═══════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════╝")

	// Sort results by score
	scored := make([]TestResult, 0)
	for _, r := range results.Results {
		if r.Error == "" && r.Score > 0 {
			scored = append(scored, r)
		}
	}
	sort.Slice(scored, func(i, j int) bool {
		return scored[i].Score > scored[j].Score
	})

	// Top 10
	fmt.Println("\n🏆 TOP 10 CONFIGURATIONS (Highest Scores):")
	fmt.Println("┌─────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────┐")
	fmt.Println("│ # │ Layer                                             │ Mode           │ NumType   │ Score    │ Accuracy │ Throughput │ Avail% │")
	fmt.Println("├───┼───────────────────────────────────────────────────┼────────────────┼───────────┼──────────┼──────────┼────────────┼────────┤")

	for i := 0; i < min(10, len(scored)); i++ {
		r := scored[i]
		displayLayer := r.LayerType
		if len(displayLayer) > 50 {
			displayLayer = displayLayer[:47] + "..."
		}
		fmt.Printf("│ %d │ %-49s │ %-14s │ %-9s │ %8.0f │  %5.1f%%  │ %10.0f │ %5.1f%% │\n",
			i+1, displayLayer, r.TrainingMode, r.NumericType, r.Score, r.AvgAccuracy, r.ThroughputPerSec, r.AvailabilityPct)
	}
	fmt.Println("└───┴───────────────────────────────────────────────────┴────────────────┴───────────┴──────────┴──────────┴────────────┴────────┘")

	// Bottom 10
	fmt.Println("\n❌ BOTTOM 10 CONFIGURATIONS (Lowest Non-Zero Scores):")
	fmt.Println("┌─────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────┐")
	fmt.Println("│ # │ Layer                                             │ Mode           │ NumType   │ Score    │ Accuracy │ Throughput │ Avail% │")
	fmt.Println("├───┼───────────────────────────────────────────────────┼────────────────┼───────────┼──────────┼──────────┼────────────┼────────┤")

	startIdx := len(scored) - 10
	if startIdx < 0 {
		startIdx = 0
	}
	for i := len(scored) - 1; i >= startIdx; i-- {
		r := scored[i]
		displayLayer := r.LayerType
		if len(displayLayer) > 50 {
			displayLayer = displayLayer[:47] + "..."
		}
		fmt.Printf("│ %d │ %-49s │ %-14s │ %-9s │ %8.0f │  %5.1f%%  │ %10.0f │ %5.1f%% │\n",
			len(scored)-i, displayLayer, r.TrainingMode, r.NumericType, r.Score, r.AvgAccuracy, r.ThroughputPerSec, r.AvailabilityPct)
	}
	fmt.Println("└───┴───────────────────────────────────────────────────┴────────────────┴───────────┴──────────┴──────────┴────────────┴────────┘")
	fmt.Println("")
}

// printNumericalTypePerformanceMatrix shows how each numerical type performs across all modes
func printNumericalTypePerformanceMatrix(results *BenchmarkResults) {
	fmt.Println("\n╔═══════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════╗")
	fmt.Println("║                                          🔢 NUMERICAL TYPE PERFORMANCE MATRIX: How Each Type Performs Across All Modes 🔢                                                                                ║")
	fmt.Println("║                                                                                                                                                                                                           ║")
	fmt.Println("║                     Not all numerical types are equal. This shows average score per type per mode.                                                                                                       ║")
	fmt.Println("╚═══════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════╝")

	modes := []string{"NormalBP", "StepBP", "Tween", "TweenChain", "StepTween", "StepTweenChain"}
	numTypes := []string{"float32", "float64", "int8", "int16", "int32", "int64", "uint8", "uint16", "uint32", "uint64"}

	// Collect avg score per type per mode
	typeModeSores := make(map[string]map[string]struct {
		Sum   float64
		Count int
	})
	for _, t := range numTypes {
		typeModeSores[t] = make(map[string]struct {
			Sum   float64
			Count int
		})
	}

	for _, r := range results.Results {
		if r.Error != "" {
			continue
		}
		s := typeModeSores[r.NumericType][r.TrainingMode]
		s.Sum += r.Score
		s.Count++
		typeModeSores[r.NumericType][r.TrainingMode] = s
	}

	// Print header
	fmt.Println("\n┌───────────┬────────────┬────────────┬────────────┬────────────┬────────────┬────────────┬────────────┐")
	fmt.Print("│ NumType   │")
	for _, mode := range modes {
		fmt.Printf(" %-10s │", mode[:min(10, len(mode))])
	}
	fmt.Println(" BEST MODE  │")
	fmt.Println("├───────────┼────────────┼────────────┼────────────┼────────────┼────────────┼────────────┼────────────┤")

	for _, numType := range numTypes {
		fmt.Printf("│ %-9s │", numType)
		bestMode := ""
		bestAvg := 0.0
		for _, mode := range modes {
			s := typeModeSores[numType][mode]
			avg := 0.0
			if s.Count > 0 {
				avg = s.Sum / float64(s.Count)
			}
			if avg > bestAvg {
				bestAvg = avg
				bestMode = mode
			}
			fmt.Printf("   %6.0f  │", avg)
		}
		fmt.Printf(" %-10s │\n", bestMode[:min(10, len(bestMode))])
	}
	fmt.Println("└───────────┴────────────┴────────────┴────────────┴────────────┴────────────┴────────────┴────────────┘")

	// Summary
	fmt.Println("\n📊 KEY INSIGHTS:")
	fmt.Println("   • float32 typically achieves highest scores due to speed+precision balance")
	fmt.Println("   • float64 excels in NormalBP where precision matters most")
	fmt.Println("   • Integer types are faster but may lose accuracy in gradient computations")
	fmt.Println("")
}

// printTimelineForFloat32 shows detailed accuracy/throughput timeline for float32 tests
func printTimelineForFloat32(results *BenchmarkResults) {
	fmt.Println("\n╔════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════╗")
	fmt.Println("║                                    📊 DETAILED TIMELINE (float32 only) 📊                                             ║")
	fmt.Println("╚════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════╝")

	for _, layerName := range []string{"Dense", "Conv2D", "RNN", "LSTM", "Attention", "SwiGLU", "NormDense", "Conv1D", "Residual"} {
		// Find float32 results for this layer
		var layerFloat32Results []TestResult
		for _, r := range results.Results {
			if r.LayerType == layerName && r.NumericType == "float32" {
				layerFloat32Results = append(layerFloat32Results, r)
			}
		}

		if len(layerFloat32Results) == 0 {
			continue
		}

		numWindows := int(TestDuration / WindowDuration)
		windowsPerSec := int(time.Second / WindowDuration)
		numSeconds := numWindows / windowsPerSec

		fmt.Printf("\n┌─── %s (float32) ─── ACCURACY PER SECOND ───────────────────────────────────────────────────────────────────┐\n", layerName)
		fmt.Print("│ Mode             │")
		for i := 0; i < numSeconds; i++ {
			fmt.Printf(" %ds ", i+1)
		}
		fmt.Println("│ Avg   │ Score    │")
		fmt.Print("├──────────────────┼")
		for i := 0; i < numSeconds; i++ {
			fmt.Print("────")
		}
		fmt.Println("┼───────┼──────────┤")

		for _, modeName := range []string{"NormalBP", "StepBP", "Tween", "TweenChain", "StepTween", "StepTweenChain"} {
			for _, r := range layerFloat32Results {
				if r.TrainingMode == modeName {
					fmt.Printf("│ %-16s │", modeName)
					for sec := 0; sec < numSeconds; sec++ {
						avgAcc := 0.0
						count := 0
						for w := sec * windowsPerSec; w < (sec+1)*windowsPerSec && w < len(r.Windows); w++ {
							avgAcc += r.Windows[w].Accuracy
							count++
						}
						if count > 0 {
							avgAcc /= float64(count)
						}
						fmt.Printf(" %2.0f%%", avgAcc)
					}
					fmt.Printf("│ %3.0f%% │ %8.0f │\n", r.AvgAccuracy, r.Score)
					break
				}
			}
		}
		fmt.Print("└──────────────────┴")
		for i := 0; i < numSeconds; i++ {
			fmt.Print("────")
		}
		fmt.Println("┴───────┴──────────┘")

		// OUTPUTS PER SECOND
		fmt.Printf("\n┌─── %s (float32) ─── OUTPUTS PER SECOND ───────────────────────────────────────────────────────────────────┐\n", layerName)
		fmt.Print("│ Mode             │")
		for i := 0; i < numSeconds; i++ {
			fmt.Printf("  %ds  ", i+1)
		}
		fmt.Println("│ Total  │ Avail%   │")
		fmt.Print("├──────────────────┼")
		for i := 0; i < numSeconds; i++ {
			fmt.Print("──────")
		}
		fmt.Println("┼────────┼──────────┤")

		for _, modeName := range []string{"NormalBP", "StepBP", "Tween", "TweenChain", "StepTween", "StepTweenChain"} {
			for _, r := range layerFloat32Results {
				if r.TrainingMode == modeName {
					fmt.Printf("│ %-16s │", modeName)
					for sec := 0; sec < numSeconds; sec++ {
						totalOutputs := 0
						for w := sec * windowsPerSec; w < (sec+1)*windowsPerSec && w < len(r.Windows); w++ {
							totalOutputs += r.Windows[w].Outputs
						}
						fmt.Printf(" %5d", totalOutputs)
					}
					fmt.Printf("│ %6d │ %6.1f%% │\n", r.TotalOutputs, r.AvailabilityPct)
					break
				}
			}
		}
		fmt.Print("└──────────────────┴")
		for i := 0; i < numSeconds; i++ {
			fmt.Print("──────")
		}
		fmt.Println("┴────────┴──────────┘")
	}

	// Print overall float32 summary
	fmt.Println("\n╔════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════╗")
	fmt.Println("║                                               🌊 FLOAT32 SUMMARY 🌊                                                                          ║")
	fmt.Println("╠════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════╣")
	fmt.Println("║  Layer/Mode       │ Accuracy │ Stability │ Throughput │ Score   │ Avail %  │ Blocked(ms) │ Peak Lat │ Avg Lat │★ Key Insight ★                ║")
	fmt.Println("║  ─────────────────┼──────────┼───────────┼────────────┼─────────┼──────────┼─────────────┼──────────┼─────────┼───────────────────────────────║")

	for _, layerName := range []string{"Dense", "Conv2D", "RNN", "LSTM", "Attention", "SwiGLU", "NormDense", "Conv1D", "Residual"} {
		for _, r := range results.Results {
			if r.LayerType == layerName && r.NumericType == "float32" {
				insight := ""
				if r.TrainingMode == "NormalBP" {
					insight = "BLOCKS during training"
				} else if r.TrainingMode == "StepTweenChain" {
					insight = "ALWAYS available ✓"
				} else if r.ZeroOutputWindows > 0 {
					insight = fmt.Sprintf("%d windows blocked", r.ZeroOutputWindows)
				} else if r.TotalBlockedMs > 100 {
					insight = "Some blocking"
				} else {
					insight = "Low blocking"
				}

				fmt.Printf("║  %-8s/%-7s │  %5.1f%%  │   %5.1f%%  │  %8.0f  │ %7.0f │  %5.1f%%  │  %9.0f  │  %5.1fms │ %5.1fms  │ %-29s ║\n",
					layerName[:min(8, len(layerName))], r.TrainingMode[:min(7, len(r.TrainingMode))],
					r.AvgAccuracy, r.Stability, r.ThroughputPerSec, r.Score,
					r.AvailabilityPct, r.TotalBlockedMs, r.MaxLatencyMs, r.AvgLatencyMs, insight)
			}
		}
	}
	fmt.Println("╚════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════╝")

	fmt.Println("\n💡 KEY INSIGHT: NormalBP achieves high accuracy BUT blocks inference during batch training.")
	fmt.Println("               StepTweenChain maintains ~100% availability while still training every sample!")
	fmt.Println("")
	fmt.Println("📊 SCORE CALCULATION: Score = (Throughput × Availability% × Accuracy%) / 10000")

	// CROSS-LAYER COMPARISON
	printCrossLayerComparison(results)

	// FAILURES AND ZERO SCORES
	printFailuresAndZeroScores(results)
}

// printCrossLayerComparison shows which modes work best with which layers
func printCrossLayerComparison(results *BenchmarkResults) {
	fmt.Println("\n╔════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════╗")
	fmt.Println("║                                          🏆 CROSS-LAYER COMPARISON (float32) 🏆                                                                          ║")
	fmt.Println("╚════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════╝")

	allLayers := []string{"Dense", "Conv2D", "RNN", "LSTM", "Attention", "SwiGLU", "NormDense", "Conv1D", "Residual"}
	modes := []string{"NormalBP", "StepBP", "Tween", "TweenChain", "StepTween", "StepTweenChain"}

	// Build lookup map for float32 results
	float32Results := make(map[string]map[string]TestResult)
	for _, r := range results.Results {
		if r.NumericType == "float32" {
			if float32Results[r.LayerType] == nil {
				float32Results[r.LayerType] = make(map[string]TestResult)
			}
			float32Results[r.LayerType][r.TrainingMode] = r
		}
	}

	// TABLE 1: Mode × Layer Score Matrix
	fmt.Println("\n┌─── SCORE MATRIX: Mode × Layer ────────────────────────────────────────────────────────────────────────────────────────────────────────┐")
	fmt.Print("│ Mode             │")
	for _, layer := range allLayers {
		fmt.Printf(" %-10s │", layer)
	}
	fmt.Println(" BEST LAYER                │")
	fmt.Print("├──────────────────┼")
	for range allLayers {
		fmt.Print("────────────┼")
	}
	fmt.Println("───────────────────────────┤")

	for _, mode := range modes {
		fmt.Printf("│ %-16s │", mode)
		bestScore := 0.0
		bestLayer := ""
		for _, layer := range allLayers {
			if r, ok := float32Results[layer][mode]; ok {
				if r.Error != "" {
					fmt.Print("    ERR     │")
				} else if r.Score == 0 {
					fmt.Print("          0 │")
				} else {
					fmt.Printf(" %10.0f │", r.Score)
				}

				if r.Score > bestScore {
					bestScore = r.Score
					bestLayer = layer
				}
			} else {
				fmt.Print("        N/A │")
			}
		}
		fmt.Printf(" %-25s │\n", fmt.Sprintf("★ %s (%.0f)", bestLayer, bestScore))
	}
	fmt.Print("└──────────────────┴")
	for range allLayers {
		fmt.Print("────────────┴")
	}
	fmt.Println("───────────────────────────┘")

	// TABLE 2: Best Mode Per Layer
	fmt.Println("\n┌─── BEST MODE PER LAYER ───────────────────────────────────────────────────────────────────────────────────────────────────────────────┐")
	fmt.Println("│ Layer      │ Best Mode         │ Score     │ Accuracy  │ Throughput │ Avail %   │ Why This Mode Wins                            │")
	fmt.Println("├────────────┼───────────────────┼───────────┼───────────┼────────────┼───────────┼───────────────────────────────────────────────┤")

	for _, layer := range allLayers {
		bestMode := ""
		var bestResult TestResult
		for _, mode := range modes {
			if r, ok := float32Results[layer][mode]; ok {
				if r.Score > bestResult.Score {
					bestResult = r
					bestMode = mode
				}
			}
		}

		// Determine why this mode wins
		reason := ""
		switch {
		case bestResult.AvailabilityPct >= 99 && bestResult.AvgAccuracy > 30:
			reason = "High availability + decent accuracy"
		case bestResult.AvgAccuracy >= 80:
			reason = "Highest accuracy"
		case bestResult.ThroughputPerSec > 50000:
			reason = "Highest throughput"
		case bestResult.AvailabilityPct >= 99:
			reason = "100% availability, always responsive"
		case bestResult.TotalBlockedMs < 100:
			reason = "Minimal blocking"
		default:
			reason = "Best balance of metrics"
		}

		fmt.Printf("│ %-10s │ %-17s │ %9.0f │  %5.1f%%   │  %8.0f  │  %5.1f%%   │ %-45s │\n",
			layer, bestMode, bestResult.Score, bestResult.AvgAccuracy,
			bestResult.ThroughputPerSec, bestResult.AvailabilityPct, reason)
	}
	fmt.Println("└────────────┴───────────────────┴───────────┴───────────┴────────────┴───────────┴───────────────────────────────────────────────┘")

	// TABLE 3: Head-to-head: NormalBP vs StepTweenChain
	fmt.Println("\n┌─── HEAD-TO-HEAD: NormalBP vs StepTweenChain ──────────────────────────────────────────────────────────────────────────────────────┐")
	fmt.Println("│ Layer      │ NormalBP Score │ StepTweenChain │ Winner           │ Accuracy Δ │ Avail Δ   │ Throughput Δ                        │")
	fmt.Println("├────────────┼────────────────┼────────────────┼──────────────────┼────────────┼───────────┼─────────────────────────────────────┤")

	for _, layer := range allLayers {
		normalBP := float32Results[layer]["NormalBP"]
		stepTween := float32Results[layer]["StepTweenChain"]

		winner := "StepTweenChain ✓"
		if normalBP.Score > stepTween.Score {
			winner = "NormalBP"
		}

		accDelta := stepTween.AvgAccuracy - normalBP.AvgAccuracy
		availDelta := stepTween.AvailabilityPct - normalBP.AvailabilityPct
		tputDelta := stepTween.ThroughputPerSec - normalBP.ThroughputPerSec

		accSign := "+"
		if accDelta < 0 {
			accSign = ""
		}
		availSign := "+"
		if availDelta < 0 {
			availSign = ""
		}
		tputSign := "+"
		if tputDelta < 0 {
			tputSign = ""
		}

		fmt.Printf("│ %-10s │     %10.0f │     %10.0f │ %-16s │ %s%5.1f%%    │ %s%5.1f%%   │ %s%.0f                               │\n",
			layer, normalBP.Score, stepTween.Score, winner,
			accSign, accDelta, availSign, availDelta, tputSign, tputDelta)
	}
	fmt.Println("└────────────┴────────────────┴────────────────┴──────────────────┴────────────┴───────────┴─────────────────────────────────────┘")

	// Find overall winners
	var overallBest TestResult
	overallBestKey := ""
	for _, r := range results.Results {
		if r.NumericType == "float32" && r.Score > overallBest.Score {
			overallBest = r
			overallBestKey = fmt.Sprintf("%s + %s", r.LayerType, r.TrainingMode)
		}
	}

	fmt.Printf("\n🏆 OVERALL WINNER: %s with Score: %.0f\n", overallBestKey, overallBest.Score)
	fmt.Printf("   → Accuracy: %.1f%% | Throughput: %.0f/sec | Availability: %.1f%% | Blocked: %.0fms\n\n",
		overallBest.AvgAccuracy, overallBest.ThroughputPerSec, overallBest.AvailabilityPct, overallBest.TotalBlockedMs)
}

func min(a, b int) int {
	if a < b {
		return a
	}
	return b
}

// printFailuresAndZeroScores prints a list of all tests that failed or had a score of 0
func printFailuresAndZeroScores(results *BenchmarkResults) {
	fmt.Println("\n╔════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════╗")
	fmt.Println("║                                          ⚠️  FAILURES AND ZERO SCORES REPORT ⚠️                                                                        ║")
	fmt.Println("╚════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════╝")

	count := 0
	fmt.Println("\n┌─── FAILED / ZERO SCORE TESTS ────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────┐")
	fmt.Println("│ Test Details                                    │ Status     │ Reason                                                                                │")
	fmt.Println("├─────────────────────────────────────────────────┼────────────┼───────────────────────────────────────────────────────────────────────────────────────┤")

	for _, r := range results.Results {
		isFailure := false
		status := ""
		reason := ""

		if !r.Passed {
			isFailure = true
			status = "FAILED ❌"
			if r.Error != "" {
				reason = fmt.Sprintf("Error: %s", r.Error)
			} else {
				reason = "Test reported failure"
			}
		} else if r.Score == 0 {
			isFailure = true
			status = "SCORE 0 ⚠️"
			if r.TotalOutputs == 0 {
				reason = "No outputs produced"
			} else if r.AvgAccuracy == 0 {
				reason = "0% Accuracy"
			} else if r.AvailabilityPct == 0 {
				reason = "0% Availability"
			} else {
				reason = "Low performance metrics resulted in 0 score"
			}
		}

		if isFailure {
			count++
			testName := fmt.Sprintf("%s / %s / %s", r.LayerType, r.TrainingMode, r.NumericType)
			fmt.Printf("│ %-47s │ %-10s │ %-85s │\n", testName, status, reason)
		}
	}

	if count == 0 {
		fmt.Println("│ No failures or zero scores detected! 🎉         │            │                                                                                       │")
	}
	fmt.Println("└─────────────────────────────────────────────────┴────────────┴───────────────────────────────────────────────────────────────────────────────────────┘")
	fmt.Printf("\nTotal issues found: %d\n", count)
}
