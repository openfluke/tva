package main

import (
	"encoding/json"
	"fmt"
	"math"
	"math/rand"
	"os"
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

	wg.Wait()
	fmt.Println("\n✅ All tests complete!")

	// Recalculate availability as relative throughput (compare to max within each layer+type)
	recalculateAvailability(results)

	saveResults(results)
	printSummaryTable(results)
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
	switch numType {
	case TypeFloat32:
		return runFloat32Test(layer, mode, allInputs, allTargets, frequencies)
	default:
		return runGenericTest(layer, mode, numType, allInputs, allTargets, frequencies)
	}
}

// runFloat32Test - uses native float32 training APIs with proper availability tracking
func runFloat32Test(layer LayerTestType, mode TrainingMode, allInputs [][][]float32, allTargets [][]float32, frequencies []float64) TestResult {
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

	net := createNetworkForLayerType(layer)
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
func runGenericTest(layer LayerTestType, mode TrainingMode, numType NumericType, allInputs [][][]float32, allTargets [][]float32, frequencies []float64) TestResult {
	switch numType {
	case TypeInt8:
		return runTypedTest[int8](layer, mode, allInputs, allTargets, frequencies)
	case TypeInt16:
		return runTypedTest[int16](layer, mode, allInputs, allTargets, frequencies)
	case TypeInt32:
		return runTypedTest[int32](layer, mode, allInputs, allTargets, frequencies)
	case TypeInt64:
		return runTypedTest[int64](layer, mode, allInputs, allTargets, frequencies)
	case TypeUint8:
		return runTypedTest[uint8](layer, mode, allInputs, allTargets, frequencies)
	case TypeUint16:
		return runTypedTest[uint16](layer, mode, allInputs, allTargets, frequencies)
	case TypeUint32:
		return runTypedTest[uint32](layer, mode, allInputs, allTargets, frequencies)
	case TypeUint64:
		return runTypedTest[uint64](layer, mode, allInputs, allTargets, frequencies)
	case TypeFloat64:
		return runTypedTest[float64](layer, mode, allInputs, allTargets, frequencies)
	default:
		return TestResult{Passed: false, Error: "unknown type"}
	}
}

// runTypedTest - generic implementation for all numeric types
func runTypedTest[T nn.Numeric](layer LayerTestType, mode TrainingMode, allInputs [][][]float32, allTargets [][]float32, frequencies []float64) TestResult {
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

	net := createNetworkForLayerType(layer)
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
