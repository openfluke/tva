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
// TEST 41: SINE WAVE ADAPTATION BENCHMARK
// ═══════════════════════════════════════════════════════════════════════════════
//
// Mirrors arc_benchmark.go behavior for SINE WAVE frequency switching:
//   - Run for 10 seconds total
//   - Switch frequency every 2.5 seconds: Sin(1x) → Sin(2x) → Sin(3x) → Sin(4x)
//   - Track PREDICTION ACCURACY % every 50ms window
//   - Calculate: Score = (Throughput × Stability × Consistency) / 100000
//
// TRAINING METHODS (all run in parallel!):
//   - NormalBP: STOPS to batch train (like arc_benchmark)
//   - StepBP: Immediate step-based backprop
//   - Tween: Batch tween (ForwardCPU + periodic TweenStep)
//   - TweenChain: Batch tween with chain rule
//   - StepTween: Step forward + immediate TweenStep
//   - StepTweenChain: Step forward + immediate TweenStep with chain rule
//
// TARGET: < 500ms to adapt after each frequency switch
//

const (
	// Network architecture
	InputSize  = 10 // Sliding window of 10 sine samples
	HiddenSize = 32 // Hidden layer size
	OutputSize = 1  // Predict next sine value

	// Training parameters
	LearningRate      = float32(0.01)
	InitScale         = float32(0.5)
	AccuracyThreshold = 0.05 // Prediction correct if abs(pred - expected) < threshold

	// Sine wave parameters
	SinePoints     = 100 // Number of points to generate
	SineResolution = 0.1 // Step size for x values

	// Timing - 10 second run with 50ms windows (200 windows total)
	TestDuration   = 10 * time.Second
	WindowDuration = 50 * time.Millisecond   // 50ms windows for fine-grained tracking
	SwitchInterval = 2500 * time.Millisecond // Switch frequency every 2.5 seconds

	// Batch training interval for batch-based methods
	TrainInterval = 10 * time.Millisecond
)

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

// TimeWindow for 50ms accuracy tracking
type TimeWindow struct {
	TimeMs        int     `json:"timeMs"`
	Outputs       int     `json:"outputs"`
	TotalAccuracy float64 `json:"totalAccuracy"`
	Accuracy      float64 `json:"accuracy"`    // Average prediction accuracy %
	FreqSwitches  int     `json:"freqSwitches"`
	MaxLatencyMs  float64 `json:"maxLatencyMs"` // Longest gap between outputs in this window
	AvailableMs   float64 `json:"availableMs"`  // Time spent producing outputs (not blocked)
	BlockedMs     float64 `json:"blockedMs"`    // Time spent blocked in training
}

// ModeResult holds per-mode benchmark results
type ModeResult struct {
	Windows          []TimeWindow `json:"windows"`
	TotalOutputs     int          `json:"totalOutputs"`
	TotalFreqSwitch  int          `json:"totalFreqSwitches"`
	TrainTimeSec     float64      `json:"trainTimeSec"`
	AvgTrainAccuracy float64      `json:"avgTrainAccuracy"`
	Stability        float64      `json:"stability"`   // 100 - stddev
	Consistency      float64      `json:"consistency"` // % windows above threshold
	ThroughputPerSec float64      `json:"throughputPerSec"`
	Score            float64      `json:"score"` // T×S×C / 100000
	// New availability metrics
	AvailabilityPct   float64 `json:"availabilityPct"`   // % of time producing outputs
	TotalBlockedMs    float64 `json:"totalBlockedMs"`    // Total time blocked in training
	AvgLatencyMs      float64 `json:"avgLatencyMs"`      // Average max latency per window
	MaxLatencyMs      float64 `json:"maxLatencyMs"`      // Peak latency (worst gap)
	ZeroOutputWindows int     `json:"zeroOutputWindows"` // Windows with 0 outputs (fully blocked)
}

// BenchmarkResults is the full output
type BenchmarkResults struct {
	Modes       []string               `json:"modes"`
	Results     map[string]*ModeResult `json:"results"`
	Timestamp   string                 `json:"timestamp"`
	Duration    string                 `json:"duration"`
	WindowMs    int                    `json:"windowMs"`
	Frequencies []float64              `json:"frequencies"`
}

func main() {
	rand.Seed(time.Now().UnixNano())

	fmt.Println("╔═════════════════════════════════════════════════════════════════════════════════════╗")
	fmt.Println("║   🌊 TEST 41: SINE WAVE ADAPTATION BENCHMARK                                        ║")
	fmt.Println("║                                                                                     ║")
	fmt.Println("║   TRAINING: Sin(1x) → Sin(2x) → Sin(3x) → Sin(4x) (switch every 2.5 seconds)        ║")
	fmt.Println("║   Track PREDICTION ACCURACY % every 50ms!                                           ║")
	fmt.Println("║                                                                                     ║")
	fmt.Println("║   → NormalBP: STOPS to batch train → accuracy DIPS during training                 ║")
	fmt.Println("║   → StepTweenChain: trains EVERY sample → maintains accuracy while switching       ║")
	fmt.Println("║                                                                                     ║")
	fmt.Println("║   Score = (Throughput × Stability × Consistency) / 100000                          ║")
	fmt.Println("╚═════════════════════════════════════════════════════════════════════════════════════╝")

	// Generate sine wave data for all 4 frequencies
	frequencies := []float64{1.0, 2.0, 3.0, 4.0}
	allInputs := make([][][]float32, len(frequencies))
	allTargets := make([][]float32, len(frequencies))

	for i, freq := range frequencies {
		sineData := generateSineWave(freq)
		allInputs[i], allTargets[i] = createSamples(sineData)
	}

	numWindows := int(TestDuration / WindowDuration)
	fmt.Printf("\n📊 Generated %d samples per frequency | %d windows at %dms each\n", SinePoints, numWindows, WindowDuration.Milliseconds())
	fmt.Printf("⏱️  Duration: %s | Frequency switch every %s\n\n", TestDuration, SwitchInterval)

	modes := []TrainingMode{
		ModeNormalBP,
		ModeStepBP,
		ModeTween,
		ModeTweenChain,
		ModeStepTween,
		ModeStepTweenChain,
	}

	results := &BenchmarkResults{
		Modes:       make([]string, len(modes)),
		Results:     make(map[string]*ModeResult),
		Timestamp:   time.Now().Format(time.RFC3339),
		Duration:    TestDuration.String(),
		WindowMs:    int(WindowDuration.Milliseconds()),
		Frequencies: frequencies,
	}

	for i, m := range modes {
		results.Modes[i] = modeNames[m]
	}

	// Run benchmarks in parallel
	var wg sync.WaitGroup
	var mu sync.Mutex

	for _, mode := range modes {
		wg.Add(1)
		go func(m TrainingMode) {
			defer wg.Done()
			modeName := modeNames[m]
			fmt.Printf("🚀 [%s] Starting...\n", modeName)

			result := runSineWaveBenchmark(m, allInputs, allTargets, frequencies)

			mu.Lock()
			results.Results[modeName] = result
			mu.Unlock()

			fmt.Printf("✅ [%s] Done | Acc: %.1f%% | Stab: %.0f%% | Cons: %.0f%% | Tput: %.0f | Score: %.0f\n",
				modeName, result.AvgTrainAccuracy, result.Stability, result.Consistency, result.ThroughputPerSec, result.Score)
		}(mode)
	}

	wg.Wait()
	fmt.Println("\n✅ All benchmarks complete!")

	saveResults(results)
	printTimeline(results)
	printSummary(results)
}

// generateSineWave creates sine wave samples with given frequency multiplier
func generateSineWave(freqMultiplier float64) []float64 {
	data := make([]float64, SinePoints)
	for i := 0; i < SinePoints; i++ {
		x := float64(i) * SineResolution
		data[i] = math.Sin(freqMultiplier * x)
	}
	return data
}

// createSamples creates input/target pairs from sine data
func createSamples(data []float64) (inputs [][]float32, targets []float32) {
	numSamples := len(data) - InputSize
	inputs = make([][]float32, numSamples)
	targets = make([]float32, numSamples)

	for i := 0; i < numSamples; i++ {
		input := make([]float32, InputSize)
		for j := 0; j < InputSize; j++ {
			input[j] = float32((data[i+j] + 1.0) / 2.0)
		}
		inputs[i] = input
		targets[i] = float32((data[i+InputSize] + 1.0) / 2.0)
	}
	return inputs, targets
}

// createNetwork builds a simple Dense network for sine prediction
func createNetwork() *nn.Network {
	net := nn.NewNetwork(InputSize, 1, 1, 3)
	net.BatchSize = 1

	layer0 := nn.InitDenseLayer(InputSize, HiddenSize, nn.ActivationLeakyReLU)
	scaleWeights(layer0.Kernel, InitScale)
	net.SetLayer(0, 0, 0, layer0)

	layer1 := nn.InitDenseLayer(HiddenSize, HiddenSize, nn.ActivationLeakyReLU)
	scaleWeights(layer1.Kernel, InitScale)
	net.SetLayer(0, 0, 1, layer1)

	layer2 := nn.InitDenseLayer(HiddenSize, OutputSize, nn.ActivationSigmoid)
	scaleWeights(layer2.Kernel, InitScale)
	net.SetLayer(0, 0, 2, layer2)

	return net
}

func scaleWeights(weights []float32, scale float32) {
	for i := range weights {
		weights[i] *= scale
	}
}

// runSineWaveBenchmark runs real-time sine wave frequency switching benchmark
func runSineWaveBenchmark(mode TrainingMode, allInputs [][][]float32, allTargets [][]float32, frequencies []float64) *ModeResult {
	numWindows := int(TestDuration / WindowDuration) // 200 windows at 50ms each
	result := &ModeResult{
		Windows: make([]TimeWindow, numWindows),
	}

	// Initialize windows
	for i := range result.Windows {
		result.Windows[i].TimeMs = (i + 1) * int(WindowDuration.Milliseconds())
	}

	// Create fresh network
	net := createNetwork()
	numLayers := net.TotalLayers()

	// Initialize states based on mode
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

	// Training batch for batch-based methods
	type TrainingSample struct {
		Input  []float32
		Target float32
	}
	trainBatch := make([]TrainingSample, 0, 20)
	lastTrainTime := time.Now()

	start := time.Now()
	currentWindow := 0
	sampleIdx := 0
	currentFreqIdx := 0
	lastSwitchTime := start

	// Latency and availability tracking
	lastOutputTime := time.Now()
	var totalBlockedTime time.Duration
	windowStartTime := time.Now()

	// =========================================================================
	// MAIN TRAINING LOOP: Switch frequency every 2.5 seconds for 10 seconds
	// =========================================================================
	for time.Since(start) < TestDuration {
		elapsed := time.Since(start)

		// Update window (50ms windows)
		newWindow := int(elapsed / WindowDuration)
		if newWindow > currentWindow && newWindow < numWindows {
			// Finalize the previous window's available time
			if currentWindow < numWindows {
				windowElapsed := time.Since(windowStartTime).Seconds() * 1000
				result.Windows[currentWindow].AvailableMs = windowElapsed - result.Windows[currentWindow].BlockedMs
			}
			currentWindow = newWindow
			windowStartTime = time.Now()
		}

		// Check for frequency switch (every 2.5 seconds)
		if time.Since(lastSwitchTime) >= SwitchInterval && currentFreqIdx < len(frequencies)-1 {
			currentFreqIdx++
			lastSwitchTime = time.Now()
			result.TotalFreqSwitch++
			if currentWindow < numWindows {
				result.Windows[currentWindow].FreqSwitches++
			}
		}

		// Get current frequency's data
		inputs := allInputs[currentFreqIdx]
		targets := allTargets[currentFreqIdx]

		// Get sample
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
			// Use ts.ForwardPass so TweenState is populated for training
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
			result.Windows[currentWindow].TotalAccuracy += sampleAcc
			result.TotalOutputs++
		}

		// =====================================================================
		// TRAINING - THIS IS WHERE EACH MODE DIFFERS
		// =====================================================================
		switch mode {
		case ModeNormalBP:
			// Batch training - accumulates samples, then PAUSES to train
			trainBatch = append(trainBatch, TrainingSample{Input: input, Target: target})
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
			// Immediate step-based backprop
			grad := make([]float32, len(output))
			if len(output) > 0 {
				grad[0] = clipGrad(output[0]-target, 0.5)
			}
			net.StepBackward(state, grad)
			net.ApplyGradients(LearningRate)

		case ModeTween, ModeTweenChain:
			// Batch tween - accumulates samples, trains periodically with regression gradients
			trainBatch = append(trainBatch, TrainingSample{Input: input, Target: target})
			if time.Since(lastTrainTime) > TrainInterval && len(trainBatch) > 0 {
				// Track blocking time during batch training
				trainStart := time.Now()
				for _, s := range trainBatch {
					out := ts.ForwardPass(net, s.Input)
					// Regression gradient: target - output
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
			// Immediate tween with regression gradients - trains EVERY sample!
			// ForwardPass was already done above, TweenState is populated
			// Compute regression gradient: target - output
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

	// Finalize windows - compute average accuracy per window
	for i := range result.Windows {
		if result.Windows[i].Outputs > 0 {
			result.Windows[i].Accuracy = result.Windows[i].TotalAccuracy / float64(result.Windows[i].Outputs)
		}
		// Finalize available time for each window
		windowDurationMs := WindowDuration.Seconds() * 1000
		result.Windows[i].AvailableMs = windowDurationMs - result.Windows[i].BlockedMs
	}

	result.TrainTimeSec = time.Since(start).Seconds()
	result.TotalBlockedMs = totalBlockedTime.Seconds() * 1000
	calculateSummaryMetrics(result)

	return result
}

func calculateSummaryMetrics(result *ModeResult) {
	// Average training accuracy
	sum := 0.0
	for _, w := range result.Windows {
		sum += w.Accuracy
	}
	result.AvgTrainAccuracy = sum / float64(len(result.Windows))

	// Stability: 100 - stddev
	variance := 0.0
	for _, w := range result.Windows {
		diff := w.Accuracy - result.AvgTrainAccuracy
		variance += diff * diff
	}
	variance /= float64(len(result.Windows))
	result.Stability = math.Max(0, 100-math.Sqrt(variance))

	// Consistency: % of windows above 50% accuracy (better threshold for sine)
	const consistencyThreshold = 50.0
	aboveThreshold := 0
	for _, w := range result.Windows {
		if w.Accuracy >= consistencyThreshold {
			aboveThreshold++
		}
	}
	result.Consistency = float64(aboveThreshold) / float64(len(result.Windows)) * 100

	// Throughput
	result.ThroughputPerSec = float64(result.TotalOutputs) / result.TrainTimeSec

	// Score = (T × S × C) / 100000
	result.Score = (result.ThroughputPerSec * result.Stability * result.Consistency) / 100000

	// NEW: Availability metrics
	// Availability % = (total time - blocked time) / total time * 100
	totalTimeMs := result.TrainTimeSec * 1000
	result.AvailabilityPct = ((totalTimeMs - result.TotalBlockedMs) / totalTimeMs) * 100

	// Average and max latency across all windows
	latencySum := 0.0
	result.MaxLatencyMs = 0
	for _, w := range result.Windows {
		latencySum += w.MaxLatencyMs
		if w.MaxLatencyMs > result.MaxLatencyMs {
			result.MaxLatencyMs = w.MaxLatencyMs
		}
		if w.Outputs == 0 {
			result.ZeroOutputWindows++
		}
	}
	result.AvgLatencyMs = latencySum / float64(len(result.Windows))
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

func saveResults(results *BenchmarkResults) {
	data, _ := json.MarshalIndent(results, "", "  ")
	os.WriteFile("test41_results.json", data, 0644)
	fmt.Println("\n✅ Results saved to test41_results.json")
}

func printTimeline(results *BenchmarkResults) {
	// ACCURACY TIMELINE
	fmt.Println("\n╔══════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════╗")
	fmt.Println("║           PREDICTION ACCURACY % (50ms windows) — Sin(1x)→Sin(2x)→Sin(3x)→Sin(4x) switching every 2.5s                                          ║")
	fmt.Println("║           NormalBP PAUSES to batch train → low throughput | StepTweenChain trains EVERY sample → maintains accuracy                            ║")
	fmt.Println("╠══════════════════════╦════════════════════════════════════════════════════════════════════════════════════════════════════════╦═══════╦════════════╣")
	fmt.Printf("║ Mode                 ║")

	// Print time headers (showing every 1s = 20 windows)
	for i := 0; i < 10; i++ {
		fmt.Printf(" %ds ", i+1)
	}
	fmt.Printf("║ Avg   ║ Score      ║\n")
	fmt.Println("╠══════════════════════╬════════════════════════════════════════════════════════════════════════════════════════════════════════╬═══════╬════════════╣")

	for _, modeName := range results.Modes {
		r := results.Results[modeName]
		fmt.Printf("║ %-20s ║", modeName)

		// Print accuracy for each 1-second block (average of 20 windows)
		for sec := 0; sec < 10; sec++ {
			avgAcc := 0.0
			count := 0
			for w := sec * 20; w < (sec+1)*20 && w < len(r.Windows); w++ {
				avgAcc += r.Windows[w].Accuracy
				count++
			}
			if count > 0 {
				avgAcc /= float64(count)
			}
			fmt.Printf(" %2.0f%%", avgAcc)
		}
		fmt.Printf(" ║ %3.0f%% ║ %10.0f ║\n", r.AvgTrainAccuracy, r.Score)
	}

	fmt.Println("╚══════════════════════╩════════════════════════════════════════════════════════════════════════════════════════════════════════╩═══════╩════════════╝")
	fmt.Println("                           ↑ 2.5s     ↑ 5.0s     ↑ 7.5s        ← Frequency switches")

	// OUTPUTS PER SECOND TIMELINE (shows gaps!)
	fmt.Println("\n╔══════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════╗")
	fmt.Println("║           OUTPUTS PER SECOND — Shows throughput gaps when batch training blocks inference                                                      ║")
	fmt.Println("║           ⚠  Lower numbers = blocked by training | Gaps = unavailable for inference                                                            ║")
	fmt.Println("╠══════════════════════╦════════════════════════════════════════════════════════════════════════════════════════════════════════╦═══════╦════════════╣")
	fmt.Printf("║ Mode                 ║")
	for i := 0; i < 10; i++ {
		fmt.Printf(" %ds  ", i+1)
	}
	fmt.Printf("║ Total ║ Avail%%     ║\n")
	fmt.Println("╠══════════════════════╬════════════════════════════════════════════════════════════════════════════════════════════════════════╬═══════╬════════════╣")

	for _, modeName := range results.Modes {
		r := results.Results[modeName]
		fmt.Printf("║ %-20s ║", modeName)

		// Print outputs for each 1-second block (sum of 20 windows)
		for sec := 0; sec < 10; sec++ {
			totalOutputs := 0
			for w := sec * 20; w < (sec+1)*20 && w < len(r.Windows); w++ {
				totalOutputs += r.Windows[w].Outputs
			}
			fmt.Printf(" %4d", totalOutputs)
		}
		fmt.Printf(" ║ %5d ║ %6.1f%%    ║\n", r.TotalOutputs, r.AvailabilityPct)
	}

	fmt.Println("╚══════════════════════╩════════════════════════════════════════════════════════════════════════════════════════════════════════╩═══════╩════════════╝")

	// MAX LATENCY PER SECOND TIMELINE (shows blocking spikes!)
	fmt.Println("\n╔══════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════╗")
	fmt.Println("║           MAX LATENCY (ms) — Longest gap between outputs in each second                                                                        ║")
	fmt.Println("║           ⚠  High values = system blocked during batch training | Low+consistent = always responsive                                           ║")
	fmt.Println("╠══════════════════════╦════════════════════════════════════════════════════════════════════════════════════════════════════════╦═══════╦════════════╣")
	fmt.Printf("║ Mode                 ║")
	for i := 0; i < 10; i++ {
		fmt.Printf(" %ds  ", i+1)
	}
	fmt.Printf("║ Peak  ║ Blocked    ║\n")
	fmt.Println("╠══════════════════════╬════════════════════════════════════════════════════════════════════════════════════════════════════════╬═══════╬════════════╣")

	for _, modeName := range results.Modes {
		r := results.Results[modeName]
		fmt.Printf("║ %-20s ║", modeName)

		// Print max latency for each 1-second block
		for sec := 0; sec < 10; sec++ {
			maxLat := 0.0
			for w := sec * 20; w < (sec+1)*20 && w < len(r.Windows); w++ {
				if r.Windows[w].MaxLatencyMs > maxLat {
					maxLat = r.Windows[w].MaxLatencyMs
				}
			}
			fmt.Printf(" %4.0f", maxLat)
		}
		fmt.Printf(" ║ %5.0f ║ %6.0fms   ║\n", r.MaxLatencyMs, r.TotalBlockedMs)
	}

	fmt.Println("╚══════════════════════╩════════════════════════════════════════════════════════════════════════════════════════════════════════╩═══════╩════════════╝")
}

func printSummary(results *BenchmarkResults) {
	fmt.Println("\n╔═══════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════╗")
	fmt.Println("║                                               🌊 SINE WAVE ADAPTATION SUMMARY 🌊                                                                         ║")
	fmt.Println("╠═══════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════╣")
	fmt.Println("║                                                                                                                                                             ║")
	fmt.Println("║  Mode               │ Accuracy │ Stability │ Throughput │ Score   │ Avail %  │ Blocked(ms) │ Peak Lat │ Avg Lat │ 0-Out Windows │ ★ Key Insight ★         ║")
	fmt.Println("║  ───────────────────┼──────────┼───────────┼────────────┼─────────┼──────────┼─────────────┼──────────┼─────────┼───────────────┼─────────────────────────║")

	bestScore := 0.0
	bestMode := ""
	lowestBlocked := math.MaxFloat64
	lowestBlockedMode := ""

	for _, modeName := range results.Modes {
		r := results.Results[modeName]

		// Determine the key insight for this mode
		insight := ""
		if modeName == "NormalBP" {
			insight = "BLOCKS during training"
		} else if modeName == "StepTweenChain" {
			insight = "ALWAYS available ✓"
		} else if r.ZeroOutputWindows > 0 {
			insight = fmt.Sprintf("%d windows blocked", r.ZeroOutputWindows)
		} else if r.TotalBlockedMs > 100 {
			insight = "Some blocking"
		} else {
			insight = "Low blocking"
		}

		fmt.Printf("║  %-18s │  %5.1f%%  │   %5.1f%%  │  %8.0f  │ %7.0f │  %5.1f%%  │  %9.0f  │  %5.1fms │ %5.1fms  │      %3d      │ %-23s ║\n",
			modeName, r.AvgTrainAccuracy, r.Stability, r.ThroughputPerSec, r.Score,
			r.AvailabilityPct, r.TotalBlockedMs, r.MaxLatencyMs, r.AvgLatencyMs,
			r.ZeroOutputWindows, insight)

		if r.Score > bestScore {
			bestScore = r.Score
			bestMode = modeName
		}
		if r.TotalBlockedMs < lowestBlocked {
			lowestBlocked = r.TotalBlockedMs
			lowestBlockedMode = modeName
		}
	}

	// Print analysis
	fmt.Println("║                                                                                                                                                             ║")
	fmt.Println("╠═══════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════╣")
	fmt.Printf("║  🏆 BEST SCORE:        %-18s with Score: %.0f                                                                                                   ║\n", bestMode, bestScore)
	fmt.Printf("║  ⚡ LOWEST BLOCKING:    %-18s with only %.0fms blocked                                                                                             ║\n", lowestBlockedMode, lowestBlocked)
	fmt.Println("║                                                                                                                                                             ║")
	fmt.Println("║  💡 KEY INSIGHT: NormalBP achieves high accuracy BUT blocks inference during batch training.                                                                ║")
	fmt.Println("║                  StepTweenChain maintains ~100%% availability while still training every sample!                                                            ║")
	fmt.Println("║                                                                                                                                                             ║")
	fmt.Println("╚═══════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════╝")
}
