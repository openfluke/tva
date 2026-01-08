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
// MUSIC PATTERN LEARNING - HARDCORE EDITION
// ═══════════════════════════════════════════════════════════════════════════════
//
// HARDCORE: 60 second test, DEEP networks, LARGE batch training
// Music generation cannot stop mid-song - blocking = ruined performance

const (
	MPInputSize  = 32  // Longer musical context
	MPHiddenSize = 128
	MPOutputSize = 24  // 2 octaves (24 notes)
	MPNumLayers  = 5

	MPLearningRate = float32(0.01)
	MPInitScale    = float32(0.3)
	MPBatchSize    = 100

	MPTestDuration   = 60 * time.Second
	MPWindowDuration = 100 * time.Millisecond
	MPSwitchInterval = 5 * time.Second
	MPTrainInterval  = 500 * time.Millisecond
	MPNoteInterval   = 15 * time.Millisecond // ~67 notes per second

	MPMaxConcurrent = 6
)

type MusicGenre int

const (
	GenreJazz MusicGenre = iota
	GenreClassical
	GenreElectronic
	GenreRandom
	GenreBlues     // NEW
	GenreMinimal   // NEW
)

var majorScale = []int{0, 2, 4, 5, 7, 9, 11}
var bluesScale = []int{0, 3, 5, 6, 7, 10}
var minorScale = []int{0, 2, 3, 5, 7, 8, 10}
var electronicPattern = []int{0, 0, 7, 5, 0, 0, 7, 3}

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

type TimeWindow struct {
	TimeMs    int     `json:"timeMs"`
	Notes     int     `json:"notes"`
	Correct   int     `json:"correct"`
	BlockedMs float64 `json:"blockedMs"`
}

type TestResult struct {
	TrainingMode    string       `json:"trainingMode"`
	Windows         []TimeWindow `json:"windows"`
	TotalNotes      int          `json:"totalNotes"`
	CorrectNotes    int          `json:"correctNotes"`
	NotesGenerated  int          `json:"notesGenerated"`
	NotesMissed     int          `json:"notesMissed"` // Notes we couldn't generate while blocked
	TrainTimeSec    float64      `json:"trainTimeSec"`
	Accuracy        float64      `json:"accuracy"`
	NotesPerSec     float64      `json:"notesPerSec"`
	AvailabilityPct float64      `json:"availabilityPct"`
	TotalBlockedMs  float64      `json:"totalBlockedMs"`
	Score           float64      `json:"score"`
	Passed          bool         `json:"passed"`
	Error           string       `json:"error,omitempty"`
}

type BenchmarkResults struct {
	Results    []TestResult `json:"results"`
	Timestamp  string       `json:"timestamp"`
	Duration   string       `json:"testDuration"`
	TotalTests int          `json:"totalTests"`
}

func main() {
	rand.Seed(time.Now().UnixNano())

	fmt.Println("╔═══════════════════════════════════════════════════════════════════════════════════════════════════════════════╗")
	fmt.Println("║   🎵 MUSIC PATTERN LEARNING: HARDCORE EDITION                                                                ║")
	fmt.Println("║                                                                                                               ║")
	fmt.Println("║   ⚠️  60 SECOND TEST | 5-LAYER DEEP NETWORK | BATCH SIZE 100                                                 ║")
	fmt.Println("║   Music generation cannot stop - blocking = ruined song                                                      ║")
	fmt.Println("╚═══════════════════════════════════════════════════════════════════════════════════════════════════════════════╝")

	modes := []TrainingMode{ModeNormalBP, ModeStepBP, ModeTween, ModeTweenChain, ModeStepTween, ModeStepTweenChain}
	
	fmt.Printf("\n📊 Running %d tests | 60s each | Batch size: %d\n\n", len(modes), MPBatchSize)

	results := &BenchmarkResults{
		Results:    make([]TestResult, 0, len(modes)),
		Timestamp:  time.Now().Format(time.RFC3339),
		Duration:   MPTestDuration.String(),
		TotalTests: len(modes),
	}

	var wg sync.WaitGroup
	var mu sync.Mutex
	sem := make(chan struct{}, MPMaxConcurrent)

	for i, mode := range modes {
		wg.Add(1)
		go func(m TrainingMode, idx int) {
			defer wg.Done()
			sem <- struct{}{}
			defer func() { <-sem }()

			modeName := modeNames[m]
			fmt.Printf("🎹 [%d/%d] Starting %s...\n", idx+1, len(modes), modeName)

			result := runMusicTest(m)
			result.TrainingMode = modeName

			mu.Lock()
			results.Results = append(results.Results, result)
			mu.Unlock()

			fmt.Printf("✅ [%d/%d] %-15s | Accuracy: %5.1f%% | Notes: %5d | Missed: %4d | Avail: %5.1f%% | Score: %.0f\n",
				idx+1, len(modes), modeName, result.Accuracy, result.NotesGenerated,
				result.NotesMissed, result.AvailabilityPct, result.Score)
		}(mode, i)
	}

	wg.Wait()
	saveResults(results)
	printSummaryTable(results)
}

func runMusicTest(mode TrainingMode) TestResult {
	result := TestResult{}
	defer func() {
		if r := recover(); r != nil {
			result.Error = fmt.Sprintf("panic: %v", r)
		}
	}()

	numWindows := int(MPTestDuration / MPWindowDuration)
	result.Windows = make([]TimeWindow, numWindows)
	for i := range result.Windows {
		result.Windows[i].TimeMs = (i + 1) * int(MPWindowDuration.Milliseconds())
	}

	net := createDeepNetwork()
	numLayers := net.TotalLayers()

	var state *nn.StepState
	if mode == ModeStepBP || mode == ModeStepTween || mode == ModeStepTweenChain {
		state = net.InitStepState(MPInputSize)
	}

	var ts *nn.TweenState
	if mode == ModeTween || mode == ModeTweenChain || mode == ModeStepTween || mode == ModeStepTweenChain {
		ts = nn.NewTweenState(net, nil)
		if mode == ModeTweenChain || mode == ModeStepTweenChain {
			ts.Config.UseChainRule = true
		}
	}

	currentGenre := GenreJazz
	noteBuffer := make([]int, MPInputSize)
	for i := range noteBuffer {
		noteBuffer[i] = rand.Intn(MPOutputSize)
	}
	noteIdx := 0

	type Sample struct {
		Input  []float32
		Target int
	}
	trainBatch := make([]Sample, 0, MPBatchSize+10)
	lastTrainTime := time.Now()
	isBlocked := false

	start := time.Now()
	currentWindow := 0
	lastSwitchTime := start
	lastNoteTime := start
	var totalBlockedTime time.Duration

	for time.Since(start) < MPTestDuration {
		elapsed := time.Since(start)

		newWindow := int(elapsed / MPWindowDuration)
		if newWindow > currentWindow && newWindow < numWindows {
			currentWindow = newWindow
		}

		if time.Since(lastSwitchTime) >= MPSwitchInterval {
			currentGenre = MusicGenre((int(currentGenre) + 1) % 6)
			lastSwitchTime = time.Now()
		}

		if time.Since(lastNoteTime) >= MPNoteInterval {
			lastNoteTime = time.Now()
			noteIdx++
			result.TotalNotes++

			// If blocked, we MISS this note - the song has a gap!
			if isBlocked {
				result.NotesMissed++
				continue
			}

			nextNote := generateNote(currentGenre, noteBuffer, noteIdx)
			input := encodeNoteBuffer(noteBuffer)

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

			predicted := argmax(output)

			result.NotesGenerated++
			if currentWindow < numWindows {
				result.Windows[currentWindow].Notes++
			}

			// Exact match or within a semitone
			if predicted == nextNote || abs(predicted-nextNote) <= 1 {
				result.CorrectNotes++
				if currentWindow < numWindows {
					result.Windows[currentWindow].Correct++
				}
			}

			copy(noteBuffer[:MPInputSize-1], noteBuffer[1:])
			noteBuffer[MPInputSize-1] = nextNote

			trainBatch = append(trainBatch, Sample{Input: input, Target: nextNote})

			switch mode {
			case ModeNormalBP:
				if len(trainBatch) >= MPBatchSize && time.Since(lastTrainTime) > MPTrainInterval {
					batches := make([]nn.TrainingBatch, len(trainBatch))
					for i, s := range trainBatch {
						target := make([]float32, MPOutputSize)
						if s.Target >= 0 && s.Target < MPOutputSize {
							target[s.Target] = 1.0
						}
						batches[i] = nn.TrainingBatch{Input: s.Input, Target: target}
					}
					isBlocked = true
					trainStart := time.Now()
					net.Train(batches, &nn.TrainingConfig{Epochs: 1, LearningRate: MPLearningRate, LossType: "crossentropy"})
					blockDuration := time.Since(trainStart)
					isBlocked = false
					totalBlockedTime += blockDuration
					trainBatch = trainBatch[:0]
					lastTrainTime = time.Now()
				}
			case ModeStepBP:
				targetVec := make([]float32, MPOutputSize)
				if nextNote >= 0 && nextNote < MPOutputSize {
					targetVec[nextNote] = 1.0
				}
				grad := make([]float32, len(output))
				for i := range grad {
					grad[i] = clipGrad(output[i]-targetVec[i], 0.5)
				}
				net.StepBackward(state, grad)
				net.ApplyGradients(MPLearningRate)
			case ModeTween, ModeTweenChain:
				if len(trainBatch) >= MPBatchSize && time.Since(lastTrainTime) > MPTrainInterval {
					isBlocked = true
					trainStart := time.Now()
					for _, s := range trainBatch {
						target := make([]float32, MPOutputSize)
						if s.Target >= 0 && s.Target < MPOutputSize {
							target[s.Target] = 1.0
						}
						out := ts.ForwardPass(net, s.Input)
						outputGrad := make([]float32, len(out))
						for i := range outputGrad {
							if i < len(target) {
								outputGrad[i] = target[i] - out[i]
							}
						}
						ts.ChainGradients[net.TotalLayers()] = outputGrad
						ts.BackwardTargets[net.TotalLayers()] = target
						ts.TweenWeightsChainRule(net, MPLearningRate)
					}
					blockDuration := time.Since(trainStart)
					isBlocked = false
					totalBlockedTime += blockDuration
					trainBatch = trainBatch[:0]
					lastTrainTime = time.Now()
				}
			case ModeStepTween, ModeStepTweenChain:
				targetVec := make([]float32, MPOutputSize)
				if nextNote >= 0 && nextNote < MPOutputSize {
					targetVec[nextNote] = 1.0
				}
				outputGrad := make([]float32, len(output))
				for i := range outputGrad {
					outputGrad[i] = targetVec[i] - output[i]
				}
				ts.ChainGradients[net.TotalLayers()] = outputGrad
				ts.BackwardTargets[net.TotalLayers()] = targetVec
				ts.TweenWeightsChainRule(net, MPLearningRate)
			}
		}
	}

	result.TrainTimeSec = time.Since(start).Seconds()
	result.TotalBlockedMs = totalBlockedTime.Seconds() * 1000

	if result.NotesGenerated > 0 {
		result.Accuracy = float64(result.CorrectNotes) / float64(result.NotesGenerated) * 100
	}
	if result.TrainTimeSec > 0 {
		result.NotesPerSec = float64(result.NotesGenerated) / result.TrainTimeSec
		totalTimeMs := result.TrainTimeSec * 1000
		result.AvailabilityPct = ((totalTimeMs - result.TotalBlockedMs) / totalTimeMs) * 100
	}

	// Heavy penalty for missed notes - gaps in music are VERY noticeable
	missedPenalty := float64(result.NotesMissed) / 20.0
	result.Score = result.Accuracy * result.NotesPerSec * (result.AvailabilityPct / 100) / 50 - missedPenalty
	if result.Score < 0 {
		result.Score = 0
	}
	if math.IsNaN(result.Score) || math.IsInf(result.Score, 0) {
		result.Score = 0
	}
	result.Passed = result.Score > 0
	return result
}

func createDeepNetwork() *nn.Network {
	config := nn.SimpleNetworkConfig{
		InputSize:  MPInputSize,
		HiddenSize: MPHiddenSize,
		OutputSize: MPOutputSize,
		Activation: nn.ActivationLeakyReLU,
		InitScale:  MPInitScale,
		NumLayers:  MPNumLayers,
		LayerType:  nn.BrainLSTM, // LSTM for sequence
		DType:      nn.DTypeFloat32,
	}
	return nn.BuildSimpleNetwork(config)
}

func generateNote(genre MusicGenre, buffer []int, idx int) int {
	lastNote := buffer[len(buffer)-1]

	switch genre {
	case GenreJazz:
		if rand.Float32() < 0.6 {
			return bluesScale[rand.Intn(len(bluesScale))]
		}
		step := rand.Intn(3) - 1
		return (lastNote + step + MPOutputSize) % MPOutputSize

	case GenreClassical:
		scaleIdx := rand.Intn(len(majorScale))
		if rand.Float32() < 0.7 {
			currentPos := findClosestScalePosition(lastNote%12, majorScale)
			newPos := (currentPos + rand.Intn(3) - 1 + len(majorScale)) % len(majorScale)
			octave := lastNote / 12
			return majorScale[newPos] + octave*12
		}
		return majorScale[scaleIdx]

	case GenreElectronic:
		patternPos := idx % len(electronicPattern)
		return electronicPattern[patternPos]

	case GenreRandom:
		return rand.Intn(MPOutputSize)

	case GenreBlues:
		return bluesScale[rand.Intn(len(bluesScale))] + (rand.Intn(2) * 12)

	case GenreMinimal:
		// Minimal: repeat patterns with slight variations
		if idx%8 < 6 {
			return lastNote // Stay on same note
		}
		return (lastNote + rand.Intn(5) - 2 + MPOutputSize) % MPOutputSize
	}

	return rand.Intn(MPOutputSize)
}

func findClosestScalePosition(note int, scale []int) int {
	minDist := 12
	pos := 0
	for i, s := range scale {
		dist := abs(note - s)
		if dist < minDist {
			minDist = dist
			pos = i
		}
	}
	return pos
}

func encodeNoteBuffer(buffer []int) []float32 {
	input := make([]float32, MPInputSize)
	for i, note := range buffer {
		input[i] = float32(note) / float32(MPOutputSize)
	}
	return input
}

func argmax(arr []float32) int {
	if len(arr) == 0 {
		return 0
	}
	maxIdx := 0
	maxVal := arr[0]
	for i, v := range arr {
		if v > maxVal {
			maxVal = v
			maxIdx = i
		}
	}
	return maxIdx
}

func abs(x int) int {
	if x < 0 {
		return -x
	}
	return x
}

func clipGrad(grad, maxVal float32) float32 {
	if grad > maxVal {
		return maxVal
	}
	if grad < -maxVal {
		return -maxVal
	}
	return grad
}

func saveResults(results *BenchmarkResults) {
	data, _ := json.MarshalIndent(results, "", "  ")
	os.WriteFile("music_patterns_hardcore_results.json", data, 0644)
	fmt.Println("\n📁 Results saved to music_patterns_hardcore_results.json")
}

func printSummaryTable(results *BenchmarkResults) {
	fmt.Println("\n╔═══════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════╗")
	fmt.Println("║                                    MUSIC PATTERN HARDCORE SUMMARY                                                                    ║")
	fmt.Println("╠═══════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════╣")
	fmt.Println("║  Training Mode    │ Accuracy │ Generated │ Missed │ Notes/s │ Blocked(ms) │ Avail% │     Score     │ Status                          ║")
	fmt.Println("╠═══════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════╣")

	for _, r := range results.Results {
		status := "✅ PASS"
		if !r.Passed {
			status = "❌ FAIL"
		}
		fmt.Printf("║  %-15s │  %5.1f%%  │    %5d  │  %4d  │   %5.0f │   %8.0f  │ %5.1f%% │ %13.1f │ %s ║\n",
			r.TrainingMode, r.Accuracy, r.NotesGenerated, r.NotesMissed,
			r.NotesPerSec, r.TotalBlockedMs, r.AvailabilityPct, r.Score, status)
	}

	fmt.Println("╚═══════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════╝")

	var best, worst *TestResult
	for i := range results.Results {
		if best == nil || results.Results[i].Score > best.Score {
			best = &results.Results[i]
		}
		if worst == nil || results.Results[i].Score < worst.Score {
			worst = &results.Results[i]
		}
	}
	if best != nil && worst != nil {
		fmt.Printf("\n🏆 BEST: %s | Score: %.1f | Accuracy: %.1f%% | Missed notes: %d\n",
			best.TrainingMode, best.Score, best.Accuracy, best.NotesMissed)
		fmt.Printf("💀 WORST: %s | Score: %.1f | Missed %d notes (gaps in the music!)\n",
			worst.TrainingMode, worst.Score, worst.NotesMissed)
	}
}
