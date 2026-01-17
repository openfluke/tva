package main

import (
	"fmt"
	"math"
	"math/rand"
	"sync"
	"time"

	"github.com/openfluke/loom/nn"
)

// THE CHAMELEON BENCHMARK v3 (FULL SUITE)
// Testing all 6 training modes on Non-Stationary Data.

const (
	InputDim      = 4
	NumClusters   = 4
	DriftInterval = 2 * time.Second
	TestDuration  = 12 * time.Second
	LearningRate  = float32(0.2)
)

// --- 1. The 6 Modes ---
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

// --- 2. The World (Drifting Clusters) ---
type WorldState struct {
	mu      sync.RWMutex
	centers [][]float32
}

func NewWorld() *WorldState {
	w := &WorldState{}
	w.Teleport()
	return w
}

func (w *WorldState) Teleport() {
	w.mu.Lock()
	defer w.mu.Unlock()
	w.centers = make([][]float32, NumClusters)
	for i := 0; i < NumClusters; i++ {
		w.centers[i] = make([]float32, InputDim)
		for d := 0; d < InputDim; d++ {
			w.centers[i][d] = rand.Float32()*2 - 1
		}
	}
}

func (w *WorldState) Sample() ([]float32, int) {
	w.mu.RLock()
	defer w.mu.RUnlock()
	cID := rand.Intn(NumClusters)
	center := w.centers[cID]
	input := make([]float32, InputDim)
	for d := 0; d < InputDim; d++ {
		input[d] = center[d] + (rand.Float32()*0.1 - 0.05)
	}
	return input, cID
}

// --- 3. The Network (Direct Mode) ---
func createNetwork() *nn.Network {
	// Input -> Dense(Identity) -> KMeans -> Output
	net := nn.NewNetwork(InputDim, 1, 1, 1)
	net.BatchSize = 1

	// Adapter (Identity)
	adapter := nn.InitDenseLayer(InputDim, InputDim, 0)
	for i := range adapter.Kernel {
		adapter.Kernel[i] = 0
	}
	for i := 0; i < InputDim; i++ {
		adapter.Kernel[i*InputDim+i] = 1.0
	}

	// KMeans
	kmeansLayer := nn.InitKMeansLayer(NumClusters, adapter, "probabilities")
	kmeansLayer.KMeansTemperature = 0.2
	kmeansLayer.KMeansLearningRate = LearningRate

	net.SetLayer(0, 0, 0, kmeansLayer)
	net.InitializeWeights()
	return net
}

type ModeResult struct {
	AverageAccuracy float32
	Recoveries      int
	TotalMovement   float32
}

// --- 4. The Loop ---
func runDriftTest(mode TrainingMode, world *WorldState) *ModeResult {
	net := createNetwork()
	state := net.InitStepState(InputDim)

	// Tween Setup
	var ts *nn.TweenState
	if mode >= ModeTween {
		ts = nn.NewTweenState(net, nil)
		if mode == ModeTweenChain || mode == ModeStepTweenChain {
			ts.Config.UseChainRule = true
		}
	}

	start := time.Now()
	lastDrift := time.Now()

	totalCorrect := 0
	totalSamples := 0
	streak := 0
	recoveries := 0
	locked := false

	// Telemetry
	var initialCenters []float32
	totalMovement := float32(0.0)

	// Helper to get center copy
	snapshot := func() []float32 {
		layer := net.GetLayer(0, 0, 0)
		if len(layer.ClusterCenters) == 0 {
			return nil
		}
		c := make([]float32, len(layer.ClusterCenters))
		copy(c, layer.ClusterCenters)
		return c
	}

	// --- RUN LOOP ---
	for time.Since(start) < TestDuration {
		// Drift Check
		if time.Since(lastDrift) > DriftInterval {
			world.Teleport()
			lastDrift = time.Now()
			streak = 0
			locked = false

			// Measure movement
			currentCenters := snapshot()
			if initialCenters != nil && currentCenters != nil {
				dist := float32(0.0)
				for i := range currentCenters {
					d := currentCenters[i] - initialCenters[i]
					dist += d * d
				}
				totalMovement += float32(math.Sqrt(float64(dist)))
			}
			initialCenters = currentCenters
		}

		input, target := world.Sample()

		// Forward
		state.SetInput(input)
		net.StepForward(state)
		output := state.GetOutput()

		// Init snapshot on first run if needed
		if initialCenters == nil {
			initialCenters = snapshot()
		}

		// Eval
		if argmax(output) == target {
			totalCorrect++
			streak++
		} else {
			streak = 0
		}
		totalSamples++

		if streak > 15 && !locked {
			recoveries++
			locked = true
		}

		// --- TRAIN (MANUAL UPDATES) ---
		switch mode {
		case ModeNormalBP, ModeStepBP:
			// Manual Backprop
			grad := make([]float32, NumClusters)
			for j := range grad {
				if j == target {
					grad[j] = 1.0 - output[j]
				} else {
					grad[j] = 0.0 - output[j]
				}
			}

			// Gradient flow triggers KMeans update internally in BackwardKMeansCPU
			net.StepBackward(state, grad)

			// Apply updates to weights (if any in adapter)
			net.ApplyGradients(LearningRate)

		case ModeTween, ModeTweenChain, ModeStepTween, ModeStepTweenChain:
			// 1. Run Tween for weights
			ts.TweenStep(net, input, target, NumClusters, LearningRate)

			// 2. FORCE K-MEANS UPDATE (Hybrid Training)
			// Because Tween only updates weights (Kernel/Bias), not ClusterCenters.
			// We force a gradient update for the Centers.
			grad := make([]float32, NumClusters)
			for j := range grad {
				if j == target {
					grad[j] = 1.0 - output[j]
				} else {
					grad[j] = 0.0 - output[j]
				}
			}

			// This triggers the BackwardKMeansCPU logic which updates centers
			net.StepBackward(state, grad)
		}
	}

	return &ModeResult{
		AverageAccuracy: float32(totalCorrect) / float32(totalSamples),
		Recoveries:      recoveries,
		TotalMovement:   totalMovement,
	}
}

func main() {
	rand.Seed(42)
	fmt.Println("🚀 Starting 'The Chameleon v3' (All Modes, Forced Updates)...")
	fmt.Printf("   Drift: Every %v | LR: %.2f\n", DriftInterval, LearningRate)

	world := NewWorld()

	modes := []TrainingMode{
		ModeNormalBP, ModeStepBP,
		ModeTween, ModeTweenChain,
		ModeStepTween, ModeStepTweenChain,
	}

	var wg sync.WaitGroup

	for _, m := range modes {
		wg.Add(1)
		go func(mode TrainingMode) {
			defer wg.Done()
			name := modeNames[mode]
			res := runDriftTest(mode, world)
			fmt.Printf("✅ [%-16s] Acc: %.1f%% | Adaptations: %d | Move: %.4f\n",
				name, res.AverageAccuracy*100, res.Recoveries, res.TotalMovement)
		}(m)
	}

	wg.Wait()
	fmt.Println("\n🏁 Drift Test Complete.")
}

func argmax(v []float32) int {
	maxI := 0
	maxV := v[0]
	for i, val := range v {
		if val > maxV {
			maxV = val
			maxI = i
		}
	}
	return maxI
}
