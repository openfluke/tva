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
// MULTI-AGENT SWARM COORDINATION BENCHMARK
// ═══════════════════════════════════════════════════════════════════════════════
//
// Multiple neural network agents must coordinate to catch targets.
// Each agent has its own network trained independently but must work together.
// Tests: How do training modes handle multi-agent synchronization?

const (
	NumAgents    = 4
	GridSize     = 16
	NumTargets   = 8
	
	MAInputSize  = 24  // Agent pos + other agents pos + nearest targets
	MAHiddenSize = 64
	MAOutputSize = 4
	MANumLayers  = 4

	MALearningRate = float32(0.01)
	MABatchSize    = 50

	MATestDuration   = 60 * time.Second
	MAWindowDuration = 100 * time.Millisecond
	MAStepInterval   = 15 * time.Millisecond
	MATrainInterval  = 300 * time.Millisecond

	MAMaxConcurrent = 6
)

type Position struct{ X, Y int }

type Agent struct {
	Net      *nn.Network
	State    *nn.StepState
	Tween    *nn.TweenState
	Pos      Position
	Catches  int
}

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

type TestResult struct {
	TrainingMode     string  `json:"trainingMode"`
	TotalCatches     int     `json:"totalCatches"`
	CatchesPerAgent  []int   `json:"catchesPerAgent"`
	Collisions       int     `json:"collisions"` // Agents bumping into each other
	CoordinationScore float64 `json:"coordinationScore"` // How well spread out
	TrainTimeSec     float64 `json:"trainTimeSec"`
	AvailabilityPct  float64 `json:"availabilityPct"`
	TotalBlockedMs   float64 `json:"totalBlockedMs"`
	Score            float64 `json:"score"`
}

type BenchmarkResults struct {
	Results    []TestResult `json:"results"`
	Timestamp  string       `json:"timestamp"`
	Duration   string       `json:"testDuration"`
}

func main() {
	rand.Seed(time.Now().UnixNano())

	fmt.Println("╔═══════════════════════════════════════════════════════════════════════════════════════════════════════════════╗")
	fmt.Println("║   🐝 MULTI-AGENT SWARM COORDINATION BENCHMARK                                                                ║")
	fmt.Println("║                                                                                                               ║")
	fmt.Println("║   4 agents × 4-layer networks | 8 targets on 16×16 grid                                                      ║")
	fmt.Println("║   Tests: coordination, collision avoidance, parallel training                                                ║")
	fmt.Println("╚═══════════════════════════════════════════════════════════════════════════════════════════════════════════════╝")

	modes := []TrainingMode{ModeNormalBP, ModeStepBP, ModeTween, ModeTweenChain, ModeStepTween, ModeStepTweenChain}
	
	fmt.Printf("\n📊 Running %d tests | 60s each | %d agents | %d targets\n\n", len(modes), NumAgents, NumTargets)

	results := &BenchmarkResults{
		Results:   make([]TestResult, 0, len(modes)),
		Timestamp: time.Now().Format(time.RFC3339),
		Duration:  MATestDuration.String(),
	}

	var wg sync.WaitGroup
	var mu sync.Mutex
	sem := make(chan struct{}, MAMaxConcurrent)

	for i, mode := range modes {
		wg.Add(1)
		go func(m TrainingMode, idx int) {
			defer wg.Done()
			sem <- struct{}{}
			defer func() { <-sem }()

			fmt.Printf("🐝 [%d/%d] Starting %s...\n", idx+1, len(modes), modeNames[m])
			result := runSwarmTest(m)
			result.TrainingMode = modeNames[m]

			mu.Lock()
			results.Results = append(results.Results, result)
			mu.Unlock()

			fmt.Printf("✅ [%d/%d] %-15s | Catches: %3d | Collisions: %3d | Coord: %5.1f%% | Avail: %5.1f%% | Score: %.0f\n",
				idx+1, len(modes), modeNames[m], result.TotalCatches, result.Collisions,
				result.CoordinationScore, result.AvailabilityPct, result.Score)
		}(mode, i)
	}

	wg.Wait()
	saveResults(results)
	printSummaryTable(results)
}

func runSwarmTest(mode TrainingMode) TestResult {
	result := TestResult{CatchesPerAgent: make([]int, NumAgents)}

	// Initialize agents
	agents := make([]*Agent, NumAgents)
	for i := 0; i < NumAgents; i++ {
		net := createAgentNetwork()
		agents[i] = &Agent{
			Net: net,
			Pos: Position{X: rand.Intn(GridSize), Y: rand.Intn(GridSize)},
		}
		
		if mode == ModeStepBP || mode == ModeStepTween || mode == ModeStepTweenChain {
			agents[i].State = net.InitStepState(MAInputSize)
		}
		if mode == ModeTween || mode == ModeTweenChain || mode == ModeStepTween || mode == ModeStepTweenChain {
			agents[i].Tween = nn.NewTweenState(net, nil)
			if mode == ModeTweenChain || mode == ModeStepTweenChain {
				agents[i].Tween.Config.UseChainRule = true
			}
		}
	}

	// Initialize targets
	targets := make([]Position, NumTargets)
	for i := 0; i < NumTargets; i++ {
		targets[i] = Position{X: rand.Intn(GridSize), Y: rand.Intn(GridSize)}
	}

	type Sample struct {
		AgentIdx int
		Input    []float32
		Target   []float32
	}
	trainBatches := make([][]Sample, NumAgents)
	for i := range trainBatches {
		trainBatches[i] = make([]Sample, 0, MABatchSize+10)
	}
	lastTrainTime := time.Now()
	isBlocked := false
	var totalBlockedTime time.Duration

	start := time.Now()
	lastStepTime := start

	for time.Since(start) < MATestDuration {
		if time.Since(lastStepTime) >= MAStepInterval {
			lastStepTime = time.Now()

			if isBlocked {
				continue
			}

			// Each agent makes a decision
			for i, agent := range agents {
				input := getSwarmInput(agent, agents, targets)
				
				var output []float32
				switch mode {
				case ModeNormalBP, ModeTween, ModeTweenChain:
					output, _ = agent.Net.ForwardCPU(input)
				case ModeStepBP:
					agent.State.SetInput(input)
					for s := 0; s < agent.Net.TotalLayers(); s++ {
						agent.Net.StepForward(agent.State)
					}
					output = agent.State.GetOutput()
				case ModeStepTween, ModeStepTweenChain:
					output = agent.Tween.ForwardPass(agent.Net, input)
				}

				// Move agent
				move := argmax(output)
				oldPos := agent.Pos
				moveAgent(agent, move)

				// Check collisions with other agents
				for j, other := range agents {
					if i != j && agent.Pos == other.Pos {
						result.Collisions++
						agent.Pos = oldPos // Undo move
					}
				}

				// Check for target catch
				for t := 0; t < len(targets); t++ {
					if agent.Pos == targets[t] {
						result.TotalCatches++
						result.CatchesPerAgent[i]++
						agent.Catches++
						// Respawn target
						targets[t] = Position{X: rand.Intn(GridSize), Y: rand.Intn(GridSize)}
					}
				}

				// Training sample
				idealTarget := getIdealSwarmMove(agent, agents, targets)
				trainBatches[i] = append(trainBatches[i], Sample{AgentIdx: i, Input: input, Target: idealTarget})

				// Step-based training (per sample)
				switch mode {
				case ModeStepBP:
					grad := make([]float32, len(output))
					for k := range grad {
						if k < len(idealTarget) {
							grad[k] = clipGrad(output[k]-idealTarget[k], 0.5)
						}
					}
					agent.Net.StepBackward(agent.State, grad)
					agent.Net.ApplyGradients(MALearningRate)
				case ModeStepTween, ModeStepTweenChain:
					outputGrad := make([]float32, len(output))
					for k := range outputGrad {
						if k < len(idealTarget) {
							outputGrad[k] = idealTarget[k] - output[k]
						}
					}
					agent.Tween.ChainGradients[agent.Net.TotalLayers()] = outputGrad
					agent.Tween.BackwardTargets[agent.Net.TotalLayers()] = idealTarget
					agent.Tween.TweenWeightsChainRule(agent.Net, MALearningRate)
				}
			}

			// Batch training for all agents
			if time.Since(lastTrainTime) > MATrainInterval {
				shouldTrain := false
				for _, batch := range trainBatches {
					if len(batch) >= MABatchSize {
						shouldTrain = true
						break
					}
				}

				if shouldTrain {
					switch mode {
					case ModeNormalBP:
						isBlocked = true
						trainStart := time.Now()
						for i, agent := range agents {
							if len(trainBatches[i]) > 0 {
								batches := make([]nn.TrainingBatch, len(trainBatches[i]))
								for j, s := range trainBatches[i] {
									batches[j] = nn.TrainingBatch{Input: s.Input, Target: s.Target}
								}
								agent.Net.Train(batches, &nn.TrainingConfig{Epochs: 1, LearningRate: MALearningRate, LossType: "mse"})
							}
							trainBatches[i] = trainBatches[i][:0]
						}
						totalBlockedTime += time.Since(trainStart)
						isBlocked = false
					case ModeTween, ModeTweenChain:
						isBlocked = true
						trainStart := time.Now()
						for i, agent := range agents {
							for _, s := range trainBatches[i] {
								out := agent.Tween.ForwardPass(agent.Net, s.Input)
								outputGrad := make([]float32, len(out))
								for k := range outputGrad {
									if k < len(s.Target) {
										outputGrad[k] = s.Target[k] - out[k]
									}
								}
								agent.Tween.ChainGradients[agent.Net.TotalLayers()] = outputGrad
								agent.Tween.BackwardTargets[agent.Net.TotalLayers()] = s.Target
								agent.Tween.TweenWeightsChainRule(agent.Net, MALearningRate)
							}
							trainBatches[i] = trainBatches[i][:0]
						}
						totalBlockedTime += time.Since(trainStart)
						isBlocked = false
					default:
						for i := range trainBatches {
							trainBatches[i] = trainBatches[i][:0]
						}
					}
					lastTrainTime = time.Now()
				}
			}
		}
	}

	result.TrainTimeSec = time.Since(start).Seconds()
	result.TotalBlockedMs = totalBlockedTime.Seconds() * 1000
	
	if result.TrainTimeSec > 0 {
		totalTimeMs := result.TrainTimeSec * 1000
		result.AvailabilityPct = ((totalTimeMs - result.TotalBlockedMs) / totalTimeMs) * 100
	}

	// Coordination score: variance in catches (lower = better coordination)
	avg := float64(result.TotalCatches) / float64(NumAgents)
	variance := 0.0
	for _, c := range result.CatchesPerAgent {
		variance += math.Pow(float64(c)-avg, 2)
	}
	variance /= float64(NumAgents)
	result.CoordinationScore = 100.0 / (1.0 + variance/100)

	// Score = catches - (blocked_seconds × 10)
	// This creates clear differentiation: 800ms blocked = -8 points penalty
	blockedPenalty := result.TotalBlockedMs / 100.0
	result.Score = float64(result.TotalCatches) - blockedPenalty
	if result.Score < 0 {
		result.Score = 0
	}
	return result
}

func createAgentNetwork() *nn.Network {
	config := nn.SimpleNetworkConfig{
		InputSize:  MAInputSize,
		HiddenSize: MAHiddenSize,
		OutputSize: MAOutputSize,
		Activation: nn.ActivationLeakyReLU,
		InitScale:  0.3,
		NumLayers:  MANumLayers,
		LayerType:  nn.BrainDense,
		DType:      nn.DTypeFloat32,
	}
	return nn.BuildSimpleNetwork(config)
}

func getSwarmInput(agent *Agent, agents []*Agent, targets []Position) []float32 {
	input := make([]float32, MAInputSize)
	input[0] = float32(agent.Pos.X) / float32(GridSize)
	input[1] = float32(agent.Pos.Y) / float32(GridSize)
	
	// Other agents positions
	idx := 2
	for _, other := range agents {
		if other != agent && idx < 10 {
			input[idx] = float32(other.Pos.X) / float32(GridSize)
			input[idx+1] = float32(other.Pos.Y) / float32(GridSize)
			idx += 2
		}
	}
	
	// Nearest targets
	for _, t := range targets[:min(4, len(targets))] {
		if idx < MAInputSize-1 {
			input[idx] = float32(t.X-agent.Pos.X) / float32(GridSize)
			input[idx+1] = float32(t.Y-agent.Pos.Y) / float32(GridSize)
			idx += 2
		}
	}
	return input
}

func getIdealSwarmMove(agent *Agent, agents []*Agent, targets []Position) []float32 {
	target := make([]float32, 4)
	
	// Find nearest target
	nearest := targets[0]
	minDist := abs(targets[0].X-agent.Pos.X) + abs(targets[0].Y-agent.Pos.Y)
	for _, t := range targets {
		dist := abs(t.X-agent.Pos.X) + abs(t.Y-agent.Pos.Y)
		if dist < minDist {
			minDist = dist
			nearest = t
		}
	}
	
	dx := nearest.X - agent.Pos.X
	dy := nearest.Y - agent.Pos.Y
	
	if abs(dx) > abs(dy) {
		if dx > 0 {
			target[3] = 1.0 // right
		} else {
			target[2] = 1.0 // left
		}
	} else if dy != 0 {
		if dy > 0 {
			target[1] = 1.0 // down
		} else {
			target[0] = 1.0 // up
		}
	}
	return target
}

func moveAgent(agent *Agent, move int) {
	switch move {
	case 0: if agent.Pos.Y > 0 { agent.Pos.Y-- }
	case 1: if agent.Pos.Y < GridSize-1 { agent.Pos.Y++ }
	case 2: if agent.Pos.X > 0 { agent.Pos.X-- }
	case 3: if agent.Pos.X < GridSize-1 { agent.Pos.X++ }
	}
}

func argmax(arr []float32) int {
	if len(arr) == 0 { return 0 }
	maxIdx, maxVal := 0, arr[0]
	for i, v := range arr {
		if v > maxVal { maxVal, maxIdx = v, i }
	}
	return maxIdx
}

func abs(x int) int { if x < 0 { return -x }; return x }
func min(a, b int) int { if a < b { return a }; return b }
func clipGrad(grad, maxVal float32) float32 {
	if grad > maxVal { return maxVal }
	if grad < -maxVal { return -maxVal }
	return grad
}

func saveResults(results *BenchmarkResults) {
	data, _ := json.MarshalIndent(results, "", "  ")
	os.WriteFile("multi_agent_results.json", data, 0644)
	fmt.Println("\n📁 Results saved to multi_agent_results.json")
}

func printSummaryTable(results *BenchmarkResults) {
	fmt.Println("\n╔═══════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════╗")
	fmt.Println("║                                    MULTI-AGENT SWARM BENCHMARK SUMMARY                                                       ║")
	fmt.Println("╠═══════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════╣")
	fmt.Println("║  Training Mode    │ Catches │ Per-Agent       │ Collisions │ Coord% │ Blocked(ms) │ Avail% │   Score   │ Status             ║")
	fmt.Println("╠═══════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════╣")

	for _, r := range results.Results {
		perAgent := fmt.Sprintf("%d/%d/%d/%d", r.CatchesPerAgent[0], r.CatchesPerAgent[1], r.CatchesPerAgent[2], r.CatchesPerAgent[3])
		fmt.Printf("║  %-15s │   %3d   │ %-15s │     %3d    │ %5.1f%% │   %8.0f  │ %5.1f%% │ %9.1f │ ✅ PASS ║\n",
			r.TrainingMode, r.TotalCatches, perAgent, r.Collisions, r.CoordinationScore, r.TotalBlockedMs, r.AvailabilityPct, r.Score)
	}

	fmt.Println("╚═══════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════╝")

	var best *TestResult
	for i := range results.Results {
		if best == nil || results.Results[i].Score > best.Score {
			best = &results.Results[i]
		}
	}
	if best != nil {
		fmt.Printf("\n🏆 BEST: %s | Score: %.1f | Catches: %d | Coordination: %.1f%%\n",
			best.TrainingMode, best.Score, best.TotalCatches, best.CoordinationScore)
	}
}
