package main

import (
	"encoding/json"
	"fmt"
	"math/rand"
	"os"
	"sync"
	"time"

	"github.com/openfluke/loom/nn"
)

// ═══════════════════════════════════════════════════════════════════════════════
// PACKET ROUTER BENCHMARK
// ═══════════════════════════════════════════════════════════════════════════════
//
// Continuous stream of packets. Each must be routed to correct destination.
// Blocking = dropped packets = lost data.

const (
	PRInputSize  = 12  // Packet header fields
	PRHiddenSize = 48
	PROutputSize = 8   // 8 possible routes
	PRNumLayers  = 3

	PRLearningRate = float32(0.02)
	PRBatchSize    = 80

	PRTestDuration   = 60 * time.Second
	PRPacketRate     = 2 * time.Millisecond  // 500 packets/second
	PRTrainInterval  = 250 * time.Millisecond

	PRMaxConcurrent = 6
)

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
	ModeNormalBP: "NormalBP", ModeStepBP: "StepBP", ModeTween: "Tween",
	ModeTweenChain: "TweenChain", ModeStepTween: "StepTween", ModeStepTweenChain: "StepTweenChain",
}

type TestResult struct {
	TrainingMode    string  `json:"trainingMode"`
	TotalPackets    int     `json:"totalPackets"`
	RoutedPackets   int     `json:"routedPackets"`
	CorrectRoutes   int     `json:"correctRoutes"`
	DroppedPackets  int     `json:"droppedPackets"` // Due to blocking
	RoutingAccuracy float64 `json:"routingAccuracy"`
	Throughput      float64 `json:"throughput"` // Packets/sec
	TotalBlockedMs  float64 `json:"totalBlockedMs"`
	Score           float64 `json:"score"`
}

type BenchmarkResults struct {
	Results   []TestResult `json:"results"`
	Timestamp string       `json:"timestamp"`
}

func generatePacket(correctRoute int) []float32 {
	packet := make([]float32, PRInputSize)
	// Simulate packet header with destination hints
	for i := range packet {
		packet[i] = rand.Float32() * 0.5
	}
	// Embed destination in header
	destField := correctRoute * (PRInputSize / PROutputSize)
	if destField < PRInputSize {
		packet[destField] = 0.8 + rand.Float32()*0.2
	}
	return packet
}

func main() {
	rand.Seed(time.Now().UnixNano())

	fmt.Println("╔═══════════════════════════════════════════════════════════════════════════════════════════════════════════════╗")
	fmt.Println("║   📡 PACKET ROUTER BENCHMARK                                                                                 ║")
	fmt.Println("║                                                                                                               ║")
	fmt.Println("║   500 packets/sec | 8 routes | Must route each packet                                                       ║")
	fmt.Println("║   BLOCKING = DROPPED PACKETS = LOST DATA                                                                    ║")
	fmt.Println("╚═══════════════════════════════════════════════════════════════════════════════════════════════════════════════╝")

	modes := []TrainingMode{ModeNormalBP, ModeStepBP, ModeTween, ModeTweenChain, ModeStepTween, ModeStepTweenChain}
	results := &BenchmarkResults{Results: make([]TestResult, 0, len(modes)), Timestamp: time.Now().Format(time.RFC3339)}

	var wg sync.WaitGroup
	var mu sync.Mutex
	sem := make(chan struct{}, PRMaxConcurrent)

	for i, mode := range modes {
		wg.Add(1)
		go func(m TrainingMode, idx int) {
			defer wg.Done()
			sem <- struct{}{}
			defer func() { <-sem }()

			fmt.Printf("📡 [%d/%d] Starting %s...\n", idx+1, len(modes), modeNames[m])
			result := runRouterTest(m)
			result.TrainingMode = modeNames[m]

			mu.Lock()
			results.Results = append(results.Results, result)
			mu.Unlock()

			fmt.Printf("✅ [%d/%d] %-15s | Packets: %5d | Dropped: %4d | Acc: %5.1f%% | Throughput: %.0f/s | Score: %.0f\n",
				idx+1, len(modes), modeNames[m], result.TotalPackets, result.DroppedPackets,
				result.RoutingAccuracy, result.Throughput, result.Score)
		}(mode, i)
	}

	wg.Wait()
	saveResults(results)
	printSummaryTable(results)
}

func runRouterTest(mode TrainingMode) TestResult {
	result := TestResult{}

	net := createNetwork()
	numLayers := net.TotalLayers()

	var state *nn.StepState
	if mode == ModeStepBP || mode == ModeStepTween || mode == ModeStepTweenChain {
		state = net.InitStepState(PRInputSize)
	}
	var ts *nn.TweenState
	if mode == ModeTween || mode == ModeTweenChain || mode == ModeStepTween || mode == ModeStepTweenChain {
		ts = nn.NewTweenState(net, nil)
		if mode == ModeTweenChain || mode == ModeStepTweenChain {
			ts.Config.UseChainRule = true
		}
	}

	type Sample struct {
		Input  []float32
		Target int
	}
	trainBatch := make([]Sample, 0, PRBatchSize+10)
	lastTrainTime := time.Now()
	isBlocked := false
	var totalBlockedTime time.Duration

	start := time.Now()
	lastPacketTime := start

	for time.Since(start) < PRTestDuration {
		if time.Since(lastPacketTime) >= PRPacketRate {
			lastPacketTime = time.Now()
			result.TotalPackets++

			correctRoute := rand.Intn(PROutputSize)
			packet := generatePacket(correctRoute)

			// IF BLOCKED, PACKET IS DROPPED
			if isBlocked {
				result.DroppedPackets++
				continue
			}

			var output []float32
			switch mode {
			case ModeNormalBP, ModeTween, ModeTweenChain:
				output, _ = net.ForwardCPU(packet)
			case ModeStepBP:
				state.SetInput(packet)
				for s := 0; s < numLayers; s++ { net.StepForward(state) }
				output = state.GetOutput()
			case ModeStepTween, ModeStepTweenChain:
				output = ts.ForwardPass(net, packet)
			}

			result.RoutedPackets++
			if argmax(output) == correctRoute {
				result.CorrectRoutes++
			}

			trainBatch = append(trainBatch, Sample{Input: packet, Target: correctRoute})
			target := make([]float32, PROutputSize)
			target[correctRoute] = 1.0

			switch mode {
			case ModeNormalBP:
				if len(trainBatch) >= PRBatchSize && time.Since(lastTrainTime) > PRTrainInterval {
					batches := make([]nn.TrainingBatch, len(trainBatch))
					for i, s := range trainBatch {
						t := make([]float32, PROutputSize); t[s.Target] = 1.0
						batches[i] = nn.TrainingBatch{Input: s.Input, Target: t}
					}
					isBlocked = true
					trainStart := time.Now()
					net.Train(batches, &nn.TrainingConfig{Epochs: 1, LearningRate: PRLearningRate, LossType: "crossentropy"})
					totalBlockedTime += time.Since(trainStart)
					isBlocked = false
					trainBatch = trainBatch[:0]; lastTrainTime = time.Now()
				}
			case ModeStepBP:
				grad := make([]float32, len(output))
				for i := range grad { if i < len(target) { grad[i] = clipGrad(output[i]-target[i], 0.5) } }
				net.StepBackward(state, grad); net.ApplyGradients(PRLearningRate)
			case ModeTween, ModeTweenChain:
				if len(trainBatch) >= PRBatchSize && time.Since(lastTrainTime) > PRTrainInterval {
					isBlocked = true
					trainStart := time.Now()
					for _, s := range trainBatch {
						t := make([]float32, PROutputSize); t[s.Target] = 1.0
						out := ts.ForwardPass(net, s.Input)
						grad := make([]float32, len(out))
						for i := range grad { if i < len(t) { grad[i] = t[i] - out[i] } }
						ts.ChainGradients[numLayers] = grad; ts.BackwardTargets[numLayers] = t
						ts.TweenWeightsChainRule(net, PRLearningRate)
					}
					totalBlockedTime += time.Since(trainStart)
					isBlocked = false
					trainBatch = trainBatch[:0]; lastTrainTime = time.Now()
				}
			case ModeStepTween, ModeStepTweenChain:
				grad := make([]float32, len(output))
				for i := range grad { if i < len(target) { grad[i] = target[i] - output[i] } }
				ts.ChainGradients[numLayers] = grad; ts.BackwardTargets[numLayers] = target
				ts.TweenWeightsChainRule(net, PRLearningRate)
			}
		}
	}

	result.TotalBlockedMs = totalBlockedTime.Seconds() * 1000
	if result.RoutedPackets > 0 {
		result.RoutingAccuracy = float64(result.CorrectRoutes) / float64(result.RoutedPackets) * 100
	}
	result.Throughput = float64(result.RoutedPackets) / PRTestDuration.Seconds()

	// Score: throughput × accuracy - drop penalty
	result.Score = result.Throughput * result.RoutingAccuracy / 100 - float64(result.DroppedPackets)
	if result.Score < 0 { result.Score = 0 }
	return result
}

func createNetwork() *nn.Network {
	return nn.BuildSimpleNetwork(nn.SimpleNetworkConfig{
		InputSize: PRInputSize, HiddenSize: PRHiddenSize, OutputSize: PROutputSize,
		Activation: nn.ActivationLeakyReLU, InitScale: 0.4, NumLayers: PRNumLayers,
		LayerType: nn.BrainDense, DType: nn.DTypeFloat32,
	})
}

func argmax(arr []float32) int {
	if len(arr) == 0 { return 0 }
	maxIdx, maxVal := 0, arr[0]
	for i, v := range arr { if v > maxVal { maxVal, maxIdx = v, i } }
	return maxIdx
}

func clipGrad(grad, maxVal float32) float32 {
	if grad > maxVal { return maxVal }
	if grad < -maxVal { return -maxVal }
	return grad
}

func saveResults(results *BenchmarkResults) {
	data, _ := json.MarshalIndent(results, "", "  ")
	os.WriteFile("router_results.json", data, 0644)
	fmt.Println("\n📁 Results saved to router_results.json")
}

func printSummaryTable(results *BenchmarkResults) {
	fmt.Println("\n╔═══════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════╗")
	fmt.Println("║                                    PACKET ROUTER BENCHMARK SUMMARY                                                                  ║")
	fmt.Println("╠═══════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════╣")
	fmt.Println("║  Training Mode    │ Total │ Routed │ Correct │ Dropped │ Accuracy │ Throughput │ Blocked(ms) │   Score   │ Status                   ║")
	fmt.Println("╠═══════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════╣")

	for _, r := range results.Results {
		status := "✅ OK"
		if r.DroppedPackets > 100 { status = "⚠️ DROPS" }
		fmt.Printf("║  %-15s │ %5d │  %5d │   %5d │   %4d  │   %5.1f%% │    %5.0f/s  │    %6.0f   │ %9.1f │ %s ║\n",
			r.TrainingMode, r.TotalPackets, r.RoutedPackets, r.CorrectRoutes,
			r.DroppedPackets, r.RoutingAccuracy, r.Throughput, r.TotalBlockedMs, r.Score, status)
	}

	fmt.Println("╚═══════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════╝")
}
