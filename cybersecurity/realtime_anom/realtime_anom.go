package main

import (
	"encoding/json"
	"fmt"
	"log"
	"math"
	"os"
	"time"

	"github.com/google/gopacket"
	"github.com/google/gopacket/layers"
	"github.com/google/gopacket/pcap"
	"github.com/openfluke/loom/nn"
)

// ═══════════════════════════════════════════════════════════════════════════════
// REAL-TIME PACKET ANOMALY DETECTION - ADVANCED BENCHMARKING EDITION
// ═══════════════════════════════════════════════════════════════════════════════
//
// Combines real-world packet sniffing with the sophisticated metrics from
// all_sine_wave.go:
//   - Track PREDICTION ACCURACY % every 50ms window
//   - Calculate: Score = (Throughput × Availability × Accuracy) / 10000
//   - All 6 Loom training modes running in parallel
//
// ═══════════════════════════════════════════════════════════════════════════════

const (
	// Network architecture
	ADInputPackets  = 16 // Previous packets
	ADFeaturesCount = 4  // Size, Protocol, SrcPort, DstPort
	ADInputSize     = ADInputPackets * ADFeaturesCount
	ADHiddenSize    = 128
	ADOutputSize    = 1 // Predicting next packet size
	ADNumLayers     = 5 // Deepnet
	ADLearningRate  = float32(0.02)
	ADInitScale     = float32(0.4)

	// Anomaly detection parameters
	AccuracyThreshold     = 0.2 // If error < 0.2, it's a "correct prediction"
	AnomalyErrorThreshold = 0.4 // Error > 0.4 is a detection
	GroundTruthThreshold  = 0.7 // What we consider a "spike"

	// Timing
	TestDuration   = 30 * time.Second
	WindowDuration = 50 * time.Millisecond // 50ms for fine-grained tracking
	TrainInterval  = 10 * time.Millisecond // For batch-based modes

	// Mode Score Factor
	ScoreFactor = 10000.0
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
	ModeNormalBP:       "NormalBP",
	ModeStepBP:         "StepBP",
	ModeTween:          "Tween",
	ModeTweenChain:     "TweenChain",
	ModeStepTween:      "StepTween",
	ModeStepTweenChain: "StepTweenChain",
}

type PacketFeature struct {
	Size    float32
	Proto   float32
	SrcPort float32
	DstPort float32
}

type TimeWindow struct {
	TimeMs           int     `json:"timeMs"`
	Outputs          int     `json:"outputs"`
	TotalAccuracy    float64 `json:"totalAccuracy"`
	Accuracy         float64 `json:"accuracy"`
	MaxLatencyMs     float64 `json:"maxLatencyMs"`
	BlockedMs        float64 `json:"blockedMs"`
	TruePositives    int     `json:"truePositives"`
	FalsePositives   int     `json:"falsePositives"`
	TotalGroundTruth int     `json:"totalGroundTruth"`
}

type ModeResult struct {
	Mode             string       `json:"mode"`
	Windows          []TimeWindow `json:"windows"`
	TotalOutputs     int          `json:"totalOutputs"`
	TotalBlockedMs   float64      `json:"totalBlockedMs"`
	AvgTrainAccuracy float64      `json:"avgAccuracy"`
	TruePositives    int          `json:"truePositives"`
	FalsePositives   int          `json:"falsePositives"`
	TotalGroundTruth int          `json:"totalGroundTruth"`
	ThroughputPerSec float64      `json:"throughput"`
	AvailabilityPct  float64      `json:"availability"`
	MaxLatencyMs     float64      `json:"maxLatency"`
	Stability        float64      `json:"stability"`
	Score            float64      `json:"score"`
}

var history []PacketFeature

func main() {
	fmt.Println("╔═════════════════════════════════════════════════════════════════════════════════════╗")
	fmt.Println("║   🛡️  LOOM CYBERSECURITY: REAL-TIME ANOMALY BENCHMARK                                ║")
	fmt.Println("║                                                                                     ║")
	fmt.Println("║   NEXT-PACKET PREDICTION | 50MS WINDOW TRACKING | 6 ADAPTIVE MODES                  ║")
	fmt.Println("║   Score = (Throughput × Availability% × Accuracy%) / 10000                          ║")
	fmt.Println("╚═════════════════════════════════════════════════════════════════════════════════════╝")

	device, err := findActiveInterface()
	if err != nil {
		log.Fatalf("❌ Error finding interface: %v", err)
	}
	fmt.Printf("🔍 Sniffing on: %s\n", device)

	handle, err := pcap.OpenLive(device, 1600, true, pcap.BlockForever)
	if err != nil {
		log.Fatalf("❌ Error: %v", err)
	}
	defer handle.Close()

	modes := []TrainingMode{
		ModeNormalBP,
		ModeStepBP,
		ModeTween,
		ModeTweenChain,
		ModeStepTween,
		ModeStepTweenChain,
	}

	packetSource := gopacket.NewPacketSource(handle, handle.LinkType())
	packets := packetSource.Packets()

	fmt.Printf("\n📊 Starting %s Benchmark...\n\n", TestDuration)

	results := make(map[TrainingMode]*ModeResult)
	networks := make(map[TrainingMode]*nn.Network)
	states := make(map[TrainingMode]*nn.StepState)
	tweens := make(map[TrainingMode]*nn.TweenState)

	numWindows := int(TestDuration / WindowDuration)
	for _, m := range modes {
		results[m] = &ModeResult{
			Mode:    modeNames[m],
			Windows: make([]TimeWindow, numWindows),
		}
		for i := range results[m].Windows {
			results[m].Windows[i].TimeMs = (i + 1) * int(WindowDuration.Milliseconds())
		}
		networks[m] = createNetwork()
		if m == ModeStepBP || m == ModeStepTween || m == ModeStepTweenChain {
			states[m] = networks[m].InitStepState(ADInputSize)
		}
		if m == ModeTween || m == ModeTweenChain || m == ModeStepTween || m == ModeStepTweenChain {
			tweens[m] = nn.NewTweenState(networks[m], nil)
			if m == ModeTweenChain || m == ModeStepTweenChain {
				tweens[m].Config.UseChainRule = true
			}
		}
	}

	// Batch buffers
	type Sample struct {
		Input  []float32
		Target float32
	}
	batches := make(map[TrainingMode][]Sample)
	lastTrainTime := make(map[TrainingMode]time.Time)
	for _, m := range modes {
		lastTrainTime[m] = time.Now()
	}

	start := time.Now()
	packetCount := 0
	history = make([]PacketFeature, ADInputPackets)

	// Latency tracking
	lastOutputTime := make(map[TrainingMode]time.Time)
	for _, m := range modes {
		lastOutputTime[m] = time.Now()
	}

	for packet := range packets {
		if time.Since(start) >= TestDuration {
			break
		}

		currentFeature := extractFeatures(packet)
		if packetCount < ADInputPackets {
			history[packetCount] = currentFeature
			packetCount++
			continue
		}

		input := flattenHistory(history)
		target := currentFeature.Size
		isGT := target > GroundTruthThreshold

		elapsed := time.Since(start)
		windowIdx := int(elapsed / WindowDuration)
		if windowIdx >= numWindows {
			break
		}

		for _, m := range modes {
			net := networks[m]
			res := results[m]
			win := &res.Windows[windowIdx]

			if isGT {
				win.TotalGroundTruth++
				res.TotalGroundTruth++
			}

			// Availability metric: time since last output (rough estimate of per-mode throughput)
			lat := time.Since(lastOutputTime[m]).Seconds() * 1000
			if lat > win.MaxLatencyMs {
				win.MaxLatencyMs = lat
			}
			lastOutputTime[m] = time.Now()

			var output []float32
			switch m {
			case ModeNormalBP, ModeTween, ModeTweenChain:
				output, _ = net.ForwardCPU(input)
			case ModeStepBP:
				st := states[m]
				st.SetInput(input)
				for i := 0; i < net.TotalLayers(); i++ {
					net.StepForward(st)
				}
				output = st.GetOutput()
			case ModeStepTween, ModeStepTweenChain:
				output = tweens[m].ForwardPass(net, input)
			}

			// 1. Regression Accuracy (for benchmark consistency)
			predictionError := math.Abs(float64(output[0] - target))
			acc := 0.0
			if predictionError < AccuracyThreshold {
				acc = 100.0
			}
			win.TotalAccuracy += acc
			win.Outputs++
			res.TotalOutputs++

			// 2. Anomaly Detection Stats (from anom.go)
			if predictionError > AnomalyErrorThreshold {
				if isGT {
					win.TruePositives++
					res.TruePositives++
				} else {
					win.FalsePositives++
					res.FalsePositives++
				}
			}

			// Training
			switch m {
			case ModeNormalBP:
				batches[m] = append(batches[m], Sample{Input: input, Target: target})
				if time.Since(lastTrainTime[m]) >= TrainInterval {
					tData := make([]nn.TrainingBatch, len(batches[m]))
					for i, s := range batches[m] {
						tData[i] = nn.TrainingBatch{Input: s.Input, Target: []float32{s.Target}}
					}
					t0 := time.Now()
					net.Train(tData, &nn.TrainingConfig{Epochs: 1, LearningRate: ADLearningRate, LossType: "mse"})
					block := time.Since(t0)
					res.TotalBlockedMs += block.Seconds() * 1000
					win.BlockedMs += block.Seconds() * 1000
					batches[m] = batches[m][:0]
					lastTrainTime[m] = time.Now()
				}
			case ModeStepBP:
				grad := []float32{clipGrad(output[0]-target, 0.5)}
				net.StepBackward(states[m], grad)
				net.ApplyGradients(ADLearningRate)
			case ModeTween, ModeTweenChain:
				batches[m] = append(batches[m], Sample{Input: input, Target: target})
				if time.Since(lastTrainTime[m]) >= TrainInterval {
					t0 := time.Now()
					ts := tweens[m]
					for _, s := range batches[m] {
						ts.ForwardPass(net, s.Input)
						ts.BackwardPassRegression(net, []float32{s.Target})
						ts.TweenWeightsChainRule(net, ADLearningRate)
					}
					block := time.Since(t0)
					res.TotalBlockedMs += block.Seconds() * 1000
					win.BlockedMs += block.Seconds() * 1000
					batches[m] = batches[m][:0]
					lastTrainTime[m] = time.Now()
				}
			case ModeStepTween, ModeStepTweenChain:
				ts := tweens[m]
				ts.BackwardPassRegression(net, []float32{target})
				ts.TweenWeightsChainRule(net, ADLearningRate)
			}
		}

		copy(history[:ADInputPackets-1], history[1:])
		history[ADInputPackets-1] = currentFeature
		packetCount++

		if packetCount%100 == 0 {
			fmt.Printf("\r📦 Packets Processed: %d ...", packetCount)
		}
	}

	totalTimeSec := time.Since(start).Seconds()
	fmt.Println("\n\n✅ Benchmark Complete. Finalizing Results...")

	// Calculate Final Metrics
	for _, m := range modes {
		res := results[m]

		// 1. Avg Accuracy
		accSum := 0.0
		for i := range res.Windows {
			if res.Windows[i].Outputs > 0 {
				res.Windows[i].Accuracy = res.Windows[i].TotalAccuracy / float64(res.Windows[i].Outputs)
			}
			accSum += res.Windows[i].Accuracy
		}
		res.AvgTrainAccuracy = accSum / float64(numWindows)

		// 2. Stability (100 - stddev)
		variance := 0.0
		for _, w := range res.Windows {
			diff := w.Accuracy - res.AvgTrainAccuracy
			variance += diff * diff
		}
		res.Stability = math.Max(0, 100-math.Sqrt(variance/float64(numWindows)))

		// 3. Throughput
		res.ThroughputPerSec = float64(res.TotalOutputs) / totalTimeSec

		// 4. Availability
		totalTimeMs := totalTimeSec * 1000
		res.AvailabilityPct = ((totalTimeMs - res.TotalBlockedMs) / totalTimeMs) * 100

		// 5. Score
		res.Score = (res.ThroughputPerSec * res.AvailabilityPct * res.AvgTrainAccuracy) / ScoreFactor

		// 6. Max Latency
		for _, w := range res.Windows {
			if w.MaxLatencyMs > res.MaxLatencyMs {
				res.MaxLatencyMs = w.MaxLatencyMs
			}
		}
	}

	printMetricsTable(results, modes)

	exportResults(results, modes, device)
}

func findActiveInterface() (string, error) {
	devices, err := pcap.FindAllDevs()
	if err != nil {
		return "", err
	}
	for _, d := range devices {
		if d.Name != "lo" && len(d.Addresses) > 0 {
			return d.Name, nil
		}
	}
	if len(devices) > 0 {
		return devices[0].Name, nil
	}
	return "", fmt.Errorf("no interfaces")
}

func extractFeatures(packet gopacket.Packet) PacketFeature {
	f := PacketFeature{
		Size: float32(len(packet.Data())) / 1500.0,
	}
	if f.Size > 1.0 {
		f.Size = 1.0
	}

	if tcpLayer := packet.Layer(layers.LayerTypeTCP); tcpLayer != nil {
		tcp, _ := tcpLayer.(*layers.TCP)
		f.Proto = 1.0
		f.SrcPort = float32(tcp.SrcPort) / 65535.0
		f.DstPort = float32(tcp.DstPort) / 65535.0
	} else if udpLayer := packet.Layer(layers.LayerTypeUDP); udpLayer != nil {
		udp, _ := udpLayer.(*layers.UDP)
		f.Proto = 0.5
		f.SrcPort = float32(udp.SrcPort) / 65535.0
		f.DstPort = float32(udp.DstPort) / 65535.0
	}
	return f
}

func flattenHistory(h []PacketFeature) []float32 {
	out := make([]float32, ADInputSize)
	for i, f := range h {
		out[i*4] = f.Size
		out[i*4+1] = f.Proto
		out[i*4+2] = f.SrcPort
		out[i*4+3] = f.DstPort
	}
	return out
}

func createNetwork() *nn.Network {
	config := nn.SimpleNetworkConfig{
		LayerType:  nn.BrainDense,
		InputSize:  ADInputSize,
		HiddenSize: ADHiddenSize,
		OutputSize: ADOutputSize,
		Activation: nn.ActivationLeakyReLU,
		InitScale:  ADInitScale,
		NumLayers:  ADNumLayers,
		DType:      nn.DTypeFloat32,
	}
	return nn.BuildSimpleNetwork(config)
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

func printMetricsTable(results map[TrainingMode]*ModeResult, modes []TrainingMode) {
	fmt.Println("\n╔═══════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════╗")
	fmt.Println("║                                        REAL-TIME ADAPTATION BENCHMARK SUMMARY                                             ║")
	fmt.Println("╠═══════════════════╦══════════╦══════════╦══════════╦══════════╦═════════╦══════════╦═════════════╦══════════╦══════════════╣")
	fmt.Println("║ Mode              ║ Detected │  GT Sigs │ FalsePos │ Accuracy │ Score   │ Avail %  │ Blocked(ms) │ Peak Lat │ Key Insight  ║")
	fmt.Println("╠═══════════════════╬══════════╬══════════╬══════════╬══════════╬═════════╬══════════╬═════════════╬══════════╬══════════════╣")

	for _, m := range modes {
		r := results[m]
		insight := "Adaptive ✓"
		if m == ModeNormalBP {
			insight = "Blocked ⚠️"
		}
		if r.AvailabilityPct < 80 {
			insight = "High Lag"
		}

		fmt.Printf("║ %-17s ║   %6d │   %6d │   %6d │  %5.1f%%  │ %7.0f │  %5.1f%%  │  %9.0f  │  %6.1fms │ %-12s ║\n",
			r.Mode, r.TruePositives, r.TotalGroundTruth, r.FalsePositives, r.AvgTrainAccuracy, r.Score,
			r.AvailabilityPct, r.TotalBlockedMs, r.MaxLatencyMs, insight)
	}

	fmt.Println("╚═══════════════════╩══════════╩══════════╩══════════╩══════════╩═════════╩══════════╩═════════════╩══════════╩══════════════╝")
}

func exportResults(results map[TrainingMode]*ModeResult, modes []TrainingMode, device string) {
	summary := struct {
		Timestamp string                 `json:"timestamp"`
		Interface string                 `json:"interface"`
		Results   map[string]*ModeResult `json:"results"`
	}{
		Timestamp: time.Now().Format(time.RFC3339),
		Interface: device,
		Results:   make(map[string]*ModeResult),
	}
	for _, m := range modes {
		summary.Results[modeNames[m]] = results[m]
	}
	data, _ := json.MarshalIndent(summary, "", "  ")
	os.WriteFile("realtime_anom_results.json", data, 0644)
	fmt.Println("💾 Full window-by-window metrics saved to realtime_anom_results.json")
}
