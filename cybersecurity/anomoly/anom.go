package main

import (
	"encoding/json"
	"fmt"
	"log"
	"math"
	"os"
	"sync"
	"time"

	"github.com/google/gopacket"
	"github.com/google/gopacket/layers"
	"github.com/google/gopacket/pcap"
	"github.com/openfluke/loom/nn"
)

// ═══════════════════════════════════════════════════════════════════════════════
// REAL-TIME PACKET ANOMALY DETECTION - HARDCORE CYBERSECURITY EDITION
// ═══════════════════════════════════════════════════════════════════════════════

const (
	ADInputPackets  = 16 // Number of previous packets to look at
	ADFeaturesCount = 4  // Size, Protocol, SrcPort, DstPort
	ADInputSize     = ADInputPackets * ADFeaturesCount
	ADHiddenSize    = 128
	ADOutputSize    = 1 // Predicting next packet size
	ADNumLayers     = 5 // Hardcore depth
	ADLearningRate  = float32(0.02)
	ADInitScale     = float32(0.4)

	// Anomaly threshold on prediction error
	AnomalyErrorThreshold = 0.4

	// Synthetic "Ground Truth" for reporting (e.g. huge packets or rare ports)
	GroundTruthThreshold = 0.7
)

type TrainingMode int

const (
	ModeNormalBP TrainingMode = iota
	ModeStepBP
	ModeTween
	ModeStepTweenChain
)

var modeNames = map[TrainingMode]string{
	ModeNormalBP:       "NormalBP",
	ModeStepBP:         "StepBP",
	ModeTween:          "Tween",
	ModeStepTweenChain: "StepTweenChain",
}

type PacketFeature struct {
	Size    float32
	Proto   float32
	SrcPort float32
	DstPort float32
}

type AnomResult struct {
	Mode               string  `json:"mode"`
	PacketsSeen        int     `json:"packetsSeen"`
	TruePositives      int     `json:"truePositives"`    // Error > Threshold && GT > Threshold
	FalsePositives     int     `json:"falsePositives"`   // Error > Threshold && GT < Threshold
	TotalGroundTruth   int     `json:"totalGroundTruth"` // GT > Threshold
	AvgPredictionError float64 `json:"avgError"`
	Throughput         float64 `json:"throughput"`
	Score              float64 `json:"score"`
}

var lastPacketTime time.Time
var featureMutex sync.Mutex
var history []PacketFeature

func main() {
	fmt.Println("╔══════════════════════════════════════════════════════════════════════════════════════╗")
	fmt.Println("║   🛡️  LOOM CYBERSECURITY: HARDCORE PACKET ANOMALY DETECTION                           ║")
	fmt.Println("║                                                                                      ║")
	fmt.Println("║   PREDICTING NEXT PACKET SIZE | 5-LAYER DEEP NETWORKS | BATCH VS ADAPTIVE            ║")
	fmt.Println("║   Anomaly = High Prediction Error on Ground-Truth Spikes (>0.7)                      ║")
	fmt.Println("╚══════════════════════════════════════════════════════════════════════════════════════╝")

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

	modes := []TrainingMode{ModeNormalBP, ModeStepBP, ModeTween, ModeStepTweenChain}
	results := make(map[TrainingMode]*AnomResult)
	networks := make(map[TrainingMode]*nn.Network)
	states := make(map[TrainingMode]*nn.StepState)
	tweens := make(map[TrainingMode]*nn.TweenState)

	// Batch buffers
	type Sample struct {
		Input  []float32
		Target float32
	}
	batches := make(map[TrainingMode][]Sample)

	for _, m := range modes {
		results[m] = &AnomResult{Mode: modeNames[m]}
		networks[m] = createNetwork()
		if m == ModeStepBP || m == ModeStepTweenChain {
			states[m] = networks[m].InitStepState(ADInputSize)
		}
		if m == ModeTween || m == ModeStepTweenChain {
			tweens[m] = nn.NewTweenState(networks[m], nil)
			if m == ModeStepTweenChain {
				tweens[m].Config.UseChainRule = true
			}
		}
		if m == ModeNormalBP || m == ModeTween {
			batches[m] = make([]Sample, 0, 50)
		}
	}

	packetSource := gopacket.NewPacketSource(handle, handle.LinkType())
	packets := packetSource.Packets()

	fmt.Println("\n� Performance Benchmarking (30 seconds capture)...\n")

	start := time.Now()
	packetCount := 0
	lastUpdate := time.Now()

	// Initial history
	history = make([]PacketFeature, ADInputPackets)

	for packet := range packets {
		currentFeature := extractFeatures(packet)

		// Wait until we have a full window
		if packetCount < ADInputPackets {
			history[packetCount] = currentFeature
			packetCount++
			continue
		}

		// Input is the previous window
		input := flattenHistory(history)
		// Target is the size of the CURRENT packet
		target := currentFeature.Size

		isGTAnomaly := target > GroundTruthThreshold

		for _, m := range modes {
			net := networks[m]
			res := results[m]
			res.PacketsSeen++
			if isGTAnomaly {
				res.TotalGroundTruth++
			}

			var output []float32
			switch m {
			case ModeNormalBP, ModeTween:
				output, _ = net.ForwardCPU(input)
			case ModeStepBP, ModeStepTweenChain:
				state := states[m]
				state.SetInput(input)
				for i := 0; i < net.TotalLayers(); i++ {
					net.StepForward(state)
				}
				output = state.GetOutput()
			}

			predictionError := math.Abs(float64(output[0] - target))
			res.AvgPredictionError = (res.AvgPredictionError * 0.99) + (predictionError * 0.01)

			// Anomaly Detection: Did we fail to predict this spike?
			if predictionError > AnomalyErrorThreshold {
				if isGTAnomaly {
					res.TruePositives++
				} else {
					res.FalsePositives++
				}
			}

			// Training
			switch m {
			case ModeNormalBP:
				batches[m] = append(batches[m], Sample{Input: input, Target: target})
				if len(batches[m]) >= 50 {
					trainingData := make([]nn.TrainingBatch, len(batches[m]))
					for i, s := range batches[m] {
						trainingData[i] = nn.TrainingBatch{Input: s.Input, Target: []float32{s.Target}}
					}
					net.Train(trainingData, &nn.TrainingConfig{Epochs: 1, LearningRate: ADLearningRate})
					batches[m] = batches[m][:0]
				}
			case ModeStepBP:
				net.StepBackward(states[m], []float32{output[0] - target})
				net.ApplyGradients(ADLearningRate)
			case ModeTween:
				batches[m] = append(batches[m], Sample{Input: input, Target: target})
				if len(batches[m]) >= 50 {
					ts := tweens[m]
					for _, s := range batches[m] {
						ts.ForwardPass(net, s.Input)
						ts.BackwardPassRegression(net, []float32{s.Target})
						ts.TweenWeightsChainRule(net, ADLearningRate)
					}
					batches[m] = batches[m][:0]
				}
			case ModeStepTweenChain:
				ts := tweens[m]
				ts.ForwardPass(net, input)
				ts.BackwardPassRegression(net, []float32{target})
				ts.TweenWeightsChainRule(net, ADLearningRate)
			}
		}

		// Slide history
		copy(history[:ADInputPackets-1], history[1:])
		history[ADInputPackets-1] = currentFeature
		packetCount++

		if time.Since(lastUpdate) >= 500*time.Millisecond {
			fmt.Printf("\r📦 Pkts: %d | GT: %d | NormalBP Err: %.3f | StepTweenChain Err: %.3f",
				packetCount, results[ModeNormalBP].TotalGroundTruth,
				results[ModeNormalBP].AvgPredictionError, results[ModeStepTweenChain].AvgPredictionError)
			lastUpdate = time.Now()
		}

		if time.Since(start) >= 30*time.Second {
			break
		}
	}

	totalTime := time.Since(start).Seconds()
	fmt.Println("\n\n✅ Capture complete. Generating Hardcore Summary...")

	summary := struct {
		Timestamp string                 `json:"timestamp"`
		Interface string                 `json:"interface"`
		Results   map[string]*AnomResult `json:"results"`
	}{
		Timestamp: time.Now().Format(time.RFC3339),
		Interface: device,
		Results:   make(map[string]*AnomResult),
	}

	for m, res := range results {
		res.Throughput = float64(res.PacketsSeen) / totalTime
		// Score = (TP / GT) * 100 - (FP / Seen) * 10
		sr := 0.0
		if res.TotalGroundTruth > 0 {
			sr = float64(res.TruePositives) / float64(res.TotalGroundTruth) * 100
		}
		penalty := float64(res.FalsePositives) / float64(res.PacketsSeen) * 100
		res.Score = sr - penalty
		if res.Score < 0 {
			res.Score = 0
		}

		summary.Results[modeNames[m]] = res
	}

	printSummaryTable(summary.Results)

	data, _ := json.MarshalIndent(summary, "", "  ")
	os.WriteFile("cybersecurity_anom_results.json", data, 0644)
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

func printSummaryTable(results map[string]*AnomResult) {
	fmt.Println("\n╔═══════════════════════════════════════════════════════════════════════════════════════════════════════════════════════╗")
	fmt.Println("║                                 HARDCORE CYBERSECURITY ANOMALY SUMMARY                                               ║")
	fmt.Println("╠═══════════════════════════════════════════════════════════════════════════════════════════════════════════════════════╣")
	fmt.Println("║  Training Mode    │ Detected │ GroundTruth │ FalsePos │ Avg Error │ Throughput (p/s) │   Score   │ Status          ║")
	fmt.Println("╠═══════════════════════════════════════════════════════════════════════════════════════════════════════════════════════╣")

	for _, name := range []string{"NormalBP", "StepBP", "Tween", "StepTweenChain"} {
		res := results[name]
		status := "✅ PASS"
		if res.Score < 10 {
			status = "⚠️ WARN"
		}
		fmt.Printf("║  %-15s │   %6d │      %6d │   %6d │   %7.3f │       %10.1f │ %9.1f │ %s ║\n",
			res.Mode, res.TruePositives, res.TotalGroundTruth, res.FalsePositives, res.AvgPredictionError, res.Throughput, res.Score, status)
	}

	fmt.Println("╚═══════════════════════════════════════════════════════════════════════════════════════════════════════════════════════╝")
}
