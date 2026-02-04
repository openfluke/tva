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

const (
	ADInputPackets  = 32 // 32 * 4 = 128 input size
	ADFeaturesCount = 4
	ADInputSize     = 128
	ADHiddenSize    = 128
	ADOutputSize    = 1
	ADNumLayers     = 5
	ADLearningRate  = float32(0.005)
	ADInitScale     = float32(0.2)

	AccuracyThreshold     = 0.2
	AnomalyErrorThreshold = 0.4
	GroundTruthThreshold  = 0.7

	TestDuration   = 30 * time.Second
	WindowDuration = 100 * time.Millisecond
	TrainInterval  = 20 * time.Millisecond

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

type LayerModeKey struct {
	Layer nn.LayerType
	Mode  TrainingMode
}

var layerTypeNames = map[nn.LayerType]string{
	nn.LayerDense:              "Dense",
	nn.LayerConv2D:             "Conv2D",
	nn.LayerMultiHeadAttention: "MHA",
	nn.LayerRNN:                "RNN",
	nn.LayerLSTM:               "LSTM",
	nn.LayerSoftmax:            "Softmax",
	nn.LayerNorm:               "Norm",
	nn.LayerResidual:           "Residual",
	nn.LayerRMSNorm:            "RMSNorm",
	nn.LayerSwiGLU:             "SwiGLU",
	nn.LayerParallel:           "Parallel",
	nn.LayerEmbedding:          "Embedding",
	nn.LayerConv1D:             "Conv1D",
	nn.LayerSequential:         "Sequential",
	nn.LayerKMeans:             "KMeans",
}

func (k LayerModeKey) String() string {
	return fmt.Sprintf("%s-%s", layerTypeNames[k.Layer], modeNames[k.Mode])
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

func main() {
	fmt.Println("╔═════════════════════════════════════════════════════════════════════════════════════╗")
	fmt.Println("║   🛡️  LOOM CYBERSECURITY: MEGA GRID BENCHMARK                                        ║")
	fmt.Println("║                                                                                     ║")
	fmt.Println("║   15 LAYERS x 6 MODES | 90 NETWORKS IN PARALLEL | 100MS WINDOW TRACKING             ║")
	fmt.Println("║   UNIFORM 128-UNIT BACKBONE | BATCH SYNC FIX                                      ║")
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
		ModeNormalBP, ModeStepBP, ModeTween, ModeTweenChain, ModeStepTween, ModeStepTweenChain,
	}

	layerTypes := []nn.LayerType{
		nn.LayerDense, nn.LayerConv2D, nn.LayerMultiHeadAttention, nn.LayerRNN, nn.LayerLSTM,
		nn.LayerSoftmax, nn.LayerNorm, nn.LayerResidual, nn.LayerRMSNorm, nn.LayerSwiGLU,
		nn.LayerParallel, nn.LayerEmbedding, nn.LayerConv1D, nn.LayerSequential, nn.LayerKMeans,
	}

	packetSource := gopacket.NewPacketSource(handle, handle.LinkType())
	packets := packetSource.Packets()

	fmt.Printf("\n📊 Starting %s Benchmark (Grid: %d Layers x %d Modes = %d Networks)...\n\n",
		TestDuration, len(layerTypes), len(modes), len(layerTypes)*len(modes))

	keys := []LayerModeKey{}
	results := make(map[LayerModeKey]*ModeResult)
	networks := make(map[LayerModeKey]*nn.Network)
	states := make(map[LayerModeKey]*nn.StepState)
	tweens := make(map[LayerModeKey]*nn.TweenState)

	numWindows := int(TestDuration / WindowDuration)
	for _, l := range layerTypes {
		for _, m := range modes {
			key := LayerModeKey{l, m}
			keys = append(keys, key)

			results[key] = &ModeResult{
				Mode:    key.String(),
				Windows: make([]TimeWindow, numWindows),
			}
			for i := range results[key].Windows {
				results[key].Windows[i].TimeMs = (i + 1) * int(WindowDuration.Milliseconds())
			}
			networks[key] = createNetwork(l)

			if m == ModeStepBP || m == ModeStepTween || m == ModeStepTweenChain {
				states[key] = networks[key].InitStepState(ADInputSize)
			}
			if m == ModeTween || m == ModeTweenChain || m == ModeStepTween || m == ModeStepTweenChain {
				tweens[key] = nn.NewTweenState(networks[key], nil)
				if m == ModeTweenChain || m == ModeStepTweenChain {
					tweens[key].Config.UseChainRule = true
				}
			}
		}
	}

	fmt.Println("🚀 Networks created reliably. Starting main processing loop...")

	type Sample struct {
		Input  []float32
		Target float32
	}
	batches := make(map[LayerModeKey][]Sample)
	lastTrainTime := make(map[LayerModeKey]time.Time)
	lastOutputTime := make(map[LayerModeKey]time.Time)
	for _, k := range keys {
		lastTrainTime[k] = time.Now()
		lastOutputTime[k] = time.Now()
	}

	start := time.Now()
	packetCount := 0
	var history []PacketFeature = make([]PacketFeature, ADInputPackets)

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

		for _, k := range keys {
			net := networks[k]
			res := results[k]
			win := &res.Windows[windowIdx]
			m := k.Mode

			if isGT {
				win.TotalGroundTruth++
				res.TotalGroundTruth++
			}

			lat := time.Since(lastOutputTime[k]).Seconds() * 1000
			if lat > win.MaxLatencyMs {
				win.MaxLatencyMs = lat
			}
			lastOutputTime[k] = time.Now()

			var output []float32
			switch m {
			case ModeNormalBP, ModeTween, ModeTweenChain:
				output, _ = net.ForwardCPU(input)
			case ModeStepBP:
				st := states[k]
				if st != nil {
					st.SetInput(input)
					for i := 0; i < net.TotalLayers(); i++ {
						net.StepForward(st)
					}
					output = st.GetOutput()
				}
			case ModeStepTween, ModeStepTweenChain:
				tw := tweens[k]
				if tw != nil {
					output = tw.ForwardPass(net, input)
				}
			}

			if len(output) > 0 {
				predictionError := math.Abs(float64(output[0] - target))
				acc := 0.0
				if predictionError < AccuracyThreshold {
					acc = 100.0
				}
				win.TotalAccuracy += acc
				win.Outputs++
				res.TotalOutputs++

				if predictionError > AnomalyErrorThreshold {
					if isGT {
						win.TruePositives++
						res.TruePositives++
					} else {
						win.FalsePositives++
						res.FalsePositives++
					}
				}
			}

			// Training
			switch m {
			case ModeNormalBP:
				batches[k] = append(batches[k], Sample{Input: input, Target: target})
				if time.Since(lastTrainTime[k]) >= TrainInterval {
					t0 := time.Now()
					// Train samples individually to ensure net.BatchSize=1 remains valid across all layers
					for _, s := range batches[k] {
						net.Train([]nn.TrainingBatch{{Input: s.Input, Target: []float32{s.Target}}}, &nn.TrainingConfig{Epochs: 1, LearningRate: ADLearningRate, LossType: "mse", Verbose: false})
					}
					block := time.Since(t0)
					res.TotalBlockedMs += block.Seconds() * 1000
					win.BlockedMs += block.Seconds() * 1000
					batches[k] = batches[k][:0]
					lastTrainTime[k] = time.Now()
				}
			case ModeStepBP:
				if len(output) > 0 && states[k] != nil {
					grad := []float32{clipGrad(output[0]-target, 0.5)}
					net.StepBackward(states[k], grad)
					net.ApplyGradients(ADLearningRate)
				}
			case ModeTween, ModeTweenChain:
				batches[k] = append(batches[k], Sample{Input: input, Target: target})
				if time.Since(lastTrainTime[k]) >= TrainInterval {
					t0 := time.Now()
					ts := tweens[k]
					if ts != nil {
						for _, s := range batches[k] {
							ts.ForwardPass(net, s.Input)
							ts.BackwardPassRegression(net, []float32{s.Target})
							ts.TweenWeightsChainRule(net, ADLearningRate)
						}
					}
					block := time.Since(t0)
					res.TotalBlockedMs += block.Seconds() * 1000
					win.BlockedMs += block.Seconds() * 1000
					batches[k] = batches[k][:0]
					lastTrainTime[k] = time.Now()
				}
			case ModeStepTween, ModeStepTweenChain:
				ts := tweens[k]
				if ts != nil {
					ts.BackwardPassRegression(net, []float32{target})
					ts.TweenWeightsChainRule(net, ADLearningRate)
				}
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

	for _, k := range keys {
		res := results[k]
		accSum := 0.0
		for i := range res.Windows {
			if res.Windows[i].Outputs > 0 {
				res.Windows[i].Accuracy = res.Windows[i].TotalAccuracy / float64(res.Windows[i].Outputs)
			}
			accSum += res.Windows[i].Accuracy
		}
		res.AvgTrainAccuracy = accSum / float64(numWindows)
		variance := 0.0
		for _, w := range res.Windows {
			diff := w.Accuracy - res.AvgTrainAccuracy
			variance += diff * diff
		}
		res.Stability = math.Max(0, 100-math.Sqrt(variance/float64(numWindows)))
		res.ThroughputPerSec = float64(res.TotalOutputs) / totalTimeSec
		totalTimeMs := totalTimeSec * 1000
		res.AvailabilityPct = ((totalTimeMs - res.TotalBlockedMs) / totalTimeMs) * 100
		res.Score = (res.ThroughputPerSec * res.AvailabilityPct * res.AvgTrainAccuracy) / ScoreFactor
		for _, w := range res.Windows {
			if w.MaxLatencyMs > res.MaxLatencyMs {
				res.MaxLatencyMs = w.MaxLatencyMs
			}
		}
	}

	printMetricsTable(results, keys)
	exportResults(results, keys, device)
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
	return "", fmt.Errorf("no interfaces found")
}

func extractFeatures(packet gopacket.Packet) PacketFeature {
	f := PacketFeature{Size: float32(len(packet.Data())) / 1500.0}
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
		if i*4+3 >= ADInputSize {
			break
		}
		out[i*4] = f.Size
		out[i*4+1] = f.Proto
		out[i*4+2] = f.SrcPort
		out[i*4+3] = f.DstPort
	}
	return out
}

func createNetwork(lType nn.LayerType) *nn.Network {
	net := nn.NewNetwork(ADInputSize, 1, 1, ADNumLayers)
	net.BatchSize = 1
	for i := 0; i < ADNumLayers; i++ {
		// All hidden layers are 128 -> 128
		inSize, outSize := ADHiddenSize, ADHiddenSize
		if i == ADNumLayers-1 {
			outSize = ADOutputSize
		}

		var layer nn.LayerConfig
		switch lType {
		case nn.LayerDense:
			layer = nn.InitDenseLayer(inSize, outSize, nn.ActivationLeakyReLU)
			if i == ADNumLayers-1 {
				layer.Activation = nn.ActivationSigmoid
			}
		case nn.LayerConv2D:
			if i == 0 {
				// 8x8x2 = 128 input. Filters 2. K=3. Pad=1. Stride=1. -> 8x8x2 = 128 output.
				layer = nn.InitConv2DLayer(8, 8, 2, 3, 1, 1, 2, nn.ActivationLeakyReLU)
			} else {
				layer = nn.InitDenseLayer(inSize, outSize, nn.ActivationLeakyReLU)
				if i == ADNumLayers-1 {
					layer.Activation = nn.ActivationSigmoid
				}
			}
		case nn.LayerMultiHeadAttention:
			if i == 1 {
				layer = nn.LayerConfig{Type: nn.LayerMultiHeadAttention, DModel: 128, NumHeads: 4, HeadDim: 32, SeqLength: 1}
			} else {
				layer = nn.InitDenseLayer(inSize, outSize, nn.ActivationLeakyReLU)
				if i == ADNumLayers-1 {
					layer.Activation = nn.ActivationSigmoid
				}
			}
		case nn.LayerRNN:
			layer = nn.InitRNNLayer(inSize, outSize, 1, 1)
		case nn.LayerLSTM:
			layer = nn.InitLSTMLayer(inSize, outSize, 1, 1)
		case nn.LayerSoftmax:
			if i == ADNumLayers-2 {
				// Softmax at penultimate layer (128->128)
				layer = nn.LayerConfig{Type: nn.LayerSoftmax, SoftmaxVariant: nn.SoftmaxStandard, Temperature: 1.0}
			} else {
				layer = nn.InitDenseLayer(inSize, outSize, nn.ActivationLeakyReLU)
				if i == ADNumLayers-1 {
					layer.Activation = nn.ActivationSigmoid
				}
			}
		case nn.LayerNorm:
			if i%2 == 1 {
				layer = nn.LayerConfig{Type: nn.LayerNorm, NormSize: 128, Gamma: make([]float32, 128), Beta: make([]float32, 128), Epsilon: 1e-5}
				for j := range layer.Gamma {
					layer.Gamma[j] = 1.0
				}
			} else {
				layer = nn.InitDenseLayer(inSize, outSize, nn.ActivationLeakyReLU)
			}
		case nn.LayerResidual:
			if i == 2 {
				layer = nn.LayerConfig{Type: nn.LayerResidual}
			} else {
				layer = nn.InitDenseLayer(inSize, outSize, nn.ActivationLeakyReLU)
			}
		case nn.LayerRMSNorm:
			if i%2 == 1 {
				layer = nn.LayerConfig{Type: nn.LayerRMSNorm, NormSize: 128, Gamma: make([]float32, 128), Epsilon: 1e-6}
				for j := range layer.Gamma {
					layer.Gamma[j] = 1.0
				}
			} else {
				layer = nn.InitDenseLayer(inSize, outSize, nn.ActivationLeakyReLU)
			}
		case nn.LayerSwiGLU:
			layer = nn.InitSwiGLUBrain(inSize, ADInitScale)
			if i == ADNumLayers-1 {
				layer = nn.InitDenseLayer(inSize, outSize, nn.ActivationSigmoid)
			}
		case nn.LayerParallel:
			if i == 1 {
				b1 := nn.InitDenseLayer(inSize, 64, nn.ActivationLeakyReLU)
				b2 := nn.InitDenseLayer(inSize, 64, nn.ActivationTanh)
				layer = nn.LayerConfig{Type: nn.LayerParallel, ParallelBranches: []nn.LayerConfig{b1, b2}} // Combined 128
			} else {
				layer = nn.InitDenseLayer(inSize, outSize, nn.ActivationLeakyReLU)
			}
		case nn.LayerEmbedding:
			if i == 0 {
				layer = nn.InitEmbeddingBrain(256, 1, ADInitScale)
			} else {
				layer = nn.InitDenseLayer(inSize, outSize, nn.ActivationLeakyReLU)
			}
		case nn.LayerConv1D:
			if i == 0 {
				layer = nn.InitConv1DLayer(128, 1, 3, 1, 1, 1, nn.ActivationLeakyReLU)
			} else {
				layer = nn.InitDenseLayer(inSize, outSize, nn.ActivationLeakyReLU)
			}
		case nn.LayerSequential:
			layer = nn.InitDenseLayer(inSize, outSize, nn.ActivationLeakyReLU)
		case nn.LayerKMeans:
			if i == 2 {
				sub := nn.InitDenseLayer(inSize, 128, nn.ActivationLeakyReLU)
				layer = nn.InitKMeansLayer(8, sub, "features")
			} else {
				layer = nn.InitDenseLayer(inSize, outSize, nn.ActivationLeakyReLU)
			}
		}

		// Set InputHeight/OutputHeight ONLY if they are not already set by specialized inits
		// This preserves spatial dimensions for Conv2D, etc. while ensuring Dense layers align.
		if layer.InputHeight == 0 {
			layer.InputHeight = inSize
		}
		if layer.OutputHeight == 0 {
			layer.OutputHeight = outSize
		}

		net.SetLayer(0, 0, i, layer)
	}
	net.InitializeWeights()
	return net
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

func printMetricsTable(results map[LayerModeKey]*ModeResult, keys []LayerModeKey) {
	fmt.Println("\n╔═══════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════╗")
	fmt.Println("║                                        REAL-TIME ADAPTATION BENCHMARK SUMMARY (90 NETS)                                  ║")
	fmt.Println("╠═══════════════════╦══════════╦══════════╦══════════╦══════════╦═════════╦══════════╦═════════════╦══════════╦══════════════╣")
	fmt.Println("║ Configuration     ║ Detected │  GT Sigs │ FalsePos │ Accuracy │ Score   │ Avail %  │ Blocked(ms) │ Peak Lat │ Key Insight  ║")
	fmt.Println("╠═══════════════════╬══════════╬══════════╬══════════╬══════════╬═════════╬══════════╬═════════════╬══════════╬══════════════╣")
	for _, k := range keys {
		r := results[k]
		insight := "Adaptive ✓"
		if k.Mode == ModeNormalBP {
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

func exportResults(results map[LayerModeKey]*ModeResult, keys []LayerModeKey, device string) {
	summary := struct {
		Timestamp string                 `json:"timestamp"`
		Interface string                 `json:"interface"`
		Results   map[string]*ModeResult `json:"results"`
	}{Timestamp: time.Now().Format(time.RFC3339), Interface: device, Results: make(map[string]*ModeResult)}
	for _, k := range keys {
		summary.Results[k.String()] = results[k]
	}
	data, _ := json.MarshalIndent(summary, "", "  ")
	os.WriteFile("realtime_anom_results.json", data, 0644)
}
