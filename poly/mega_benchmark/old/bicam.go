package main

import (
	"encoding/json"
	"fmt"
	"log"
	"math"
	"os"
	"reflect"
	"sync"
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
	SyncInterval   = 10

	ScoreFactor = 10000.0

	UseSyntheticSource = true // Turbo mode for max throughput
)

type TrainingMode int

const (
	ModeNormalBP TrainingMode = iota
	ModeStepBP
	ModeStepBP_Full
	ModeTween
	ModeTweenChain
	ModeStepTween
	ModeStepTween_Full
	ModeStepTweenChain
	ModeStepTweenChain_Full
)

var modeNames = map[TrainingMode]string{
	ModeNormalBP:    "NormalBP",
	ModeStepBP:      "StepBP",
	ModeStepBP_Full: "StepBP_Full",
	ModeTween:       "Tween",
	ModeTweenChain:  "TweenChain",
	//ModeStepTween:           "StepTween",
	ModeStepTween_Full: "StepTween_Full",
	//ModeStepTweenChain:      "StepTweenChain",
	ModeStepTweenChain_Full: "StepTweenChain_Full",
}

type LayerModeKey struct {
	Layer nn.LayerType
	Mode  TrainingMode
	Dual  bool // True = Bicameral, False = Single
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
	config := "Single"
	if k.Dual {
		config = "Dual"
	}
	return fmt.Sprintf("%s-%s-%s", layerTypeNames[k.Layer], modeNames[k.Mode], config)
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
	Name             string       `json:"name"`
	Layer            string       `json:"layer"`
	Mode             string       `json:"mode"`
	Config           string       `json:"config"`
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

type TrainPacket struct {
	Key    LayerModeKey
	Input  []float32
	Target float32
}

type Sample struct {
	Input  []float32
	Target float32
}

func main() {
	fmt.Println("╔═════════════════════════════════════════════════════════════════════════════════════╗")
	fmt.Println("║   🛡️  LOOM CYBER-BICAMERAL MEGA GRID BENCHMARK v4.0                                  ║")
	fmt.Println("║   15 LAYERS x 6 MODES x 2 CONFIGS | 180 NETWORKS | SINGLE VS BICAMERAL              ║")
	fmt.Println("║   METRICS: THROUGHPUT, AVAILABILITY, ACCURACY, LHI SCORE                            ║")
	fmt.Println("╚═════════════════════════════════════════════════════════════════════════════════════╝")

	var device string
	var err error
	var packetSource *gopacket.PacketSource
	var packets chan gopacket.Packet

	if !UseSyntheticSource {
		device, err = findActiveInterface()
		if err != nil {
			log.Fatalf("❌ Error finding interface: %v", err)
		}
		fmt.Printf("🔍 Sniffing on: %s\n", device)
		handle, err := pcap.OpenLive(device, 1600, true, pcap.BlockForever)
		if err != nil {
			log.Fatalf("❌ Error: %v", err)
		}
		defer handle.Close()
		packetSource = gopacket.NewPacketSource(handle, handle.LinkType())
		packets = packetSource.Packets()
	} else {
		device = "synthetic"
		fmt.Println("⚡ Running in SYNTHETIC TURBO MODE (Max Throughput)")
	}

	modes := []TrainingMode{
		ModeNormalBP,
		ModeStepBP, ModeStepBP_Full,
		ModeTween, ModeTweenChain,
		ModeStepTween, ModeStepTween_Full,
		ModeStepTweenChain, ModeStepTweenChain_Full,
	}
	layerTypes := []nn.LayerType{
		nn.LayerDense, nn.LayerConv2D, nn.LayerMultiHeadAttention, nn.LayerRNN, nn.LayerLSTM,
		nn.LayerSoftmax, nn.LayerNorm, nn.LayerResidual, nn.LayerRMSNorm, nn.LayerSwiGLU,
		nn.LayerParallel, nn.LayerEmbedding, nn.LayerConv1D, nn.LayerSequential, nn.LayerKMeans,
	}

	fmt.Printf("\n📊 Starting %s Benchmark...\n\n", TestDuration)

	keys := []LayerModeKey{}
	results := make(map[LayerModeKey]*ModeResult)

	// Data structures for Single & Bicameral
	leftNets := make(map[LayerModeKey]*nn.Network)
	rightNets := make(map[LayerModeKey]*nn.Network) // Only used for Dual

	statesL := make(map[LayerModeKey]*nn.StepState)
	statesR := make(map[LayerModeKey]*nn.StepState)

	tweensL := make(map[LayerModeKey]*nn.TweenState)
	tweensR := make(map[LayerModeKey]*nn.TweenState)

	numWindows := int(TestDuration / WindowDuration)

	// Async training channels for Bicameral mode
	trainChan := make(chan TrainPacket, 10000) // Large buffer for all nets

	startInit := time.Now()
	for _, l := range layerTypes {
		for _, m := range modes {
			for _, dual := range []bool{false, true} {
				key := LayerModeKey{l, m, dual}
				keys = append(keys, key)

				configName := "Single"
				if dual {
					configName = "Bicameral"
				}

				results[key] = &ModeResult{
					Name: key.String(), Layer: layerTypeNames[l], Mode: modeNames[m], Config: configName,
					Windows: make([]TimeWindow, numWindows),
				}
				for i := range results[key].Windows {
					results[key].Windows[i].TimeMs = (i + 1) * int(WindowDuration.Milliseconds())
				}

				leftNets[key] = createNetwork(l)
				if dual {
					rightNets[key] = createNetwork(l)
					DeepMirror(leftNets[key], rightNets[key])
				}

				// Initialize States based on mode
				if m == ModeStepBP || m == ModeStepBP_Full || m == ModeStepTween || m == ModeStepTween_Full || m == ModeStepTweenChain || m == ModeStepTweenChain_Full {
					statesL[key] = leftNets[key].InitStepState(ADInputSize)
					if dual {
						statesR[key] = rightNets[key].InitStepState(ADInputSize)
					}
				}

				// Initialize Tweens
				if m == ModeTween || m == ModeTweenChain || m == ModeStepTween || m == ModeStepTween_Full || m == ModeStepTweenChain || m == ModeStepTweenChain_Full {
					tweensL[key] = nn.NewTweenState(leftNets[key], nil)
					if dual {
						tweensR[key] = nn.NewTweenState(rightNets[key], nil)
					}

					if m == ModeTweenChain || m == ModeStepTweenChain || m == ModeStepTweenChain_Full {
						tweensL[key].Config.UseChainRule = true
						if dual {
							tweensR[key].Config.UseChainRule = true
						}
					}
				}
			}
		}
	}

	// Background Learner for Bicameral Nets
	var mu sync.Mutex
	// We need to keep this mostly as-is, BUT since we are parallel now, this learner should run concurrently
	var bgWg sync.WaitGroup
	bgWg.Add(1)
	go func() {
		defer bgWg.Done()
		trainStep := make(map[LayerModeKey]int)
		for p := range trainChan {
			k := p.Key
			net := rightNets[k] // Train Right Hemisphere
			mode := k.Mode

			switch mode {
			case ModeNormalBP:
				net.Train([]nn.TrainingBatch{{Input: p.Input, Target: []float32{p.Target}}}, &nn.TrainingConfig{Epochs: 1, LearningRate: ADLearningRate, LossType: "mse", Verbose: false})
			case ModeStepBP:
				st := statesR[k]
				st.SetInput(p.Input)
				net.StepForward(st) // Single Step
				out := st.GetOutput()
				if len(out) > 0 {
					grad := []float32{clipGrad(out[0]-p.Target, 0.5)}
					net.StepBackward(st, grad)
					net.ApplyGradients(ADLearningRate)
				}
			case ModeStepBP_Full:
				st := statesR[k]
				st.SetInput(p.Input)
				for i := 0; i < net.TotalLayers(); i++ {
					net.StepForward(st)
				}
				out := st.GetOutput()
				if len(out) > 0 {
					grad := []float32{clipGrad(out[0]-p.Target, 0.5)}
					net.StepBackward(st, grad)
					net.ApplyGradients(ADLearningRate)
				}
			case ModeTween, ModeTweenChain, ModeStepTween, ModeStepTween_Full, ModeStepTweenChain, ModeStepTweenChain_Full:
				tw := tweensR[k]
				if mode == ModeTween || mode == ModeTweenChain || mode == ModeStepTween_Full || mode == ModeStepTweenChain_Full {
					tw.ForwardPass(net, p.Input)
				} else {
					//tw.StepForward(net, p.Input)
				}
				tw.BackwardPassRegression(net, []float32{p.Target})
				if mode == ModeTweenChain || mode == ModeStepTweenChain || mode == ModeStepTweenChain_Full {
					tw.TweenWeightsChainRule(net, ADLearningRate)
				} else {
					tw.TweenWeights(net, ADLearningRate)
				}
			}

			trainStep[k]++
			if trainStep[k]%SyncInterval == 0 {
				mu.Lock()
				DeepMirror(net, leftNets[k]) // Right -> Left Sync
				mu.Unlock()
			}
		}
	}()

	// 🚀 PARALLEL WORKER SETUP
	inputChans := make(map[LayerModeKey]chan PacketFeature)
	resultChan := make(chan *ModeResult, len(keys))
	var wg sync.WaitGroup

	for _, k := range keys {
		inputChans[k] = make(chan PacketFeature, 1000) // Buffer to allow some drift
		wg.Add(1)
		go runNetworkWorker(k, inputChans[k], &wg, resultChan, trainChan, leftNets[k], statesL[k], tweensL[k])
	}

	initDuration := time.Since(startInit)
	fmt.Printf("🚀 Networks created reliably in %v. Starting Parallel Turbo Mode...\n", initDuration)

	start := time.Now()
	packetCount := 0

	// Synthetic Loop (Controller)
	for {
		if time.Since(start) >= TestDuration {
			break
		}

		var currentFeature PacketFeature
		if UseSyntheticSource {
			// Fast synthetic generation
			currentFeature = PacketFeature{
				Size:    float32(math.Mod(float64(packetCount), 100)) / 100.0,
				Proto:   1.0,
				SrcPort: 0.5,
				DstPort: 0.5,
			}
		} else {
			// Real packet consumption
			select {
			case packet := <-packets:
				currentFeature = extractFeatures(packet)
			default:
				// Non-blocking wait if no packets
				continue
			}
		}

		// BROADCAST TO ALL WORKERS
		for _, k := range keys {
			// Non-blocking send
			select {
			case inputChans[k] <- currentFeature:
			default:
				// Worker full, skip frame (simulates drop)
			}
		}

		packetCount++
		if packetCount%10000 == 0 {
			fmt.Printf("\r📦 Packets Broadcast: %d | Time: %.1fs", packetCount, time.Since(start).Seconds())
		}
	}

	// Stop Workers
	for _, ch := range inputChans {
		close(ch)
	}

	fmt.Println("\nWaiting for workers to finish...")
	wg.Wait()
	close(resultChan)

	// Stop Background Learner
	close(trainChan)
	bgWg.Wait() // Wait for BG learner to drain

	fmt.Println("Aggregating Results...")
	// Collect Results
	for res := range resultChan {
		// Map result back to 'results' map using Name
		for k, v := range results {
			if v.Name == res.Name {
				results[k] = res

				// Re-calculate aggregate stats (redundant if worker did it, but good for safety)
				// Actually worker did NOT calculate final 'AvgTrainAccuracy' over all windows or 'Score'.
				// Worker only filled windows and basic counters.
				// We should finalize here.

				totalTimeSec := time.Since(start).Seconds()
				res.ThroughputPerSec = float64(res.TotalOutputs) / totalTimeSec
				totalTimeMs := totalTimeSec * 1000
				res.AvailabilityPct = ((totalTimeMs - res.TotalBlockedMs) / totalTimeMs) * 100
				if res.AvailabilityPct < 0 {
					res.AvailabilityPct = 0
				}

				accSum := 0.0
				validWins := 0
				for i := range res.Windows {
					if res.Windows[i].Outputs > 0 {
						res.Windows[i].Accuracy = res.Windows[i].TotalAccuracy / float64(res.Windows[i].Outputs)
						accSum += res.Windows[i].Accuracy
						validWins++
					}
				}
				if validWins > 0 {
					res.AvgTrainAccuracy = accSum / float64(validWins) // Use valid wins only? Or all windows (inc zero)?
					// Original code used numWindows const.
					res.AvgTrainAccuracy = accSum / float64(numWindows) // Stick to original formula
				}

				res.Score = (res.ThroughputPerSec * res.AvailabilityPct * res.AvgTrainAccuracy) / ScoreFactor

				// Stability Calc
				variance := 0.0
				for _, w := range res.Windows {
					diff := w.Accuracy - res.AvgTrainAccuracy
					variance += diff * diff
				}
				res.Stability = math.Max(0, 100-math.Sqrt(variance/float64(numWindows)))

				break
			}
		}
	}

	printMetricsTable(results, keys)
	exportResults(results, keys, device)
}

func DeepMirror(n1, n2 *nn.Network) {
	for i := range n1.Layers {
		mirrorLayerConfig(&n1.Layers[i], &n2.Layers[i])
	}
}

func mirrorLayerConfig(l1, l2 *nn.LayerConfig) {
	if l1 == nil || l2 == nil {
		return
	}
	v1 := reflect.ValueOf(l1).Elem()
	v2 := reflect.ValueOf(l2).Elem()
	for j := 0; j < v1.NumField(); j++ {
		f1 := v1.Field(j)
		f2 := v2.Field(j)
		if f1.Kind() == reflect.Slice && f1.Type().Elem().Kind() == reflect.Float32 {
			s1 := f1.Interface().([]float32)
			s2 := f2.Interface().([]float32)
			if len(s1) > 0 && len(s1) == len(s2) {
				for k := range s1 {
					avg := (s1[k] + s2[k]) / 2.0
					s1[k] = avg
					s2[k] = avg
				}
			}
			continue
		}
		if f1.Kind() == reflect.Slice && f1.Type().Elem().Name() == "LayerConfig" {
			for k := 0; k < f1.Len(); k++ {
				mirrorLayerConfig(f1.Index(k).Addr().Interface().(*nn.LayerConfig), f2.Index(k).Addr().Interface().(*nn.LayerConfig))
			}
			continue
		}
		if f1.Kind() == reflect.Ptr && f1.Type().Elem().Name() == "LayerConfig" {
			if !f1.IsNil() && !f2.IsNil() {
				mirrorLayerConfig(f1.Interface().(*nn.LayerConfig), f2.Interface().(*nn.LayerConfig))
			}
			continue
		}
	}
}

func printMetricsTable(results map[LayerModeKey]*ModeResult, keys []LayerModeKey) {
	fmt.Println("\n╔═════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════╗")
	fmt.Printf("║ %-25s │ %-12s │ %-10s │ %8s │ %8s │ %8s │ %8s │ %9s │ %12s ║\n", "Layer Type", "Mode", "Config", "Score", "Avail", "Acc", "TP/FP", "Block(ms)", "Throughput")
	fmt.Println("╠══════════════════════════╪══════════════╪════════════╪══════════╪══════════╪══════════╪══════════╪═══════════╪══════════════╣")
	for _, k := range keys {
		r := results[k]
		fmt.Printf("║ %-25s │ %-12s │ %-10s │ %8.1f │ %7.1f%% │ %7.1f%% │ %3d/%-3d │ %9.0f │ %8.0f Hz ║\n",
			r.Layer, r.Mode, r.Config, r.Score, r.AvailabilityPct, r.AvgTrainAccuracy, r.TruePositives, r.FalsePositives, r.TotalBlockedMs, r.ThroughputPerSec)
	}
	fmt.Println("╚══════════════════════════╧══════════════╧════════════╧══════════╧══════════╧══════════╧══════════╧═══════════╧══════════════╝")
}

func exportResults(results map[LayerModeKey]*ModeResult, keys []LayerModeKey, device string) {
	summary := struct {
		Update  string
		Results []ModeResult
	}{Update: time.Now().Format(time.RFC3339), Results: []ModeResult{}}
	for _, k := range keys {
		summary.Results = append(summary.Results, *results[k])
	}
	data, _ := json.MarshalIndent(summary, "", "  ")
	os.WriteFile("rt_bicameral_mega_results.json", data, 0644)
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

func createNetwork(lType nn.LayerType) *nn.Network {
	net := nn.NewNetwork(ADInputSize, 1, 1, ADNumLayers)
	net.BatchSize = 1
	for i := 0; i < ADNumLayers; i++ {
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
				layer = nn.LayerConfig{Type: nn.LayerParallel, ParallelBranches: []nn.LayerConfig{b1, b2}}
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

func deepCopy(src []float32) []float32 { dst := make([]float32, len(src)); copy(dst, src); return dst }
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

func runNetworkWorker(
	k LayerModeKey,
	inputChan <-chan PacketFeature,
	wg *sync.WaitGroup,
	finalResultChan chan<- *ModeResult,
	trainChan chan<- TrainPacket,
	net *nn.Network,
	st *nn.StepState,
	tw *nn.TweenState,
) {
	defer wg.Done()

	// Initialize Result for this worker
	numWindows := int(TestDuration / WindowDuration)
	configName := "Single"
	if k.Dual {
		configName = "Bicameral"
	}
	res := &ModeResult{
		Name: k.String(), Layer: layerTypeNames[k.Layer], Mode: modeNames[k.Mode], Config: configName,
		Windows: make([]TimeWindow, numWindows),
	}

	packetCount := 0
	var history []PacketFeature = make([]PacketFeature, ADInputPackets)
	var batches []Sample

	start := time.Now()
	lastTrainTime := start
	lastOutputTime := start

	for currentFeature := range inputChan {
		// Update History
		if packetCount < ADInputPackets {
			history[packetCount] = currentFeature
			packetCount++
			continue
		}
		copy(history[:ADInputPackets-1], history[1:])
		history[ADInputPackets-1] = currentFeature
		packetCount++

		// Prepare Input
		input := flattenHistory(history)
		target := currentFeature.Size
		isGT := target > GroundTruthThreshold

		// Time Window Logic
		elapsed := time.Since(start)
		windowIdx := int(elapsed / WindowDuration)
		if windowIdx >= numWindows {
			continue // Drain
		}
		win := &res.Windows[windowIdx]

		if isGT {
			win.TotalGroundTruth++
			res.TotalGroundTruth++
		}

		// Latency
		lat := time.Since(lastOutputTime).Seconds() * 1000
		if lat > win.MaxLatencyMs {
			win.MaxLatencyMs = lat
		}
		lastOutputTime = time.Now()

		// --- INFERENCE ---
		var output []float32

		switch k.Mode {
		case ModeNormalBP, ModeTween, ModeTweenChain:
			output, _ = net.ForwardCPU(input)
		case ModeStepBP:
			st.SetInput(input)
			net.StepForward(st)
			output = st.GetOutput()
		case ModeStepBP_Full:
			st.SetInput(input)
			for i := 0; i < net.TotalLayers(); i++ {
				net.StepForward(st)
			}
			output = st.GetOutput()
		case ModeStepTween_Full, ModeStepTweenChain_Full:
			output = tw.ForwardPass(net, input)
		case ModeStepTween, ModeStepTweenChain:
			//output = tw.StepForward(net, input)
		}

		// Metrics
		if len(output) > 0 {
			err := math.Abs(float64(output[0] - target))
			acc := 0.0
			if err < AccuracyThreshold {
				acc = 100.0
			}
			win.TotalAccuracy += acc
			win.Outputs++
			res.TotalOutputs++

			if err > AnomalyErrorThreshold {
				if isGT {
					win.TruePositives++
					res.TruePositives++
				} else {
					win.FalsePositives++
					res.FalsePositives++
				}
			}
		}

		// --- TRAINING / DISPATCH ---
		if !k.Dual {
			// Single Mode: Blocking Train
			switch k.Mode {
			case ModeNormalBP, ModeTween, ModeTweenChain:
				batches = append(batches, Sample{Input: input, Target: target})
				if time.Since(lastTrainTime) >= TrainInterval {
					t0 := time.Now()
					for _, s := range batches {
						if k.Mode == ModeNormalBP {
							net.Train([]nn.TrainingBatch{{Input: s.Input, Target: []float32{s.Target}}}, &nn.TrainingConfig{Epochs: 1, LearningRate: ADLearningRate, LossType: "mse", Verbose: false})
						} else {
							tw.ForwardPass(net, s.Input)
							tw.BackwardPassRegression(net, []float32{s.Target})
							if k.Mode == ModeTweenChain {
								tw.TweenWeightsChainRule(net, ADLearningRate)
							} else {
								tw.TweenWeights(net, ADLearningRate)
							}
						}
					}
					dur := float64(time.Since(t0).Milliseconds())
					res.TotalBlockedMs += dur
					win.BlockedMs += dur
					batches = nil // clear
					lastTrainTime = time.Now()
				}
			case ModeStepBP, ModeStepBP_Full:
				t0 := time.Now()
				net.Train([]nn.TrainingBatch{{Input: input, Target: []float32{target}}}, &nn.TrainingConfig{Epochs: 1, LearningRate: ADLearningRate, LossType: "mse"})
				dur := float64(time.Since(t0).Milliseconds())
				res.TotalBlockedMs += dur
				win.BlockedMs += dur

			case ModeStepTween, ModeStepTween_Full, ModeStepTweenChain, ModeStepTweenChain_Full:
				t0 := time.Now()
				tw.BackwardPassRegression(net, []float32{target})
				if k.Mode == ModeStepTweenChain || k.Mode == ModeStepTweenChain_Full {
					tw.TweenWeightsChainRule(net, ADLearningRate)
				} else {
					tw.TweenWeights(net, ADLearningRate)
				}
				dur := float64(time.Since(t0).Milliseconds())
				res.TotalBlockedMs += dur
				win.BlockedMs += dur
			}
		} else {
			// Bicameral Mode: Non-blocking Dispatch
			select {
			case trainChan <- TrainPacket{Key: k, Input: deepCopy(input), Target: target}:
			default:
				// Dropped packet (buffer full)
			}
		}
	}

	// Done
	finalResultChan <- res
}
