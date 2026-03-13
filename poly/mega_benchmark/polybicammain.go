package main

import (
	"encoding/json"
	"fmt"
	"math"
	"os"
	"sync"
	"time"

	"github.com/openfluke/loom/poly"
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

	TestDuration   = 10 * time.Second // Reduced for quick verification
	WindowDuration = 100 * time.Millisecond
	TrainInterval  = 20 * time.Millisecond
	SyncInterval   = 10

	ScoreFactor = 10000.0

	UseSyntheticSource = true
)

type Mode int

const (
	ModeNormalBP Mode = iota
	ModeStepBP
	ModeStepBP_Full
	ModeTween
	ModeTweenChain
	ModeStepTween_Full
	ModeStepTweenChain_Full
)

var modeNames = map[Mode]string{
	ModeNormalBP:            "NormalBP",
	ModeStepBP:              "StepBP",
	ModeStepBP_Full:         "StepBP_Full",
	ModeTween:               "Tween",
	ModeTweenChain:          "TweenChain",
	ModeStepTween_Full:      "StepTween_Full",
	ModeStepTweenChain_Full: "StepTweenChain_Full",
}

type LayerModeKey struct {
	Layer poly.LayerType
	Mode  Mode
	Dual  bool // True = Bicameral, False = Single
}

var layerTypeNames = map[poly.LayerType]string{
	poly.LayerDense:              "Dense",
	poly.LayerMultiHeadAttention: "MHA",
	poly.LayerRNN:                "RNN",
	poly.LayerLSTM:               "LSTM",
	poly.LayerSoftmax:            "Softmax",
	poly.LayerLayerNorm:          "Norm",
	poly.LayerRMSNorm:            "RMSNorm",
	poly.LayerSwiGLU:             "SwiGLU",
	poly.LayerParallel:           "Parallel",
	poly.LayerEmbedding:          "Embedding",
	poly.LayerSequential:         "Sequential",
	poly.LayerKMeans:             "KMeans",
	poly.LayerCNN1:               "CNN1",
}

func (k LayerModeKey) String() string {
	config := "Single"
	if k.Dual {
		config = "Dual"
	}
	return fmt.Sprintf("%s-%s-%s", layerTypeNames[k.Layer], modeNames[k.Mode], config)
}

type PacketFeature struct {
	Magnitude float32
	Frequency float32
	Phase     float32
	Entropy   float32
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
	LayerType        string       `json:"layer"`
	Mode             string       `json:"mode"`
	Config           string       `json:"config"`
	Windows          []TimeWindow `json:"windows"`
	TotalOutputs     int          `json:"totalOutputs"`
	TotalBlockedMs   float64      `json:"totalBlockedMs"`
	AvgTrainAccuracy float64      `json:"avgAccuracy"`
	TruePositives    int          `json:"truePositives"`
	FalsePositives   int          `json:"falsePositives"`
	TotalGroundTruth int          `json:"totalGroundTruth"`
	Throughput       float64      `json:"throughput"`
	Availability     float64      `json:"availability"`
	Latency          float64      `json:"latency"`
	Stability        float64      `json:"stability"`
	Score            float64      `json:"score"`
}

type TrainPacket struct {
	Key    LayerModeKey
	Input  *poly.Tensor[float32]
	Target *poly.Tensor[float32]
}

func main() {
	fmt.Println("╔═════════════════════════════════════════════════════════════════════════════════════╗")
	fmt.Println("║   🛡️  LOOM POLY-BICAMERAL MEGA GRID BENCHMARK v5.0                                  ║")
	fmt.Println("║   13 LAYERS x 7 MODES x 2 CONFIGS | 182 NETWORKS | SINGLE VS BICAMERAL              ║")
	fmt.Println("║   METRICS: THROUGHPUT, AVAILABILITY, ACCURACY, POLY SCORE                           ║")
	fmt.Println("╚═════════════════════════════════════════════════════════════════════════════════════╝")

	modes := []Mode{
		ModeNormalBP, ModeStepBP, ModeStepBP_Full,
		ModeTween, ModeTweenChain,
		ModeStepTween_Full, ModeStepTweenChain_Full,
	}
	layerTypes := []poly.LayerType{
		poly.LayerDense, poly.LayerMultiHeadAttention, poly.LayerRNN, poly.LayerLSTM,
		poly.LayerSoftmax, poly.LayerLayerNorm, poly.LayerRMSNorm, poly.LayerSwiGLU,
		poly.LayerParallel, poly.LayerEmbedding, poly.LayerSequential, poly.LayerKMeans,
		poly.LayerCNN1,
	}

	keys := []LayerModeKey{}
	results := make(map[LayerModeKey]*ModeResult)

	leftNets := make(map[LayerModeKey]*poly.VolumetricNetwork)
	rightNets := make(map[LayerModeKey]*poly.VolumetricNetwork)

	numWindows := int(TestDuration / WindowDuration)
	trainChan := make(chan TrainPacket, 10000)

	fmt.Printf("\n📊 Starting %s Benchmark...\n\n", TestDuration)

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
					Name: key.String(), LayerType: layerTypeNames[l], Mode: modeNames[m], Config: configName,
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
			}
		}
	}

	var mu sync.Mutex
	var bgWg sync.WaitGroup
	bgWg.Add(1)
	go func() {
		defer bgWg.Done()
		trainStep := make(map[LayerModeKey]int)
		for p := range trainChan {
			k := p.Key
			net := rightNets[k]
			mode := k.Mode

			switch mode {
			case ModeNormalBP, ModeStepBP, ModeStepBP_Full:
				poly.Train(net, []poly.TrainingBatch[float32]{{Input: p.Input, Target: p.Target}}, &poly.TrainingConfig{Epochs: 1, LearningRate: ADLearningRate, LossType: "mse", Verbose: false})
			case ModeTween, ModeStepTween_Full:
				s := poly.NewTargetPropState[float32](net, &poly.TargetPropConfig{UseChainRule: false, LearningRate: ADLearningRate})
				poly.TargetPropForward(net, s, p.Input)
				poly.TargetPropBackward(net, s, p.Target)
				s.CalculateLinkBudgets()
				poly.ApplyTargetPropGaps(net, s, ADLearningRate)
			case ModeTweenChain, ModeStepTweenChain_Full:
				s := poly.NewTargetPropState[float32](net, &poly.TargetPropConfig{UseChainRule: true, LearningRate: ADLearningRate})
				poly.TargetPropForward(net, s, p.Input)
				poly.TargetPropBackward(net, s, p.Target)
				s.CalculateLinkBudgets()
				poly.ApplyTargetPropGaps(net, s, ADLearningRate)
			}

			trainStep[k]++
			if trainStep[k]%SyncInterval == 0 {
				mu.Lock()
				DeepMirror(net, leftNets[k])
				mu.Unlock()
			}
		}
	}()

	resultChan := make(chan *ModeResult, len(keys))
	var wg sync.WaitGroup
	stopChan := make(chan struct{})

	for _, k := range keys {
		wg.Add(1)
		go func(key LayerModeKey) {
			defer wg.Done()
			runNetworkWorker(key.Layer, key.Mode, key.Dual, resultChan, stopChan, leftNets[key], trainChan)
		}(k)
	}

	initDuration := time.Since(startInit)
	fmt.Printf("🚀 Networks created in %v. Starting Parallel Benchmark...\n", initDuration)

	time.Sleep(TestDuration)
	close(stopChan)

	wg.Wait()
	close(resultChan)
	close(trainChan)
	bgWg.Wait()

	finalResults := make(map[LayerModeKey]*ModeResult)
	for res := range resultChan {
		// Matching back is tricky since key isn't in result. 
		// We'll use the Name string.
		for _, k := range keys {
			if k.String() == res.Name {
				finalResults[k] = res
				
				// Final score calculation
				res.Score = (res.Throughput * res.Availability * res.AvgTrainAccuracy) / ScoreFactor
				break
			}
		}
	}

	printMetricsTable(finalResults, keys)
	exportResults(finalResults, keys)
}

func DeepMirror(n1, n2 *poly.VolumetricNetwork) {
	for i := range n1.Layers {
		l1 := &n1.Layers[i]
		l2 := &n2.Layers[i]
		if l1.WeightStore != nil && l2.WeightStore != nil {
			if len(l1.WeightStore.Master) == len(l2.WeightStore.Master) {
				copy(l2.WeightStore.Master, l1.WeightStore.Master)
			}
		}
	}
}

func printMetricsTable(results map[LayerModeKey]*ModeResult, keys []LayerModeKey) {
	fmt.Println("\n\u2554\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u25ae\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u25ae\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u25ae\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u25ae\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u25ae\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u25ae\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u25ae\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u25ae\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2557")
	fmt.Printf("\u2551 %-25s \u2502 %-12s \u2502 %-10s \u2502 %8s \u2502 %8s \u2502 %8s \u2502 %8s \u2502 %9s \u2502 %12s \u2551\n", "Layer Type", "Mode", "Config", "Score", "Avail", "Acc", "TP/FP", "Block(ms)", "Throughput")
	fmt.Println("\u2560\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u25d4\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u25d4\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u25d4\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u25d4\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u25d4\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u25d4\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u25d4\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u25d4\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2563")
	sortKeys := make([]LayerModeKey, 0, len(results))
	for k := range results {
		sortKeys = append(sortKeys, k)
	}
	// (Skipping sort for brevity)
	for _, k := range keys {
		r, ok := results[k]
		if !ok { continue }
		fmt.Printf("\u2551 %-25s \u2502 %-12s \u2502 %-10s \u2502 %8.1f \u2502 %7.1f%% \u2502 %7.1f%% \u2502 %3d/%-3d \u2502 %9.0f \u2502 %8.0f Hz \u2551\n",
			r.LayerType, r.Mode, r.Config, r.Score, r.Availability, r.AvgTrainAccuracy, r.TruePositives, r.FalsePositives, r.TotalBlockedMs, r.Throughput)
	}
	fmt.Println("\u255a\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2569\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2569\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2569\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2569\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2569\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2569\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2569\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2569\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u255d")
}

func exportResults(results map[LayerModeKey]*ModeResult, keys []LayerModeKey) {
	summary := struct {
		Update  string
		Results []ModeResult
	}{Update: time.Now().Format(time.RFC3339), Results: []ModeResult{}}
	for _, k := range keys {
		if r, ok := results[k]; ok {
			summary.Results = append(summary.Results, *r)
		}
	}
	data, _ := json.MarshalIndent(summary, "", "  ")
	os.WriteFile("rt_poly_bicameral_mega_results.json", data, 0644)
}

func createNetwork(lType poly.LayerType) *poly.VolumetricNetwork {
	net := poly.NewVolumetricNetwork(1, 1, 1, ADNumLayers)
	
	currentIn := ADInputSize
	for i := 0; i < ADNumLayers; i++ {
		l := net.GetLayer(0, 0, 0, i)
		l.Type = poly.LayerDense
		l.Activation = poly.ActivationTanh
		l.DType = poly.DTypeFloat32

		inDim := currentIn
		outDim := ADHiddenSize
		if i == ADNumLayers-1 {
			outDim = ADOutputSize
			l.Activation = poly.ActivationSigmoid
		}

		l.InputHeight = inDim
		l.OutputHeight = outDim
		l.SeqLength = 1

		// Theme logic
		switch lType {
		case poly.LayerRNN, poly.LayerLSTM, poly.LayerDense:
			l.Type = lType
		case poly.LayerMultiHeadAttention:
			if i == 1 {
				l.Type = poly.LayerMultiHeadAttention
			}
		case poly.LayerKMeans:
			if i == 2 {
				l.Type = poly.LayerKMeans
				l.NumClusters = 8
				outDim = 8 // Output of KMeans is cluster probs
			}
		case poly.LayerSoftmax:
			if i == ADNumLayers-2 {
				l.Type = poly.LayerSoftmax
				outDim = inDim
			}
		case poly.LayerLayerNorm, poly.LayerRMSNorm:
			if i%2 == 1 {
				l.Type = lType
				outDim = inDim
			}
		case poly.LayerCNN1:
			if i == 0 {
				l.Type = poly.LayerCNN1
				l.InputChannels = ADInputSize
				l.InputHeight = 1
				l.Filters = ADHiddenSize
				l.KernelSize = 1
				l.Padding = 0
				l.Stride = 1
				outDim = ADHiddenSize
				l.OutputHeight = 1 // Filters(128) * OutLen(1) = 128
			}
		case poly.LayerSwiGLU:
			if i < ADNumLayers-1 {
				l.Type = poly.LayerSwiGLU
			}
		}

		if l.Type != poly.LayerCNN1 {
			l.InputChannels = inDim
			l.Filters = outDim
		}
		
		l.KernelSize = 3
		l.Stride = 1
		l.Padding = 1
		l.DModel = inDim
		l.HeadDim = 32
		l.NumHeads = 4
		l.NumKVHeads = 4
		l.SeqLength = 1
		l.VocabSize = 256
		l.EmbeddingDim = inDim
		if l.NumClusters == 0 { l.NumClusters = 8 }

		wCount := 0
		switch l.Type {
		case poly.LayerDense:
			wCount = inDim * outDim
		case poly.LayerMultiHeadAttention:
			d := l.DModel
			kv := l.NumKVHeads * l.HeadDim
			wCount = 2*d*d + 2*d*kv + 2*d + 2*kv
		case poly.LayerSwiGLU:
			wCount = 3*inDim*outDim + 2*outDim + inDim
		case poly.LayerRMSNorm:
			wCount = inDim // RMSNorm weights are same as input dim
		case poly.LayerRNN:
			wCount = outDim*inDim + outDim*outDim + outDim
		case poly.LayerLSTM:
			gate := outDim*inDim + outDim*outDim + outDim
			wCount = 4 * gate
		case poly.LayerLayerNorm:
			wCount = 2 * inDim
		case poly.LayerEmbedding:
			wCount = l.VocabSize * l.EmbeddingDim
		case poly.LayerKMeans:
			wCount = l.NumClusters * inDim
		case poly.LayerCNN1:
			wCount = l.Filters * l.InputChannels * l.KernelSize
		default:
			// No weights for Softmax, etc.
		}

		if wCount > 0 {
			l.WeightStore = poly.NewWeightStore(wCount)
			l.WeightStore.Randomize(time.Now().UnixNano(), 0.1)
		}
		
		currentIn = outDim
	}
	return net
}

func flattenHistory(history []PacketFeature) []float32 {
	data := make([]float32, ADInputSize)
	for i := 0; i < len(history); i++ {
		if i*4+3 >= ADInputSize { break }
		data[i*4] = history[i].Magnitude
		data[i*4+1] = history[i].Frequency
		data[i*4+2] = history[i].Phase
		data[i*4+3] = history[i].Entropy
	}
	return data
}

func syntheticDataSource(stopChan <-chan struct{}) <-chan PacketFeature {
	out := make(chan PacketFeature, 100)
	go func() {
		defer close(out)
		count := 0
		for {
			select {
			case <-stopChan:
				return
			default:
				out <- PacketFeature{
					Magnitude: float32(math.Mod(float64(count), 100)) / 100.0,
					Frequency: 1.0,
					Phase:     0.5,
					Entropy:   0.5,
				}
				count++
				time.Sleep(10 * time.Microsecond)
			}
		}
	}()
	return out
}

func runNetworkWorker(lType poly.LayerType, mode Mode, isBicameral bool, resultChan chan<- *ModeResult, stopChan <-chan struct{}, net *poly.VolumetricNetwork, trainChan chan<- TrainPacket) {
	inputChan := syntheticDataSource(stopChan)

	configName := "Single"
	if isBicameral { configName = "Bicameral" }
	key := LayerModeKey{lType, mode, isBicameral}
	
	res := &ModeResult{
		Name: key.String(), LayerType: layerTypeNames[lType], Mode: modeNames[mode], Config: configName,
		Windows: make([]TimeWindow, int(TestDuration/WindowDuration)),
	}

	packetCount := 0
	history := make([]PacketFeature, ADInputPackets)
	start := time.Now()

	for currentFeature := range inputChan {
		if packetCount < ADInputPackets {
			history[packetCount] = currentFeature
			packetCount++
			continue
		}

		copy(history, history[1:])
		history[ADInputPackets-1] = currentFeature

		input := poly.NewTensorFromSlice(flattenHistory(history), 1, ADInputSize)
		target := poly.NewTensorFromSlice([]float32{currentFeature.Magnitude}, 1, ADOutputSize)
		isGT := currentFeature.Magnitude > GroundTruthThreshold

		elapsed := time.Since(start)
		winIdx := int(elapsed / WindowDuration)
		if winIdx >= len(res.Windows) { break }
		win := &res.Windows[winIdx]

		if isGT {
			win.TotalGroundTruth++
			res.TotalGroundTruth++
		}

		// Inference
		latStart := time.Now()
		output, _, _ := poly.ForwardPolymorphic(net, input)
		lat := time.Since(latStart).Seconds() * 1000
		if lat > win.MaxLatencyMs { win.MaxLatencyMs = lat }

		// Metrics
		if len(output.Data) > 0 {
			err := math.Abs(float64(output.Data[0] - currentFeature.Magnitude))
			acc := 0.0
			if err < AccuracyThreshold { acc = 100.0 }
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

		// Training
		if isBicameral {
			select {
			case trainChan <- TrainPacket{Key: key, Input: input, Target: target}:
			default:
				win.BlockedMs += 1.0 // Simple block simulation
				res.TotalBlockedMs += 1.0
			}
		} else {
			t0 := time.Now()
			switch mode {
			case ModeNormalBP, ModeStepBP, ModeStepBP_Full:
				poly.Train(net, []poly.TrainingBatch[float32]{{Input: input, Target: target}}, &poly.TrainingConfig{Epochs: 1, LearningRate: ADLearningRate, LossType: "mse", Verbose: false})
			case ModeTween, ModeStepTween_Full:
				s := poly.NewTargetPropState[float32](net, &poly.TargetPropConfig{UseChainRule: false, LearningRate: ADLearningRate})
				poly.TargetPropForward(net, s, input)
				poly.TargetPropBackward(net, s, target)
				s.CalculateLinkBudgets()
				poly.ApplyTargetPropGaps(net, s, ADLearningRate)
			case ModeTweenChain, ModeStepTweenChain_Full:
				s := poly.NewTargetPropState[float32](net, &poly.TargetPropConfig{UseChainRule: true, LearningRate: ADLearningRate})
				poly.TargetPropForward(net, s, input)
				poly.TargetPropBackward(net, s, target)
				s.CalculateLinkBudgets()
				poly.ApplyTargetPropGaps(net, s, ADLearningRate)
			}
			block := time.Since(t0).Seconds() * 1000
			win.BlockedMs += block
			res.TotalBlockedMs += block
		}

		packetCount++
	}

	duration := time.Since(start).Seconds()
	res.Throughput = float64(res.TotalOutputs) / duration
	res.Availability = 100.0 * (duration*1000 - res.TotalBlockedMs) / (duration * 1000)
	if res.Availability < 0 { res.Availability = 0 }
	
	accSum := 0.0
	for _, w := range res.Windows {
		if w.Outputs > 0 {
			accSum += w.TotalAccuracy / float64(w.Outputs)
		}
	}
	res.AvgTrainAccuracy = accSum / float64(len(res.Windows))

	resultChan <- res
}
