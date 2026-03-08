package main

import (
	"fmt"
	"sync"
	"time"

	"github.com/openfluke/loom/poly"
)

type Result struct {
	Label    string
	Forward  time.Duration
	Backward time.Duration
	Total    time.Duration
	SizeKb   float64
	PctSaved float64
	SimTax   float64
}

func main() {
	fmt.Println("=== M-POLY-VTD LSTM Exhaustive Multi-Numerical Benchmark ===")

	batchSize := 1
	inputSize := 32
	hiddenSize := 64
	seqLength := 10
	iterations := 20
	depth := 2

	allTypes := []struct {
		Label string
		Type  poly.DType
	}{
		{"Pure FLOAT64", poly.DTypeFloat64},
		{"Pure FLOAT32", poly.DTypeFloat32},
		{"Pure INT64", poly.DTypeInt64},
		{"Pure UINT64", poly.DTypeUint64},
		{"Pure INT32", poly.DTypeInt32},
		{"Pure UINT32", poly.DTypeUint32},
		{"Pure INT16", poly.DTypeInt16},
		{"Pure UINT16", poly.DTypeUint16},
		{"Pure INT8", poly.DTypeInt8},
		{"Pure UINT8", poly.DTypeUint8},
		{"Pure BFLOAT16", poly.DTypeBFloat16},
		{"Pure FP4", poly.DTypeFP4},
		{"Pure BINARY", poly.DTypeBinary},
	}

	results := make(chan Result, 20)
	var wg sync.WaitGroup

	// Baseline size is FLOAT64 (8 bytes per param)
	wsSize := 4 * (hiddenSize*inputSize + hiddenSize*hiddenSize + hiddenSize)
	baselineSize := float64(depth*wsSize*8) / 1024.0

	for _, s := range allTypes {
		wg.Add(1)
		go func(label string, dtype poly.DType) {
			defer wg.Done()
			net := poly.NewVolumetricNetwork(1, 1, 1, depth)
			for i := 0; i < depth; i++ {
				l := net.GetLayer(0, 0, 0, i)
				l.Type = poly.LayerLSTM
				l.Activation = poly.ActivationTanh
				currentIn := hiddenSize; if i == 0 { currentIn = inputSize }
				l.InputHeight = currentIn
				l.OutputHeight = hiddenSize
				l.SeqLength = seqLength
				l.WeightStore = poly.NewWeightStore(4 * (hiddenSize*currentIn + hiddenSize*hiddenSize + hiddenSize))
				l.DType = dtype
				l.WeightStore.Scale = 0.01
			}
			res := runBenchmark(label, net, batchSize, inputSize, hiddenSize, seqLength, iterations)
			results <- res
		}(s.Label, s.Type)
	}

	wg.Wait()
	close(results)

	resList := []Result{}
	for res := range results {
		res.PctSaved = (1.0 - (res.SizeKb / baselineSize)) * 100.0
		resList = append(resList, res)
	}

	fmt.Println("\n| Scenario              | Forward (avg) | Training (total) | Size (KB) | % Saved | Sim Tax |")
	fmt.Println("|-----------------------|---------------|------------------|-----------|---------|---------|")
	for _, res := range resList {
		fmt.Printf("| %-21s | %-13v | %-16v | %-9.1f | %-7.1f%% | %-7.1f%% |\n",
			res.Label, res.Forward, res.Total, res.SizeKb, res.PctSaved, res.SimTax)
	}
}

func runBenchmark(label string, net *poly.VolumetricNetwork, batch, in, hidden, seq, iter int) Result {
	input := poly.NewTensor[float32](batch, seq, in)
	gradOut := poly.NewTensor[float32](batch, seq, hidden)

	sizeKb := float64(net.CalculateTotalMemory()) / 1024.0
	poly.ForwardPolymorphic(net, input)

	var totalLayerForward time.Duration
	start := time.Now()
	for i := 0; i < iter; i++ {
		_, _, layerTimes := poly.ForwardPolymorphic(net, input)
		for _, d := range layerTimes { totalLayerForward += d }
	}
	fwd := time.Since(start) / time.Duration(iter)
	avgLayerForward := totalLayerForward / time.Duration(iter)

	start = time.Now()
	for i := 0; i < iter; i++ {
		hist_in := make([]*poly.Tensor[float32], len(net.Layers))
		hist_pre := make([]*poly.Tensor[float32], len(net.Layers))
		curr := input
		for idx := range net.Layers {
			l := &net.Layers[idx]
			hist_in[idx] = curr
			pre, post := poly.DispatchLayer(l, curr, nil)
			hist_pre[idx] = pre
			curr = post
		}
		_, grads, _ := poly.BackwardPolymorphic(net, gradOut, hist_in, hist_pre)
		for idx := range net.Layers {
			l := &net.Layers[idx]
			if l.WeightStore != nil && grads[idx][1] != nil {
				gW := poly.ConvertTensor[float32, float32](grads[idx][1])
				l.WeightStore.ApplyGradients(gW, 0.001)
			}
		}
	}
	bwd := time.Since(start) / time.Duration(iter)

	simTax := 0.0
	if fwd > 0 { simTax = (1.0 - (float64(avgLayerForward)/float64(fwd))) * 100.0 }

	return Result{Label: label, Forward: fwd, Total: fwd+bwd, SizeKb: sizeKb, SimTax: simTax}
}
