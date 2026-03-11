package main

import (
	"flag"
	"fmt"
	"strings"
	"time"

	"github.com/openfluke/loom/poly"
)

type Result struct {
	Label        string
	Forward      time.Duration
	ForwardTile  time.Duration
	Backward     time.Duration
	Total        time.Duration
	SizeKb       float64
	PctSaved     float64
	SimTax       float64 // % of time spent in dispatch/overhead vs pure math
}

func main() {
	layerFlag := flag.String("layer", "all", "Specify layer to benchmark (all, dense, cnn1, cnn2, cnn3, embedding, lstm, mha, rmsnorm, layernorm, residual, rnn, sequential, swiglu)")
	flag.Parse()

	target := strings.ToLower(*layerFlag)

	fmt.Println("=== M-POLY-VTD Truly Exhaustive Multi-Numerical Benchmark ===")

	allTypes := []struct {
		Label string
		Type  poly.DType
	}{
		{"Pure FLOAT64", poly.DTypeFloat64},
		{"Pure FLOAT32", poly.DTypeFloat32},
		{"Pure FLOAT16", poly.DTypeFloat16},
		{"Pure BFLOAT16", poly.DTypeBFloat16},
		{"Pure FP8 (E4M3)", poly.DTypeFP8E4M3},
		{"Pure FP8 (E5M2)", poly.DTypeFP8E5M2},
		{"Pure INT64", poly.DTypeInt64},
		{"Pure INT32", poly.DTypeInt32},
		{"Pure INT16", poly.DTypeInt16},
		{"Pure INT8", poly.DTypeInt8},
		{"Pure UINT64", poly.DTypeUint64},
		{"Pure UINT32", poly.DTypeUint32},
		{"Pure UINT16", poly.DTypeUint16},
		{"Pure UINT8", poly.DTypeUint8},
		{"Pure INT4", poly.DTypeInt4},
		{"Pure UINT4", poly.DTypeUint4},
		{"Pure FP4", poly.DTypeFP4},
		{"Pure INT2", poly.DTypeInt2},
		{"Pure UINT2", poly.DTypeUint2},
		{"Pure TERNARY", poly.DTypeTernary},
		{"Pure BINARY", poly.DTypeBinary},
	}

	// 1. DENSE BENCHMARKS
	if target == "all" || target == "dense" {
		runBenchSet("DENSE", poly.LayerDense, allTypes)
	}

	// 2. CNN3 BENCHMARKS
	if target == "all" || target == "cnn3" {
		runBenchSet("CNN3", poly.LayerCNN3, allTypes)
	}

	// 3. CNN2 BENCHMARKS
	if target == "all" || target == "cnn2" {
		runBenchSet("CNN2", poly.LayerCNN2, allTypes)
	}

	// 4. CNN1 BENCHMARKS
	if target == "all" || target == "cnn1" {
		runBenchSet("CNN1", poly.LayerCNN1, allTypes)
	}

	// 5. RNN BENCHMARKS
	if target == "all" || target == "rnn" {
		runBenchSet("RNN", poly.LayerRNN, allTypes)
	}

	// 6. EMBEDDING BENCHMARKS
	if target == "all" || target == "embedding" {
		runBenchSet("EMBEDDING", poly.LayerEmbedding, allTypes)
	}

	// 6. LSTM BENCHMARKS
	if target == "all" || target == "lstm" {
		runBenchSet("LSTM", poly.LayerLSTM, allTypes)
	}

	// 7. MHA BENCHMARKS
	if target == "all" || target == "mha" {
		runBenchSet("MHA", poly.LayerMultiHeadAttention, allTypes)
	}

	// 8. RMSNORM BENCHMARKS
	if target == "all" || target == "rmsnorm" {
		runBenchSet("RMSNORM", poly.LayerRMSNorm, allTypes)
	}

	// 9. LAYERNORM BENCHMARKS
	if target == "all" || target == "layernorm" {
		runBenchSet("LAYERNORM", poly.LayerLayerNorm, allTypes)
	}

	// 10. RESIDUAL BENCHMARKS
	if target == "all" || target == "residual" {
		runBenchSet("RESIDUAL", poly.LayerResidual, allTypes)
	}

	// 11. SEQUENTIAL BENCHMARKS
	if target == "all" || target == "sequential" {
		runBenchSet("SEQUENTIAL", poly.LayerSequential, allTypes)
	}

	// 12. SWIGLU BENCHMARKS
	if target == "all" || target == "swiglu" {
		runBenchSet("SWIGLU", poly.LayerSwiGLU, allTypes)
	}
}

func runBenchSet(title string, lType poly.LayerType, allTypes []struct {
	Label string
	Type  poly.DType
}) {
	fmt.Printf("\n=== %s Benchmarks ===\n", title)

	batchSize := 1
	iterations := 20
	depth := 4
	if lType == poly.LayerEmbedding || lType == poly.LayerLSTM || lType == poly.LayerMultiHeadAttention || lType == poly.LayerRMSNorm || lType == poly.LayerLayerNorm || lType == poly.LayerResidual || lType == poly.LayerRNN || lType == poly.LayerSequential || lType == poly.LayerSwiGLU {
		depth = 1
	}

	// Dimensions
	inputSize := 256
	outputSize := 256
	if lType == poly.LayerCNN3 {
		inputSize = 512
		outputSize = 512
	}

	resList := []Result{}

	// Individual Type Benchmarks
	for _, s := range allTypes {
		net := poly.NewVolumetricNetwork(1, 1, 1, depth)
		for i := 0; i < depth; i++ {
			l := net.GetLayer(0, 0, 0, i)
			l.Type = lType
			l.DType = s.Type
			l.Activation = poly.ActivationReLU

			if lType == poly.LayerCNN3 {
				l.InputChannels = 8
				l.InputDepth = 4
				l.InputHeight = 4
				l.InputWidth = 4
				l.Filters = 8
				l.KernelSize = 3
				l.Stride = 1
				l.Padding = 1
				l.OutputDepth = 4
				l.OutputHeight = 4
				l.OutputWidth = 4
				l.WeightStore = poly.NewWeightStore(l.Filters * l.InputChannels * l.KernelSize * l.KernelSize * l.KernelSize)
			} else if lType == poly.LayerCNN2 {
				l.InputChannels = 8
				l.InputHeight = 8
				l.InputWidth = 8
				l.Filters = 8
				l.KernelSize = 3
				l.Stride = 1
				l.Padding = 1
				l.OutputHeight = 8
				l.OutputWidth = 8
				l.WeightStore = poly.NewWeightStore(l.Filters * l.InputChannels * l.KernelSize * l.KernelSize)
			} else if lType == poly.LayerCNN1 {
				l.InputChannels = 8
				l.InputHeight = 64
				l.Filters = 8
				l.KernelSize = 3
				l.Stride = 1
				l.Padding = 1
				l.OutputHeight = 64
				l.WeightStore = poly.NewWeightStore(l.Filters * l.InputChannels * l.KernelSize)
			} else if lType == poly.LayerEmbedding {
				l.VocabSize = 1024
				l.EmbeddingDim = 256
				l.WeightStore = poly.NewWeightStore(l.VocabSize * l.EmbeddingDim)
			} else if lType == poly.LayerLSTM {
				l.InputHeight = 128
				l.OutputHeight = 256
				l.SeqLength = 16
				ihSize := l.OutputHeight * l.InputHeight
				hhSize := l.OutputHeight * l.OutputHeight
				bSize := l.OutputHeight
				l.WeightStore = poly.NewWeightStore(4 * (ihSize + hhSize + bSize))
			} else if lType == poly.LayerRNN {
				l.InputHeight = 128
				l.OutputHeight = 128
				l.SeqLength = 16
				ihSize := l.OutputHeight * l.InputHeight
				hhSize := l.OutputHeight * l.OutputHeight
				bSize := l.OutputHeight
				l.WeightStore = poly.NewWeightStore(ihSize + hhSize + bSize)
			} else if lType == poly.LayerMultiHeadAttention {
				l.DModel = 128
				l.NumHeads = 4
				l.NumKVHeads = 4
				l.HeadDim = 32
				l.MaxSeqLen = 512
				kvDim := l.NumKVHeads * l.HeadDim
				l.WeightStore = poly.NewWeightStore(2*(l.DModel*l.DModel+l.DModel) + 2*(l.DModel*kvDim+kvDim))
			} else if lType == poly.LayerRMSNorm {
				l.OutputHeight = 1024
				l.WeightStore = poly.NewWeightStore(l.OutputHeight)
			} else if lType == poly.LayerLayerNorm {
				l.OutputHeight = 1024
				l.WeightStore = poly.NewWeightStore(2 * l.OutputHeight)
			} else if lType == poly.LayerResidual {
				l.OutputHeight = 1024
				l.WeightStore = poly.NewWeightStore(0) // No weights
			} else if lType == poly.LayerSequential {
				l.SequentialLayers = []poly.VolumetricLayer{
					{Type: poly.LayerDense, InputHeight: 256, OutputHeight: 256, WeightStore: poly.NewWeightStore(256 * 256)},
					{Type: poly.LayerDense, InputHeight: 256, OutputHeight: 256, WeightStore: poly.NewWeightStore(256 * 256)},
				}
				l.WeightStore = poly.NewWeightStore(0) // Sequential itself has no weights
			} else if lType == poly.LayerSwiGLU {
				l.InputHeight = 128
				l.OutputHeight = 256
				ihSize := 128 * 256
				l.WeightStore = poly.NewWeightStore(3*ihSize + 2*256 + 128)
			} else {
				l.InputHeight = inputSize
				l.OutputHeight = outputSize
				l.WeightStore = poly.NewWeightStore(inputSize * outputSize)
			}
			l.WeightStore.Scale = 0.01
		}
		res := runBenchmark(s.Label, net, batchSize, lType, inputSize, outputSize, iterations)

		// Calculate PctSaved
		baselineWeights := depth * inputSize * outputSize
		if lType == poly.LayerLSTM {
			ihSize := outputSize * inputSize
			hhSize := outputSize * outputSize
			bSize := outputSize
			baselineWeights = depth * 4 * (ihSize + hhSize + bSize)
		} else if lType == poly.LayerMultiHeadAttention {
			dModel := 128
			kvDim := 128 // assuming numHeads == numKVHeads
			baselineWeights = depth * (2*(dModel*dModel+dModel) + 2*(dModel*kvDim+kvDim))
		} else if lType == poly.LayerRNN {
			baselineWeights = depth * (128*128 + 128*128 + 128)
		} else if lType == poly.LayerRMSNorm {
			baselineWeights = depth * 1024
			baselineWeights = depth * 2048
		} else if lType == poly.LayerResidual {
			baselineWeights = 0
		} else if lType == poly.LayerSequential {
			baselineWeights = depth * (256*256 + 256*256)
		} else if lType == poly.LayerSwiGLU {
			baselineWeights = depth * (128*256*3 + 256*2 + 128)
		}
		baselineSize := float64(baselineWeights*4) / 1024.0
		res.PctSaved = (1.0 - (res.SizeKb / baselineSize)) * 100.0
		resList = append(resList, res)
	}

	// Hybrid / Mixed
	mixedDepth := len(allTypes)
	if lType == poly.LayerEmbedding || lType == poly.LayerLSTM || lType == poly.LayerMultiHeadAttention || lType == poly.LayerRMSNorm || lType == poly.LayerLayerNorm || lType == poly.LayerResidual || lType == poly.LayerRNN || lType == poly.LayerSequential || lType == poly.LayerSwiGLU {
		mixedDepth = 1
	}
	net := poly.NewVolumetricNetwork(1, 1, 1, mixedDepth)
	for i := 0; i < mixedDepth; i++ {
		l := net.GetLayer(0, 0, 0, i)
		l.Type = lType
		l.DType = allTypes[i].Type
		l.Activation = poly.ActivationReLU

		if lType == poly.LayerCNN3 {
			l.InputChannels = 8
			l.InputDepth = 4
			l.InputHeight = 4
			l.InputWidth = 4
			l.Filters = 8
			l.KernelSize = 3
			l.Stride = 1
			l.Padding = 1
			l.OutputDepth = 4
			l.OutputHeight = 4
			l.OutputWidth = 4
			l.WeightStore = poly.NewWeightStore(l.Filters * l.InputChannels * l.KernelSize * l.KernelSize * l.KernelSize)
		} else if lType == poly.LayerCNN2 {
			l.InputChannels = 8
			l.InputHeight = 8
			l.InputWidth = 8
			l.Filters = 8
			l.KernelSize = 3
			l.Stride = 1
			l.Padding = 1
			l.OutputHeight = 8
			l.OutputWidth = 8
			l.WeightStore = poly.NewWeightStore(l.Filters * l.InputChannels * l.KernelSize * l.KernelSize)
		} else if lType == poly.LayerCNN1 {
			l.InputChannels = 8
			l.InputHeight = 64
			l.Filters = 8
			l.KernelSize = 3
			l.Stride = 1
			l.Padding = 1
			l.OutputHeight = 64
			l.WeightStore = poly.NewWeightStore(l.Filters * l.InputChannels * l.KernelSize)
		} else if lType == poly.LayerEmbedding {
			l.VocabSize = 1024
			l.EmbeddingDim = 256
			l.WeightStore = poly.NewWeightStore(l.VocabSize * l.EmbeddingDim)
		} else if lType == poly.LayerLSTM {
			l.InputHeight = 64
			l.OutputHeight = 64
			l.SeqLength = 8
			ihSize := l.OutputHeight * l.InputHeight
			hhSize := l.OutputHeight * l.OutputHeight
			bSize := l.OutputHeight
			l.WeightStore = poly.NewWeightStore(4 * (ihSize + hhSize + bSize))
		} else if lType == poly.LayerMultiHeadAttention {
			l.DModel = 128
			l.NumHeads = 4
			l.NumKVHeads = 4
			l.HeadDim = 32
			l.MaxSeqLen = 512
			kvDim := l.NumKVHeads * l.HeadDim
			l.WeightStore = poly.NewWeightStore(2*(l.DModel*l.DModel+l.DModel) + 2*(l.DModel*kvDim+kvDim))
		} else if lType == poly.LayerRNN {
			l.InputHeight = 128
			l.OutputHeight = 128
			l.SeqLength = 16
			ihSize := l.OutputHeight * l.InputHeight
			hhSize := l.OutputHeight * l.OutputHeight
			bSize := l.OutputHeight
			l.WeightStore = poly.NewWeightStore(ihSize + hhSize + bSize)
		} else if lType == poly.LayerRMSNorm {
			l.OutputHeight = 1024
			l.WeightStore = poly.NewWeightStore(l.OutputHeight)
		} else if lType == poly.LayerLayerNorm {
			l.OutputHeight = 1024
			l.WeightStore = poly.NewWeightStore(2 * l.OutputHeight)
		} else if lType == poly.LayerResidual {
			l.OutputHeight = 1024
			l.WeightStore = poly.NewWeightStore(0)
		} else if lType == poly.LayerSequential {
			l.SequentialLayers = []poly.VolumetricLayer{
				{Type: poly.LayerDense, InputHeight: 256, OutputHeight: 256, WeightStore: poly.NewWeightStore(256 * 256)},
				{Type: poly.LayerDense, InputHeight: 256, OutputHeight: 256, WeightStore: poly.NewWeightStore(256 * 256)},
			}
			l.WeightStore = poly.NewWeightStore(0)
		} else if lType == poly.LayerSwiGLU {
			l.InputHeight = 128
			l.OutputHeight = 256
			ihSize := 128 * 256
			l.WeightStore = poly.NewWeightStore(3*ihSize + 2*256 + 128)
		} else {
			l.InputHeight = inputSize
			l.OutputHeight = outputSize
			l.WeightStore = poly.NewWeightStore(inputSize * outputSize)
		}
		l.WeightStore.Scale = 0.01
	}
	res := runBenchmark("Exhaustive Multi-Type", net, batchSize, lType, inputSize, outputSize, iterations)
	baselineWeights := mixedDepth * inputSize * outputSize
	if lType == poly.LayerLSTM {
		ihSize := outputSize * inputSize
		hhSize := outputSize * outputSize
		bSize := outputSize
		baselineWeights = mixedDepth * 4 * (ihSize + hhSize + bSize)
	} else if lType == poly.LayerMultiHeadAttention {
		dModel := 128
		kvDim := 128
		baselineWeights = mixedDepth * (2*(dModel*dModel+dModel) + 2*(dModel*kvDim+kvDim))
	} else if lType == poly.LayerRMSNorm {
		baselineWeights = mixedDepth * 1024
	} else if lType == poly.LayerLayerNorm {
		baselineWeights = mixedDepth * 2048
	} else if lType == poly.LayerRNN {
		baselineWeights = mixedDepth * (128*128 + 128*128 + 128)
	} else if lType == poly.LayerResidual {
		baselineWeights = 0
	} else if lType == poly.LayerSequential {
		baselineWeights = mixedDepth * (256*256 + 256*256)
	} else if lType == poly.LayerSwiGLU {
		baselineWeights = mixedDepth * (128*256*3 + 256*2 + 128)
	}
	currentBaseline := (float64(baselineWeights) * 4) / 1024.0
	res.PctSaved = (1.0 - (res.SizeKb / currentBaseline)) * 100.0
	resList = append(resList, res)

	fmt.Println("| Scenario              | Forward (avg) | Forward (Tile)| Training (total) | Size (KB) | % Saved | Sim Tax |")
	fmt.Println("|-----------------------|---------------|---------------|------------------|-----------|---------|---------|")
	for _, res := range resList {
		fmt.Printf("| %-21s | %-13v | %-13v | %-16v | %-9.1f | %-7.1f%% | %-7.1f%% |\n",
			res.Label, res.Forward, res.ForwardTile, res.Total, res.SizeKb, res.PctSaved, res.SimTax)
	}
}

func runBenchmark(label string, net *poly.VolumetricNetwork, batch int, lType poly.LayerType, in, out, iter int) Result {
	var input *poly.Tensor[float32]
	var gradOut *poly.Tensor[float32]

	if lType == poly.LayerCNN3 {
		input = poly.NewTensor[float32](batch, 8, 4, 4, 4)
		gradOut = poly.NewTensor[float32](batch, 8, 4, 4, 4)
	} else if lType == poly.LayerCNN2 {
		input = poly.NewTensor[float32](batch, 8, 8, 8)
		gradOut = poly.NewTensor[float32](batch, 8, 8, 8)
	} else if lType == poly.LayerCNN1 {
		input = poly.NewTensor[float32](batch, 8, 64)
		gradOut = poly.NewTensor[float32](batch, 8, 64)
	} else if lType == poly.LayerEmbedding {
		input = poly.NewTensor[float32](batch, 64) // seqLen = 64
		// Fill input with random token IDs
		for i := 0; i < len(input.Data); i++ {
			input.Data[i] = float32(i % 1024)
		}
		gradOut = poly.NewTensor[float32](batch, 64, 256)
	} else if lType == poly.LayerLSTM || lType == poly.LayerRNN {
		input = poly.NewTensor[float32](batch, 16, 128) // seqLen=16, inputSize=128
		gradOut = poly.NewTensor[float32](batch, 16, 256) // seqLen=16, hiddenSize=256
		if lType == poly.LayerRNN {
			gradOut = poly.NewTensor[float32](batch, 16, 128)
		}
	} else if lType == poly.LayerMultiHeadAttention {
		input = poly.NewTensor[float32](batch, 16, 128) // seqLen=16, dModel=128
		gradOut = poly.NewTensor[float32](batch, 16, 128)
	} else if lType == poly.LayerRMSNorm || lType == poly.LayerLayerNorm || lType == poly.LayerResidual {
		input = poly.NewTensor[float32](batch, 1024)
		gradOut = poly.NewTensor[float32](batch, 1024)
	} else if lType == poly.LayerSwiGLU {
		input = poly.NewTensor[float32](batch, 128)
		gradOut = poly.NewTensor[float32](batch, 128)
	} else {
		input = poly.NewTensor[float32](batch, in)
		gradOut = poly.NewTensor[float32](batch, out)
	}

	sizeBytes := net.CalculateTotalMemory()
	sizeKb := float64(sizeBytes) / 1024.0

	depth := len(net.Layers)
	// 1. Standard Forward
	var fwdAvg, fwdTiledAvg, bwdAvg time.Duration

	// Standard Forward
	var preActs, postActs []*poly.Tensor[float32]
	start := time.Now()
	for i := 0; i < iter; i++ {
		currentInput := input
		for d := 0; d < depth; d++ {
			layer := &net.Layers[d]
			layer.UseTiling = false
			var pre, post *poly.Tensor[float32]
			if lType == poly.LayerResidual {
				// For benchmark purposes, use input as skip
				pre, post = poly.ResidualForwardPolymorphic(layer, currentInput, currentInput)
			} else {
				pre, post = poly.DispatchLayer(layer, currentInput, nil)
			}
			if i == 0 {
				preActs = append(preActs, pre)
				postActs = append(postActs, post)
			}
			currentInput = post
		}
	}
	fwdAvg = time.Since(start) / time.Duration(iter)

	// 2. Tiled Forward
	start = time.Now()
	for i := 0; i < iter; i++ {
		currentInput := input
		for d := 0; d < depth; d++ {
			layer := &net.Layers[d]
			layer.UseTiling = true
			layer.TileSize = 1024
			var post *poly.Tensor[float32]
			if lType == poly.LayerResidual {
				_, post = poly.ResidualForwardPolymorphic(layer, currentInput, currentInput)
			} else {
				_, post = poly.DispatchLayer(layer, currentInput, nil)
			}
			currentInput = post
		}
	}
	fwdTiledAvg = time.Since(start) / time.Duration(iter)

	// 3. Backward
	start = time.Now()
	for i := 0; i < iter; i++ {
		currentGrad := gradOut
		for d := depth - 1; d >= 0; d-- {
			layer := &net.Layers[d]
			var in *poly.Tensor[float32]
			if d == 0 {
				in = input
			} else {
				in = postActs[d-1]
			}
			var gIn *poly.Tensor[float32]
			if lType == poly.LayerResidual {
				gIn, _ = poly.ResidualBackwardPolymorphic(layer, currentGrad, in, preActs[d])
			} else if lType == poly.LayerSequential {
				gIn, _ = poly.SequentialBackwardPolymorphic(layer, currentGrad, in, preActs[d])
			} else if lType == poly.LayerSwiGLU {
				gIn, _ = poly.SwiGLUBackwardPolymorphic(layer, currentGrad, in, preActs[d])
			} else {
				gIn, _ = poly.DispatchLayerBackward(layer, currentGrad, in, nil, preActs[d])
			}
			currentGrad = gIn
		}
	}
	bwdAvg = time.Since(start) / time.Duration(iter)

	return Result{
		Label:       label,
		Forward:     fwdAvg,
		ForwardTile: fwdTiledAvg,
		Backward:    bwdAvg,
		Total:       fwdAvg + bwdAvg,
		SizeKb:      sizeKb,
		SimTax:      0.0,
	}
}
