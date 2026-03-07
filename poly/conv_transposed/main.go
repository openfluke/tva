package main

import (
	"fmt"
	"math/rand"
	"time"
	"sync"
	"github.com/openfluke/loom/poly"
)

type Result struct {
	Label     string
	Forward   time.Duration
	SizeKb    float64
	PctSaved  float64
}

func main() {
	fmt.Println("=== M-POLY-VTD Modular ConvTransposed Benchmark ===")
	
	allTypes := []struct {
		Label string
		Type  poly.DType
	}{
		{"Pure FLOAT64", poly.DTypeFloat64},
		{"Pure INT64", poly.DTypeInt64},
		{"Pure UINT64", poly.DTypeUint64},
		{"Pure FLOAT32", poly.DTypeFloat32},
		{"Pure INT32", poly.DTypeInt32},
		{"Pure UINT32", poly.DTypeUint32},
		{"Pure FLOAT16", poly.DTypeFloat16},
		{"Pure BFLOAT16", poly.DTypeBFloat16},
		{"Pure INT16", poly.DTypeInt16},
		{"Pure UINT16", poly.DTypeUint16},
		{"Pure INT8", poly.DTypeInt8},
		{"Pure UINT8", poly.DTypeUint8},
		{"Pure INT4", poly.DTypeInt4},
		{"Pure UINT4", poly.DTypeUint4},
		{"Pure FP4", poly.DTypeFP4},
		{"Pure INT2", poly.DTypeInt2},
		{"Pure UINT2", poly.DTypeUint2},
		{"Pure TERNARY", poly.DTypeTernary},
		{"Pure BINARY", poly.DTypeBinary},
	}

	runBenchmark("1D Transposed Conv (Upsampling)", poly.LayerConvTransposed1D, allTypes)
	runBenchmark("2D Transposed Conv (Upsampling)", poly.LayerConvTransposed2D, allTypes)
	runBenchmark("3D Transposed Conv (Upsampling)", poly.LayerConvTransposed3D, allTypes)
}

func runBenchmark(title string, lType poly.LayerType, types []struct{Label string; Type poly.DType}) {
	fmt.Printf("\n--- %s ---\n", title)
	fmt.Printf("| Scenario              | Forward (avg) | Size (KB) | %% Saved |\n")
	fmt.Printf("|-----------------------|---------------|-----------|---------|\n")

	var inH, inW, inD, inC int
	var outH, outW, outD int
	var totalParams int
	
	filters := 16
	kSize := 3
	stride := 2
	padding := 1
	inC = 8

	switch lType {
	case poly.LayerConvTransposed1D:
		inW = 64
		outW = (inW-1)*stride - 2*padding + kSize + 1
		totalParams = filters * inC * kSize
	case poly.LayerConvTransposed2D:
		inH, inW = 16, 16
		outH = (inH-1)*stride - 2*padding + kSize + 1
		outW = (inW-1)*stride - 2*padding + kSize + 1
		totalParams = filters * inC * kSize * kSize
	case poly.LayerConvTransposed3D:
		inD, inH, inW = 8, 8, 8
		outD = (inD-1)*stride - 2*padding + kSize + 1
		outH = (inH-1)*stride - 2*padding + kSize + 1
		outW = (inW-1)*stride - 2*padding + kSize + 1
		totalParams = filters * inC * kSize * kSize * kSize
	}

	baselineSize := float64(totalParams * 4) / 1024.0

	results := make(chan Result, len(types))
	var wg sync.WaitGroup

	for _, s := range types {
		wg.Add(1)
		go func(label string, dtype poly.DType) {
			defer wg.Done()
			
			layer := &poly.VolumetricLayer{
				Type:          lType,
				DType:         dtype,
				InputDepth:    inD,
				InputHeight:   inH,
				InputWidth:    inW,
				InputChannels: inC,
				OutputDepth:   outD,
				OutputHeight:  outH,
				OutputWidth:   outW,
				Filters:       filters,
				KernelSize:    kSize,
				Stride:        stride,
				Padding:       padding,
				OutputPadding: 1,
			}
			layer.WeightStore = poly.NewWeightStore(totalParams)
			
			var input *poly.Tensor[float32]
			switch lType {
			case poly.LayerConvTransposed1D:
				input = poly.NewTensor[float32](1, inC, inW)
			case poly.LayerConvTransposed2D:
				input = poly.NewTensor[float32](1, inC, inH, inW)
			case poly.LayerConvTransposed3D:
				input = poly.NewTensor[float32](1, inC, inD, inH, inW)
			}
			for i := range input.Data { input.Data[i] = rand.Float32() }
			
			poly.DispatchLayer(layer, input)
			
			start := time.Now()
			for i := 0; i < 3; i++ {
				poly.DispatchLayer(layer, input)
			}
			avgForward := time.Since(start) / 3

			currSizeKB := float64(layer.WeightStore.SizeInBytes(dtype)) / 1024.0
			percentSaved := (1.0 - (currSizeKB / baselineSize)) * 100.0

			results <- Result{
				Label:    label,
				Forward:  avgForward,
				SizeKb:   currSizeKB,
				PctSaved: percentSaved,
			}
		}(s.Label, s.Type)
	}

	wg.Wait()
	close(results)

	orderedResults := make([]Result, len(types))
	for r := range results {
		for i, v := range types {
			if v.Label == r.Label {
				orderedResults[i] = r
				break
			}
		}
	}

	for _, r := range orderedResults {
		if len(r.Label) > 0 {
			fmt.Printf("| %-21s | %-13v | %-9.1f | %-7.1f %% |\n",
				r.Label, r.Forward, r.SizeKb, r.PctSaved)
		}
	}
}
