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
	Total     time.Duration
	SizeKb    float64
	PctSaved  float64
	SimTax    float64 
}

func main() {
	fmt.Println("=== M-POLY-VTD Truly Exhaustive MHA Benchmark ===")
	
	seqLen := 32
	dModel := 128
	numHeads := 4
	headDim := 32 // dModel / numHeads
	iterations := 5
	
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

	fmt.Printf("\n| Scenario              | Forward (avg) | Total Time | Size (KB) | %% Saved | Sim Tax |\n")
	fmt.Printf("|-----------------------|---------------|------------|-----------|---------|---------|\n")

	// Baseline size for FP32: (QW + KW + VW + OW + Biases)
	// (dModel * dModel) * 4 + (dModel * 4)
	totalParams := (dModel * dModel) * 4 + (dModel * 4)
	baselineSize := float64(totalParams * 4) / 1024.0

	results := make(chan Result, len(allTypes))
	var wg sync.WaitGroup

	for _, s := range allTypes {
		wg.Add(1)
		go func(label string, dtype poly.DType) {
			defer wg.Done()
			
			layer := &poly.VolumetricLayer{
				Type:         poly.LayerMultiHeadAttention,
				DType:        dtype,
				DModel:       dModel,
				NumHeads:     numHeads,
				NumKVHeads:   numHeads,
				HeadDim:      headDim,
				SeqLength:    seqLen,
				RoPEFreqBase: 10000,
			}
			layer.WeightStore = poly.NewWeightStore(totalParams)
			layer.WeightStore.Scale = 0.01
			
			input := poly.NewTensor[float32](seqLen, dModel)
			for i := range input.Data { input.Data[i] = rand.Float32() }
			
			// Warmup
			poly.DispatchLayer(layer, input)
			
			fStart := time.Now()
			for i := 0; i < iterations; i++ {
				poly.DispatchLayer(layer, input)
			}
			avgForward := time.Since(fStart) / time.Duration(iterations)

			totalStart := time.Now()
			poly.DispatchLayer(layer, input)
			trainTotal := time.Since(totalStart)

			currSizeKB := float64(layer.WeightStore.SizeInBytes(dtype)) / 1024.0
			percentSaved := (1.0 - (currSizeKB / baselineSize)) * 100.0
			
			expected := avgForward
			simTax := 0.0
			if trainTotal > expected {
				simTax = float64(trainTotal-expected) / float64(trainTotal) * 100.0
			}

			results <- Result{
				Label:    label,
				Forward:  avgForward,
				Total:    trainTotal,
				SizeKb:   currSizeKB,
				PctSaved: percentSaved,
				SimTax:   simTax,
			}
		}(s.Label, s.Type)
	}

	wg.Wait()
	close(results)

	orderedResults := make([]Result, len(allTypes))
	for r := range results {
		for i, v := range allTypes {
			if v.Label == r.Label {
				orderedResults[i] = r
				break
			}
		}
	}

	for _, r := range orderedResults {
		if len(r.Label) > 0 {
			fmt.Printf("| %-21s | %-13v | %-10v | %-9.1f | %-7.1f %% | %-7.1f %% |\n",
				r.Label, r.Forward, r.Total, r.SizeKb, r.PctSaved, r.SimTax)
		}
	}
}
