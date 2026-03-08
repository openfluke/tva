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
	Backward  time.Duration
	Total     time.Duration
	SizeKb    float64
	PctSaved  float64
	SimTax    float64 
}

func main() {
	fmt.Println("=== M-POLY-VTD Truly Exhaustive Normalization Benchmark ===")
	
	batchSize := 32
	normSize := 512
	iterations := 50
	
	allTypes := []struct {
		Label string
		Type  poly.DType
	}{
		// 64-bit
		{"Pure FLOAT64", poly.DTypeFloat64},
		{"Pure INT64", poly.DTypeInt64},
		{"Pure UINT64", poly.DTypeUint64},
		// 32-bit
		{"Pure FLOAT32", poly.DTypeFloat32},
		{"Pure INT32", poly.DTypeInt32},
		{"Pure UINT32", poly.DTypeUint32},
		// 16-bit
		{"Pure FLOAT16", poly.DTypeFloat16},
		{"Pure BFLOAT16", poly.DTypeBFloat16},
		{"Pure INT16", poly.DTypeInt16},
		{"Pure UINT16", poly.DTypeUint16},
		// 8-bit
		{"Pure FP8 (E4M3)", poly.DTypeFP8E4M3},
		{"Pure FP8 (E5M2)", poly.DTypeFP8E5M2},
		{"Pure INT8", poly.DTypeInt8},
		{"Pure UINT8", poly.DTypeUint8},
		// 4-bit
		{"Pure INT4", poly.DTypeInt4},
		{"Pure UINT4", poly.DTypeUint4},
		{"Pure FP4", poly.DTypeFP4},
		// 2-bit
		{"Pure INT2", poly.DTypeInt2},
		{"Pure UINT2", poly.DTypeUint2},
		{"Pure TERNARY", poly.DTypeTernary},
		// 1-bit
		{"Pure BINARY", poly.DTypeBinary},
	}

	fmt.Printf("\n| Scenario              | Forward (avg) | Training (total) | Size (KB) | %% Saved | Sim Tax |\n")
	fmt.Printf("|-----------------------|---------------|------------------|-----------|---------|---------|\n")

	baselineSize := float64(normSize * 2 * 4) / 1024.0 // FP32 Gamma + Beta (matches dense benchmark base)

	results := make(chan Result, len(allTypes))
	var wg sync.WaitGroup

	for _, s := range allTypes {
		wg.Add(1)
		go func(label string, dtype poly.DType) {
			defer wg.Done()
			
			layer := &poly.VolumetricLayer{
				Type:         poly.LayerLayerNorm,
				DType:        dtype,
				OutputHeight: normSize,
			}
			layer.WeightStore = poly.NewWeightStore(normSize * 2)
			layer.WeightStore.Scale = 0.1
			
			input := poly.NewTensor[float32](batchSize, normSize)
			for i := range input.Data { input.Data[i] = rand.Float32() }
			
			gradOutput := poly.NewTensor[float32](batchSize, normSize)
			for i := range gradOutput.Data { gradOutput.Data[i] = rand.Float32() }

			// Warmup
			poly.DispatchLayer(layer, input, nil)
			
			fStart := time.Now()
			for i := 0; i < iterations; i++ {
				poly.DispatchLayer(layer, input, nil)
			}
			avgForward := time.Since(fStart) / time.Duration(iterations)

			trainStart := time.Now()
			pre, _ := poly.DispatchLayer(layer, input, nil)
			poly.DispatchLayerBackward(layer, gradOutput, input, nil, pre)
			trainTotal := time.Since(trainStart)

			currSizeKB := float64(layer.WeightStore.SizeInBytes(dtype)) / 1024.0
			percentSaved := (1.0 - (currSizeKB / baselineSize)) * 100.0
			
			// Approximation of simulation tax: if training takes 2x forward time usually, anything above is tax
			expectedTrain := avgForward * 3
			simTax := 0.0
			if trainTotal > expectedTrain {
				simTax = float64(trainTotal-expectedTrain) / float64(trainTotal) * 100.0
			}

			results <- Result{
				Label:    label,
				Forward:  avgForward,
				Backward: trainTotal,
				Total:    trainTotal,
				SizeKb:   currSizeKB,
				PctSaved: percentSaved,
				SimTax:   simTax,
			}
		}(s.Label, s.Type)
	}

	wg.Wait()
	close(results)

	// Collect and print preserving original order using slice manipulation
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
		if len(r.Label) > 0 { // Check if valid
			fmt.Printf("| %-21s | %-13v | %-16v | %-9.1f | %-7.1f %% | %-7.1f %% |\n",
				r.Label, r.Forward, r.Total, r.SizeKb, r.PctSaved, r.SimTax)
		}
	}
}
