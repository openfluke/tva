package main

import (
	"fmt"
	"math/rand"
	"time"
	"sync"
	"github.com/openfluke/loom/poly"
)

type Result struct {
	Label    string
	Forward  time.Duration
	Backward time.Duration
	Total    time.Duration
	SizeKb   float64
}

func main() {
	fmt.Println("=== M-POLY-VTD Residual Layer Benchmark ===")
	
	batchSize := 32
	normSize := 512
	iterations := 100
	
	allTypes := []struct {
		Label string
		Type  poly.DType
	}{
		{"FLOAT64", poly.DTypeFloat64},
		{"FLOAT32", poly.DTypeFloat32},
		{"INT32", poly.DTypeInt32},
		{"INT16", poly.DTypeInt16},
		{"INT8", poly.DTypeInt8},
	}

	fmt.Printf("\n| Scenario              | Forward (avg) | Training (total) | Size (KB) |\n")
	fmt.Printf("|-----------------------|---------------|------------------|-----------|\n")

	results := make(chan Result, len(allTypes))
	var wg sync.WaitGroup

	for _, s := range allTypes {
		wg.Add(1)
		go func(label string, dtype poly.DType) {
			defer wg.Done()
			
			// Sequential block with Residual
			// Dense (identity) -> Residual
			layer := &poly.VolumetricLayer{
				Type: poly.LayerSequential,
				DType: dtype,
				SequentialLayers: []poly.VolumetricLayer{
					{
						Type: poly.LayerDense,
						InputHeight: normSize,
						OutputHeight: normSize,
						WeightStore: poly.NewWeightStore(normSize*normSize + normSize),
						DType: dtype,
					},
					{
						Type: poly.LayerResidual,
						DType: dtype,
					},
				},
			}
			// Initialize dense weights to identity (approx)
			w := layer.SequentialLayers[0].WeightStore.Master
			for i := 0; i < normSize; i++ {
				w[i*normSize+i] = 1.0
			}

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

			results <- Result{
				Label:    label,
				Forward:  avgForward,
				Backward: trainTotal,
				Total:    trainTotal,
				SizeKb:   float64(layer.SequentialLayers[0].WeightStore.SizeInBytes(dtype)) / 1024.0,
			}
		}(s.Label, s.Type)
	}

	wg.Wait()
	close(results)

	// Collect and print preserving original order
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
			fmt.Printf("| %-21s | %-13v | %-16v | %-9.1f |\n",
				r.Label, r.Forward, r.Total, r.SizeKb)
		}
	}
	
	// Final functional validation check
	checkResidualFunctional()
}

func checkResidualFunctional() {
	fmt.Println("\nFunctional Validation:")
	layer := &poly.VolumetricLayer{Type: poly.LayerResidual}
	input := poly.NewTensor[float32](1, 4)
	input.Data = []float32{1, 2, 3, 4}
	skip := poly.NewTensor[float32](1, 4)
	skip.Data = []float32{10, 20, 30, 40}
	
	_, output := poly.DispatchLayer(layer, input, skip)
	fmt.Printf("Input: %v, Skip: %v, Output: %v\n", input.Data, skip.Data, output.Data)
}
