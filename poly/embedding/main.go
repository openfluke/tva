package main

import (
	"fmt"
	"math/rand"
	"time"
	"github.com/openfluke/loom/poly"
)

type Result struct {
	Label     string
	Forward   time.Duration
	SizeMb    float64
	PctSaved  float64
}

func main() {
	fmt.Println("=== M-POLY-VTD Truly Exhaustive Embedding Benchmark ===")
	
	// Reduced dimensions for memory safety during dev
	vocabSize := 10000 
	embeddingDim := 1024
	seqLen := 1
	iterations := 10
	
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

	fmt.Printf("\n| Scenario              | Forward (avg) | Size (MB) | %% Saved |\n")
	fmt.Printf("|-----------------------|---------------|-----------|---------|\n")

	totalParams := vocabSize * embeddingDim
	baselineSize := float64(totalParams * 4) / (1024.0 * 1024.0)

	results := []Result{}

	// Run SEQUENTIALLY to save RAM (no 30GB spikes!)
	for _, s := range allTypes {
		layer := &poly.VolumetricLayer{
			Type:         poly.LayerEmbedding,
			DType:        s.Type,
			VocabSize:    vocabSize,
			EmbeddingDim: embeddingDim,
		}
		layer.WeightStore = poly.NewWeightStore(totalParams)
		
		input := poly.NewTensor[float32](seqLen)
		for i := range input.Data { input.Data[i] = float32(rand.Intn(vocabSize)) }
		
		// Warmup
		poly.DispatchLayer(layer, input)
		
		fStart := time.Now()
		for i := 0; i < iterations; i++ {
			poly.DispatchLayer(layer, input)
		}
		avgForward := time.Since(fStart) / time.Duration(iterations)

		currSizeMB := float64(layer.WeightStore.SizeInBytes(s.Type)) / (1024.0 * 1024.0)
		percentSaved := (1.0 - (currSizeMB / baselineSize)) * 100.0
		
		results = append(results, Result{
			Label:    s.Label,
			Forward:  avgForward,
			SizeMb:   currSizeMB,
			PctSaved: percentSaved,
		})
		
		fmt.Printf("| %-21s | %-13v | %-9.1f | %-7.1f %% |\n",
			s.Label, avgForward, currSizeMB, percentSaved)
		
		// Signal GC to keep memory tight
		layer.WeightStore = nil
	}
}
