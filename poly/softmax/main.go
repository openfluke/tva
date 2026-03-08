package main

import (
	"fmt"
	"math/rand"
	"time"
	"github.com/openfluke/loom/poly"
)

func main() {
	fmt.Println("=== M-POLY-VTD Truly Exhaustive Softmax Cross-Matrix Benchmark ===")
	
	size := 128
	iterations := 20
	
	allTypes := []struct {
		Label string
		Type  poly.DType
	}{
		{"FLOAT64", poly.DTypeFloat64},
		{"INT64", poly.DTypeInt64},
		{"UINT64", poly.DTypeUint64},
		{"FLOAT32", poly.DTypeFloat32},
		{"INT32", poly.DTypeInt32},
		{"UINT32", poly.DTypeUint32},
		{"FLOAT16", poly.DTypeFloat16},
		{"BFLOAT16", poly.DTypeBFloat16},
		{"INT16", poly.DTypeInt16},
		{"UINT16", poly.DTypeUint16},
		{"INT8", poly.DTypeInt8},
		{"UINT8", poly.DTypeUint8},
		{"INT4", poly.DTypeInt4},
		{"UINT4", poly.DTypeUint4},
		{"FP4", poly.DTypeFP4},
		{"INT2", poly.DTypeInt2},
		{"UINT2", poly.DTypeUint2},
		{"TERNARY", poly.DTypeTernary},
		{"BINARY", poly.DTypeBinary},
	}

	variants := []struct {
		Label string
		Type  poly.SoftmaxType
		Setup func(*poly.VolumetricLayer)
	}{
		{"Standard", poly.SoftmaxStandard, nil},
		{"Grid", poly.SoftmaxGrid, func(l *poly.VolumetricLayer) {
			l.SoftmaxRows = 8
			l.SoftmaxCols = 16
		}},
		{"Masked", poly.SoftmaxMasked, func(l *poly.VolumetricLayer) {
			l.Mask = make([]bool, size)
			for i := range l.Mask { l.Mask[i] = i%4 != 0 } // 75% active
		}},
		{"Gumbel", poly.SoftmaxGumbel, nil},
		{"Sparse", poly.SoftmaxSparse, nil},
		{"Entmax", poly.SoftmaxEntmax, func(l *poly.VolumetricLayer) {
			l.EntmaxAlpha = 1.5
		}},
		{"Hierarchical", poly.SoftmaxHierarchical, func(l *poly.VolumetricLayer) {
			l.HierarchyLevels = []int{8, 16}
		}},
	}

	fmt.Printf("\n| DType     | " )
	for _, v := range variants {
		fmt.Printf("%-11s | ", v.Label)
	}
	fmt.Printf("\n|-----------|" )
	for range variants {
		fmt.Printf("------------|" )
	}
	fmt.Printf("\n" )

	for _, t := range allTypes {
		fmt.Printf("| %-9s | ", t.Label)
		for _, v := range variants {
			layer := &poly.VolumetricLayer{
				Type:        poly.LayerSoftmax,
				DType:       t.Type,
				Temperature: 1.0,
				SoftmaxType: v.Type,
			}
			if v.Setup != nil { v.Setup(layer) }
			
			input := poly.NewTensor[float32](size)
			for i := range input.Data { input.Data[i] = rand.Float32() * 4 - 2 }
			
			// Measure
			fStart := time.Now()
			for i := 0; i < iterations; i++ {
				poly.DispatchLayer(layer, input, nil)
			}
			avgForward := time.Since(fStart) / time.Duration(iterations)
			
			// Concise output
			us := avgForward.Microseconds()
			if us < 1 {
				fmt.Printf("%-11s | ", "<1µs")
			} else {
				fmt.Printf("%-7dµs  | ", us)
			}
		}
		fmt.Printf("\n")
	}
}
