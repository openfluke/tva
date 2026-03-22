package main

import (
	"fmt"
	"time"

	"github.com/openfluke/loom/poly"
)

func main() {
	fmt.Println("=== CNN3 Multi-Core Tiling Experiment ===")
	
	iterations := 5
	
	// Setup Layer
	l := poly.VolumetricLayer{
		Network:       poly.NewVolumetricNetwork(1, 1, 1, 1),
		Type:          poly.LayerCNN3,
		InputChannels: 32,
		InputDepth:    32,
		InputHeight:   32,
		InputWidth:    32,
		Filters:       32,
		OutputDepth:   32,
		OutputHeight:  32,
		OutputWidth:   32,
		KernelSize:    3,
		Stride:        1,
		Padding:       1,
		DType:         poly.DTypeFloat32,
		WeightStore:   poly.NewWeightStore(32 * 32 * 3 * 3 * 3),
		UseTiling:     true,
		TileSize:      0, // Auto-detect
	}

	for i := range l.WeightStore.Master {
		l.WeightStore.Master[i] = 0.1
	}

	input := poly.NewTensor[float32](1, l.InputChannels, l.InputDepth, l.InputHeight, l.InputWidth)
	for i := range input.Data {
		input.Data[i] = 0.5
	}

	// 0. Normal (Non-Tiled)
	l.UseTiling = false
	fmt.Print("Running Normal (Non-Tiled)...  ")
	start := time.Now()
	var post0 *poly.Tensor[float32]
	for i := 0; i < iterations; i++ {
		_, post0 = poly.CNN3ForwardPolymorphic(&l, input)
	}
	tNormal := time.Since(start) / time.Duration(iterations)
	fmt.Printf("%v\n", tNormal)

	// 1. Single-Core Tiled
	l.UseTiling = true
	l.TileSize = 0 // reset to allow auto-detect
	l.Network.EnableMultiCoreTiling = false
	l.SyncToCPU()
	fmt.Printf("Running Single-Core Tiled (TileSize: %d)...    ", l.TileSize)
	start = time.Now()
	var post1 *poly.Tensor[float32]
	for i := 0; i < iterations; i++ {
		_, post1 = poly.CNN3ForwardPolymorphic(&l, input)
	}
	tSingle := time.Since(start) / time.Duration(iterations)
	fmt.Printf("%v\n", tSingle)

	// 2. Multi-Core Tiled
	l.UseTiling = true
	l.TileSize = 0 // reset to allow auto-detect
	l.Network.EnableMultiCoreTiling = true
	l.SyncToCPU()
	fmt.Printf("Running Multi-Core Tiled (TileSize: %d)... ", l.TileSize)
	start = time.Now()
	var post2 *poly.Tensor[float32]
	for i := 0; i < iterations; i++ {
		_, post2 = poly.CNN3ForwardPolymorphic(&l, input)
	}
	tMulti := time.Since(start) / time.Duration(iterations)
	fmt.Printf("%v\n", tMulti)

	// Parity Check
	maxDiff01 := 0.0
	maxDiff02 := 0.0
	for i := range post0.Data {
		d01 := float64(post0.Data[i] - post1.Data[i])
		if d01 < 0 { d01 = -d01 }
		if d01 > maxDiff01 { maxDiff01 = d01 }

		d02 := float64(post0.Data[i] - post2.Data[i])
		if d02 < 0 { d02 = -d02 }
		if d02 > maxDiff02 { maxDiff02 = d02 }
	}

	fmt.Println("\n=== Performance Results ===")
	fmt.Printf("| %-20s | %-12s | %-12s | %-12s |\n", "Implementation", "Time", "Speedup (vs Normal)", "Parity (vs Normal)")
	fmt.Println("|----------------------|--------------|-------------------|-------------------|")
	fmt.Printf("| %-20s | %-12v | %-17s | %-17v |\n", "Normal (Native)", tNormal, "1.00x", "BASE")
	fmt.Printf("| %-20s | %-12v | %-17.2fx | %-17e |\n", "Single-Core Tiled", tSingle, float64(tNormal)/float64(tSingle), maxDiff01)
	fmt.Printf("| %-20s | %-12v | %-17.2fx | %-17e |\n", "Multi-Core Tiled", tMulti, float64(tNormal)/float64(tMulti), maxDiff02)

	fmt.Println("\n=== Numerical Sample (First 3 elements) ===")
	fmt.Printf("| %-20s | %-40s |\n", "Implementation", "Sample Values")
	fmt.Println("|----------------------|------------------------------------------|")
	fmt.Printf("| %-20s | %.6f, %.6f, %.6f |\n", "Normal (Native)", post0.Data[0], post0.Data[1], post0.Data[2])
	fmt.Printf("| %-20s | %.6f, %.6f, %.6f |\n", "Single-Core Tiled", post1.Data[0], post1.Data[1], post1.Data[2])
	fmt.Printf("| %-20s | %.6f, %.6f, %.6f |\n", "Multi-Core Tiled", post2.Data[0], post2.Data[1], post2.Data[2])
	
	if maxDiff01 < 1e-6 && maxDiff02 < 1e-6 {
		fmt.Println("\n✅ All parity checks passed!")
	} else {
		fmt.Println("\n❌ Parity check failure detected!")
	}
}
