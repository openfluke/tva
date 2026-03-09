package main

import (
	"fmt"
	"math/rand"
	"time"
	"github.com/openfluke/loom/poly"
)

func main() {
	fmt.Println("🚀 Running Poly Tiling Benchmarks...")
	fmt.Println("------------------------------------")

	runCNNBenchmarks()
	runMonsterBenchmark()
	runDeepBenchmark()

	fmt.Println("\n✅ All Benchmarks Completed.")
}

func runCNNBenchmarks() {
	fmt.Println("\n--- CNN Benchmarks (64x64x16) ---")
	inH, inW, inC := 64, 64, 16
	filters, kSize := 32, 3
	
	setupCNN := func(ltype poly.LayerType) *poly.VolumetricLayer {
		wCount := filters * inC * kSize * kSize
		inD, outD := 1, 1
		if ltype == poly.LayerCNN3 { 
			wCount *= kSize 
			inD = 16
			outD = 16
		}
		layer := &poly.VolumetricLayer{
			Type:          ltype,
			InputDepth:    inD,
			InputHeight:   inH,
			InputWidth:    inW,
			InputChannels: inC,
			Filters:       filters,
			KernelSize:    kSize,
			Stride:        1,
			Padding:       1,
			OutputDepth:   outD,
			OutputHeight:  inH,
			OutputWidth:   inW,
			DType:         poly.DTypeFloat32,
			WeightStore:   poly.NewWeightStore(wCount),
			TileSize:      8,
		}
		layer.WeightStore.Randomize(42)
		return layer
	}

	types := []poly.LayerType{poly.LayerCNN1, poly.LayerCNN2, poly.LayerCNN3}
	names := []string{"CNN1", "CNN2", "CNN3"}

	for i, lt := range types {
		layer := setupCNN(lt)
		input := poly.NewTensor[float32](1, inC, inH, inW)
		if lt == poly.LayerCNN3 {
			input = poly.NewTensor[float32](1, inC, 16, inH, inW)
		}
		for j := range input.Data { input.Data[j] = rand.Float32() }

		// No Tiling
		layer.UseTiling = false
		start := time.Now()
		for j := 0; j < 10; j++ {
			poly.DispatchLayer(layer, input, nil)
		}
		noTileDur := time.Since(start) / 10

		// Tiling
		layer.UseTiling = true
		start = time.Now()
		for j := 0; j < 10; j++ {
			poly.DispatchLayer(layer, input, nil)
		}
		tileDur := time.Since(start) / 10

		fmt.Printf("%s: No Tiling: %v | Tiling: %v | Speedup: %.2fx\n", 
			names[i], noTileDur, tileDur, float64(noTileDur)/float64(tileDur))
	}
}

func runMonsterBenchmark() {
	fmt.Println("\n--- CNN3 Monster Benchmark (135M+ Params) ---")
	filters, inC, kSize := 2560, 2048, 3
	inD, inH, inW := 2, 2, 2
	outD, outH, outW := 2, 2, 2

	layer := &poly.VolumetricLayer{
		Type:          poly.LayerCNN3,
		InputDepth:    inD,
		InputHeight:   inH,
		InputWidth:    inW,
		InputChannels: inC,
		Filters:       filters,
		KernelSize:    kSize,
		Stride:        1,
		Padding:       1,
		OutputDepth:   outD,
		OutputHeight:  outH,
		OutputWidth:   outW,
		DType:         poly.DTypeFloat32,
		WeightStore:   poly.NewWeightStore(filters * inC * kSize * kSize * kSize),
		TileSize:      8,
	}
	
	input := poly.NewTensor[float32](1, inC, inD, inH, inW)

	// No Tiling
	layer.UseTiling = false
	fmt.Print("Running Monster (No Tiling)... ")
	start := time.Now()
	poly.CNN3ForwardPolymorphic(layer, input)
	noTileDur := time.Since(start)
	fmt.Println(noTileDur)

	// Tiling
	layer.UseTiling = true
	fmt.Print("Running Monster (Tiling)...    ")
	start = time.Now()
	poly.CNN3ForwardPolymorphic(layer, input)
	tileDur := time.Since(start)
	fmt.Println(tileDur)

	fmt.Printf("Monster Speedup: %.2fx\n", float64(noTileDur)/float64(tileDur))
}

func runDeepBenchmark() {
	fmt.Println("\n--- Forward 10-Deep Benchmark ---")
	depth, rows, cols, lpc := 1, 1, 1, 10
	n := poly.NewVolumetricNetwork(depth, rows, cols, lpc)
	dModel := 1024
	for i := range n.Layers {
		n.InitDenseCell(0, 0, 0, i, dModel, poly.ActivationReLU, 0.02)
	}

	input := poly.NewTensor[float32](1, dModel)
	for i := range input.Data { input.Data[i] = rand.Float32() }

	// No Tiling
	n.UseTiling = false
	start := time.Now()
	for i := 0; i < 50; i++ {
		poly.ForwardPolymorphic(n, input)
	}
	noTileDur := time.Since(start) / 50

	// Tiling
	n.UseTiling = true
	start = time.Now()
	for i := 0; i < 50; i++ {
		poly.ForwardPolymorphic(n, input)
	}
	tileDur := time.Since(start) / 50

	fmt.Printf("10-Deep: No Tiling: %v | Tiling: %v | Speedup: %.2fx\n", 
		noTileDur, tileDur, float64(noTileDur)/float64(tileDur))
}
