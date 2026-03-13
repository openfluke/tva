package main

import (
	"fmt"
	"math/rand"
	"time"
	"github.com/openfluke/loom/poly"
)

func main() {
	fmt.Println("🚀 Running Multi-Type Monster Benchmark (135M+ Params)...")
	fmt.Println("---------------------------------------------------------")

	// 135M+ Parameters: 2560 filters * 2048 inC * 3*3*3 = 141.5M parameters
	filters, inC, kSize := 2560, 2048, 3
	inD, inH, inW := 2, 2, 2
	outD, outH, outW := 2, 2, 2
	
	totalParams := filters * inC * kSize * kSize * kSize
	fmt.Printf("Total Parameters: %.2f Million\n", float64(totalParams)/1e6)

	dtypes := []poly.DType{
		poly.DTypeFloat64,
		poly.DTypeFloat32,
		poly.DTypeFloat16,
		poly.DTypeBFloat16,
		poly.DTypeInt8,
		poly.DTypeFP4,
		poly.DTypeInt2,
		poly.DTypeBinary,
	}

	fmt.Printf("%-10s | %-12s | %-12s | %-12s | %-12s\n", "DType", "Memory (MB)", "NoTile (s)", "Tiled (s)", "Speedup")
	fmt.Println("---------------------------------------------------------------------------------")

	for _, dt := range dtypes {
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
			DType:         dt,
			WeightStore:   poly.NewWeightStore(totalParams),
			TileSize:      8,
		}
		
		// Initialize weights and triggers metamorphosis
		layer.WeightStore.Randomize(42, 0.1)
		layer.WeightStore.Morph(dt)
		
		memMB := float64(layer.WeightStore.SizeInBytes(dt)) / (1024 * 1024)
		
		input := poly.NewTensor[float32](1, inC, inD, inH, inW)
		for j := range input.Data { input.Data[j] = rand.Float32() }

		// benchmark No Tiling
		layer.UseTiling = false
		start := time.Now()
		poly.CNN3ForwardPolymorphic(layer, input)
		noTileDur := time.Since(start)

		// benchmark Tiling
		layer.UseTiling = true
		start = time.Now()
		poly.CNN3ForwardPolymorphic(layer, input)
		tileDur := time.Since(start)

		speedup := float64(noTileDur) / float64(tileDur)

		fmt.Printf("%-10v | %-12.1f | %-12.3f | %-12.3f | %-12.2fx\n", 
			dt, memMB, noTileDur.Seconds(), tileDur.Seconds(), speedup)
	}

	fmt.Println("---------------------------------------------------------------------------------")
	fmt.Println("✅ Multi-Type Benchmark Completed.")
}
