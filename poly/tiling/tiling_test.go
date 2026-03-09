package main

import (
	"math/rand"
	"testing"
	"github.com/openfluke/loom/poly"
)

func TestTilingFidelity(t *testing.T) {
	inH, inW, inC := 32, 32, 8
	filters, kSize := 16, 3
	
	setupCNN := func() *poly.VolumetricLayer {
		layer := &poly.VolumetricLayer{
			Type:          poly.LayerCNN2,
			InputHeight:   inH,
			InputWidth:    inW,
			InputChannels: inC,
			Filters:       filters,
			KernelSize:    kSize,
			Stride:        1,
			Padding:       1,
			OutputHeight:  inH,
			OutputWidth:   inW,
			DType:         poly.DTypeFloat32,
			WeightStore:   poly.NewWeightStore(filters * inC * kSize * kSize),
			TileSize:      8,
		}
		layer.WeightStore.Randomize(42)
		return layer
	}

	layer := setupCNN()
	input := poly.NewTensor[float32](1, inC, inH, inW)
	for i := range input.Data { input.Data[i] = rand.Float32() }

	layer.UseTiling = false
	_, postNoTile := poly.DispatchLayer(layer, input, nil)

	layer.UseTiling = true
	_, postTile := poly.DispatchLayer(layer, input, nil)

	for i := range postNoTile.Data {
		diff := postNoTile.Data[i] - postTile.Data[i]
		if diff < 0 { diff = -diff }
		if diff > 1e-5 {
			t.Errorf("Fidelity mismatch at %d: %f vs %f", i, postNoTile.Data[i], postTile.Data[i])
			break
		}
	}
}

func BenchmarkCNNTiling(b *testing.B) {
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

		b.Run(names[i]+"_NoTiling", func(b *testing.B) {
			layer.UseTiling = false
			for j := 0; j < b.N; j++ {
				poly.DispatchLayer(layer, input, nil)
			}
		})
		b.Run(names[i]+"_Tiling", func(b *testing.B) {
			layer.UseTiling = true
			for j := 0; j < b.N; j++ {
				poly.DispatchLayer(layer, input, nil)
			}
		})
	}
}

func BenchmarkCNN3Monster(b *testing.B) {
	// 135M+ Parameters: 2560 filters * 2048 inC * 3*3*3 = 141.5M parameters
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

	b.Run("CNN3_Monster_NoTiling", func(b *testing.B) {
		layer.UseTiling = false
		for i := 0; i < b.N; i++ {
			poly.CNN3ForwardPolymorphic(layer, input)
		}
	})

	b.Run("CNN3_Monster_Tiling", func(b *testing.B) {
		layer.UseTiling = true
		for i := 0; i < b.N; i++ {
			poly.CNN3ForwardPolymorphic(layer, input)
		}
	})
}

func BenchmarkForward10Deep(b *testing.B) {
	depth, rows, cols, lpc := 1, 1, 1, 10
	n := poly.NewVolumetricNetwork(depth, rows, cols, lpc)
	dModel := 1024
	for i := range n.Layers {
		n.InitDenseCell(0, 0, 0, i, dModel, poly.ActivationReLU, 0.02)
	}

	input := poly.NewTensor[float32](1, dModel)
	for i := range input.Data { input.Data[i] = rand.Float32() }

	b.Run("Forward_NoTiling_10Deep", func(b *testing.B) {
		n.UseTiling = false
		b.ResetTimer()
		for i := 0; i < b.N; i++ {
			poly.ForwardPolymorphic(n, input)
		}
	})

	b.Run("Forward_Tiling_10Deep", func(b *testing.B) {
		n.UseTiling = true
		b.ResetTimer()
		for i := 0; i < b.N; i++ {
			poly.ForwardPolymorphic(n, input)
		}
	})
}
