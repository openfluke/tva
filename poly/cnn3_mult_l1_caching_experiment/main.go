package main

import (
	"fmt"
	"time"

	"github.com/openfluke/loom/poly"
)

type typeConfig struct {
	name      string
	dtype     poly.DType
	scale     float32
	tolerance float64
}

type result struct {
	tNormal    time.Duration
	tSingle    time.Duration
	tMulti     time.Duration
	tileSize   int
	maxDiff01  float64
	maxDiff02  float64
	parity01   bool
	parity02   bool
	sample     [3]float32
}

func runDType(cfg typeConfig, iterations int) result {
	ws := poly.NewWeightStore(32 * 32 * 3 * 3 * 3)
	for i := range ws.Master {
		ws.Master[i] = 0.1
	}
	ws.Scale = cfg.scale
	if cfg.dtype != poly.DTypeFloat32 {
		ws.Morph(cfg.dtype)
	}

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
		DType:         cfg.dtype,
		WeightStore:   ws,
		UseTiling:     true,
		TileSize:      0,
	}

	input := poly.NewTensor[float32](1, 32, 32, 32, 32)
	for i := range input.Data {
		input.Data[i] = 0.5
	}

	// Normal (non-tiled)
	l.UseTiling = false
	var post0 *poly.Tensor[float32]
	start := time.Now()
	for i := 0; i < iterations; i++ {
		_, post0 = poly.CNN3ForwardPolymorphic(&l, input)
	}
	tNormal := time.Since(start) / time.Duration(iterations)

	// Single-core tiled
	l.UseTiling = true
	l.TileSize = 0
	l.Network.EnableMultiCoreTiling = false
	l.SyncToCPU()
	tileSize := l.TileSize
	var post1 *poly.Tensor[float32]
	start = time.Now()
	for i := 0; i < iterations; i++ {
		_, post1 = poly.CNN3ForwardPolymorphic(&l, input)
	}
	tSingle := time.Since(start) / time.Duration(iterations)

	// Multi-core tiled
	l.UseTiling = true
	l.TileSize = 0
	l.Network.EnableMultiCoreTiling = true
	l.SyncToCPU()
	var post2 *poly.Tensor[float32]
	start = time.Now()
	for i := 0; i < iterations; i++ {
		_, post2 = poly.CNN3ForwardPolymorphic(&l, input)
	}
	tMulti := time.Since(start) / time.Duration(iterations)

	// Parity
	maxDiff01, maxDiff02 := 0.0, 0.0
	for i := range post0.Data {
		d01 := float64(post0.Data[i] - post1.Data[i])
		if d01 < 0 {
			d01 = -d01
		}
		if d01 > maxDiff01 {
			maxDiff01 = d01
		}
		d02 := float64(post0.Data[i] - post2.Data[i])
		if d02 < 0 {
			d02 = -d02
		}
		if d02 > maxDiff02 {
			maxDiff02 = d02
		}
	}

	return result{
		tNormal:   tNormal,
		tSingle:   tSingle,
		tMulti:    tMulti,
		tileSize:  tileSize,
		maxDiff01: maxDiff01,
		maxDiff02: maxDiff02,
		parity01:  maxDiff01 <= cfg.tolerance,
		parity02:  maxDiff02 <= cfg.tolerance,
		sample:    [3]float32{post0.Data[0], post0.Data[1], post0.Data[2]},
	}
}

func parityMark(ok bool) string {
	if ok {
		return "PASS"
	}
	return "FAIL"
}

func main() {
	fmt.Println("=== CNN3 Multi-Core Tiling — All Numerical Types ===")

	iterations := 3

	types := []typeConfig{
		// 64-bit floats (8 bytes/weight → smallest tile)
		{"Float64",   poly.DTypeFloat64,  1.0,  1e-5},
		// 32-bit floats (4 bytes/weight)
		{"Float32",   poly.DTypeFloat32,  1.0,  1e-5},
		{"Float16",   poly.DTypeFloat16,  1.0,  1e-5}, // stored as float32
		{"BFloat16",  poly.DTypeBFloat16, 1.0,  1e-5}, // stored as float32
		// 8-bit floats (1 byte/weight → larger tile possible)
		{"FP8-E4M3",  poly.DTypeFP8E4M3, 0.01, 1e-5},
		{"FP8-E5M2",  poly.DTypeFP8E5M2, 0.01, 1e-5},
		// 64-bit integers (8 bytes/weight)
		{"Int64",     poly.DTypeInt64,  0.01, 1e-5},
		{"Uint64",    poly.DTypeUint64, 0.01, 1e-5},
		// 32-bit integers (4 bytes/weight)
		{"Int32",     poly.DTypeInt32,  0.01, 1e-5},
		{"Uint32",    poly.DTypeUint32, 0.01, 1e-5},
		// 16-bit integers (2 bytes/weight)
		{"Int16",     poly.DTypeInt16,  0.01, 1e-5},
		{"Uint16",    poly.DTypeUint16, 0.01, 1e-5},
		// 8-bit integers (1 byte/weight)
		{"Int8",      poly.DTypeInt8,   0.01, 1e-5},
		{"Uint8",     poly.DTypeUint8,  0.01, 1e-5},
		// Sub-byte — all stored as []int8 (1 byte/weight in RAM)
		{"Int4",      poly.DTypeInt4,   0.01, 1e-5},
		{"Uint4",     poly.DTypeUint4,  0.01, 1e-5},
		{"FP4",       poly.DTypeFP4,    0.01, 1e-5},
		{"Int2",      poly.DTypeInt2,   0.01, 1e-5},
		{"Uint2",     poly.DTypeUint2,  0.01, 1e-5},
		{"Ternary",   poly.DTypeTernary, 0.1, 1e-5},
		{"Binary",    poly.DTypeBinary,  0.1, 1e-5},
	}

	// Header
	fmt.Println()
	fmt.Printf("| %-10s | %-5s | %-14s | %-14s | %-14s | %-7s | %-7s | %-7s | %-8s | %-8s |\n",
		"DType", "Tile", "Normal", "Single-Core", "Multi-Core", "1C-Spd", "MC-Spd", "MaxDiff", "1C-Par", "MC-Par")
	fmt.Println("|------------|-------|----------------|----------------|----------------|---------|---------|----------|----------|----------|")

	allPass := true
	for _, cfg := range types {
		fmt.Printf("  running %-10s ...\r", cfg.name)
		r := runDType(cfg, iterations)
		if !r.parity01 || !r.parity02 {
			allPass = false
		}
		fmt.Printf("| %-10s | %-5d | %-14v | %-14v | %-14v | %-7.2fx | %-7.2fx | %-8.2e | %-8s | %-8s |\n",
			cfg.name,
			r.tileSize,
			r.tNormal,
			r.tSingle,
			r.tMulti,
			float64(r.tNormal)/float64(r.tSingle),
			float64(r.tNormal)/float64(r.tMulti),
			r.maxDiff02,
			parityMark(r.parity01),
			parityMark(r.parity02),
		)
	}

	fmt.Println()
	if allPass {
		fmt.Println("✅ All parity checks passed across all numerical types!")
	} else {
		fmt.Println("❌ One or more parity checks FAILED — review table above.")
	}
}
