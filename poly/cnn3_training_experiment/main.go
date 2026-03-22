// Package main: CNN3 training experiment — all 6 execution modes × 21 numerical types.
//
// Verification chain per run:
//   1. Create network from JSON architecture spec  (BuildNetworkFromJSON)
//   2. Apply target dtype / scale to the layer
//   3. Snapshot initial master weights (W0)
//   4. Train for N epochs
//   5. For GPU modes: SyncWeightsFromGPU → master weights
//   6. Snapshot trained master weights (Wt)
//   7. SerializeNetwork → JSON bytes → write to temp file → read back → DeserializeNetwork
//   8. Snapshot reloaded master weights (Wr)
//
// Pass criteria:
//   trainOK  : maxDiff(Wt, W0) > 0          — training changed at least one weight
//   saveOK   : maxDiff(Wr, Wt) == 0         — save/reload is bit-exact (base64 raw bytes)
package main

import (
	"encoding/json"
	"fmt"
	"math"
	"os"
	"time"

	"github.com/openfluke/loom/poly"
)

// ---- Network geometry ----
const (
	batchSz              = 1
	inC                  = 4
	inD, inH, inW        = 4, 4, 4
	filters              = 4
	kSize                = 3
	stride               = 1
	padding              = 1
	outD, outH, outW     = inD, inH, inW
	inputElements        = batchSz * inC * inD * inH * inW   // 256
	outputElements       = batchSz * filters * outD * outH * outW // 256
	weightCount          = filters * inC * kSize * kSize * kSize  // 432
	trainingEpochs       = 5
)

// ---- JSON architecture spec (dtype set at runtime) ----
type layerSpecJSON struct {
	Z int    `json:"z"`
	Y int    `json:"y"`
	X int    `json:"x"`
	L int    `json:"l"`
	Type          string `json:"type"`
	Activation    string `json:"activation"`
	DType         string `json:"dtype"`
	InputChannels int    `json:"input_channels"`
	InputDepth    int    `json:"input_depth"`
	InputHeight   int    `json:"input_height"`
	InputWidth    int    `json:"input_width"`
	Filters       int    `json:"filters"`
	OutputDepth   int    `json:"output_depth"`
	OutputHeight  int    `json:"output_height"`
	OutputWidth   int    `json:"output_width"`
	KernelSize    int    `json:"kernel_size"`
	Stride        int    `json:"stride"`
	Padding       int    `json:"padding"`
}

type netSpecJSON struct {
	ID            string          `json:"id"`
	Depth         int             `json:"depth"`
	Rows          int             `json:"rows"`
	Cols          int             `json:"cols"`
	LayersPerCell int             `json:"layers_per_cell"`
	Layers        []layerSpecJSON `json:"layers"`
}

// dtypeStr maps DType to the string understood by ParseDType in serialization.go.
func dtypeStr(d poly.DType) string {
	switch d {
	case poly.DTypeFloat64:  return "FLOAT64"
	case poly.DTypeFloat32:  return "FLOAT32"
	case poly.DTypeFloat16:  return "FLOAT16"
	case poly.DTypeBFloat16: return "BFLOAT16"
	case poly.DTypeFP8E4M3:  return "FP8E4M3"
	case poly.DTypeFP8E5M2:  return "FP8E5M2"
	case poly.DTypeInt64:    return "INT64"
	case poly.DTypeUint64:   return "UINT64"
	case poly.DTypeInt32:    return "INT32"
	case poly.DTypeUint32:   return "UINT32"
	case poly.DTypeInt16:    return "INT16"
	case poly.DTypeUint16:   return "UINT16"
	case poly.DTypeInt8:     return "INT8"
	case poly.DTypeUint8:    return "UINT8"
	case poly.DTypeInt4:     return "INT4"
	case poly.DTypeUint4:    return "UINT4"
	case poly.DTypeFP4:      return "FP4"
	case poly.DTypeInt2:     return "INT2"
	case poly.DTypeUint2:    return "UINT2"
	case poly.DTypeTernary:  return "TERNARY"
	case poly.DTypeBinary:   return "BINARY"
	default:                 return "FLOAT32"
	}
}

// gpuCompatible returns true for types whose SyncToGPU creates float32 GPU buffers.
// Sub-byte types (Int4 and smaller) are skipped for GPU training.
func gpuCompatible(d poly.DType) bool {
	switch d {
	case poly.DTypeInt4, poly.DTypeUint4, poly.DTypeFP4,
		poly.DTypeInt2, poly.DTypeUint2,
		poly.DTypeTernary, poly.DTypeBinary:
		return false
	}
	return true
}

// buildNetworkFromSpec creates a CNN3 network using BuildNetworkFromJSON,
// then applies the target dtype and scale on top of the random init weights.
func buildNetworkFromSpec(dtype poly.DType, scale float32) (*poly.VolumetricNetwork, error) {
	spec := netSpecJSON{
		ID: "cnn3_train_test", Depth: 1, Rows: 1, Cols: 1, LayersPerCell: 1,
		Layers: []layerSpecJSON{{
			Type: "CNN3", Activation: "LINEAR", DType: "FLOAT32",
			InputChannels: inC, InputDepth: inD, InputHeight: inH, InputWidth: inW,
			Filters: filters, OutputDepth: outD, OutputHeight: outH, OutputWidth: outW,
			KernelSize: kSize, Stride: stride, Padding: padding,
		}},
	}
	raw, err := json.Marshal(spec)
	if err != nil {
		return nil, err
	}
	net, err := poly.BuildNetworkFromJSON(raw)
	if err != nil {
		return nil, err
	}
	// Apply target dtype + scale (architecture spec always uses FLOAT32 to get random init).
	l := &net.Layers[0]
	l.DType = dtype
	l.WeightStore.Scale = scale
	if dtype != poly.DTypeFloat32 {
		l.WeightStore.Morph(dtype)
	}
	return net, nil
}

// makeBatch creates a deterministic training batch (same every run).
func makeBatch() poly.TrainingBatch[float32] {
	input := poly.NewTensor[float32](batchSz, inC, inD, inH, inW)
	for i := range input.Data {
		// simple deterministic fill
		input.Data[i] = float32(i%13)*0.1 - 0.6
	}
	target := poly.NewTensor[float32](batchSz, filters, outD, outH, outW)
	for i := range target.Data {
		target.Data[i] = float32(i%7)*0.15 - 0.45
	}
	return poly.TrainingBatch[float32]{Input: input, Target: target}
}

// cloneMaster returns a copy of WeightStore.Master.
func cloneMaster(ws *poly.WeightStore) []float32 {
	out := make([]float32, len(ws.Master))
	copy(out, ws.Master)
	return out
}

// maxDiff returns the max absolute difference between two float32 slices.
func maxDiff(a, b []float32) float64 {
	d := 0.0
	for i := range a {
		if v := math.Abs(float64(a[i] - b[i])); v > d {
			d = v
		}
	}
	return d
}

// saveToDiskAndReload serializes net to a temp file, reads it back, and deserializes.
func saveToDiskAndReload(net *poly.VolumetricNetwork) (*poly.VolumetricNetwork, error) {
	data, err := poly.SerializeNetwork(net)
	if err != nil {
		return nil, fmt.Errorf("serialize: %w", err)
	}
	tmp, err := os.CreateTemp("", "poly_cnn3_*.json")
	if err != nil {
		return nil, fmt.Errorf("tempfile: %w", err)
	}
	path := tmp.Name()
	defer os.Remove(path)

	if _, err := tmp.Write(data); err != nil {
		tmp.Close()
		return nil, fmt.Errorf("write: %w", err)
	}
	tmp.Close()

	// Read back from disk to truly simulate loading from file.
	diskData, err := os.ReadFile(path)
	if err != nil {
		return nil, fmt.Errorf("readfile: %w", err)
	}
	reloaded, err := poly.DeserializeNetwork(diskData)
	if err != nil {
		return nil, fmt.Errorf("deserialize: %w", err)
	}
	return reloaded, nil
}

type runResult struct {
	loss0    float64
	lossN    float64
	duration time.Duration
	trainOK  bool  // training changed at least one master weight
	saveOK   bool  // save/reload preserved master weights bit-exactly
	skipped  bool  // mode not applicable for this dtype
	err      string
}

func runMode(dtype poly.DType, scale float32, mode poly.TrainingMode, batch poly.TrainingBatch[float32]) runResult {
	// Skip GPU modes for sub-byte dtypes.
	if mode.IsGPU() && !gpuCompatible(dtype) {
		return runResult{skipped: true}
	}

	net, err := buildNetworkFromSpec(dtype, scale)
	if err != nil {
		return runResult{err: err.Error()}
	}

	// Snapshot initial master weights.
	w0 := cloneMaster(net.Layers[0].WeightStore)

	tcfg := &poly.TrainingConfig{
		Epochs:       trainingEpochs,
		LearningRate: 0.001,
		LossType:     "mse",
		Verbose:      false,
		Mode:         mode,
	}

	start := time.Now()
	result, terr := poly.Train[float32](net, []poly.TrainingBatch[float32]{batch}, tcfg)
	dur := time.Since(start)
	if terr != nil {
		return runResult{err: terr.Error(), duration: dur}
	}

	// For GPU modes: sync GPU weights back to CPU master before comparing/saving.
	if mode.IsGPU() {
		if serr := poly.SyncWeightsFromGPU(net); serr != nil {
			return runResult{err: "SyncWeightsFromGPU: " + serr.Error(), duration: dur}
		}
	}

	// Snapshot trained master weights.
	wt := cloneMaster(net.Layers[0].WeightStore)
	trainOK := maxDiff(wt, w0) > 0

	// Save to disk and reload.
	reloaded, rerr := saveToDiskAndReload(net)
	if rerr != nil {
		return runResult{
			loss0: result.LossHistory[0], lossN: result.FinalLoss,
			duration: dur, trainOK: trainOK,
			err: rerr.Error(),
		}
	}

	// Snapshot reloaded master weights and compare.
	wr := cloneMaster(reloaded.Layers[0].WeightStore)
	saveOK := maxDiff(wr, wt) == 0

	return runResult{
		loss0:    result.LossHistory[0],
		lossN:    result.FinalLoss,
		duration: dur,
		trainOK:  trainOK,
		saveOK:   saveOK,
	}
}

func mark(ok bool) string {
	if ok { return "PASS" }
	return "FAIL"
}

type typeConfig struct {
	name  string
	dtype poly.DType
	scale float32
}

func main() {
	fmt.Println("=== CNN3 Training — All Modes × All Numerical Types ===")
	fmt.Println()
	fmt.Println("Workflow per run:")
	fmt.Println("  1. BuildNetworkFromJSON (architecture JSON → random init weights)")
	fmt.Println("  2. Apply target dtype + scale")
	fmt.Println("  3. Train (5 epochs, LR=0.001, MSE)")
	fmt.Println("  4. GPU modes: SyncWeightsFromGPU → master")
	fmt.Println("  5. SerializeNetwork → temp file on disk")
	fmt.Println("  6. DeserializeNetwork ← disk")
	fmt.Println("  Pass: trainOK = master weights changed  |  saveOK = reloaded == trained (bit-exact)")
	fmt.Println()

	batch := makeBatch()

	// Check GPU once.
	testNet := poly.NewVolumetricNetwork(1, 1, 1, 1)
	gpuAvail := testNet.InitWGPU() == nil
	if gpuAvail {
		sc, mc := poly.CNN3GPUTileSizes(testNet.GPUContext)
		fmt.Printf("GPU ready — SC tile=%d  MC tile=%d  MaxInvocations=%d\n\n",
			sc, mc, testNet.GPUContext.Limits.MaxComputeInvocationsPerWorkgroup)
	} else {
		fmt.Println("No GPU detected — GPU modes skipped.\n")
	}

	allModes := []poly.TrainingMode{
		poly.TrainingModeCPUNormal,
		poly.TrainingModeCPUSC,
		poly.TrainingModeCPUMC,
		poly.TrainingModeGPUNormal,
		poly.TrainingModeGPUSC,
		poly.TrainingModeGPUMC,
	}

	types := []typeConfig{
		{"Float32",  poly.DTypeFloat32,  1.0},
		{"Float64",  poly.DTypeFloat64,  1.0},
		{"Float16",  poly.DTypeFloat16,  1.0},
		{"BFloat16", poly.DTypeBFloat16, 1.0},
		{"FP8-E4M3", poly.DTypeFP8E4M3, 0.01},
		{"FP8-E5M2", poly.DTypeFP8E5M2, 0.01},
		{"Int64",    poly.DTypeInt64,    0.01},
		{"Uint64",   poly.DTypeUint64,   0.01},
		{"Int32",    poly.DTypeInt32,    0.01},
		{"Uint32",   poly.DTypeUint32,   0.01},
		{"Int16",    poly.DTypeInt16,    0.01},
		{"Uint16",   poly.DTypeUint16,   0.01},
		{"Int8",     poly.DTypeInt8,     0.01},
		{"Uint8",    poly.DTypeUint8,    0.01},
		{"Int4",     poly.DTypeInt4,     0.01},
		{"Uint4",    poly.DTypeUint4,    0.01},
		{"FP4",      poly.DTypeFP4,      0.01},
		{"Int2",     poly.DTypeInt2,     0.01},
		{"Uint2",    poly.DTypeUint2,    0.01},
		{"Ternary",  poly.DTypeTernary,  0.1},
		{"Binary",   poly.DTypeBinary,   0.1},
	}

	fmt.Printf("| %-10s | %-13s | %-10s | %-10s | %-8s | %-7s | %-10s |\n",
		"DType", "Mode", "Loss[0]", "Loss[N]", "Time", "Train↑", "Save/Reload")
	fmt.Println("|------------|---------------|------------|------------|----------|---------|------------|")

	overallPass := true
	for _, cfg := range types {
		for _, mode := range allModes {
			if mode.IsGPU() && !gpuAvail {
				continue
			}
			fmt.Printf("  running %-10s %-13s ...\r", cfg.name, mode.String())

			r := runMode(cfg.dtype, cfg.scale, mode, batch)

			if r.skipped {
				fmt.Printf("| %-10s | %-13s | %-10s | %-10s | %-8s | %-7s | %-10s |\n",
					cfg.name, mode.String(), "—", "—", "—", "SKIP", "SKIP")
				continue
			}
			if r.err != "" {
				fmt.Printf("| %-10s | %-13s | %-10s | %-10s | %-8v | %-7s | %-10s |\n",
					cfg.name, mode.String(), "ERR", "ERR",
					r.duration.Round(time.Millisecond), "ERR", r.err[:min(len(r.err), 20)])
				overallPass = false
				continue
			}

			if !r.trainOK || !r.saveOK {
				overallPass = false
			}
			fmt.Printf("| %-10s | %-13s | %-10.4e | %-10.4e | %-8v | %-7s | %-10s |\n",
				cfg.name, mode.String(),
				r.loss0, r.lossN,
				r.duration.Round(time.Millisecond),
				mark(r.trainOK),
				mark(r.saveOK),
			)
		}
		fmt.Println("|------------|---------------|------------|------------|----------|---------|------------|")
	}

	fmt.Println()
	if overallPass {
		fmt.Println("✅ All training + save/reload checks passed!")
	} else {
		fmt.Println("❌ One or more checks FAILED — see table above.")
	}
	fmt.Println()
	fmt.Println("Notes:")
	fmt.Println("  Train↑     = training changed at least one master weight (maxDiff > 0)")
	fmt.Println("  Save/Reload = DeserializeNetwork produces bit-exact master weights (maxDiff == 0)")
	fmt.Println("  SKIP        = sub-byte dtype + GPU mode (SyncToGPU doesn't create float32 buffers)")
}

func min(a, b int) int {
	if a < b { return a }
	return b
}
