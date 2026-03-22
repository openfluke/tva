// Package main: CNN3 training experiment — all 6 execution modes × 21 numerical types.
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

func gpuCompatible(d poly.DType) bool {
	return true
}

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
	l := &net.Layers[0]
	l.DType = dtype
	l.WeightStore.Scale = scale
	if dtype != poly.DTypeFloat32 {
		l.WeightStore.Morph(dtype)
	}
	return net, nil
}

func makeBatch() poly.TrainingBatch[float32] {
	input := poly.NewTensor[float32](batchSz, inC, inD, inH, inW)
	for i := range input.Data {
		input.Data[i] = float32(i%13)*0.1 - 0.6
	}
	target := poly.NewTensor[float32](batchSz, filters, outD, outH, outW)
	for i := range target.Data {
		target.Data[i] = float32(i%7)*0.15 - 0.45
	}
	return poly.TrainingBatch[float32]{Input: input, Target: target}
}

func cloneMaster(ws *poly.WeightStore) []float32 {
	out := make([]float32, len(ws.Master))
	copy(out, ws.Master)
	return out
}

func maxDiff(a, b []float32) float64 {
	d := 0.0
	for i := range a {
		if v := math.Abs(float64(a[i] - b[i])); v > d {
			d = v
		}
	}
	return d
}

func saveToDiskAndReload(net *poly.VolumetricNetwork) (*poly.VolumetricNetwork, int, error) {
	data, err := poly.SerializeNetwork(net)
	if err != nil {
		return nil, 0, fmt.Errorf("serialize: %w", err)
	}
	tmp, err := os.CreateTemp("", "poly_cnn3_*.json")
	if err != nil {
		return nil, 0, fmt.Errorf("tempfile: %w", err)
	}
	path := tmp.Name()
	defer os.Remove(path)

	if _, err := tmp.Write(data); err != nil {
		tmp.Close()
		return nil, 0, fmt.Errorf("write: %w", err)
	}
	tmp.Close()

	diskData, err := os.ReadFile(path)
	if err != nil {
		return nil, 0, fmt.Errorf("readfile: %w", err)
	}
	reloaded, err := poly.DeserializeNetwork(diskData)
	if err != nil {
		return nil, 0, fmt.Errorf("deserialize: %w", err)
	}
	return reloaded, len(data), nil
}

type runResult struct {
	loss0    float64
	lossN    float64
	duration time.Duration
	trainOK  bool
	saveOK   bool
	fileSize float64
	ramSize  float64
	skipped  bool
	err      string
}

func runMode(dtype poly.DType, scale float32, mode poly.TrainingMode, batch poly.TrainingBatch[float32]) runResult {
	if mode.IsGPU() && !gpuCompatible(dtype) {
		return runResult{skipped: true}
	}

	net, err := buildNetworkFromSpec(dtype, scale)
	if err != nil {
		return runResult{err: err.Error()}
	}

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

	if mode.IsGPU() {
		if serr := poly.SyncWeightsFromGPU(net); serr != nil {
			return runResult{err: "SyncWeightsFromGPU: " + serr.Error(), duration: dur}
		}
	}

	wt := cloneMaster(net.Layers[0].WeightStore)
	trainOK := maxDiff(wt, w0) > 0

	reloaded, byteCount, rerr := saveToDiskAndReload(net)
	if rerr != nil {
		return runResult{
			loss0: result.LossHistory[0], lossN: result.FinalLoss,
			duration: dur, trainOK: trainOK,
			err: rerr.Error(),
		}
	}

	wr := cloneMaster(reloaded.Layers[0].WeightStore)
	
	// Prepare expected version accounting for quantization
	expected := wt
	if dtype != poly.DTypeFloat32 {
		expected = make([]float32, len(wt))
		for i, v := range wt {
			expected[i] = poly.SimulatePrecision(v, dtype, net.Layers[0].WeightStore.Scale)
		}
	}
	saveOK := maxDiff(wr, expected) == 0

	// RAM estimate
	weightCount := len(wt)
	ramBytes := int64(weightCount * 4) // Master
	if dtype != poly.DTypeFloat32 {
		bits := poly.DTypeBits(dtype)
		ramBytes += int64(math.Ceil(float64(weightCount) * float64(bits) / 8.0))
	}

	return runResult{
		loss0:    result.LossHistory[0],
		lossN:    result.FinalLoss,
		duration: dur,
		trainOK:  trainOK,
		saveOK:   saveOK,
		fileSize: float64(byteCount) / 1024.0,
		ramSize:  float64(ramBytes) / 1024.0,
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
	fmt.Println("  5. SerializeNetwork → temp file on disk (Quantized if DType != FP32)")
	fmt.Println("  6. DeserializeNetwork ← disk")
	fmt.Println("  Pass: trainOK = master weights changed  |  saveOK = reloaded == trained (quantized bit-exact)")
	fmt.Println()

	batch := makeBatch()

	testNet := poly.NewVolumetricNetwork(1, 1, 1, 1)
	gpuAvail := testNet.InitWGPU() == nil
	if gpuAvail {
		sc, mc := poly.CNN3GPUTileSizes(testNet.GPUContext)
		fmt.Printf("GPU ready — SC tile=%d  MC tile=%d  MaxInvocations=%d\n\n",
			sc, mc, testNet.GPUContext.Limits.MaxComputeInvocationsPerWorkgroup)
	} else {
		fmt.Println("No GPU detected — GPU modes skipped.")
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

	fmt.Printf("| %-10s | %-13s | %-10s | %-10s | %-8s | %-7s | %-11s | %-8s | %-8s |\n",
		"DType", "Mode", "Loss[0]", "Loss[N]", "Time", "Train↑", "Save/Reload", "File", "RAM")
	fmt.Println("|------------|---------------|------------|------------|----------|---------|-------------|----------|----------|")

	overallPass := true
	for _, cfg := range types {
		for _, mode := range allModes {
			if mode.IsGPU() && !gpuAvail {
				continue
			}
			fmt.Printf("  running %-10s %-13s ...\r", cfg.name, mode.String())

			r := runMode(cfg.dtype, cfg.scale, mode, batch)

			if r.skipped {
				fmt.Printf("| %-10s | %-13s | %-10s | %-10s | %-8s | %-7s | %-11s | %-8s | %-8s |\n",
					cfg.name, mode.String(), "—", "—", "—", "SKIP", "SKIP", "—", "—")
				continue
			}
			if r.err != "" {
				fmt.Printf("| %-10s | %-13s | %-10s | %-10s | %-8v | %-7s | %-11s | %-8s | %-8s |\n",
					cfg.name, mode.String(), "ERR", "ERR",
					r.duration.Round(time.Millisecond), "ERR", r.err[:min(len(r.err), 5)], "—", "—")
				overallPass = false
				continue
			}
			fmt.Printf("| %-10s | %-13s | %-10.4e | %-10.4e | %-8s | %-7s | %-11s | %-8.1fKB | %-8.1fKB |\n",
				cfg.name, mode.String(),
				r.loss0, r.lossN,
				r.duration.Round(time.Millisecond),
				mark(r.trainOK),
				mark(r.saveOK),
				r.fileSize,
				r.ramSize,
			)
			if !r.trainOK || !r.saveOK {
				overallPass = false
			}
		}
		fmt.Println("|------------|---------------|------------|------------|----------|---------|-------------|----------|----------|")
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
	fmt.Println("  Save/Reload = DeserializeNetwork produces bit-exact master weights (accounting for quantization loss)")
	fmt.Println("  File Size  = Serialized JSON size on disk (reflects packing)")
	fmt.Println("  RAM Size   = Estimated memory for Master + Active weight buffers")
}

func min(a, b int) int {
	if a < b { return a }
	return b
}
