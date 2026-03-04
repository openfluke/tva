package main

import (
	"fmt"
	"math"

	"github.com/openfluke/loom/nn"
)

// ─────────────────────────────────────────────────────────────────────────────
// Helpers
// ─────────────────────────────────────────────────────────────────────────────

func pass(label string)      { fmt.Printf("  ✅  PASS  %s\n", label) }
func fail(label, msg string) { fmt.Printf("  ❌  FAIL  %s — %s\n", label, msg) }

func approxEq(a, b, tol float32) bool {
	d := a - b
	if d < 0 {
		d = -d
	}
	return d <= tol
}

// ─────────────────────────────────────────────────────────────────────────────
// 1. E2M1 value table — every positive nibble must decode correctly
// ─────────────────────────────────────────────────────────────────────────────

func testDecodeTable() {
	fmt.Println("\n[1] E2M1 decode table")

	// Ground-truth reference values (positive half, bits 0b000–0b111)
	ref := []struct {
		nibble uint8
		want   float32
	}{
		{0b0000, 0.0},
		{0b0001, 0.5},
		{0b0010, 1.0},
		{0b0011, 1.5},
		{0b0100, 2.0},
		{0b0101, 3.0},
		{0b0110, 4.0},
		{0b0111, 6.0},
		// Negative mirrors
		{0b1000, -0.0},
		{0b1001, -0.5},
		{0b1010, -1.0},
		{0b1011, -1.5},
		{0b1100, -2.0},
		{0b1101, -3.0},
		{0b1110, -4.0},
		{0b1111, -6.0},
	}

	allOK := true
	for _, r := range ref {
		got := nn.E2M1ToFloat32(r.nibble)
		// Use math.Abs to handle -0.0 == 0.0
		if math.Abs(float64(got-r.want)) > 1e-6 {
			fail(fmt.Sprintf("nibble 0b%04b", r.nibble),
				fmt.Sprintf("want %v, got %v", r.want, got))
			allOK = false
		}
	}
	if allOK {
		pass("all 16 nibble values decode correctly")
	}
}

// ─────────────────────────────────────────────────────────────────────────────
// 2. MultiplyE2M1 — spot-check key products
// ─────────────────────────────────────────────────────────────────────────────

func testMultiply() {
	fmt.Println("\n[2] MultiplyE2M1 bitwise gate")

	cases := []struct {
		a, b uint8
		desc string
		// expected float value of the nibble product
		wantF float32
	}{
		// 0 × anything = 0
		{0b0000, 0b0111, "0 × 6 = 0", 0.0},
		{0b0111, 0b0000, "6 × 0 = 0", 0.0},
		// 1 × 1 = 1
		{0b0010, 0b0010, "1 × 1 = 1", 1.0},
		// 1 × 2 = 2
		{0b0010, 0b0100, "1 × 2 = 2", 2.0},
		// 2 × 3 = 6 (saturates exactly at max E2M1)
		{0b0100, 0b0101, "2 × 3 = 6", 6.0},
		// sign rules: negative × positive = negative
		{0b1010, 0b0010, "-1 × 1 = -1", -1.0},
		{0b1010, 0b1010, "-1 × -1 = 1", 1.0},
		// 0.5 × 2 = 1
		{0b0001, 0b0100, "0.5 × 2 = 1", 1.0},
		// 0.5 × 0.5 = 0.25 → rounds to 0 in E2M1 (sub-smallest representable)
		// smallest positive is 0.5, so 0.25 underflows to 0
		{0b0001, 0b0001, "0.5 × 0.5 → 0 (underflow)", 0.0},
	}

	allOK := true
	for _, c := range cases {
		r := nn.MultiplyE2M1(c.a, c.b)
		got := nn.E2M1ToFloat32(r)
		if math.Abs(float64(got-c.wantF)) > 1e-6 {
			fail(c.desc, fmt.Sprintf("want %.4f, got %.4f (nibble 0b%04b)", c.wantF, got, r))
			allOK = false
		}
	}
	if allOK {
		pass("all multiply cases correct")
	}
}

// ─────────────────────────────────────────────────────────────────────────────
// 3. PackedWeights round-trip
// ─────────────────────────────────────────────────────────────────────────────

func testPackRoundTrip() {
	fmt.Println("\n[3] PackedWeights pack / unpack round-trip")

	// Use the 8 representable positive magnitudes as weights.
	// After quantisation + unpack they should come back exactly.
	weights := []float32{0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0}
	pw := nn.NewPackedWeights(weights, 8, 1)

	allOK := true
	for i, want := range weights {
		nibble := pw.Get(i)
		got := nn.E2M1ToFloat32(nibble)
		if math.Abs(float64(got-want)) > 1e-6 {
			fail(fmt.Sprintf("index %d", i),
				fmt.Sprintf("want %.2f, got %.2f", want, got))
			allOK = false
		}
	}
	if allOK {
		pass("representable magnitudes survive pack/unpack intact")
	}

	// Negative weights test
	fmt.Println("    (negative weights)")
	weightsNeg := []float32{-0.5, -1.0, -2.0, -4.0, -6.0}
	pwNeg := nn.NewPackedWeights(weightsNeg, 5, 1)
	allOK = true
	for i, want := range weightsNeg {
		nibble := pwNeg.Get(i)
		got := nn.E2M1ToFloat32(nibble)
		if math.Abs(float64(got-want)) > 1e-6 {
			fail(fmt.Sprintf("neg index %d", i),
				fmt.Sprintf("want %.2f, got %.2f", want, got))
			allOK = false
		}
	}
	if allOK {
		pass("negative representable values survive pack/unpack intact")
	}
}

// ─────────────────────────────────────────────────────────────────────────────
// 4. QuantiseInputRowFP4 — check scale and approximate reconstruction
// ─────────────────────────────────────────────────────────────────────────────

func testQuantiseRow() {
	fmt.Println("\n[4] QuantiseInputRowFP4 — scale & reconstruction")

	// Row of 16 values spanning the full fp4 range.
	row := make([]float32, 16)
	for i := range row {
		row[i] = float32(i) * 0.4 // 0, 0.4, 0.8, … 6.0
	}

	nibbles, scales := nn.QuantiseInputRowFP4(row)

	// Reconstruct and check MSE vs original.
	mse := float64(0)
	for i, v := range row {
		recon := nn.E2M1ToFloat32(nibbles[i]) * scales[0]
		diff := float64(v - recon)
		mse += diff * diff
	}
	mse /= float64(len(row))
	rmse := math.Sqrt(mse)

	// RMSE should be well below 0.5 (the step size of the largest bin).
	const threshold = 0.5
	if rmse < threshold {
		pass(fmt.Sprintf("RMSE %.4f < %.2f (acceptable quantisation error)", rmse, threshold))
	} else {
		fail("reconstruction RMSE",
			fmt.Sprintf("RMSE %.4f ≥ %.2f — quantisation error too large", rmse, threshold))
	}

	_ = scales
}

// ─────────────────────────────────────────────────────────────────────────────
// 5. ForwardRowFP4 — compare against float32 reference for a tiny matmul
// ─────────────────────────────────────────────────────────────────────────────

func testForwardRow() {
	fmt.Println("\n[5] ForwardRowFP4 vs float32 reference (4-in × 4-out)")

	// A 4×4 weight matrix of simple representable values.
	weights := []float32{
		1.0, -1.0, 2.0, 0.5,
		0.5, 1.0, -0.5, 1.0,
		-1.0, 2.0, 1.0, -1.0,
		2.0, 0.5, -1.0, 2.0,
	}
	input := []float32{1.0, 0.5, -1.0, 2.0}

	// Float32 reference matmul (weights stored row-major as [inRow][outCol])
	ref := make([]float32, 4)
	for o := 0; o < 4; o++ {
		for i := 0; i < 4; i++ {
			ref[o] += input[i] * weights[i*4+o]
		}
	}

	// FP4 path
	pw := nn.NewPackedWeights(weights, 4, 4)
	nibbles, scales := nn.QuantiseInputRowFP4(input)
	got := nn.ForwardRowFP4(nibbles, scales, pw, nil)

	fmt.Printf("    ref:  %v\n", ref)
	fmt.Printf("    fp4:  %v\n", got)

	allOK := true
	for o := 0; o < 4; o++ {
		// Allow ≤15% relative error or ≤0.3 absolute (quantisation noise)
		absErr := float32(math.Abs(float64(got[o] - ref[o])))
		relErr := float32(0)
		if ref[o] != 0 {
			relErr = float32(math.Abs(float64((got[o] - ref[o]) / ref[o])))
		}
		if absErr > 0.3 && relErr > 0.15 {
			fail(fmt.Sprintf("output[%d]", o),
				fmt.Sprintf("ref=%.4f fp4=%.4f (absErr=%.4f relErr=%.2f%%)",
					ref[o], got[o], absErr, relErr*100))
			allOK = false
		}
	}
	if allOK {
		pass("fp4 output within acceptable tolerance of float32 reference")
	}
}

// ─────────────────────────────────────────────────────────────────────────────
// 6. DenseForwardFP4 — full layer forward pass, compare with denseForwardCPU
// ─────────────────────────────────────────────────────────────────────────────

func testDenseForwardFP4() {
	fmt.Println("\n[6] DenseForwardFP4 vs DenseForward (8-in × 4-out, batch=2)")

	const (
		inputSize  = 8
		outputSize = 4
		batchSize  = 2
	)

	// Construct a weight matrix of representable E2M1 values.
	weights := make([]float32, inputSize*outputSize)
	for i := range weights {
		// Cycle through representable magnitudes with alternating signs.
		mags := []float32{0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0}
		sign := float32(1)
		if i%3 == 0 {
			sign = -1
		}
		weights[i] = sign * mags[i%len(mags)]
	}
	bias := []float32{0.1, -0.1, 0.2, -0.2}
	input := []float32{
		// batch 0
		1.0, -0.5, 2.0, 1.5, -1.0, 3.0, 0.5, -2.0,
		// batch 1
		0.5, 1.0, -1.5, 2.0, 1.0, -0.5, -1.0, 1.5,
	}

	// Float32 reference
	refConfig := nn.InitDenseLayer(inputSize, outputSize, nn.ActivationType(-1))
	copy(refConfig.Kernel, weights)
	copy(refConfig.Bias, bias)

	// Use the exported *generic* DenseForward so we can call it directly.
	inputT := nn.NewTensorFromSlice(input, batchSize*inputSize)
	wT := nn.NewTensorFromSlice(weights, inputSize*outputSize)
	bT := nn.NewTensorFromSlice(bias, outputSize)
	refPre, _ := nn.DenseForward(inputT, wT, bT, inputSize, outputSize, batchSize, nn.ActivationType(-1))
	refPreData := refPre.Data

	// FP4 path
	pw := nn.NewPackedWeights(weights, inputSize, outputSize)
	fp4Pre, _ := nn.DenseForwardFP4(input, pw, bias, batchSize, nn.ActivationType(-1))

	fmt.Printf("    ref preAct: %v\n", refPreData)
	fmt.Printf("    fp4 preAct: %v\n", fp4Pre)

	allOK := true
	for i, r := range refPreData {
		g := fp4Pre[i]
		absErr := float32(math.Abs(float64(g - r)))
		relErr := float32(0)
		if r != 0 {
			relErr = float32(math.Abs(float64((g - r) / r)))
		}
		if absErr > 1.0 && relErr > 0.20 {
			fail(fmt.Sprintf("batch element[%d]", i),
				fmt.Sprintf("ref=%.4f fp4=%.4f (absErr=%.4f relErr=%.2f%%)",
					r, g, absErr, relErr*100))
			allOK = false
		}
	}
	if allOK {
		pass("DenseForwardFP4 output within 20% of float32 reference")
	}
}

// ─────────────────────────────────────────────────────────────────────────────
// 7. Micro-scale grouping — verify 16-element group boundary correctness
// ─────────────────────────────────────────────────────────────────────────────

func testMicroScaleGroups() {
	fmt.Println("\n[7] Micro-scale group boundary (32 weights, 2 groups)")

	const n = 32
	// Group 0: small values (scale ~1.0)
	// Group 1: large values (scale ~6.0)
	weights := make([]float32, n)
	for i := 0; i < 16; i++ {
		weights[i] = float32(i%8) * 0.1 // 0..0.7
	}
	for i := 16; i < 32; i++ {
		weights[i] = float32((i-16)%8) * 0.75 // 0..5.25
	}

	pw := nn.NewPackedWeights(weights, n, 1)

	if pw.NumRowGroups != 2 {
		fail("row-group count", fmt.Sprintf("want 2, got %d", pw.NumRowGroups))
		return
	}
	pass(fmt.Sprintf("two micro-scale row-groups created (scales[0]=%.4f, scales[1]=%.4f)",
		pw.Scales[0], pw.Scales[1]))

	// Group 1 scale should be significantly larger than group 0.
	if pw.Scales[1] > pw.Scales[0] {
		pass("group-1 scale > group-0 scale (large values produce larger scale)")
	} else {
		fail("scale ordering", fmt.Sprintf("group0=%.4f group1=%.4f expects group1 > group0", pw.Scales[0], pw.Scales[1]))
	}
}

// ─────────────────────────────────────────────────────────────────────────────
// main
// ─────────────────────────────────────────────────────────────────────────────

func main() {
	fmt.Println("════════════════════════════════════════════════════════════")
	fmt.Println("  NVFP4 (E2M1) — Pure-Go CPU Emulation Test Suite")
	fmt.Println("════════════════════════════════════════════════════════════")

	testDecodeTable()
	testMultiply()
	testPackRoundTrip()
	testQuantiseRow()
	testForwardRow()
	testDenseForwardFP4()
	testMicroScaleGroups()

	fmt.Println("\n════════════════════════════════════════════════════════════")
	fmt.Println("  Done.")
	fmt.Println("════════════════════════════════════════════════════════════")
}
