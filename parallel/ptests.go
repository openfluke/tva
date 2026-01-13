package main

import (
	"fmt"
	"math"
	"math/rand"
	"os"

	"github.com/openfluke/loom/nn"
)

const linearActivation = nn.ActivationType(-1)

type testCase struct {
	name string
	run  func() error
}

func main() {
	rand.Seed(1)

	tests := []testCase{
		{name: "concat forward (dense)", run: testConcatForwardDense},
		{name: "add forward/backward (dense)", run: testAddForwardBackwardDense},
		{name: "avg forward/backward (dense)", run: testAvgForwardBackwardDense},
		{name: "grid_scatter forward/backward (dense)", run: testGridScatterForwardBackwardDense},
		{name: "concat backward includes Conv1D grads", run: testConcatBackwardConv1DGradSplit},
		{name: "grid_scatter backward includes Conv1D grads", run: testGridScatterBackwardConv1DGradSplit},
		{name: "filter forward (gated weighted sum)", run: testFilterForwardGating},
		{name: "filter backward uses gate weights", run: testFilterBackwardUsesGateWeights},
	}

	passed := 0
	failed := 0

	for _, tc := range tests {
		if err := tc.run(); err != nil {
			fmt.Printf("FAIL: %s: %v\n", tc.name, err)
			failed++
		} else {
			fmt.Printf("PASS: %s\n", tc.name)
			passed++
		}
	}

	fmt.Printf("\nResult: %d passed, %d failed\n", passed, failed)
	if failed > 0 {
		os.Exit(1)
	}
}

func testConcatForwardDense() error {
	branch0 := nn.InitDenseLayer(3, 2, linearActivation)
	setDenseWeights(&branch0, []float32{
		1, 0,
		0, 1,
		0, 0,
	}, []float32{0, 0})

	branch1 := nn.InitDenseLayer(3, 1, linearActivation)
	setDenseWeights(&branch1, []float32{
		1,
		1,
		1,
	}, []float32{0})

	parallel := nn.LayerConfig{
		Type:             nn.LayerParallel,
		CombineMode:      "concat",
		ParallelBranches: []nn.LayerConfig{branch0, branch1},
	}

	net := buildParallelNet(3, parallel)
	input := []float32{1, 2, 3}
	out, _ := net.ForwardCPU(input)

	expected := []float32{1, 2, 6}
	if !approxEqualSlice(out, expected, 1e-5) {
		return fmt.Errorf("unexpected concat output: got %v want %v", out, expected)
	}
	return nil
}

func testAddForwardBackwardDense() error {
	branch0 := nn.InitDenseLayer(2, 2, linearActivation)
	setDenseWeights(&branch0, []float32{
		1, 0,
		0, 1,
	}, []float32{0, 0})

	branch1 := nn.InitDenseLayer(2, 2, linearActivation)
	setDenseWeights(&branch1, []float32{
		1, 1,
		1, 1,
	}, []float32{0, 0})

	parallel := nn.LayerConfig{
		Type:             nn.LayerParallel,
		CombineMode:      "add",
		ParallelBranches: []nn.LayerConfig{branch0, branch1},
	}

	net := buildParallelNet(2, parallel)
	input := []float32{1, 2}
	out, _ := net.ForwardCPU(input)

	expected := []float32{4, 5}
	if !approxEqualSlice(out, expected, 1e-5) {
		return fmt.Errorf("unexpected add output: got %v want %v", out, expected)
	}

	before0 := cloneSlice(net.Layers[0].ParallelBranches[0].Kernel)
	before1 := cloneSlice(net.Layers[0].ParallelBranches[1].Kernel)

	net.BackwardCPU([]float32{1, 1})
	net.UpdateWeights(0.1)

	after0 := net.Layers[0].ParallelBranches[0].Kernel
	after1 := net.Layers[0].ParallelBranches[1].Kernel

	if sumAbsDiff(before0, after0) <= 0 {
		return fmt.Errorf("branch 0 weights did not change")
	}
	if sumAbsDiff(before1, after1) <= 0 {
		return fmt.Errorf("branch 1 weights did not change")
	}

	return nil
}

func testAvgForwardBackwardDense() error {
	branch0 := nn.InitDenseLayer(2, 2, linearActivation)
	setDenseWeights(&branch0, []float32{
		1, 0,
		0, 1,
	}, []float32{0, 0})

	branch1 := nn.InitDenseLayer(2, 2, linearActivation)
	setDenseWeights(&branch1, []float32{
		1, 1,
		1, 1,
	}, []float32{0, 0})

	parallel := nn.LayerConfig{
		Type:             nn.LayerParallel,
		CombineMode:      "avg",
		ParallelBranches: []nn.LayerConfig{branch0, branch1},
	}

	net := buildParallelNet(2, parallel)
	input := []float32{1, 2}
	out, _ := net.ForwardCPU(input)

	expected := []float32{2, 2.5}
	if !approxEqualSlice(out, expected, 1e-5) {
		return fmt.Errorf("unexpected avg output: got %v want %v", out, expected)
	}

	before0 := cloneSlice(net.Layers[0].ParallelBranches[0].Kernel)
	before1 := cloneSlice(net.Layers[0].ParallelBranches[1].Kernel)

	net.BackwardCPU([]float32{1, 1})
	net.UpdateWeights(0.1)

	after0 := net.Layers[0].ParallelBranches[0].Kernel
	after1 := net.Layers[0].ParallelBranches[1].Kernel

	if sumAbsDiff(before0, after0) <= 0 {
		return fmt.Errorf("branch 0 weights did not change")
	}
	if sumAbsDiff(before1, after1) <= 0 {
		return fmt.Errorf("branch 1 weights did not change")
	}

	return nil
}

func testGridScatterForwardBackwardDense() error {
	branch0 := nn.InitDenseLayer(2, 2, linearActivation)
	setDenseWeights(&branch0, []float32{
		1, 0,
		0, 1,
	}, []float32{0, 0})

	branch1 := nn.InitDenseLayer(2, 1, linearActivation)
	setDenseWeights(&branch1, []float32{
		1,
		1,
	}, []float32{0})

	parallel := nn.LayerConfig{
		Type:             nn.LayerParallel,
		CombineMode:      "grid_scatter",
		GridOutputRows:   1,
		GridOutputCols:   2,
		GridOutputLayers: 1,
		GridPositions: []nn.GridPosition{
			{BranchIndex: 0, TargetRow: 0, TargetCol: 0, TargetLayer: 0},
			{BranchIndex: 1, TargetRow: 0, TargetCol: 1, TargetLayer: 0},
		},
		ParallelBranches: []nn.LayerConfig{branch0, branch1},
	}

	net := buildParallelNet(2, parallel)
	input := []float32{1, 2}
	out, _ := net.ForwardCPU(input)

	expected := []float32{1, 2, 3}
	if !approxEqualSlice(out, expected, 1e-5) {
		return fmt.Errorf("unexpected grid_scatter output: got %v want %v", out, expected)
	}

	before0 := cloneSlice(net.Layers[0].ParallelBranches[0].Kernel)
	before1 := cloneSlice(net.Layers[0].ParallelBranches[1].Kernel)

	net.BackwardCPU([]float32{1, 1, 1})
	net.UpdateWeights(0.1)

	after0 := net.Layers[0].ParallelBranches[0].Kernel
	after1 := net.Layers[0].ParallelBranches[1].Kernel

	if sumAbsDiff(before0, after0) <= 0 {
		return fmt.Errorf("branch 0 weights did not change")
	}
	if sumAbsDiff(before1, after1) <= 0 {
		return fmt.Errorf("branch 1 weights did not change")
	}

	return nil
}

func testConcatBackwardConv1DGradSplit() error {
	conv := nn.InitConv1DLayer(4, 1, 3, 1, 1, 1, linearActivation)
	setConv1DKernel(&conv, []float32{1, 1, 1}, []float32{0})

	dense := nn.InitDenseLayer(4, 2, linearActivation)
	setDenseWeights(&dense, []float32{
		1, 1,
		1, 1,
		1, 1,
		1, 1,
	}, []float32{0, 0})

	parallel := nn.LayerConfig{
		Type:             nn.LayerParallel,
		CombineMode:      "concat",
		ParallelBranches: []nn.LayerConfig{conv, dense},
	}

	net := buildParallelNet(4, parallel)
	input := []float32{1, 2, 3, 4}
	out, _ := net.ForwardCPU(input)

	grad := make([]float32, len(out))
	for i := range grad {
		grad[i] = 1
	}

	net.BackwardCPU(grad)
	grads := net.KernelGradients()
	if grads == nil || len(grads) == 0 || grads[0] == nil || len(grads[0]) == 0 {
		return fmt.Errorf("no kernel grads produced for Conv1D concat backward")
	}

	return nil
}

func testGridScatterBackwardConv1DGradSplit() error {
	conv := nn.InitConv1DLayer(4, 1, 3, 1, 1, 1, linearActivation)
	setConv1DKernel(&conv, []float32{1, 1, 1}, []float32{0})

	dense := nn.InitDenseLayer(4, 2, linearActivation)
	setDenseWeights(&dense, []float32{
		1, 1,
		1, 1,
		1, 1,
		1, 1,
	}, []float32{0, 0})

	parallel := nn.LayerConfig{
		Type:             nn.LayerParallel,
		CombineMode:      "grid_scatter",
		GridOutputRows:   1,
		GridOutputCols:   2,
		GridOutputLayers: 1,
		GridPositions: []nn.GridPosition{
			{BranchIndex: 0, TargetRow: 0, TargetCol: 0, TargetLayer: 0},
			{BranchIndex: 1, TargetRow: 0, TargetCol: 1, TargetLayer: 0},
		},
		ParallelBranches: []nn.LayerConfig{conv, dense},
	}

	net := buildParallelNet(4, parallel)
	input := []float32{1, 2, 3, 4}
	out, _ := net.ForwardCPU(input)

	grad := make([]float32, len(out))
	for i := range grad {
		grad[i] = 1
	}

	net.BackwardCPU(grad)
	grads := net.KernelGradients()
	if grads == nil || len(grads) == 0 || grads[0] == nil || len(grads[0]) == 0 {
		return fmt.Errorf("no kernel grads produced for Conv1D grid_scatter backward")
	}

	return nil
}

func testFilterForwardGating() error {
	branch0 := nn.InitDenseLayer(2, 2, linearActivation)
	setDenseWeights(&branch0, []float32{
		1, 0,
		0, 1,
	}, []float32{0, 0})

	branch1 := nn.InitDenseLayer(2, 2, linearActivation)
	setDenseWeights(&branch1, []float32{
		0, 0,
		0, 0,
	}, []float32{0, 0})

	gate := nn.InitDenseLayer(2, 2, linearActivation)
	setDenseWeights(&gate, []float32{
		0, 0,
		0, 0,
	}, []float32{2, 0})

	parallel := nn.LayerConfig{
		Type:              nn.LayerParallel,
		CombineMode:       "filter",
		ParallelBranches:  []nn.LayerConfig{branch0, branch1},
		FilterGateConfig:  &gate,
		FilterSoftmax:     nn.SoftmaxStandard,
		FilterTemperature: 1,
	}

	net := buildParallelNet(2, parallel)
	input := []float32{1, 2}
	out, _ := net.ForwardCPU(input)

	gateWeight0 := softmax2(2, 0)
	expected := []float32{input[0] * gateWeight0, input[1] * gateWeight0}
	if !approxEqualSlice(out, expected, 1e-5) {
		return fmt.Errorf("unexpected filter output: got %v want %v", out, expected)
	}

	return nil
}

func testFilterBackwardUsesGateWeights() error {
	branch0 := nn.InitDenseLayer(2, 2, linearActivation)
	setDenseWeights(&branch0, []float32{
		1, 0,
		0, 1,
	}, []float32{0, 0})

	branch1 := nn.InitDenseLayer(2, 2, linearActivation)
	setDenseWeights(&branch1, []float32{
		1, 0,
		0, 1,
	}, []float32{0, 0})

	gate := nn.InitDenseLayer(2, 2, linearActivation)
	setDenseWeights(&gate, []float32{
		0, 0,
		0, 0,
	}, []float32{2, 0})

	parallel := nn.LayerConfig{
		Type:              nn.LayerParallel,
		CombineMode:       "filter",
		ParallelBranches:  []nn.LayerConfig{branch0, branch1},
		FilterGateConfig:  &gate,
		FilterSoftmax:     nn.SoftmaxStandard,
		FilterTemperature: 1,
	}

	net := buildParallelNet(2, parallel)
	input := []float32{1, 2}
	net.ForwardCPU(input)

	before0 := cloneSlice(net.Layers[0].ParallelBranches[0].Kernel)
	before1 := cloneSlice(net.Layers[0].ParallelBranches[1].Kernel)

	net.BackwardCPU([]float32{1, 1})
	net.UpdateWeights(0.1)

	after0 := net.Layers[0].ParallelBranches[0].Kernel
	after1 := net.Layers[0].ParallelBranches[1].Kernel

	delta0 := sumAbsDiff(before0, after0)
	delta1 := sumAbsDiff(before1, after1)

	gateWeight0 := softmax2(2, 0)
	gateWeight1 := 1 - gateWeight0
	if math.Abs(float64(gateWeight0-gateWeight1)) < 0.1 {
		return fmt.Errorf("gate weights unexpectedly uniform: w0=%.4f w1=%.4f", gateWeight0, gateWeight1)
	}

	if math.Abs(float64(delta0-delta1)) < 1e-6 {
		return fmt.Errorf("branch updates identical; expected gate-weighted grads (delta0=%.6f delta1=%.6f)", delta0, delta1)
	}

	return nil
}

func buildParallelNet(inputSize int, layer nn.LayerConfig) *nn.Network {
	net := nn.NewNetwork(inputSize, 1, 1, 1)
	net.BatchSize = 1
	net.SetLayer(0, 0, 0, layer)
	return net
}

func setDenseWeights(cfg *nn.LayerConfig, weights, bias []float32) {
	cfg.Kernel = cloneSlice(weights)
	cfg.Bias = cloneSlice(bias)
}

func setConv1DKernel(cfg *nn.LayerConfig, kernel, bias []float32) {
	cfg.Kernel = cloneSlice(kernel)
	cfg.Bias = cloneSlice(bias)
}

func cloneSlice(in []float32) []float32 {
	out := make([]float32, len(in))
	copy(out, in)
	return out
}

func sumAbsDiff(a, b []float32) float32 {
	if len(a) != len(b) {
		return float32(math.Inf(1))
	}
	var sum float32
	for i := range a {
		diff := a[i] - b[i]
		if diff < 0 {
			diff = -diff
		}
		sum += diff
	}
	return sum
}

func approxEqualSlice(a, b []float32, tol float32) bool {
	if len(a) != len(b) {
		return false
	}
	for i := range a {
		diff := a[i] - b[i]
		if diff < 0 {
			diff = -diff
		}
		if diff > tol {
			return false
		}
	}
	return true
}

func softmax2(a, b float32) float32 {
	max := a
	if b > max {
		max = b
	}
	ea := math.Exp(float64(a - max))
	eb := math.Exp(float64(b - max))
	return float32(ea / (ea + eb))
}
