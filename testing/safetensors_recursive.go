package main

import (
	"fmt"
	"math"
	"math/rand"
	"os"
	"path/filepath"

	"github.com/openfluke/loom/nn"
)

func main() {
	fmt.Println("╔══════════════════════════════════════════════════════════════╗")
	fmt.Println("║   SafeTensors Recursive & Full Layer Support Test           ║")
	fmt.Println("╚══════════════════════════════════════════════════════════════╝")

	if err := runRecursiveTest(); err != nil {
		fmt.Printf("\n❌ TEST FAILED: %v\n", err)
		os.Exit(1)
	}

	fmt.Println("\n✅ TEST PASSED: All recursive and complex layers saved/loaded correctly.")
}

func runRecursiveTest() error {
	// 1. Construct a complex network with nested layers
	net := createComplexNetwork()

	// 2. Save to Safetensors
	tmpDir := os.TempDir()
	filepath := filepath.Join(tmpDir, "recursive_test_model.safetensors")
	defer os.Remove(filepath)

	fmt.Printf("Saving network to %s...\n", filepath)
	if err := net.SaveWeightsToSafetensors(filepath); err != nil {
		return fmt.Errorf("SaveWeightsToSafetensors failed: %w", err)
	}

	// 3. Load into a fresh network structure
	// We need to re-create the structure because SafeTensors doesn't store structure
	loadedNet := createComplexNetwork()

	// Reset weights in loadedNet to zero/random to ensure we are actually loading something
	resetWeights(loadedNet)

	fmt.Println("Loading network weights...")
	if err := loadedNet.LoadWeightsFromSafetensors(filepath); err != nil {
		return fmt.Errorf("LoadWeightsFromSafetensors failed: %w", err)
	}

	// 4. Verify Weights
	fmt.Println("Verifying weights...")
	if err := compareNetworks(net, loadedNet); err != nil {
		return fmt.Errorf("Comparison failed: %w", err)
	}

	return nil
}

func createComplexNetwork() *nn.Network {
	// Network: 1 row, 1 col, 1 layer (initially)
	// We will manually populate it to have specific complex layers
	net := nn.NewNetwork(1, 1, 1, 1)

	// Layer 0: Parallel Layer containing [Dense, Sequential[Conv1D, RNN]]

	// Branch 1: Dense
	dense := nn.InitDenseLayer(8, 4, nn.ActivationSigmoid)
	dense.Kernel = randomize(8 * 4)
	dense.Bias = randomize(4)

	// Branch 2: Sequential
	conv1d := nn.InitConv1DLayer(8, 1, 2, 1, 0, 1, nn.ActivationScaledReLU) // In=8, Out=2
	// InitConv1DLayer might not set all fields for direct use, let's manually ensure weights
	conv1d.Kernel = randomize(1 * 1 * 2) // Filters=1, InCh=1, Kern=2 matches Init arguments
	conv1d.Bias = randomize(1)

	rnn := nn.LayerConfig{
		Type:         nn.LayerRNN,
		HiddenSize:   4,
		RNNInputSize: 2, // Output of Conv1D
		WeightIH:     randomize(4 * 2),
		WeightHH:     randomize(4 * 4),
		BiasH:        randomize(4),
	}

	seq := nn.LayerConfig{
		Type:             nn.LayerSequential,
		ParallelBranches: []nn.LayerConfig{conv1d, rnn}, // Sequential reuses ParallelBranches field
	}

	parallel := nn.LayerConfig{
		Type:             nn.LayerParallel,
		CombineMode:      "concat",
		ParallelBranches: []nn.LayerConfig{dense, seq},
	}

	// Top-level MoE Layer (Filter Combine Mode)
	gate := nn.InitDenseLayer(4, 2, nn.ActivationSigmoid) // 2 branches
	gate.Kernel = randomize(4 * 2)
	gate.Bias = randomize(2)

	moe := nn.LayerConfig{
		Type:             nn.LayerParallel,
		CombineMode:      "filter",
		ParallelBranches: []nn.LayerConfig{parallel, parallel}, // Use parallel branch structure twice
		FilterGateConfig: &gate,
	}

	net.Layers[0] = moe

	return net
}

func randomize(n int) []float32 {
	d := make([]float32, n)
	for i := range d {
		d[i] = rand.Float32()*2 - 1
	}
	return d
}

func resetWeights(n *nn.Network) {
	// Zero out the first layer (Parallel) recursively
	// Since we know the structure, we can just be destructive
	// simpler: createComplexNetwork returns random weights.
	// We want 'loadedNet' to be different before loading.
	// But since createComplexNetwork calls randomize(), it IS different (assuming seed ticks).
	// We'll trust random() to produce different values.
}

func compareNetworks(n1, n2 *nn.Network) error {
	// Helper to compare float slices
	compare := func(path string, v1, v2 []float32) error {
		if len(v1) != len(v2) {
			return fmt.Errorf("%s length mismatch: %d vs %d", path, len(v1), len(v2))
		}
		for i := range v1 {
			if math.Abs(float64(v1[i]-v2[i])) > 1e-6 {
				return fmt.Errorf("%s mismatch at [%d]: %f vs %f", path, i, v1[i], v2[i])
			}
		}
		return nil
	}

	// Root Layer (MoE)
	moe1 := n1.Layers[0]
	moe2 := n2.Layers[0]

	if moe1.Type != nn.LayerParallel || moe2.Type != nn.LayerParallel {
		return fmt.Errorf("Top layer type mismatch (expected Parallel/MoE)")
	}

	// Check Gate Weights
	if err := compare("MoE.Gate.Weight", moe1.FilterGateConfig.Kernel, moe2.FilterGateConfig.Kernel); err != nil {
		return err
	}
	if err := compare("MoE.Gate.Bias", moe1.FilterGateConfig.Bias, moe2.FilterGateConfig.Bias); err != nil {
		return err
	}

	// Check Branch 0 (Parallel)
	l1 := moe1.ParallelBranches[0]
	l2 := moe2.ParallelBranches[0]

	// Parallel Layer
	if l1.Type != nn.LayerParallel || l2.Type != nn.LayerParallel {
		return fmt.Errorf("Branch 0 type mismatch")
	}

	// Branch 0.0 (Dense)
	b1_0 := l1.ParallelBranches[0]
	b2_0 := l2.ParallelBranches[0]
	if err := compare("Branch0.0.Dense.Weight", b1_0.Kernel, b2_0.Kernel); err != nil {
		return err
	}
	if err := compare("Branch0.0.Dense.Bias", b1_0.Bias, b2_0.Bias); err != nil {
		return err
	}

	// Branch 0.1 (Sequential)
	b1_1 := l1.ParallelBranches[1]
	b2_1 := l2.ParallelBranches[1]

	// Seq Layer 0 (Conv1D)
	seq1_0 := b1_1.ParallelBranches[0]
	seq2_0 := b2_1.ParallelBranches[0]
	if err := compare("Branch0.1.Seq0.Conv1D.Weight", seq1_0.Kernel, seq2_0.Kernel); err != nil {
		return err
	}
	if err := compare("Branch0.1.Seq0.Conv1D.Bias", seq1_0.Bias, seq2_0.Bias); err != nil {
		return err
	}

	// Seq Layer 1 (RNN)
	seq1_1 := b1_1.ParallelBranches[1]
	seq2_1 := b2_1.ParallelBranches[1]
	if err := compare("Branch0.1.Seq1.RNN.WeightIH", seq1_1.WeightIH, seq2_1.WeightIH); err != nil {
		return err
	}
	if err := compare("Branch0.1.Seq1.RNN.WeightHH", seq1_1.WeightHH, seq2_1.WeightHH); err != nil {
		return err
	}
	if err := compare("Branch0.1.Seq1.RNN.Bias", seq1_1.BiasH, seq2_1.BiasH); err != nil {
		return err
	}

	return nil
}
