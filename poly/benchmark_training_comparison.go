package main

import (
	"fmt"
	"math/rand"
	"time"

	"github.com/openfluke/loom/poly"
)

func main() {
	fmt.Println("=== M-POLY-VTD Multi-Architecture Training Showdown ===")
	rand.Seed(42)

	// Pre-init GPU once and share across all tests to eliminate per-test init overhead.
	sharedNet := poly.NewVolumetricNetwork(1, 1, 1, 1)
	gpuInitStart := time.Now()
	gpuInitErr := sharedNet.InitWGPU()
	gpuInitTime := time.Since(gpuInitStart)
	if gpuInitErr != nil {
		fmt.Printf("⚠️  GPU unavailable: %v — running CPU-only\n", gpuInitErr)
	} else {
		fmt.Printf("✅ GPU initialised in %v (shared across all tests)\n\n", gpuInitTime)
	}
	sharedCtx := sharedNet.GPUContext // nil if GPU failed

	// All tests use the same config; only per-test batchSize/numBatches differ.
	baseConfig := &poly.TrainingConfig{
		Epochs:       20,
		LearningRate: 0.01,
		LossType:     "mse",
		Verbose:      false,
	}

	fmt.Println("Layers tested: Dense MLP · CNN 1D · CNN 2D · CNN 3D · RMSNorm MLP · Deep Transformer MLP")
	fmt.Println("(SwiGLU / MHA / Embedding: GPU backward not yet in DispatchBackwardLayer — skipped here)")
	fmt.Println()

	// ── 1. Large Dense MLP ──────────────────────────────────────────────────
	runTest("Dense MLP (128→512→512→8)",
		func() *poly.VolumetricNetwork { return createLargeDenseNet() },
		func(nb, bs int) []poly.TrainingBatch[float32] { return synth(nb, bs, []int{128}, []int{8}) },
		baseConfig, sharedCtx,
		8,  // numBatches
		64, // batchSize
	)

	// ── 2. Large CNN 1D ──────────────────────────────────────────────────────
	runTest("CNN 1D (3ch×128→32f→64f→Dense→8)",
		func() *poly.VolumetricNetwork { return createLargeCNN1DNet() },
		func(nb, bs int) []poly.TrainingBatch[float32] { return synth(nb, bs, []int{3, 128}, []int{8}) },
		baseConfig, sharedCtx,
		8, 32,
	)

	// ── 3. Large CNN 2D ──────────────────────────────────────────────────────
	runTest("CNN 2D (3ch×32×32→16f→32f→Dense→8)",
		func() *poly.VolumetricNetwork { return createLargeCNN2DNet() },
		func(nb, bs int) []poly.TrainingBatch[float32] { return synth(nb, bs, []int{3, 32, 32}, []int{8}) },
		baseConfig, sharedCtx,
		8, 16,
	)

	// ── 4. Large CNN 3D ──────────────────────────────────────────────────────
	runTest("CNN 3D (2ch×8×8×8→8f→Dense→8)",
		func() *poly.VolumetricNetwork { return createLargeCNN3DNet() },
		func(nb, bs int) []poly.TrainingBatch[float32] { return synth(nb, bs, []int{2, 8, 8, 8}, []int{8}) },
		baseConfig, sharedCtx,
		8, 8,
	)

	// ── 5. RMSNorm MLP ───────────────────────────────────────────────────────
	runTest("RMSNorm MLP (128→Dense512→Norm→Dense512→8)",
		func() *poly.VolumetricNetwork { return createRMSNormNet() },
		func(nb, bs int) []poly.TrainingBatch[float32] { return synth(nb, bs, []int{128}, []int{8}) },
		baseConfig, sharedCtx,
		8, 64,
	)

	// ── 6. Deep Dense MLP ────────────────────────────────────────────────────
	runTest("Deep Dense MLP (128→512→512→512→512→8)",
		func() *poly.VolumetricNetwork { return createDeepDenseNet() },
		func(nb, bs int) []poly.TrainingBatch[float32] { return synth(nb, bs, []int{128}, []int{8}) },
		baseConfig, sharedCtx,
		8, 64,
	)
}

// ── Runner ────────────────────────────────────────────────────────────────────

func runTest(
	name string,
	netFn func() *poly.VolumetricNetwork,
	dataFn func(int, int) []poly.TrainingBatch[float32],
	cfg *poly.TrainingConfig,
	sharedCtx *poly.WGPUContext,
	numBatches, batchSize int,
) {
	fmt.Printf("--- %s ---\n", name)

	batches := dataFn(numBatches, batchSize)

	// CPU
	nCPU := netFn()
	cfg.UseGPU = false
	tCPU := time.Now()
	resCPU, err := poly.Train(nCPU, batches, cfg)
	cpuDur := time.Since(tCPU)
	if err != nil {
		fmt.Printf("  CPU error: %v\n\n", err)
		return
	}

	// GPU — share the pre-initialised context
	if sharedCtx == nil {
		fmt.Printf("  CPU: %v | GPU: skipped (no device)\n\n", cpuDur)
		return
	}
	nGPU := netFn()
	// Copy weights so both networks start identically
	for i := range nCPU.Layers {
		if nCPU.Layers[i].WeightStore != nil && nGPU.Layers[i].WeightStore != nil {
			copy(nGPU.Layers[i].WeightStore.Master, nCPU.Layers[i].WeightStore.Master)
		}
	}
	nGPU.GPUContext = sharedCtx
	nGPU.UseGPU = true
	// SyncToGPU so weights land in VRAM before training starts
	if err := nGPU.SyncToGPU(); err != nil {
		fmt.Printf("  GPU sync error: %v\n\n", err)
		return
	}

	cfg.UseGPU = true
	tGPU := time.Now()
	resGPU, err := poly.Train(nGPU, batches, cfg)
	gpuDur := time.Since(tGPU)
	cfg.UseGPU = false
	if err != nil {
		fmt.Printf("  GPU error: %v\n\n", err)
		return
	}

	// Results
	initLoss := resCPU.LossHistory[0]
	finalCPU := resCPU.LossHistory[len(resCPU.LossHistory)-1]
	finalGPU := resGPU.LossHistory[len(resGPU.LossHistory)-1]
	cpuImprv := 0.0
	gpuImprv := 0.0
	if initLoss > 0 {
		cpuImprv = (initLoss - finalCPU) / initLoss * 100
		gpuImprv = (initLoss - finalGPU) / initLoss * 100
	}
	speedup := float64(cpuDur) / float64(gpuDur)

	fmt.Printf("  | %-12s | %-14s | %-14s | %-8s |\n", "Metric", "CPU", "GPU", "")
	fmt.Printf("  | %-12s | %-14v | %-14v | Speedup: %.2fx |\n", "Time", cpuDur.Round(time.Millisecond), gpuDur.Round(time.Millisecond), speedup)
	fmt.Printf("  | %-12s | %-14.6f | %-14.6f | CPU: %+.1f%% / GPU: %+.1f%% |\n", "Final Loss", finalCPU, finalGPU, cpuImprv, gpuImprv)
	fmt.Println()
}

// ── Network Factories ─────────────────────────────────────────────────────────

func createLargeDenseNet() *poly.VolumetricNetwork {
	n := poly.NewVolumetricNetwork(1, 1, 1, 3)
	setupDenseL(n.GetLayer(0, 0, 0, 0), 128, 512, poly.ActivationReLU)
	setupDenseL(n.GetLayer(0, 0, 0, 1), 512, 512, poly.ActivationReLU)
	setupDenseL(n.GetLayer(0, 0, 0, 2), 512, 8, poly.ActivationLinear)
	randNet(n)
	return n
}

func createDeepDenseNet() *poly.VolumetricNetwork {
	n := poly.NewVolumetricNetwork(1, 1, 1, 5)
	setupDenseL(n.GetLayer(0, 0, 0, 0), 128, 512, poly.ActivationReLU)
	setupDenseL(n.GetLayer(0, 0, 0, 1), 512, 512, poly.ActivationReLU)
	setupDenseL(n.GetLayer(0, 0, 0, 2), 512, 512, poly.ActivationReLU)
	setupDenseL(n.GetLayer(0, 0, 0, 3), 512, 512, poly.ActivationReLU)
	setupDenseL(n.GetLayer(0, 0, 0, 4), 512, 8, poly.ActivationLinear)
	randNet(n)
	return n
}

func createLargeCNN1DNet() *poly.VolumetricNetwork {
	n := poly.NewVolumetricNetwork(1, 1, 1, 3)

	l0 := n.GetLayer(0, 0, 0, 0)
	l0.Type = poly.LayerCNN1
	l0.InputChannels = 3
	l0.InputHeight = 128
	l0.Filters = 32
	l0.KernelSize = 3
	l0.Stride = 1
	l0.Padding = 1
	l0.OutputHeight = 128
	l0.Activation = poly.ActivationReLU
	l0.WeightStore = poly.NewWeightStore(32 * 3 * 3)

	l1 := n.GetLayer(0, 0, 0, 1)
	l1.Type = poly.LayerCNN1
	l1.InputChannels = 32
	l1.InputHeight = 128
	l1.Filters = 64
	l1.KernelSize = 3
	l1.Stride = 1
	l1.Padding = 1
	l1.OutputHeight = 128
	l1.Activation = poly.ActivationReLU
	l1.WeightStore = poly.NewWeightStore(64 * 32 * 3)

	setupDenseL(n.GetLayer(0, 0, 0, 2), 64*128, 8, poly.ActivationLinear)
	randNet(n)
	return n
}

func createLargeCNN2DNet() *poly.VolumetricNetwork {
	n := poly.NewVolumetricNetwork(1, 1, 1, 3)

	l0 := n.GetLayer(0, 0, 0, 0)
	l0.Type = poly.LayerCNN2
	l0.InputChannels = 3
	l0.InputHeight = 32
	l0.InputWidth = 32
	l0.Filters = 16
	l0.KernelSize = 3
	l0.Stride = 1
	l0.Padding = 1
	l0.OutputHeight = 32
	l0.OutputWidth = 32
	l0.Activation = poly.ActivationReLU
	l0.WeightStore = poly.NewWeightStore(16 * 3 * 3 * 3)

	l1 := n.GetLayer(0, 0, 0, 1)
	l1.Type = poly.LayerCNN2
	l1.InputChannels = 16
	l1.InputHeight = 32
	l1.InputWidth = 32
	l1.Filters = 32
	l1.KernelSize = 3
	l1.Stride = 1
	l1.Padding = 1
	l1.OutputHeight = 32
	l1.OutputWidth = 32
	l1.Activation = poly.ActivationReLU
	l1.WeightStore = poly.NewWeightStore(32 * 16 * 3 * 3)

	setupDenseL(n.GetLayer(0, 0, 0, 2), 32*32*32, 8, poly.ActivationLinear)
	randNet(n)
	return n
}

func createLargeCNN3DNet() *poly.VolumetricNetwork {
	n := poly.NewVolumetricNetwork(1, 1, 1, 2)

	l0 := n.GetLayer(0, 0, 0, 0)
	l0.Type = poly.LayerCNN3
	l0.InputChannels = 2
	l0.InputDepth = 8
	l0.InputHeight = 8
	l0.InputWidth = 8
	l0.Filters = 8
	l0.KernelSize = 3
	l0.Stride = 1
	l0.Padding = 1
	l0.OutputDepth = 8
	l0.OutputHeight = 8
	l0.OutputWidth = 8
	l0.Activation = poly.ActivationReLU
	l0.WeightStore = poly.NewWeightStore(8 * 2 * 3 * 3 * 3)

	setupDenseL(n.GetLayer(0, 0, 0, 1), 8*8*8*8, 8, poly.ActivationLinear)
	randNet(n)
	return n
}

func createRMSNormNet() *poly.VolumetricNetwork {
	n := poly.NewVolumetricNetwork(1, 1, 1, 4)
	setupDenseL(n.GetLayer(0, 0, 0, 0), 128, 512, poly.ActivationLinear)

	lNorm := n.GetLayer(0, 0, 0, 1)
	lNorm.Type = poly.LayerRMSNorm
	lNorm.InputHeight = 512
	lNorm.OutputHeight = 512
	lNorm.WeightStore = poly.NewWeightStore(512)

	setupDenseL(n.GetLayer(0, 0, 0, 2), 512, 512, poly.ActivationReLU)
	setupDenseL(n.GetLayer(0, 0, 0, 3), 512, 8, poly.ActivationLinear)
	randNet(n)
	return n
}

// ── Helpers ───────────────────────────────────────────────────────────────────

func setupDenseL(l *poly.VolumetricLayer, in, out int, act poly.ActivationType) {
	l.Type = poly.LayerDense
	l.InputHeight = in
	l.OutputHeight = out
	l.Activation = act
	l.WeightStore = poly.NewWeightStore(in*out + out)
}

func randNet(n *poly.VolumetricNetwork) {
	for i := range n.Layers {
		l := &n.Layers[i]
		if l.WeightStore != nil {
			for j := range l.WeightStore.Master {
				l.WeightStore.Master[j] = (rand.Float32()*2 - 1) * 0.05
			}
		}
	}
}

func synth(numBatches, batchSize int, inShape, targetShape []int) []poly.TrainingBatch[float32] {
	batches := make([]poly.TrainingBatch[float32], numBatches)
	for b := 0; b < numBatches; b++ {
		fullIn := append([]int{batchSize}, inShape...)
		fullTgt := append([]int{batchSize}, targetShape...)
		inp := poly.NewTensor[float32](fullIn...)
		tgt := poly.NewTensor[float32](fullTgt...)
		for i := range inp.Data {
			inp.Data[i] = rand.Float32()*2 - 1
		}
		for i := range tgt.Data {
			tgt.Data[i] = rand.Float32()*2 - 1
		}
		batches[b] = poly.TrainingBatch[float32]{Input: inp, Target: tgt}
	}
	return batches
}
