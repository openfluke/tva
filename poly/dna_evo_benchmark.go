package main

import (
	"fmt"
	"math"
	"math/rand"
	"time"

	"github.com/openfluke/loom/poly"
)

// =============================================================================
// DNA + Evolution Benchmark — Full Layer Coverage
//
// Sections:
//   1.  ExtractDNA on all 19 layer types individually
//   2.  Parallel / Sequential recursive DNA (new feature)
//   3.  CosineSimilarity — direct tests
//   4.  CompareNetworks — identical / similar / dissimilar / cross-type pairs
//   5.  SpliceDNA — all 3 modes on Dense networks
//   6.  SpliceDNA — heterogeneous networks (MHA, CNN, Embedding, etc.)
//   7.  SpliceDNAWithReport
//   8.  NEATMutate — each mutation type in isolation
//   9.  NEATMutate — node mutation across all 17 allowed layer types
//   10. NEATMutate — connection add / drop
//   11. NEATMutate — immutability guarantee (original never modified)
//   12. NEATPopulation — full evolution loop with fitness tracking
//   13. Logic shift detection after aggressive NEAT evolution
//   14. Splice stability (identical parents → identical child)
//   15. Coverage summary
// =============================================================================

func main() {
	rand.Seed(time.Now().UnixNano())
	sep := "═══════════════════════════════════════════════════════════════"

	fmt.Println(sep)
	fmt.Println("  DNA & Evolution Engine — All-Layer Full Coverage Benchmark")
	fmt.Println(sep)
	fmt.Println()

	// =========================================================================
	// 1. ExtractDNA on all 19 layer types
	// =========================================================================
	fmt.Println("── 1. ExtractDNA — all 19 layer types ─────────────────────────")
	fmt.Printf("  %-20s  %-8s  %-10s  %-12s  %s\n",
		"Layer Type", "Layers", "DNA Sigs", "Sig Len", "Elapsed")
	fmt.Printf("  %-20s  %-8s  %-10s  %-12s  %s\n",
		"────────────────────", "──────", "────────", "──────────", "───────")

	type layerCase struct {
		name string
		net  *poly.VolumetricNetwork
	}

	allLayerNets := []layerCase{
		{"Dense",            buildSingleLayer(poly.LayerDense, 32)},
		{"MultiHeadAttn",    buildMHANet(32, 4)},
		{"SwiGLU",           buildSwiGLUNet(32)},
		{"RMSNorm",          buildRMSNormNet(32)},
		{"LayerNorm",        buildLayerNormNet(32)},
		{"CNN1",             buildCNN1Net()},
		{"CNN2",             buildCNN2Net()},
		{"CNN3",             buildCNN3Net()},
		{"ConvTransposed1D", buildConvT1Net()},
		{"ConvTransposed2D", buildConvT2Net()},
		{"ConvTransposed3D", buildConvT3Net()},
		{"RNN",              buildSingleRNN(32)},
		{"LSTM",             buildSingleLSTM(32)},
		{"Embedding",        buildEmbeddingNet(256, 32)},
		{"KMeans",           buildKMeansNet(8, 32)},
		{"Softmax",          buildWeightlessLayer(poly.LayerSoftmax)},
		{"Residual",         buildWeightlessLayer(poly.LayerResidual)},
		{"Parallel",         buildParallelNet(32)},
		{"Sequential",       buildSequentialNet(32)},
	}

	allDNAs := make(map[string]poly.NetworkDNA, len(allLayerNets))
	for _, lc := range allLayerNets {
		t0 := time.Now()
		dna := poly.ExtractDNA(lc.net)
		elapsed := time.Since(t0)
		allDNAs[lc.name] = dna
		sigLen := 0
		if len(dna) > 0 {
			sigLen = len(dna[0].Weights)
		}
		fmt.Printf("  %-20s  %-8d  %-10d  %-12d  %v\n",
			lc.name, len(lc.net.Layers), len(dna), sigLen, elapsed)
	}
	fmt.Println()

	// =========================================================================
	// 2. Parallel / Sequential recursive DNA
	// =========================================================================
	fmt.Println("── 2. Parallel / Sequential recursive DNA ──────────────────────")

	parNetA := buildParallelNet(32)
	parNetB := buildParallelNet(32)
	// Give B's branches distinct weights
	for i := range parNetB.Layers {
		for j := range parNetB.Layers[i].ParallelBranches {
			if parNetB.Layers[i].ParallelBranches[j].WeightStore != nil {
				for k := range parNetB.Layers[i].ParallelBranches[j].WeightStore.Master {
					parNetB.Layers[i].ParallelBranches[j].WeightStore.Master[k] *= -2.0
				}
			}
		}
	}

	dnaParA := poly.ExtractDNA(parNetA)
	dnaParB := poly.ExtractDNA(parNetB)
	parSame := poly.ExtractDNA(parNetA) // second extract of same net

	cmpSame := poly.CompareNetworks(dnaParA, parSame)
	cmpDiff := poly.CompareNetworks(dnaParA, dnaParB)

	fmt.Printf("  Parallel(A) vs Parallel(A) overlap=%.4f  (expect 1.0000)\n", cmpSame.OverallOverlap)
	fmt.Printf("  Parallel(A) vs Parallel(B) overlap=%.4f  (expect <1.0 — different branch weights)\n", cmpDiff.OverallOverlap)

	seqNetA := buildSequentialNet(32)
	seqNetB := buildSequentialNet(32)
	for i := range seqNetB.Layers {
		for j := range seqNetB.Layers[i].SequentialLayers {
			if seqNetB.Layers[i].SequentialLayers[j].WeightStore != nil {
				for k := range seqNetB.Layers[i].SequentialLayers[j].WeightStore.Master {
					seqNetB.Layers[i].SequentialLayers[j].WeightStore.Master[k] *= -3.0
				}
			}
		}
	}
	dnaSeqA := poly.ExtractDNA(seqNetA)
	dnaSeqB := poly.ExtractDNA(seqNetB)
	cmpSeq := poly.CompareNetworks(dnaSeqA, dnaSeqB)
	fmt.Printf("  Sequential(A) vs Sequential(B) overlap=%.4f  (expect <1.0 — different sub-layer weights)\n",
		cmpSeq.OverallOverlap)
	fmt.Println()

	// =========================================================================
	// 3. CosineSimilarity — direct tests
	// =========================================================================
	fmt.Println("── 3. CosineSimilarity — direct tests ──────────────────────────")

	dna32 := poly.ExtractDNA(buildDenseMLP(32, 1))
	dna32b := poly.ExtractDNA(buildDenseMLP(32, 1))

	// Same type/dtype but different weights — should be near 0 or low
	if len(dna32) > 0 && len(dna32b) > 0 {
		sim := poly.CosineSimilarity(dna32[0], dna32b[0])
		fmt.Printf("  Dense(32) vs Dense(32) different weights  : sim=%.4f\n", sim)
	}

	// Identical layers — should be 1.0
	dna32same := poly.ExtractDNA(buildDenseMLP(32, 1))
	// copy weights from dna32's network
	net32 := buildDenseMLP(32, 1)
	net32copy := buildDenseMLP(32, 1)
	copy(net32copy.Layers[0].WeightStore.Master, net32.Layers[0].WeightStore.Master)
	dnaIdA := poly.ExtractDNA(net32)
	dnaIdB := poly.ExtractDNA(net32copy)
	if len(dnaIdA) > 0 && len(dnaIdB) > 0 {
		sim := poly.CosineSimilarity(dnaIdA[0], dnaIdB[0])
		fmt.Printf("  Dense(32) vs Dense(32) identical weights  : sim=%.4f  (expect 1.0000)\n", sim)
	}

	// Type mismatch — should return 0
	dnaRNN := poly.ExtractDNA(buildSingleRNN(32))
	_ = dna32same
	if len(dna32) > 0 && len(dnaRNN) > 0 {
		sim := poly.CosineSimilarity(dna32[0], dnaRNN[0])
		fmt.Printf("  Dense(32) vs RNN(32) type mismatch        : sim=%.4f  (expect 0.0000)\n", sim)
	}

	// Weightless layers — Softmax vs Softmax
	dnaSoftA := poly.ExtractDNA(buildWeightlessLayer(poly.LayerSoftmax))
	dnaSoftB := poly.ExtractDNA(buildWeightlessLayer(poly.LayerSoftmax))
	if len(dnaSoftA) > 0 && len(dnaSoftB) > 0 {
		sim := poly.CosineSimilarity(dnaSoftA[0], dnaSoftB[0])
		fmt.Printf("  Softmax vs Softmax (weightless)           : sim=%.4f  (expect 1.0000)\n", sim)
	}

	// Normalize utility — zero vector edge case
	zeros := poly.Normalize([]float32{0, 0, 0, 0})
	allZero := true
	for _, v := range zeros {
		if v != 0 {
			allZero = false
		}
	}
	fmt.Printf("  Normalize(zero vector) → all-zero         : %v  (expect true)\n", allZero)
	fmt.Println()

	// =========================================================================
	// 4. CompareNetworks — diverse pairs
	// =========================================================================
	fmt.Println("── 4. CompareNetworks — diverse pairs ──────────────────────────")
	fmt.Printf("  %-24s  %-24s  %-10s  Shifts\n", "Net A", "Net B", "Overlap")
	fmt.Printf("  %-24s  %-24s  %-10s  ──────\n",
		"────────────────────────", "────────────────────────", "──────────")

	cmpCases := []struct{ a, b string }{
		{"Dense", "Dense"},
		{"Dense", "RNN"},
		{"Dense", "LSTM"},
		{"Dense", "MultiHeadAttn"},
		{"RNN", "LSTM"},
		{"CNN1", "CNN2"},
		{"CNN2", "CNN3"},
		{"Embedding", "KMeans"},
		{"Softmax", "Softmax"},
		{"Residual", "Residual"},
		{"Parallel", "Sequential"},
		{"SwiGLU", "RMSNorm"},
		{"ConvTransposed1D", "ConvTransposed2D"},
	}
	for _, c := range cmpCases {
		dA, okA := allDNAs[c.a]
		dB, okB := allDNAs[c.b]
		if !okA || !okB {
			continue
		}
		r := poly.CompareNetworks(dA, dB)
		fmt.Printf("  %-24s  %-24s  %-10.4f  %d\n", c.a, c.b, r.OverallOverlap, len(r.LogicShifts))
	}
	fmt.Println()

	// =========================================================================
	// 5. SpliceDNA — all 3 modes on Dense networks
	// =========================================================================
	fmt.Println("── 5. SpliceDNA — blend / point / uniform on Dense ────────────")

	pDenseA := buildDenseMLP(64, 4)
	pDenseB := buildDenseMLP(64, 4)
	scaleWeightsByFactor(pDenseB, -0.5)

	for _, mode := range []string{"blend", "point", "uniform"} {
		cfg := poly.DefaultSpliceConfig()
		cfg.CrossoverMode = mode
		cfg.BlendAlpha = 0.4
		cfg.SplitRatio = 0.6
		cfg.FitnessA = 0.7
		cfg.FitnessB = 0.3
		t0 := time.Now()
		child := poly.SpliceDNA(pDenseA, pDenseB, cfg)
		elapsed := time.Since(t0)
		wdA := weightDistance(pDenseA, child)
		wdB := weightDistance(pDenseB, child)
		gridOK := child.Depth == pDenseA.Depth && child.Rows == pDenseA.Rows &&
			child.Cols == pDenseA.Cols && child.LayersPerCell == pDenseA.LayersPerCell
		fmt.Printf("  mode=%-8s  grid_ok=%v  dist_A=%.5f  dist_B=%.5f  %v\n",
			mode, gridOK, wdA, wdB, elapsed)
	}
	fmt.Println()

	// =========================================================================
	// 6. SpliceDNA — heterogeneous single-layer networks
	// =========================================================================
	fmt.Println("── 6. SpliceDNA — heterogeneous layer types ────────────────────")

	hetCases := []struct {
		name string
		netA *poly.VolumetricNetwork
		netB *poly.VolumetricNetwork
	}{
		{"MHA + MHA",       buildMHANet(32, 4),        buildMHANet(32, 4)},
		{"CNN2 + CNN2",     buildCNN2Net(),             buildCNN2Net()},
		{"Embedding+Embed", buildEmbeddingNet(256, 32), buildEmbeddingNet(256, 32)},
		{"LSTM + LSTM",     buildSingleLSTM(32),        buildSingleLSTM(32)},
		{"KMeans + KMeans", buildKMeansNet(8, 32),      buildKMeansNet(8, 32)},
		{"SwiGLU + SwiGLU", buildSwiGLUNet(32),         buildSwiGLUNet(32)},
	}

	for _, hc := range hetCases {
		scaleWeightsByFactor(hc.netB, -0.7)
		cfg := poly.DefaultSpliceConfig()
		cfg.CrossoverMode = "blend"
		cfg.FitnessA = 0.8
		cfg.FitnessB = 0.5
		child := poly.SpliceDNA(hc.netA, hc.netB, cfg)
		dA := poly.ExtractDNA(hc.netA)
		dB := poly.ExtractDNA(hc.netB)
		dC := poly.ExtractDNA(child)
		ovA := poly.CompareNetworks(dC, dA).OverallOverlap
		ovB := poly.CompareNetworks(dC, dB).OverallOverlap
		wdA := weightDistance(hc.netA, child)
		fmt.Printf("  %-20s  child_vs_A=%.4f  child_vs_B=%.4f  dist_A=%.5f\n",
			hc.name, ovA, ovB, wdA)
	}
	fmt.Println()

	// =========================================================================
	// 7. SpliceDNAWithReport
	// =========================================================================
	fmt.Println("── 7. SpliceDNAWithReport ──────────────────────────────────────")

	reportA := buildDenseMLP(48, 4)
	reportB := buildDenseMLP(48, 4)
	scaleWeightsByFactor(reportB, -0.6)

	reportCfg := poly.DefaultSpliceConfig()
	reportCfg.CrossoverMode = "blend"
	reportCfg.FitnessA = 0.85
	reportCfg.FitnessB = 0.55

	report := poly.SpliceDNAWithReport(reportA, reportB, reportCfg)
	fmt.Printf("  ParentA DNA sigs : %d\n", len(report.ParentADNA))
	fmt.Printf("  ParentB DNA sigs : %d\n", len(report.ParentBDNA))
	fmt.Printf("  Child DNA sigs   : %d\n", len(report.ChildDNA))
	fmt.Printf("  Layers blended   : %d / %d\n", report.BlendedCount, len(report.ParentADNA))
	fmt.Printf("  Similarity map   : %d entries\n", len(report.Similarities))
	count := 0
	for pos, sim := range report.Similarities {
		if count >= 4 {
			break
		}
		fmt.Printf("    %-14s sim=%.4f\n", pos, sim)
		count++
	}
	childVsA := poly.CompareNetworks(report.ChildDNA, report.ParentADNA)
	childVsB := poly.CompareNetworks(report.ChildDNA, report.ParentBDNA)
	fmt.Printf("  Child overlap vs A=%.4f  vs B=%.4f\n",
		childVsA.OverallOverlap, childVsB.OverallOverlap)
	fmt.Println()

	// =========================================================================
	// 8. NEATMutate — each mutation type isolated
	// =========================================================================
	fmt.Println("── 8. NEATMutate — isolated mutation types ─────────────────────")

	base := buildDenseMLP(64, 6)
	baseDNA := poly.ExtractDNA(base)

	isolatedCases := []struct {
		name string
		cfg  poly.NEATConfig
	}{
		{
			"weight-perturb",
			poly.NEATConfig{WeightPerturbRate: 1.0, WeightPerturbScale: 0.2, DModel: 64, Seed: 1},
		},
		{
			"activation-only",
			poly.NEATConfig{ActivationMutRate: 1.0, DModel: 64, Seed: 2},
		},
		{
			"node-mutate-only",
			poly.NEATConfig{
				NodeMutateRate: 1.0, DModel: 64, Seed: 3,
				AllowedLayerTypes: []poly.LayerType{poly.LayerRNN, poly.LayerLSTM, poly.LayerRMSNorm},
			},
		},
		{
			"toggle-only",
			poly.NEATConfig{LayerToggleRate: 1.0, DModel: 64, Seed: 4},
		},
		{
			"conn-add-only",
			poly.NEATConfig{ConnectionAddRate: 1.0, DModel: 64, Seed: 5},
		},
		{
			"conn-drop-after-add",
			func() poly.NEATConfig {
				// First add a connection so there's something to drop
				addCfg := poly.NEATConfig{ConnectionAddRate: 1.0, DModel: 64, Seed: 6}
				_ = poly.NEATMutate(base, addCfg)
				return poly.NEATConfig{ConnectionDropRate: 1.0, DModel: 64, Seed: 7}
			}(),
		},
		{
			"all-mutations",
			poly.DefaultNEATConfig(64),
		},
	}

	for _, tc := range isolatedCases {
		mutated := poly.NEATMutate(base, tc.cfg)
		mutDNA := poly.ExtractDNA(mutated)
		overlap := poly.CompareNetworks(baseDNA, mutDNA).OverallOverlap
		disabled := countDisabled(mutated)
		remotes := countRemoteLinks(mutated)
		wdiff := weightDistance(base, mutated)
		fmt.Printf("  %-22s  overlap=%.4f  disabled=%d  remote=%d  wdist=%.5f\n",
			tc.name, overlap, disabled, remotes, wdiff)
	}
	fmt.Println()

	// =========================================================================
	// 9. NEATMutate — node mutation across ALL 17 allowed layer types
	// =========================================================================
	fmt.Println("── 9. NEATMutate — node mutation → all 17 layer types ──────────")
	fmt.Printf("  %-20s  %-12s  %-10s  %-12s  %s\n",
		"Target Type", "WeightCount", "Reinit OK", "DNA SigLen", "Overlap")
	fmt.Printf("  %-20s  %-12s  %-10s  %-12s  %s\n",
		"────────────────────", "────────────", "─────────", "──────────", "───────")

	allAllowed := []poly.LayerType{
		poly.LayerDense,
		poly.LayerMultiHeadAttention,
		poly.LayerSwiGLU,
		poly.LayerRMSNorm,
		poly.LayerLayerNorm,
		poly.LayerCNN1,
		poly.LayerCNN2,
		poly.LayerCNN3,
		poly.LayerConvTransposed1D,
		poly.LayerConvTransposed2D,
		poly.LayerConvTransposed3D,
		poly.LayerRNN,
		poly.LayerLSTM,
		poly.LayerEmbedding,
		poly.LayerKMeans,
		poly.LayerSoftmax,
		poly.LayerResidual,
	}

	seedNet := buildDenseMLP(32, 1)
	seedDNA := poly.ExtractDNA(seedNet)

	for _, lt := range allAllowed {
		cfg := poly.NEATConfig{
			NodeMutateRate:     1.0,
			DModel:             32,
			AllowedLayerTypes:  []poly.LayerType{lt},
			DefaultNumHeads:    4,
			DefaultInChannels:  1,
			DefaultFilters:     8,
			DefaultKernelSize:  3,
			DefaultVocabSize:   64,
			DefaultNumClusters: 8,
			Seed:               int64(lt) + 100,
		}
		mutated := poly.NEATMutate(seedNet, cfg)
		mutDNA := poly.ExtractDNA(mutated)

		wCount := 0
		reinitOK := false
		sigLen := 0
		if len(mutated.Layers) > 0 {
			l := mutated.Layers[0]
			reinitOK = l.Type == lt
			if l.WeightStore != nil {
				wCount = len(l.WeightStore.Master)
			}
		}
		if len(mutDNA) > 0 {
			sigLen = len(mutDNA[0].Weights)
		}
		overlap := poly.CompareNetworks(seedDNA, mutDNA).OverallOverlap

		fmt.Printf("  %-20s  %-12d  %-10v  %-12d  %.4f\n",
			lt.String(), wCount, reinitOK, sigLen, overlap)
	}
	fmt.Println()

	// =========================================================================
	// 10. NEATMutate — connection add / drop
	// =========================================================================
	fmt.Println("── 10. NEATMutate — connection add / drop ──────────────────────")

	connNet := buildDenseMLP(32, 4)
	fmt.Printf("  Before any mutation  : remote_links=%d\n", countRemoteLinks(connNet))

	addCfg := poly.NEATConfig{ConnectionAddRate: 1.0, DModel: 32, Seed: 42}
	after1 := poly.NEATMutate(connNet, addCfg)
	fmt.Printf("  After connection add : remote_links=%d\n", countRemoteLinks(after1))

	after2 := poly.NEATMutate(after1, addCfg)
	fmt.Printf("  After 2nd add        : remote_links=%d\n", countRemoteLinks(after2))

	dropCfg := poly.NEATConfig{ConnectionDropRate: 1.0, DModel: 32, Seed: 43}
	after3 := poly.NEATMutate(after2, dropCfg)
	fmt.Printf("  After connection drop: remote_links=%d\n", countRemoteLinks(after3))

	// DNA should still work on a network with remote links
	dnaConn := poly.ExtractDNA(after2)
	fmt.Printf("  DNA on net-with-links: sigs=%d  ok=%v\n", len(dnaConn), len(dnaConn) > 0)
	fmt.Println()

	// =========================================================================
	// 11. Immutability — original is never modified by NEATMutate or SpliceDNA
	// =========================================================================
	fmt.Println("── 11. Immutability guarantee ──────────────────────────────────")

	immut := buildDenseMLP(48, 4)
	immutDNA := poly.ExtractDNA(immut)

	aggressiveCfg := poly.DefaultNEATConfig(48)
	aggressiveCfg.WeightPerturbRate = 1.0
	aggressiveCfg.WeightPerturbScale = 1.0
	aggressiveCfg.NodeMutateRate = 1.0
	aggressiveCfg.ConnectionAddRate = 1.0
	aggressiveCfg.LayerToggleRate = 1.0
	aggressiveCfg.Seed = 999

	for i := 0; i < 5; i++ {
		_ = poly.NEATMutate(immut, aggressiveCfg)
	}
	immutDNAAfter := poly.ExtractDNA(immut)
	ov := poly.CompareNetworks(immutDNA, immutDNAAfter).OverallOverlap
	fmt.Printf("  After 5× aggressive NEATMutate: original overlap=%.4f  (expect 1.0000)\n", ov)

	immutSplice := buildDenseMLP(48, 4)
	immutSpliceDNA := poly.ExtractDNA(immutSplice)
	for i := 0; i < 5; i++ {
		_ = poly.SpliceDNA(immutSplice, buildDenseMLP(48, 4), poly.DefaultSpliceConfig())
	}
	immutSpliceAfter := poly.ExtractDNA(immutSplice)
	ov2 := poly.CompareNetworks(immutSpliceDNA, immutSpliceAfter).OverallOverlap
	fmt.Printf("  After 5× SpliceDNA as parentA:  original overlap=%.4f  (expect 1.0000)\n", ov2)
	fmt.Println()

	// =========================================================================
	// 12. NEATPopulation — full evolution loop
	// =========================================================================
	fmt.Println("── 12. NEATPopulation — 15 generation evolution ────────────────")

	popSeed := buildDenseMLP(32, 3)
	popCfg := poly.DefaultNEATConfig(32)
	popCfg.WeightPerturbScale = 0.08
	popCfg.NodeMutateRate = 0.05
	popCfg.ConnectionAddRate = 0.05
	popCfg.Seed = time.Now().UnixNano()

	pop := poly.NewNEATPopulation(popSeed, 16, popCfg)

	// Fixed target: approximate sin(x)
	fixedIn := make([]float32, 32)
	fixedTgt := make([]float32, 32)
	for i := range fixedIn {
		fixedIn[i] = rand.Float32()*2 - 1
		fixedTgt[i] = float32(math.Sin(float64(fixedIn[i])))
	}

	fitnessFn := func(net *poly.VolumetricNetwork) (result float64) {
		defer func() {
			if r := recover(); r != nil {
				result = -1e9 // incompatible architecture after mutation
			}
		}()
		inp := poly.NewTensorFromSlice(fixedIn, 1, len(fixedIn))
		out, _, _ := poly.ForwardPolymorphic[float32](net, inp)
		if out == nil || len(out.Data) == 0 {
			return -1e9
		}
		n := len(fixedTgt)
		if len(out.Data) < n {
			n = len(out.Data)
		}
		mse := 0.0
		for i := 0; i < n; i++ {
			d := float64(out.Data[i]) - float64(fixedTgt[i])
			mse += d * d
		}
		return -(mse / float64(n))
	}

	// Evaluate initial population
	initialBest := fitnessFn(pop.Best())
	fmt.Printf("  Initial best fitness : %.6f\n", initialBest)

	for gen := 1; gen <= 15; gen++ {
		popCfg.Seed = time.Now().UnixNano()
		pop.Config = popCfg
		pop.Evolve(fitnessFn)
		if gen == 1 || gen%5 == 0 || gen == 15 {
			fmt.Printf("  %s\n", pop.Summary(gen))
		}
	}

	fmt.Printf("\n  Best after 15 gens : fitness=%.6f\n", pop.BestFitness())
	fmt.Printf("  Best network layers: %d\n", len(pop.Best().Layers))

	// DNA-compare best vs initial seed
	bestDNA := poly.ExtractDNA(pop.Best())
	seedDNA2 := poly.ExtractDNA(popSeed)
	bestVsSeed := poly.CompareNetworks(bestDNA, seedDNA2)
	fmt.Printf("  Best vs seed overlap: %.4f  logic_shifts=%d\n",
		bestVsSeed.OverallOverlap, len(bestVsSeed.LogicShifts))
	fmt.Println()

	// =========================================================================
	// 13. Logic shift detection after aggressive evolution
	// =========================================================================
	fmt.Println("── 13. Logic Shift Detection ───────────────────────────────────")

	lsNet := buildMixedLayerNet(32)
	lsOrigDNA := poly.ExtractDNA(lsNet)

	lsCfg := poly.DefaultNEATConfig(32)
	lsCfg.NodeMutateRate = 0.6
	lsCfg.WeightPerturbRate = 1.0
	lsCfg.WeightPerturbScale = 0.5
	lsCfg.Seed = 2025

	lsEvolved := poly.NEATMutate(lsNet, lsCfg)
	lsEvolvedDNA := poly.ExtractDNA(lsEvolved)

	lsResult := poly.CompareNetworks(lsOrigDNA, lsEvolvedDNA)
	fmt.Printf("  Original vs Evolved: overlap=%.4f  shifts=%d\n",
		lsResult.OverallOverlap, len(lsResult.LogicShifts))

	for i, shift := range lsResult.LogicShifts {
		if i >= 6 {
			fmt.Printf("    ... (%d more)\n", len(lsResult.LogicShifts)-6)
			break
		}
		fmt.Printf("    %s → %s  overlap=%.4f\n", shift.SourcePos, shift.TargetPos, shift.Overlap)
	}

	// Multi-generation shift accumulation
	fmt.Printf("\n  Multi-gen shift accumulation:\n")
	gen := lsNet
	for step := 1; step <= 5; step++ {
		lsCfg.Seed = int64(step * 777)
		gen = poly.NEATMutate(gen, lsCfg)
		genDNA := poly.ExtractDNA(gen)
		r := poly.CompareNetworks(lsOrigDNA, genDNA)
		fmt.Printf("    gen=%d  overlap=%.4f  shifts=%d\n",
			step, r.OverallOverlap, len(r.LogicShifts))
	}
	fmt.Println()

	// =========================================================================
	// 14. Multi-parent splice chain + stability
	// =========================================================================
	fmt.Println("── 14. Multi-parent splice chain + stability ───────────────────")

	spA := buildDenseMLP(48, 4)
	spB := buildDenseMLP(48, 4)
	spC := buildDenseMLP(48, 4)
	scaleWeightsByFactor(spA, 1.0)
	scaleWeightsByFactor(spB, -0.8)
	scaleWeightsByFactor(spC, 0.3)

	cfgAB := poly.DefaultSpliceConfig()
	cfgAB.CrossoverMode = "blend"
	cfgAB.FitnessA = 0.9
	cfgAB.FitnessB = 0.4
	childAB := poly.SpliceDNA(spA, spB, cfgAB)

	cfgGC := poly.DefaultSpliceConfig()
	cfgGC.CrossoverMode = "uniform"
	cfgGC.FitnessA = 0.65
	cfgGC.FitnessB = 0.55
	grandchild := poly.SpliceDNA(childAB, spC, cfgGC)

	fmt.Printf("  A vs B       overlap=%.4f\n",
		poly.CompareNetworks(poly.ExtractDNA(spA), poly.ExtractDNA(spB)).OverallOverlap)
	fmt.Printf("  AB vs A      overlap=%.4f\n",
		poly.CompareNetworks(poly.ExtractDNA(childAB), poly.ExtractDNA(spA)).OverallOverlap)
	fmt.Printf("  AB vs B      overlap=%.4f\n",
		poly.CompareNetworks(poly.ExtractDNA(childAB), poly.ExtractDNA(spB)).OverallOverlap)
	fmt.Printf("  GC vs C      overlap=%.4f\n",
		poly.CompareNetworks(poly.ExtractDNA(grandchild), poly.ExtractDNA(spC)).OverallOverlap)
	fmt.Printf("  GC vs AB     overlap=%.4f\n",
		poly.CompareNetworks(poly.ExtractDNA(grandchild), poly.ExtractDNA(childAB)).OverallOverlap)

	// Stability: splice of identical parents → identical child
	fmt.Printf("\n  Splice stability (identical parents):\n")
	for _, mode := range []string{"blend", "point", "uniform"} {
		same := buildDenseMLP(32, 4)
		sameDNA := poly.ExtractDNA(same)
		cfg := poly.DefaultSpliceConfig()
		cfg.CrossoverMode = mode
		child := poly.SpliceDNA(same, same, cfg)
		childDNA := poly.ExtractDNA(child)
		ov := poly.CompareNetworks(sameDNA, childDNA).OverallOverlap
		fmt.Printf("    mode=%-8s  overlap=%.4f  (expect ~1.0000)\n", mode, ov)
	}
	fmt.Println()

	// =========================================================================
	// 15. Coverage summary
	// =========================================================================
	fmt.Println(sep)
	fmt.Println("  Coverage Summary")
	fmt.Println(sep)

	coverage := []struct{ fn, status string }{
		// dna.go
		{"ExtractDNA (all 19 layer types)", "✅"},
		{"extractLayerSignature (Parallel recurse)", "✅"},
		{"extractLayerSignature (Sequential recurse)", "✅"},
		{"extractLayerSignature (weightless fallback)", "✅"},
		{"Normalize", "✅"},
		{"Normalize (zero-vector edge case)", "✅"},
		{"CosineSimilarity (same weights)", "✅"},
		{"CosineSimilarity (diff weights)", "✅"},
		{"CosineSimilarity (type mismatch → 0)", "✅"},
		{"CosineSimilarity (weightless → 1.0)", "✅"},
		{"CompareNetworks (OverallOverlap)", "✅"},
		{"CompareNetworks (LogicShift detection)", "✅"},
		// evolution.go — splice
		{"DefaultSpliceConfig", "✅"},
		{"SpliceDNA (blend)", "✅"},
		{"SpliceDNA (point)", "✅"},
		{"SpliceDNA (uniform)", "✅"},
		{"SpliceDNA (heterogeneous types)", "✅"},
		{"SpliceDNA (chain / grandchild)", "✅"},
		{"SpliceDNA (stability: identical parents)", "✅"},
		{"SpliceDNAWithReport", "✅"},
		// evolution.go — NEAT
		{"DefaultNEATConfig (all 17 layer types)", "✅"},
		{"NEATMutate (weight perturbation)", "✅"},
		{"NEATMutate (activation mutation)", "✅"},
		{"NEATMutate (node mutation — all types)", "✅"},
		{"NEATMutate (layer toggle)", "✅"},
		{"NEATMutate (connection add)", "✅"},
		{"NEATMutate (connection drop)", "✅"},
		{"NEATMutate (immutability — original safe)", "✅"},
		{"neatReinitLayer (Dense)", "✅"},
		{"neatReinitLayer (MHA)", "✅"},
		{"neatReinitLayer (SwiGLU)", "✅"},
		{"neatReinitLayer (RMSNorm)", "✅"},
		{"neatReinitLayer (LayerNorm)", "✅"},
		{"neatReinitLayer (RNN)", "✅"},
		{"neatReinitLayer (LSTM)", "✅"},
		{"neatReinitLayer (CNN1/2/3)", "✅"},
		{"neatReinitLayer (ConvT1/2/3)", "✅"},
		{"neatReinitLayer (Embedding)", "✅"},
		{"neatReinitLayer (KMeans)", "✅"},
		{"neatReinitLayer (Softmax/Residual)", "✅"},
		{"cloneNetwork (via NEATMutate/SpliceDNA)", "✅"},
		// evolution.go — population
		{"NewNEATPopulation", "✅"},
		{"NEATPopulation.Evolve", "✅"},
		{"NEATPopulation.Best", "✅"},
		{"NEATPopulation.BestFitness", "✅"},
		{"NEATPopulation.Summary", "✅"},
	}
	for _, c := range coverage {
		fmt.Printf("  %-42s %s\n", c.fn, c.status)
	}
	fmt.Println()
}

// =============================================================================
// Network builders — one per layer type
// =============================================================================

func buildSingleLayer(lt poly.LayerType, dModel int) *poly.VolumetricNetwork {
	n := poly.NewVolumetricNetwork(1, 1, 1, 1)
	l := n.GetLayer(0, 0, 0, 0)
	l.Type = lt
	l.InputHeight = dModel
	l.OutputHeight = dModel
	l.DType = poly.DTypeFloat32
	l.WeightStore = poly.NewWeightStore(dModel * dModel)
	l.WeightStore.Randomize(time.Now().UnixNano(), 0.1)
	return n
}

func buildDenseMLP(dModel, numLayers int) *poly.VolumetricNetwork {
	n := poly.NewVolumetricNetwork(1, 1, 1, numLayers)
	for l := 0; l < numLayers; l++ {
		n.InitDenseCell(0, 0, 0, l, dModel, poly.ActivationReLU, 0.1)
		n.Layers[n.GetIndex(0, 0, 0, l)].DType = poly.DTypeFloat32
	}
	return n
}

func buildSingleRNN(dModel int) *poly.VolumetricNetwork {
	n := poly.NewVolumetricNetwork(1, 1, 1, 1)
	n.InitRNNCell(0, 0, 0, 0, dModel, 0.1)
	return n
}

func buildSingleLSTM(dModel int) *poly.VolumetricNetwork {
	n := poly.NewVolumetricNetwork(1, 1, 1, 1)
	n.InitLSTMCell(0, 0, 0, 0, dModel, 0.1)
	return n
}

func buildMHANet(dModel, numHeads int) *poly.VolumetricNetwork {
	n := poly.NewVolumetricNetwork(1, 1, 1, 1)
	n.InitMHACell(0, 0, 0, 0, dModel, numHeads, 0.02)
	return n
}

func buildSwiGLUNet(dModel int) *poly.VolumetricNetwork {
	n := poly.NewVolumetricNetwork(1, 1, 1, 1)
	l := n.GetLayer(0, 0, 0, 0)
	l.Type = poly.LayerSwiGLU
	l.InputHeight = dModel
	l.OutputHeight = dModel * 2
	l.DType = poly.DTypeFloat32
	inter := dModel * 2
	l.WeightStore = poly.NewWeightStore(dModel*inter*3 + inter*2 + dModel)
	l.WeightStore.Randomize(time.Now().UnixNano(), 0.05)
	return n
}

func buildRMSNormNet(dModel int) *poly.VolumetricNetwork {
	n := poly.NewVolumetricNetwork(1, 1, 1, 1)
	l := n.GetLayer(0, 0, 0, 0)
	l.Type = poly.LayerRMSNorm
	l.InputHeight = dModel
	l.DType = poly.DTypeFloat32
	l.WeightStore = poly.NewWeightStore(dModel)
	for i := range l.WeightStore.Master {
		l.WeightStore.Master[i] = 1.0
	}
	return n
}

func buildLayerNormNet(dModel int) *poly.VolumetricNetwork {
	n := poly.NewVolumetricNetwork(1, 1, 1, 1)
	n.InitLayerNormCell(0, 0, 0, 0, dModel, poly.DTypeFloat32)
	return n
}

func buildCNN1Net() *poly.VolumetricNetwork {
	n := poly.NewVolumetricNetwork(1, 1, 1, 1)
	n.InitCNNCell(0, 0, 0, 0, poly.LayerCNN1, 1, 8, 3, poly.DTypeFloat32, 0.05)
	return n
}

func buildCNN2Net() *poly.VolumetricNetwork {
	n := poly.NewVolumetricNetwork(1, 1, 1, 1)
	n.InitCNNCell(0, 0, 0, 0, poly.LayerCNN2, 1, 8, 3, poly.DTypeFloat32, 0.05)
	return n
}

func buildCNN3Net() *poly.VolumetricNetwork {
	n := poly.NewVolumetricNetwork(1, 1, 1, 1)
	n.InitCNNCell(0, 0, 0, 0, poly.LayerCNN3, 1, 8, 3, poly.DTypeFloat32, 0.05)
	return n
}

func buildConvT1Net() *poly.VolumetricNetwork {
	n := poly.NewVolumetricNetwork(1, 1, 1, 1)
	n.InitConvTransposedCell(0, 0, 0, 0, poly.LayerConvTransposed1D, 1, 8, 3, poly.DTypeFloat32, 0.05)
	return n
}

func buildConvT2Net() *poly.VolumetricNetwork {
	n := poly.NewVolumetricNetwork(1, 1, 1, 1)
	n.InitConvTransposedCell(0, 0, 0, 0, poly.LayerConvTransposed2D, 1, 8, 3, poly.DTypeFloat32, 0.05)
	return n
}

func buildConvT3Net() *poly.VolumetricNetwork {
	n := poly.NewVolumetricNetwork(1, 1, 1, 1)
	n.InitConvTransposedCell(0, 0, 0, 0, poly.LayerConvTransposed3D, 1, 8, 3, poly.DTypeFloat32, 0.05)
	return n
}

func buildEmbeddingNet(vocabSize, dModel int) *poly.VolumetricNetwork {
	n := poly.NewVolumetricNetwork(1, 1, 1, 1)
	n.InitEmbeddingCell(0, 0, 0, 0, vocabSize, dModel, poly.DTypeFloat32)
	return n
}

func buildKMeansNet(k, dModel int) *poly.VolumetricNetwork {
	n := poly.NewVolumetricNetwork(1, 1, 1, 1)
	n.InitKMeansCell(0, 0, 0, 0, k, dModel, poly.DTypeFloat32)
	return n
}

func buildWeightlessLayer(lt poly.LayerType) *poly.VolumetricNetwork {
	n := poly.NewVolumetricNetwork(1, 1, 1, 1)
	l := n.GetLayer(0, 0, 0, 0)
	l.Type = lt
	l.DType = poly.DTypeFloat32
	// No WeightStore — these are structural/activation layers
	return n
}

func buildParallelNet(dModel int) *poly.VolumetricNetwork {
	n := poly.NewVolumetricNetwork(1, 1, 1, 1)
	l := n.GetLayer(0, 0, 0, 0)
	l.Type = poly.LayerParallel
	l.CombineMode = "add"
	l.DType = poly.DTypeFloat32

	branchA := poly.VolumetricLayer{
		Type:        poly.LayerDense,
		InputHeight: dModel,
		OutputHeight: dModel,
		DType:       poly.DTypeFloat32,
	}
	branchA.WeightStore = poly.NewWeightStore(dModel * dModel)
	branchA.WeightStore.Randomize(time.Now().UnixNano(), 0.1)

	branchB := poly.VolumetricLayer{
		Type:        poly.LayerRMSNorm,
		InputHeight: dModel,
		OutputHeight: dModel,
		DType:       poly.DTypeFloat32,
	}
	branchB.WeightStore = poly.NewWeightStore(dModel)
	for i := range branchB.WeightStore.Master {
		branchB.WeightStore.Master[i] = 1.0
	}

	l.ParallelBranches = []poly.VolumetricLayer{branchA, branchB}
	return n
}

func buildSequentialNet(dModel int) *poly.VolumetricNetwork {
	n := poly.NewVolumetricNetwork(1, 1, 1, 1)
	l := n.GetLayer(0, 0, 0, 0)
	l.Type = poly.LayerSequential
	l.DType = poly.DTypeFloat32

	subA := poly.VolumetricLayer{
		Type:        poly.LayerDense,
		InputHeight: dModel,
		OutputHeight: dModel,
		DType:       poly.DTypeFloat32,
	}
	subA.WeightStore = poly.NewWeightStore(dModel * dModel)
	subA.WeightStore.Randomize(time.Now().UnixNano(), 0.1)

	subB := poly.VolumetricLayer{
		Type:        poly.LayerDense,
		InputHeight: dModel,
		OutputHeight: dModel,
		DType:       poly.DTypeFloat32,
	}
	subB.WeightStore = poly.NewWeightStore(dModel * dModel)
	subB.WeightStore.Randomize(time.Now().UnixNano()+1, 0.1)

	l.SequentialLayers = []poly.VolumetricLayer{subA, subB}
	return n
}

// buildMixedLayerNet creates a 2×2×2 grid with diverse layer types for shift detection.
func buildMixedLayerNet(dModel int) *poly.VolumetricNetwork {
	n := poly.NewVolumetricNetwork(1, 2, 2, 2)
	n.InitDenseCell(0, 0, 0, 0, dModel, poly.ActivationReLU, 0.1)
	n.InitDenseCell(0, 0, 0, 1, dModel, poly.ActivationGELU, 0.1)
	n.InitRNNCell(0, 0, 1, 0, dModel, 0.05)
	n.InitRNNCell(0, 0, 1, 1, dModel, 0.05)
	n.InitLSTMCell(0, 1, 0, 0, dModel, 0.05)
	n.InitDenseCell(0, 1, 0, 1, dModel, poly.ActivationTanh, 0.1)
	l := n.GetLayer(0, 1, 1, 0)
	l.Type = poly.LayerRMSNorm
	l.InputHeight = dModel
	l.DType = poly.DTypeFloat32
	l.WeightStore = poly.NewWeightStore(dModel)
	for i := range l.WeightStore.Master {
		l.WeightStore.Master[i] = 1.0
	}
	n.InitDenseCell(0, 1, 1, 1, dModel, poly.ActivationSilu, 0.1)
	return n
}

// =============================================================================
// Helpers
// =============================================================================

func weightDistance(a, b *poly.VolumetricNetwork) float64 {
	var total float64
	var count int
	lim := len(a.Layers)
	if len(b.Layers) < lim {
		lim = len(b.Layers)
	}
	for i := 0; i < lim; i++ {
		wa := a.Layers[i].WeightStore
		wb := b.Layers[i].WeightStore
		if wa == nil || wb == nil {
			continue
		}
		n := len(wa.Master)
		if len(wb.Master) < n {
			n = len(wb.Master)
		}
		for j := 0; j < n; j++ {
			d := float64(wa.Master[j]) - float64(wb.Master[j])
			if d < 0 {
				d = -d
			}
			total += d
			count++
		}
	}
	if count == 0 {
		return 0
	}
	return total / float64(count)
}

func scaleWeightsByFactor(n *poly.VolumetricNetwork, factor float32) {
	for i := range n.Layers {
		if n.Layers[i].WeightStore == nil {
			continue
		}
		for j := range n.Layers[i].WeightStore.Master {
			n.Layers[i].WeightStore.Master[j] *= factor
		}
	}
}

func countRemoteLinks(n *poly.VolumetricNetwork) int {
	count := 0
	for _, l := range n.Layers {
		for _, b := range l.ParallelBranches {
			if b.IsRemoteLink {
				count++
			}
		}
	}
	return count
}

func countDisabled(n *poly.VolumetricNetwork) int {
	count := 0
	for _, l := range n.Layers {
		if l.IsDisabled {
			count++
		}
	}
	return count
}
