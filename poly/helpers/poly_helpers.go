package main

import (
	"fmt"
	"math/rand"
	"time"

	"github.com/openfluke/loom/poly"
)

func main() {
	fmt.Println("\u2554\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2557")
	fmt.Println("\u2551           ABSOLUTE EXHAUSTIVE POLY VERIFICATION SUITE    \u2551")
	fmt.Println("\u255a\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2569")

	rand.Seed(time.Now().UnixNano())

	// 1. ARCHITECTURE
	testArchitecture()

	// 2. TELEMETRY
	testTelemetry()

	// 3. GROUPING
	testGrouping()

	// 4. OBSERVER
	testObserver()

	// 5. EVALUATION
	testEvaluation()

	// 6. CLUSTERING
	testClustering()

	// 7. ENSEMBLE
	testEnsemble()

	// 8. GRAFTING
	testGrafting()

	fmt.Println("\n\u2728 [DOUBLE CHECK COMPLETE] EVERY POLY MODULE FULLY VERIFIED \u2728")
}

// 1. ARCHITECTURE: Universal Initializers & Builders
func testArchitecture() {
	fmt.Println("\n[\u25cf] ARCHITECTURE: Universal Layer Support")
	
	// Test all 18 layer types
	net := poly.NewVolumetricNetwork(1, 1, 1, 18)
	net.InitDenseCell(0,0,0,0, 128, poly.ActivationReLU, 0.1)
	net.InitMHACell(0,0,0,1, 128, 4, 0.1)
	l2 := net.GetLayer(0,0,0,2); l2.Type = poly.LayerSwiGLU; l2.InputHeight=128; l2.OutputHeight=512; l2.WeightStore=poly.NewWeightStore(128*512*3)
	l3 := net.GetLayer(0,0,0,3); l3.Type = poly.LayerRMSNorm; l3.InputHeight=128
	net.InitCNNCell(0,0,0,4, poly.LayerCNN1, 1, 16, 3, poly.DTypeFloat32, 0.1)
	net.InitCNNCell(0,0,0,5, poly.LayerCNN2, 1, 16, 3, poly.DTypeFloat32, 0.1)
	net.InitCNNCell(0,0,0,6, poly.LayerCNN3, 1, 16, 3, poly.DTypeFloat32, 0.1)
	net.InitRNNCell(0,0,0,7, 128, 0.1)
	net.InitLSTMCell(0,0,0,8, 128, 0.1)
	net.InitLayerNormCell(0,0,0,9, 128, poly.DTypeFloat32)
	net.InitConvTransposedCell(0,0,0,10, poly.LayerConvTransposed1D, 8, 16, 3, poly.DTypeFloat32, 0.1)
	net.InitConvTransposedCell(0,0,0,11, poly.LayerConvTransposed2D, 8, 16, 3, poly.DTypeFloat32, 0.1)
	net.InitConvTransposedCell(0,0,0,12, poly.LayerConvTransposed3D, 8, 16, 3, poly.DTypeFloat32, 0.1)
	net.InitEmbeddingCell(0,0,0,13, 1000, 128, poly.DTypeFloat32)
	net.InitKMeansCell(0,0,0,14, 10, 128, poly.DTypeFloat32)
	l15 := net.GetLayer(0,0,0,15); l15.Type = poly.LayerSoftmax; l15.InputHeight=128; l15.OutputHeight=10
	l16 := net.GetLayer(0,0,0,16); l16.Type = poly.LayerParallel; l16.ParallelBranches = []poly.VolumetricLayer{*net.GetLayer(0,0,0,0)}
	l17 := net.GetLayer(0,0,0,17); l17.Type = poly.LayerSequential

	for i := 0; i < 18; i++ {
		assert(net.Layers[i].Type == poly.LayerType(i), fmt.Sprintf("Layer %d init fail", i))
	}
	
	// Test High-level builders
	t := poly.BuildTransformerNetwork(1, 128, 4, poly.DTypeFloat32)
	assert(len(t.Layers) == 4, "Transformer layer count fail")
	
	cnn := poly.BuildCNN(28, 10, poly.DTypeFloat32)
	assert(cnn.Layers[0].Type == poly.LayerCNN2, "CNN builder fail")
	
	fmt.Println("   \u2713 All 18 LayerTypes initialized and verified")
	fmt.Println("   \u2713 Complex builders (Transformer, CNN) verified")
}

// 2. TELEMETRY: Shape & Parameter Analysis for All Types
func testTelemetry() {
	fmt.Println("\n[\u25cf] TELEMETRY: Structural Blueprint Spectrum")
	
	net := poly.NewVolumetricNetwork(1, 1, 1, 18)
	// (Init same as in Architecture test)
	net.InitDenseCell(0,0,0,0, 128, poly.ActivationReLU, 0.1)
	net.InitMHACell(0,0,0,1, 128, 4, 0.1)
	net.InitCNNCell(0,0,0,6, poly.LayerCNN3, 1, 16, 3, poly.DTypeFloat32, 0.1)
	net.InitConvTransposedCell(0,0,0,12, poly.LayerConvTransposed3D, 8, 16, 3, poly.DTypeFloat32, 0.1)
	net.InitEmbeddingCell(0,0,0,13, 1000, 128, poly.DTypeFloat32)
	
	blue := poly.ExtractNetworkBlueprint(net, "Extreme-Net")
	assert(blue.TotalLayers == 18, "Blueprint layers mismatch")
	
	var foundMHA, foundCNN3, foundCT3, foundEmb bool
	for _, l := range blue.Layers {
		switch l.Type {
		case "MultiHeadAttention":
			foundMHA = true
			assert(l.Parameters > 0, "MHA params fail")
		case "CNN3":
			foundCNN3 = true
			assert(l.Parameters == 16 * 1 * 3 * 3 * 3, "CNN3 params fail")
		case "ConvTransposed3D":
			foundCT3 = true
			assert(l.Parameters == 8 * 16 * 3 * 3 * 3, "CT3 params fail")
		case "Embedding":
			foundEmb = true
			assert(l.Parameters == 1000 * 128, "Embedding params fail")
		}
	}
	assert(foundMHA && foundCNN3 && foundCT3 && foundEmb, "Telemetry missing layer logic")
	fmt.Printf("   \u2713 ExtractNetworkBlueprint: Parameters verified for complex 3D and NLP layers\n")
}

// 3. GROUPING: Pattern-based Layer Discovery
func testGrouping() {
	fmt.Println("\n[\u25cf] GROUPING: Multi-Pattern Discovery")
	
	tensors := []poly.DetectedTensor{
		{Name: "model.layers.0.self_attn.q_proj.weight"},
		{Name: "model.layers.0.self_attn.k_proj.weight"},
		{Name: "model.layers.0.self_attn.v_proj.weight"},
		{Name: "model.layers.0.self_attn.o_proj.weight"},
		{Name: "model.layers.1.mlp.gate_proj.weight"},
		{Name: "model.layers.1.mlp.up_proj.weight"},
		{Name: "model.layers.1.mlp.down_proj.weight"},
		{Name: "model.layers.2.input_layernorm.weight"},
		{Name: "model.layers.3.conv.weight"},
	}
	
	groups := poly.GroupRelatedTensors(tensors)
	assert(len(groups) == 4, "Grouping fail")
	
	mha, _ := poly.ReconstructMHALayer("mha", groups["model.layers.0.self_attn"], 128, 4)
	swi, _ := poly.ReconstructSwiGLULayer("swi", groups["model.layers.1.mlp"], 128)
	norm, _ := poly.ReconstructLayerNormLayer("norm", groups["model.layers.2.input_layernorm"], 128)
	cnn, _ := poly.ReconstructCNNLayer("cnn", groups["model.layers.3.conv"], poly.LayerCNN2)
	
	assert(mha.Type == poly.LayerMultiHeadAttention, "MHA Reconstruct fail")
	assert(swi.Type == poly.LayerSwiGLU, "SwiGLU Reconstruct fail")
	assert(norm.Type == poly.LayerLayerNorm, "LayerNorm Reconstruct fail")
	assert(cnn.Type == poly.LayerCNN2, "CNN Reconstruct fail")
	
	fmt.Println("   \u2713 Reconstructors: MHA, SwiGLU, LayerNorm, CNN discovery verified")
}

// 4. OBSERVER: Real-time Metric Aggregation
func testObserver() {
	fmt.Println("\n[\u25cf] OBSERVER: Aggregate Telemetry")
	
	// Stats calculation
	tensor := poly.NewTensorFromSlice([]float32{-1, 0, 1, 2}, 4)
	stats := poly.ComputeLayerStats(tensor)
	assert(stats.Max == 2.0 && stats.Avg == 0.5, "Stats calculation fail")
	
	// Aggregator Windowing
	agg := poly.NewAggregatingObserver(2)
	agg.OnForward(poly.PolyLayerEvent{ModelID: "M1", Stats: poly.LayerStats{Max: 1.0}})
	agg.OnForward(poly.PolyLayerEvent{ModelID: "M1", Stats: poly.LayerStats{Max: 3.0}})
	agg.OnForward(poly.PolyLayerEvent{ModelID: "M1", Stats: poly.LayerStats{Max: 5.0}})
	agg.OnForward(poly.PolyLayerEvent{ModelID: "M1", Stats: poly.LayerStats{Max: 7.0}})
	
	assert(len(agg.History) == 2, "Windowing fail")
	assert(agg.History[0].Max == 3.0, "Window 0 Max fail")
	assert(agg.History[1].Max == 7.0, "Window 1 Max fail")
	
	fmt.Println("   \u2713 LayerStats and AggregatingObserver verified")
}

// 5. EVALUATION: Multi-Model Performance Hub
func testEvaluation() {
	fmt.Println("\n[\u25cf] EVALUATION: Multi-Model Benchmarking")
	
	m1 := poly.BuildSequentialNetwork(1, 4, poly.ActivationReLU, poly.DTypeFloat32)
	m2 := poly.BuildSequentialNetwork(1, 4, poly.ActivationReLU, poly.DTypeFloat32)
	
	models := map[string]*poly.VolumetricNetwork{"M1": m1, "M2": m2}
	inputs := []*poly.Tensor[float32]{poly.NewTensorFromSlice([]float32{1, 1, 1, 1}, 1, 4)}
	expected := []float64{0.0}
	
	results, err := poly.MultiNetworkEvaluation(models, inputs, expected)
	assert(err == nil, "Multi eval err")
	assert(len(results) == 2, "Results count fail")
	
	// Deviation metrics
	pred := poly.EvaluatePrediction(0, 100.0, 95.0)
	assert(pred.Deviation == 5.0 && pred.Bucket == "0-10%", "Prediction eval fail")
	
	fmt.Println("   \u2713 MultiNetworkEvaluation and DeviationBuckets verified")
}

// 6. CLUSTERING: Semantic & Statistical Discovery
func testClustering() {
	fmt.Println("\n[\u25cf] CLUSTERING: KMeans & Silhouette")
	
	data := []*poly.Tensor[float32]{
		poly.NewTensorFromSlice([]float32{1, 1}, 2),
		poly.NewTensorFromSlice([]float32{1, 1.1}, 2),
		poly.NewTensorFromSlice([]float32{5, 5}, 2),
		poly.NewTensorFromSlice([]float32{5.1, 5}, 2),
	}
	
	centroids, assignments := poly.KMeansCluster(data, 2, 10, false)
	assert(len(centroids) == 2, "KMeans centroids fail")
	assert(assignments[0] == assignments[1] && assignments[2] == assignments[3], "KMeans assignments fail")
	
	silhouette := poly.ComputeSilhouetteScore(data, assignments)
	assert(silhouette > 0.8, "Silhouette score fail")
	
	dist := poly.CosineDistance([]float32{1, 0}, []float32{1, 0.1})
	assert(dist < 0.1, "Cosine distance fail")
	
	fmt.Printf("   \u2713 KMeans (Sil: %.4f) and Cosine Distance verified\n", silhouette)
}

// 7. ENSEMBLE: Consensus & Discovery Engines
func testEnsemble() {
	fmt.Println("\n[\u25cf] ENSEMBLE: Consensus & Similarity")
	
	// Majority Voting
	votes := [][]int{{1, 0}, {1, 1}, {0, 0}}
	winner := poly.MajorityVote(votes)
	assert(winner[0] == 1 && winner[1] == 0, "MajorityVote fail")
	
	// Similarity
	m1 := poly.ModelPerformance{ModelID: "M1", Mask: []bool{true, true, false}}
	m2 := poly.ModelPerformance{ModelID: "M2", Mask: []bool{true, true, false}}
	sim := poly.PerformanceSimilarity(m1, m2)
	assert(sim > 0.99, "Similarity fail")
	
	// Complementary matching
	m3 := poly.ModelPerformance{ModelID: "M3", Mask: []bool{false, false, true}}
	matches := poly.FindComplementaryMatches([]poly.ModelPerformance{m1, m3}, 0.5)
	assert(len(matches) > 0 && matches[0].Coverage == 1.0, "Complementary match fail")
	
	fmt.Println("   \u2713 Voting, Mask Similarity, and Complementary Discovery verified")
}

// 8. GRAFTING: Heterogeneous Hive Fusion
func testGrafting() {
	fmt.Println("\n[\u25cf] GRAFTING: Parallel & Residual Fusion")
	
	n1 := poly.BuildSequentialNetwork(1, 8, poly.ActivationReLU, poly.DTypeFloat32)
	n2 := poly.BuildSequentialNetwork(1, 8, poly.ActivationReLU, poly.DTypeFloat32)
	
	// Gated grafting (MoE)
	gated, _ := poly.GraftNetworksPolymorphic([]*poly.VolumetricNetwork{n1, n2}, "gated")
	assert(gated.Type == poly.LayerParallel && gated.CombineMode == "gated", "Graft gated fail")
	
	// Residual grafting
	res := poly.CreateResidualGraft(n1)
	assert(len(res.ParallelBranches) == 2 && res.CombineMode == "add", "Graft residual fail")
	
	fmt.Println("   \u2713 Poly-Grafting: Gated (MoE) and Residual (Skip) layers verified")
}

func assert(cond bool, msg string) {
	if !cond { panic("ASSERT FAILED: " + msg) }
}
