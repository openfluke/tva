package main

import (
	"fmt"
	"math/rand"
	"time"

	"github.com/openfluke/loom/poly"
)

func main() {
	fmt.Println("╔══════════════════════════════════════════════════════════════════════════════╗")
	fmt.Println("║               POLYMORPHIC MODULE INTEGRATION TEST SUITE                      ║")
	fmt.Println("╚══════════════════════════════════════════════════════════════════════════════╝")

	rand.Seed(time.Now().UnixNano())

	// 1. Architecture Testing
	fmt.Println("\n[1] Testing Architecture Generation...")
	net := poly.BuildRandomNetwork(1, 1, 1, 4, 128)
	fmt.Printf("\u2713 Created random network with %d layers\n", len(net.Layers))
	for i, l := range net.Layers {
		fmt.Printf("   Layer %d: %s (DType: %v)\n", i, l.Type, l.DType)
	}

	// 2. Observer & Telemetry Testing
	fmt.Println("\n[2] Testing Observer Telemetry...")
	obs := &poly.ConsoleObserver{}
	for i := range net.Layers {
		net.Layers[i].Observer = obs
	}

	input := poly.NewTensor[float32](1, 128)
	for i := range input.Data { input.Data[i] = rand.Float32() }
	
	fmt.Println("   Starting Forward Pass with Telemetry:")
	output, _, _ := poly.ForwardPolymorphic(net, input)
	fmt.Printf("\u2713 Forward pass complete. Output size: %d\n", len(output.Data))

	// 3. Evaluation Testing
	fmt.Println("\n[3] Testing Evaluation Metrics...")
	expected := []float64{0.0} // Dummy expected class
	metrics, err := poly.EvaluateNetworkPolymorphic(net, []*poly.Tensor[float32]{input}, expected)
	if err != nil {
		fmt.Printf("\u2717 Evaluation failed: %v\n", err)
	} else {
		metrics.PrintSummary()
		fmt.Println("\u2713 Evaluation metrics working correctly")
	}

	// 4. Clustering Testing
	fmt.Println("\n[4] Testing Polymorphic Clustering...")
	clusterData := make([]*poly.Tensor[float32], 10)
	for i := 0; i < 10; i++ {
		clusterData[i] = poly.NewTensor[float32](128)
		for j := range clusterData[i].Data { clusterData[i].Data[j] = rand.Float32() }
	}
	centroids, assignments := poly.KMeansCluster(clusterData, 3, 10, true)
	fmt.Printf("\u2713 Clustering complete: %d centroids, %d assignments\n", len(centroids), len(assignments))
	fmt.Printf("   Assignments: %v\n", assignments)

	// 5. Ensemble Testing
	fmt.Println("\n[5] Testing Ensemble Discovery...")
	perf := []poly.ModelPerformance{
		{ModelID: "Model_A", Mask: []bool{true, true, false, false}},
		{ModelID: "Model_B", Mask: []bool{false, false, true, true}},
	}
	matches := poly.FindComplementaryMatches(perf, 0.5)
	poly.PrintEnsembleReport(matches, 5)
	fmt.Println("\u2713 Ensemble discovery working correctly")

	// 6. Grafting Testing
	fmt.Println("\n[6] Testing Network Grafting...")
	net1 := poly.BuildSequentialNetwork(2, 64, poly.ActivationReLU, poly.DTypeFloat32)
	net2 := poly.BuildSequentialNetwork(2, 64, poly.ActivationTanh, poly.DTypeFloat32)
	grafted, err := poly.GraftNetworksPolymorphic([]*poly.VolumetricNetwork{net1, net2}, "concat")
	if err != nil {
		fmt.Printf("\u2717 Grafting failed: %v\n", err)
	} else {
		fmt.Printf("\u2713 Grafting complete: Created Parallel layer with %d branches\n", len(grafted.ParallelBranches))
	}

	// 7. Grouping Testing
	fmt.Println("\n[7] Testing Weight Grouping...")
	tensors := []poly.DetectedTensor{
		{Name: "model.layers.0.self_attn.q_proj.weight", CanLoad: false},
		{Name: "model.layers.0.self_attn.k_proj.weight", CanLoad: false},
		{Name: "model.layers.0.self_attn.v_proj.weight", CanLoad: false},
		{Name: "model.layers.0.self_attn.o_proj.weight", CanLoad: false},
	}
	groups := poly.GroupRelatedTensors(tensors)
	fmt.Printf("\u2713 Found %d groupings\n", len(groups))
	for name, group := range groups {
		mha, err := poly.ReconstructMHALayer(name, group, 128, 4)
		if err == nil {
			fmt.Printf("   \u2713 Successfully reconstructed MHA: %s\n", mha.Type)
		}
	}

	fmt.Println("\n\u2728 ALL POLYMORPHIC MODULE TESTS COMPLETED \u2728")
}
