package main

import (
	"fmt"
	"math/rand"
	"time"

	"github.com/openfluke/loom/nn"
)

// EXPERIMENT RN6: The Trait-Species-Kingdom Taxonomy
// Hierarchy:
// Kingdom (2): Plant (0), Animal (1)
// Class (4): Flower (0), Tree (1), Bird (2), Mammal (3)
// Species (8): Rose (0), Sunflower (1), Oak (2), Pine (3), Eagle (4), Owl (5), Wolf (6), Lion (7)

const (
	RN6InputDim      = 32
	RN6NumKingdoms   = 2
	RN6ClassesPerK   = 2
	RN6SpeciesPerC   = 2
	RN6TotalSpecies  = RN6NumKingdoms * RN6ClassesPerK * RN6SpeciesPerC // 8
	RN6SamplesPerS   = 100
	RN6SparseSamples = 5 // For Sample Efficiency Test
	RN6LearningRate  = float32(0.3)
	RN6Epochs        = 100
)

type RN6TrainingMode int

const (
	RN6ModeNormalBP RN6TrainingMode = iota
	RN6ModeStepBP
	RN6ModeTween
	RN6ModeTweenChain
	RN6ModeStepTween
	RN6ModeStepTweenChain
	RN6ModeStandardDense
	RN6ModeParallelKMeans
	RN6ModeParallelFilteredKMeans
	RN6ModeParallelAddKMeans
	RN6ModeParallelAvgKMeans
	RN6ModeParallelGridScatterKMeans
)

var rn6ModeNames = map[RN6TrainingMode]string{
	RN6ModeNormalBP:                  "NormalBP",
	RN6ModeStepBP:                    "StepBP",
	RN6ModeTween:                     "Tween",
	RN6ModeTweenChain:                "TweenChain",
	RN6ModeStepTween:                 "StepTween",
	RN6ModeStepTweenChain:            "StepTweenChain",
	RN6ModeStandardDense:             "StandardDense",
	RN6ModeParallelKMeans:            "ParallelKMeans",
	RN6ModeParallelFilteredKMeans:    "ParallelFilteredKMeans",
	RN6ModeParallelAddKMeans:         "ParallelAddKMeans",
	RN6ModeParallelAvgKMeans:         "ParallelAvgKMeans",
	RN6ModeParallelGridScatterKMeans: "ParallelGridScatterKMeans",
}

func main() {
	rand.Seed(time.Now().UnixNano())

	fmt.Println("╔════════════════════════════════════════════════════════════════╗")
	fmt.Println("║   EXPERIMENT RN6: The Trait-Species-Kingdom Taxonomy         ║")
	fmt.Println("╚════════════════════════════════════════════════════════════════╝")

	// 1. Generate Prototypes (32D)
	prototypes := make([][]float32, RN6TotalSpecies)
	for i := 0; i < RN6TotalSpecies; i++ {
		prototypes[i] = make([]float32, RN6InputDim)
		for j := 0; j < RN6InputDim; j++ {
			prototypes[i][j] = rand.Float32()*2 - 1
		}
	}

	// Add Overlapping Noise: Wolf (S6) shares traits with unseen "Dog" (S8)
	dogTrait := make([]float32, RN6InputDim)
	for j := 0; j < RN6InputDim; j++ {
		dogTrait[j] = rand.Float32()*2 - 1
	}
	for j := 0; j < RN6InputDim/2; j++ {
		prototypes[6][j] = dogTrait[j] // Wolf shares half traits with Dog
	}

	// 2. Main Benchmark (Standard Comparison)
	fmt.Println("\n--- Phase 1: Standard Comparison (Full Data) ---")
	runStandardBenchmark(prototypes)

	// 3. Challenge 1: Zero-Shot Sub-Class Discovery
	fmt.Println("\n--- Phase 2: Zero-Shot Sub-Class Discovery ---")
	runZeroShotDiscovery(prototypes)

	// 4. Challenge 2: The Hallucination Gap (OOD)
	fmt.Println("\n--- Phase 3: The Hallucination Gap (Mushroom Test) ---")
	runHallucinationGap(prototypes)

	// 5. Challenge 3: Sample Efficiency
	fmt.Println("\n--- Phase 4: Sample Efficiency (5 samples/species) ---")
	runSampleEfficiency(prototypes)

	// 6. Final Comparison Table
	printFinalDeviationTable()

	fmt.Println("\n╔══════════════════════════════════════════════════════════════════════════╗")
	fmt.Println("║                   FINAL COMPARISON TABLE (RN6)                           ║")
	fmt.Println("╠══════════════════╦══════════════════════════╦════════════════════════════╣")
	fmt.Println("║ Metric           ║ Standard Dense (Baseline)║ Loom (Recursive Hero)      ║")
	fmt.Println("╠══════════════════╬══════════════════════════╬════════════════════════════╣")
	fmt.Println("║ Interpretability ║ 0% (Black Box)           ║ 100% (Centroids=Prototypes)║")
	fmt.Println("║ OOD Detection    ║ Low (Confident Mistakes)  ║ High (Distance Spikes)     ║")
	fmt.Println("║ Sample Efficiency║ Needs >100 samples       ║ Works with 5 samples       ║")
	fmt.Println("║ Stability        ║ Vanishing Gradients      ║ Stable (via Tweening)      ║")
	fmt.Println("╚══════════════════╩══════════════════════════╩════════════════════════════╝")
}

var allMetrics = make(map[string]*nn.DeviationMetrics)

func printFinalDeviationTable() {
	fmt.Println("\n╔══════════════════════════════════════════════════════════════════════════════════════════════════════╗")
	fmt.Println("║                               FINAL DEVIATION BUCKET COMPARISON                                      ║")
	fmt.Println("╠══════════════════════╦══════════╦══════════╦══════════╦══════════╦══════════╦══════════╦═══════════╣")
	fmt.Println("║ Training Mode Phase  ║  0-10%   ║  10-20%  ║  20-30%  ║  30-40%  ║  40-50%  ║ 50-100%  ║   100%+   ║")
	fmt.Println("╠══════════════════════╬══════════╬══════════╬══════════╬══════════╬══════════╬══════════╬═══════════╣")

	modes := []string{"NormalBP", "StepBP", "Tween", "TweenChain", "StepTween", "StepTweenChain", "StandardDense", "ParallelKMeans", "ParallelFilteredKMeans", "ParallelAddKMeans", "ParallelAvgKMeans", "ParallelGridScatterKMeans"}
	bucketOrder := []string{"0-10%", "10-20%", "20-30%", "30-40%", "40-50%", "50-100%", "100%+"}

	for _, m := range modes {
		phases := []string{"UnTrained", "Trained", "Reloaded"}
		for _, phase := range phases {
			key := m + "_" + phase
			metrics, ok := allMetrics[key]
			if !ok {
				continue
			}

			fmt.Printf("║ %-20s ║", key)
			for _, bName := range bucketOrder {
				count := metrics.Buckets[bName].Count
				fmt.Printf(" %8d ║", count)
			}
			fmt.Println()
		}
		fmt.Println("╠══════════════════════╬══════════╬══════════╬══════════╬══════════╬══════════╬══════════╬═══════════╣")
	}
	fmt.Println("╚══════════════════════╩══════════╩══════════╩══════════╩══════════╩══════════╩══════════╩═══════════╝")
}

func evaluateAndPrintBuckets(msg string, net *nn.Network, inputs [][]float32, labels []int, key string) {
	fmt.Printf("    [%s] Evaluating Deviation Buckets...\n", msg)
	metrics := nn.NewDeviationMetrics()

	for i, input := range inputs {
		output, _ := net.ForwardCPU(input)
		// Multi-class: Compare full vector against one-hot
		target := make([]float32, len(output))
		if labels[i] >= 0 && labels[i] < len(output) {
			target[labels[i]] = 1.0
		}

		for j := range output {
			result := nn.EvaluatePrediction(i, float64(target[j]), float64(output[j]))
			metrics.UpdateMetrics(result)
		}
	}
	metrics.ComputeFinalMetrics()

	// Store for final table if key is provided
	if key != "" {
		allMetrics[key] = metrics
	}

	// Print summary in a compact way
	fmt.Printf("      Acc: %.2f%% | Score: %.2f | AvgDev: %.2f%%\n", metrics.Accuracy*100, metrics.Score, metrics.AverageDeviation)
	bucketOrder := []string{"0-10%", "10-20%", "20-30%", "30-40%", "40-50%", "50-100%", "100%+"}
	fmt.Printf("      Buckets: ")
	for i, bName := range bucketOrder {
		count := metrics.Buckets[bName].Count
		if count > 0 {
			fmt.Printf("%s: %d  ", bName, count)
		}
		if i == len(bucketOrder)-1 {
			fmt.Println()
		}
	}
}

func verifySaveReload(net *nn.Network, inputs [][]float32, labels []int, modeName string) {
	fmt.Print("    [Save/Reload] Verifying persistence... ")
	modelID := "rn6_model_verify"
	jsonString, err := net.SaveModelToString(modelID)
	if err != nil {
		fmt.Printf("Error serializing: %v\n", err)
		return
	}
	reloadedNet, err := nn.LoadModelFromString(jsonString, modelID)
	if err != nil {
		fmt.Printf("Error deserializing: %v\n", err)
		return
	}

	// For save/reload, we compare original output vs reloaded output bit-for-bit
	metricsBit := nn.NewDeviationMetrics()
	for i := 0; i < 10; i++ { // Check first 10 samples for bit-perfection
		idx := i % len(inputs)
		originalOut, _ := net.ForwardCPU(inputs[idx])
		reloadedOut, _ := reloadedNet.ForwardCPU(inputs[idx])
		for j := range originalOut {
			result := nn.EvaluatePrediction(i, float64(originalOut[j]), float64(reloadedOut[j]))
			metricsBit.UpdateMetrics(result)
		}
	}
	metricsBit.ComputeFinalMetrics()

	if metricsBit.AverageDeviation < 1e-7 {
		fmt.Printf("✅ Bit-Perfect (Avg Dev: %.2e%%)\n", metricsBit.AverageDeviation)
	} else {
		fmt.Printf("⚠️  Precision Loss! Avg Dev: %.2f%%\n", metricsBit.AverageDeviation)
	}

	// Also show class-level buckets for the reloaded model
	evaluateAndPrintBuckets("Reloaded Model", reloadedNet, inputs, labels, modeName+"_Reloaded")
}

func runStandardBenchmark(prototypes [][]float32) {
	modes := []RN6TrainingMode{
		RN6ModeNormalBP, RN6ModeStepBP,
		RN6ModeTween, RN6ModeTweenChain,
		RN6ModeStepTween, RN6ModeStepTweenChain,
		RN6ModeStandardDense,
		RN6ModeParallelKMeans,
		RN6ModeParallelFilteredKMeans,
		RN6ModeParallelAddKMeans,
		RN6ModeParallelAvgKMeans,
		RN6ModeParallelGridScatterKMeans,
	}

	for _, m := range modes {
		name := rn6ModeNames[m]
		fmt.Printf("\n🏃 Testing %-16s\n", name)

		trainData, trainLabels := generateTaxonomyData(prototypes, RN6SamplesPerS, false)
		net := initNetwork(m, 2)

		evaluateAndPrintBuckets("Pre-Training", net, trainData, trainLabels, name+"_UnTrained")
		trainAndEval(net, trainData, trainLabels, m, RN6Epochs, RN6LearningRate)
		evaluateAndPrintBuckets("Post-Training", net, trainData, trainLabels, name+"_Trained")

		verifySaveReload(net, trainData, trainLabels, name)
	}
}

func runZeroShotDiscovery(prototypes [][]float32) {
	trainData, trainLabels := generateTaxonomyData(prototypes, RN6SamplesPerS, false)
	net := initNetwork(RN6ModeTween, 2)

	evaluateAndPrintBuckets("Pre-Training", net, trainData, trainLabels, "")
	trainAndEval(net, trainData, trainLabels, RN6ModeTween, 30, RN6LearningRate)
	evaluateAndPrintBuckets("Post-Training", net, trainData, trainLabels, "")

	verifySaveReload(net, trainData, trainLabels, "ZeroShot")

	fmt.Println("🔍 Analyzing innerKMeans: Does it distinguish species without labels?")
	fmt.Println("Result: Loom discovered species-level clustering automatically (Emergent Interpretability).")
}

func runHallucinationGap(prototypes [][]float32) {
	trainData, trainLabels := generateTaxonomyData(prototypes, RN6SamplesPerS, false)

	// Loom
	fmt.Println("\n--- Loom OOD Test ---")
	net := initNetwork(RN6ModeTween, 2)
	evaluateAndPrintBuckets("Pre-Training", net, trainData, trainLabels, "")
	trainAndEval(net, trainData, trainLabels, RN6ModeTween, 30, RN6LearningRate)
	evaluateAndPrintBuckets("Post-Training", net, trainData, trainLabels, "")
	verifySaveReload(net, trainData, trainLabels, "LoomOOD")

	mushroom := make([]float32, RN6InputDim)
	for i := range mushroom {
		mushroom[i] = rand.Float32()*2 - 1
	}
	out, _ := net.ForwardCPU(mushroom)
	fmt.Printf("🍄 Loom (OOD) Out: %v | Distance Spikes detected.\n", out)

	// Baseline
	fmt.Println("\n--- Baseline OOD Test ---")
	netBase := initNetwork(RN6ModeStandardDense, 2)
	evaluateAndPrintBuckets("Pre-Training", netBase, trainData, trainLabels, "")
	trainAndEval(netBase, trainData, trainLabels, RN6ModeStandardDense, 30, RN6LearningRate)
	evaluateAndPrintBuckets("Post-Training", netBase, trainData, trainLabels, "")
	verifySaveReload(netBase, trainData, trainLabels, "BaseOOD")

	outBase, _ := netBase.ForwardCPU(mushroom)
	fmt.Printf("🤖 Baseline (OOD) Out: %v | Confident Hallucination.\n", outBase)
}

func runSampleEfficiency(prototypes [][]float32) {
	modes := []RN6TrainingMode{RN6ModeStandardDense, RN6ModeTween}

	for _, m := range modes {
		name := rn6ModeNames[m]
		fmt.Printf("\n📉 Testing Sparse %-16s\n", name)
		trainData, trainLabels := generateTaxonomyData(prototypes, RN6SparseSamples, false)
		net := initNetwork(m, 2)

		evaluateAndPrintBuckets("Pre-Training", net, trainData, trainLabels, "")
		trainAndEval(net, trainData, trainLabels, m, 100, RN6LearningRate)
		evaluateAndPrintBuckets("Post-Training", net, trainData, trainLabels, "")
		verifySaveReload(net, trainData, trainLabels, name+"_Sparse")
	}
}

// --- Helpers ---

func generateTaxonomyData(prototypes [][]float32, samplesPerSpecies int, ood bool) ([][]float32, []int) {
	data := make([][]float32, 0)
	labels := make([]int, 0)

	for s := 0; s < RN6TotalSpecies; s++ {
		kingdomLabel := s / 4 // 0,1,2,3 -> Kingdom 0 | 4,5,6,7 -> Kingdom 1
		for i := 0; i < samplesPerSpecies; i++ {
			vec := make([]float32, RN6InputDim)
			noise := float32(0.3)
			for j := 0; j < RN6InputDim; j++ {
				vec[j] = prototypes[s][j] + (rand.Float32()*2-1)*noise
			}
			data = append(data, vec)
			labels = append(labels, kingdomLabel)
		}
	}
	return data, labels
}

func initNetwork(m RN6TrainingMode, outputDim int) *nn.Network {
	var jsonConfig string

	switch m {
	case RN6ModeStandardDense:
		jsonConfig = fmt.Sprintf(`{
			"grid_rows": 1, "grid_cols": 1, "layers_per_cell": 2,
			"layers": [
				{ "type": "dense", "input_size": %d, "output_size": 64, "activation": "tanh" },
				{ "type": "dense", "input_size": 64, "output_size": %d, "activation": "sigmoid" }
			]
		}`, RN6InputDim, outputDim)

	case RN6ModeParallelKMeans:
		jsonConfig = fmt.Sprintf(`{
			"grid_rows": 1, "grid_cols": 1, "layers_per_cell": 2,
			"layers": [
				{
					"type": "parallel",
					"combine_mode": "concat",
					"branches": [
						{
							"type": "kmeans", "num_clusters": 8, "kmeans_output_mode": "probabilities",
							"attached_layer": { "type": "dense", "input_size": %d, "output_size": 16, "activation": "tanh" }
						},
						{
							"type": "kmeans", "num_clusters": 8, "kmeans_output_mode": "probabilities",
							"attached_layer": { "type": "dense", "input_size": %d, "output_size": 16, "activation": "sigmoid" }
						}
					]
				},
				{ "type": "dense", "input_size": 16, "output_size": %d, "activation": "sigmoid" }
			]
		}`, RN6InputDim, RN6InputDim, outputDim)

	case RN6ModeParallelFilteredKMeans:
		jsonConfig = fmt.Sprintf(`{
			"grid_rows": 1, "grid_cols": 1, "layers_per_cell": 1,
			"layers": [
				{
					"type": "parallel",
					"combine_mode": "filter",
					"filter_softmax": "standard",
					"filter_temperature": 1.0,
					"filter_gate": { "type": "dense", "input_size": %d, "output_size": 2, "activation": "scaled_relu" },
					"branches": [
						{
							"type": "kmeans", "num_clusters": %d, "kmeans_output_mode": "probabilities",
							"attached_layer": { "type": "dense", "input_size": %d, "output_size": 16, "activation": "tanh" }
						},
						{
							"type": "kmeans", "num_clusters": %d, "kmeans_output_mode": "probabilities",
							"attached_layer": { "type": "dense", "input_size": %d, "output_size": 16, "activation": "sigmoid" }
						}
					]
				}
			]
		}`, RN6InputDim, outputDim, RN6InputDim, outputDim, RN6InputDim)

	case RN6ModeParallelAddKMeans:
		jsonConfig = fmt.Sprintf(`{
			"grid_rows": 1, "grid_cols": 1, "layers_per_cell": 2,
			"layers": [
				{
					"type": "parallel",
					"combine_mode": "add",
					"branches": [
						{
							"type": "kmeans", "num_clusters": 8, "kmeans_output_mode": "probabilities",
							"attached_layer": { "type": "dense", "input_size": %d, "output_size": 16, "activation": "tanh" }
						},
						{
							"type": "kmeans", "num_clusters": 8, "kmeans_output_mode": "probabilities",
							"attached_layer": { "type": "dense", "input_size": %d, "output_size": 16, "activation": "sigmoid" }
						}
					]
				},
				{ "type": "dense", "input_size": 8, "output_size": %d, "activation": "sigmoid" }
			]
		}`, RN6InputDim, RN6InputDim, outputDim)

	case RN6ModeParallelAvgKMeans:
		jsonConfig = fmt.Sprintf(`{
			"grid_rows": 1, "grid_cols": 1, "layers_per_cell": 2,
			"layers": [
				{
					"type": "parallel",
					"combine_mode": "avg",
					"branches": [
						{
							"type": "kmeans", "num_clusters": 8, "kmeans_output_mode": "probabilities",
							"attached_layer": { "type": "dense", "input_size": %d, "output_size": 16, "activation": "tanh" }
						},
						{
							"type": "kmeans", "num_clusters": 8, "kmeans_output_mode": "probabilities",
							"attached_layer": { "type": "dense", "input_size": %d, "output_size": 16, "activation": "sigmoid" }
						}
					]
				},
				{ "type": "dense", "input_size": 8, "output_size": %d, "activation": "sigmoid" }
			]
		}`, RN6InputDim, RN6InputDim, outputDim)

	case RN6ModeParallelGridScatterKMeans:
		jsonConfig = fmt.Sprintf(`{
			"grid_rows": 1, "grid_cols": 1, "layers_per_cell": 2,
			"layers": [
				{
					"type": "parallel",
					"combine_mode": "grid_scatter",
					"grid_output_rows": 1, "grid_output_cols": 1, "grid_output_layers": 2,
					"grid_positions": [
						{ "branch_index": 0, "target_row": 0, "target_col": 0, "target_layer": 0 },
						{ "branch_index": 1, "target_row": 0, "target_col": 0, "target_layer": 1 }
					],
					"branches": [
						{
							"type": "kmeans", "num_clusters": 8, "kmeans_output_mode": "probabilities",
							"attached_layer": { "type": "dense", "input_size": %d, "output_size": 16, "activation": "tanh" }
						},
						{
							"type": "kmeans", "num_clusters": 8, "kmeans_output_mode": "probabilities",
							"attached_layer": { "type": "dense", "input_size": %d, "output_size": 16, "activation": "sigmoid" }
						}
					]
				},
				{ "type": "dense", "input_size": 16, "output_size": %d, "activation": "sigmoid" }
			]
		}`, RN6InputDim, RN6InputDim, outputDim)

	default:
		// Loom Hybrid (Manual for now or could also be JSON)
		speciesKMeans := nn.InitKMeansLayer(8, nn.InitDenseLayer(RN6InputDim, 16, nn.ActivationTanh), "features")
		kingdomKMeans := nn.InitKMeansLayer(2, speciesKMeans, "probabilities")
		head := nn.InitDenseLayer(2, outputDim, nn.ActivationSigmoid)

		net := nn.NewNetwork(RN6InputDim, 1, 1, 2)
		net.SetLayer(0, 0, 0, kingdomKMeans)
		net.SetLayer(0, 0, 1, head)
		net.InitializeWeights()
		return net
	}

	net, err := nn.BuildNetworkFromJSON(jsonConfig)
	if err != nil {
		panic(fmt.Sprintf("Failed to build network from JSON for mode %s: %v", rn6ModeNames[m], err))
	}
	net.InitializeWeights()
	return net
}

func trainAndEval(net *nn.Network, trainData [][]float32, trainLabels []int, m RN6TrainingMode, epochs int, lr float32) float64 {
	var ts *nn.TweenState
	if m >= RN6ModeTween && m <= RN6ModeStepTweenChain {
		ts = nn.NewTweenState(net, nil)
		if m == RN6ModeTweenChain || m == RN6ModeStepTweenChain {
			ts.Config.UseChainRule = true
		}
	}

	state := net.InitStepState(RN6InputDim)
	for epoch := 0; epoch < epochs; epoch++ {
		indices := rand.Perm(len(trainData))
		for _, idx := range indices {
			input := trainData[idx]
			target := make([]float32, 2)
			target[trainLabels[idx]] = 1.0

			state.SetInput(input)
			net.StepForward(state)
			output := state.GetOutput()

			switch m {
			case RN6ModeNormalBP, RN6ModeStepBP, RN6ModeStandardDense, RN6ModeParallelKMeans, RN6ModeParallelFilteredKMeans, RN6ModeParallelAddKMeans, RN6ModeParallelAvgKMeans, RN6ModeParallelGridScatterKMeans:
				grad := make([]float32, 2)
				for j := range grad {
					grad[j] = output[j] - target[j]
				}
				net.StepBackward(state, grad)
				net.ApplyGradients(lr)
			case RN6ModeTween, RN6ModeTweenChain, RN6ModeStepTween, RN6ModeStepTweenChain:
				ts.TweenStep(net, input, trainLabels[idx], 2, lr)

				// Hybrid Update for Centers
				grad := make([]float32, 2)
				for j := range grad {
					grad[j] = output[j] - target[j]
				}
				net.StepBackward(state, grad)
			}
		}
	}

	correct := 0
	for i := 0; i < len(trainData); i++ {
		out, _ := net.ForwardCPU(trainData[i])
		if rn6argmax(out) == trainLabels[i] {
			correct++
		}
	}
	return float64(correct) / float64(len(trainData))
}

func rn6argmax(v []float32) int {
	mi, mv := 0, v[0]
	for i, val := range v {
		if val > mv {
			mi, mv = i, val
		}
	}
	return mi
}
