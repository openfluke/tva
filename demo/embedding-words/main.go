package main

import (
	"fmt"
	"math"
	"math/rand"
	"strings"
	"time"

	"github.com/openfluke/loom/nn"
)

// Embedding Word Similarity Demo
// Demonstrates Embedding layer learning word relationships
// Note: Embedding GPU support may be limited, demo uses CPU only

const (
	VocabSize     = 100  // Small vocabulary
	EmbedDim      = 32   // Embedding dimension
	ContextWindow = 3    // Words before/after target
	NumSamples    = 5000 // Training pairs
	Epochs        = 100
	LearningRate  = 0.01
	BatchSize     = 20
)

func main() {
	rand.Seed(time.Now().UnixNano())

	fmt.Println("╔════════════════════════════════════════════════════════════════╗")
	fmt.Println("║   Embedding Demo: Word Co-occurrence Learning                 ║")
	fmt.Println("╚════════════════════════════════════════════════════════════════╝")

	// Generate dataset
	fmt.Println("\n[1/3] Generating word co-occurrence data...")
	trainData, trainLabels := generateWordDataset()
	fmt.Printf("      Generated %d context-target pairs (vocab=%d)\n",
		len(trainData), VocabSize)

	// Train model (CPU only)
	fmt.Println("\n[2/3] Training on CPU...")
	jsonConfig := buildNetworkConfig()
	net, err := nn.BuildNetworkFromJSON(jsonConfig)
	if err != nil {
		panic(err)
	}
	net.InitializeWeights()

	startCPU := time.Now()
	accCPU := trainNetwork(net, trainData, trainLabels)
	timeCPU := time.Since(startCPU)

	fmt.Printf("      Training complete: Accuracy=%.2f%%, Time=%v\n", accCPU*100, timeCPU)

	// Show learned relationships
	fmt.Println("\n[3/3] Analyzing learned word embeddings...")
	showWordSimilarities(net)

	fmt.Println("\n✅ Embedding Demo Complete!")
	fmt.Println("\nKey Insight: Words that appear in similar contexts")
	fmt.Println("should have similar embeddings (distributional semantics)")
}

func generateWordDataset() ([][]float32, []int) {
	categories := [][]int{
		makeRange(0, 20),   // Animals
		makeRange(20, 40),  // Food
		makeRange(40, 60),  // Colors
		makeRange(60, 80),  // Actions
		makeRange(80, 100), // Objects
	}

	data := make([][]float32, 0)
	labels := make([]int, 0)

	for i := 0; i < NumSamples; i++ {
		catIdx := rand.Intn(len(categories))
		category := categories[catIdx]

		sequenceLen := 2*ContextWindow + 1
		sequence := make([]int, sequenceLen)

		for j := 0; j < sequenceLen; j++ {
			if rand.Float64() < 0.8 {
				sequence[j] = category[rand.Intn(len(category))]
			} else {
				sequence[j] = rand.Intn(VocabSize)
			}
		}

		target := sequence[ContextWindow]
		context := make([]float32, 2*ContextWindow)
		for j := 0; j < ContextWindow; j++ {
			context[j] = float32(sequence[j])
		}
		for j := 0; j < ContextWindow; j++ {
			context[ContextWindow+j] = float32(sequence[ContextWindow+1+j])
		}

		data = append(data, context)
		labels = append(labels, target)
	}

	return data, labels
}

func makeRange(min, max int) []int {
	result := make([]int, max-min)
	for i := range result {
		result[i] = min + i
	}
	return result
}

func buildNetworkConfig() string {
	contextSize := 2 * ContextWindow
	return fmt.Sprintf(`{
		"grid_rows": 1,
		"grid_cols": 1,
		"layers_per_cell": 3,
		"batch_size": %d,
		"layers": [
			{
				"type": "embedding",
				"vocab_size": %d,
				"embedding_dim": %d
			},
			{
				"type": "dense",
				"input_size": %d,
				"output_size": 64,
				"activation": "tanh"
			},
			{
				"type": "dense",
				"input_size": 64,
				"output_size": %d,
				"activation": "sigmoid"
			}
		]
	}`, BatchSize, VocabSize, EmbedDim, EmbedDim*contextSize, VocabSize)
}

func trainNetwork(net *nn.Network, trainData [][]float32, trainLabels []int) float64 {
	batches := createBatches(trainData, trainLabels, BatchSize)

	config := &nn.TrainingConfig{
		Epochs:          Epochs,
		LearningRate:    LearningRate,
		UseGPU:          false, // CPU only for embedding demo
		LossType:        "cross_entropy",
		PrintEveryBatch: 0,
		Verbose:         false,
	}

	_, err := net.Train(batches, config)
	if err != nil {
		fmt.Printf("Warning: Training error: %v\n", err)
		return 0
	}

	return evaluateAccuracy(net, trainData, trainLabels)
}

func createBatches(data [][]float32, labels []int, batchSize int) []nn.TrainingBatch {
	indices := rand.Perm(len(data))
	numBatches := len(data) / batchSize
	batches := make([]nn.TrainingBatch, numBatches)

	contextSize := 2 * ContextWindow
	for b := 0; b < numBatches; b++ {
		input := make([]float32, batchSize*contextSize)
		target := make([]float32, batchSize*VocabSize)

		for i := 0; i < batchSize; i++ {
			idx := indices[b*batchSize+i]
			copy(input[i*contextSize:], data[idx])
			target[i*VocabSize+labels[idx]] = 1.0
		}

		batches[b] = nn.TrainingBatch{Input: input, Target: target}
	}

	return batches
}

func evaluateAccuracy(net *nn.Network, data [][]float32, labels []int) float64 {
	correct := 0
	for i, input := range data {
		output, _ := net.ForwardCPU(input)
		predicted := argmax(output)
		if predicted == labels[i] {
			correct++
		}
	}
	return float64(correct) / float64(len(data))
}

func argmax(vec []float32) int {
	maxIdx := 0
	maxVal := vec[0]
	for i, val := range vec {
		if val > maxVal {
			maxIdx = i
			maxVal = val
		}
	}
	return maxIdx
}

func showWordSimilarities(net *nn.Network) {
	embLayer := net.GetLayer(0, 0, 0)
	if embLayer == nil || len(embLayer.EmbeddingWeights) == 0 {
		fmt.Println("      Could not extract embeddings")
		return
	}

	embeddings := embLayer.EmbeddingWeights

	categories := map[string][]int{
		"Animals": {0, 5, 10, 15},
		"Food":    {20, 25, 30, 35},
		"Colors":  {40, 45, 50, 55},
		"Actions": {60, 65, 70, 75},
		"Objects": {80, 85, 90, 95},
	}

	fmt.Println("\n  Word Similarity Analysis (Cosine Similarity):")
	fmt.Println(strings.Repeat("─", 60))

	for catName, words := range categories {
		fmt.Printf("\n  %s category:\n", catName)

		similarities := make([]float64, 0)
		for i := 0; i < len(words); i++ {
			for j := i + 1; j < len(words); j++ {
				sim := cosineSimilarity(embeddings, words[i], words[j], EmbedDim)
				similarities = append(similarities, sim)
			}
		}

		avgSim := 0.0
		for _, sim := range similarities {
			avgSim += sim
		}
		if len(similarities) > 0 {
			avgSim /= float64(len(similarities))
		}

		fmt.Printf("    Avg within-category similarity: %.4f\n", avgSim)
	}

	fmt.Println("\n  Cross-category comparison:")
	word1 := 5  // Animal
	word2 := 25 // Food
	crossSim := cosineSimilarity(embeddings, word1, word2, EmbedDim)
	fmt.Printf("    Animal-Food similarity: %.4f (should be lower)\n", crossSim)
}

func cosineSimilarity(embeddings []float32, idx1, idx2, dim int) float64 {
	start1 := idx1 * dim
	start2 := idx2 * dim

	dotProduct := 0.0
	norm1 := 0.0
	norm2 := 0.0

	for i := 0; i < dim; i++ {
		v1 := float64(embeddings[start1+i])
		v2 := float64(embeddings[start2+i])

		dotProduct += v1 * v2
		norm1 += v1 * v1
		norm2 += v2 * v2
	}

	if norm1 == 0 || norm2 == 0 {
		return 0
	}

	return dotProduct / (math.Sqrt(norm1) * math.Sqrt(norm2))
}
