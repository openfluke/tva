package main

import (
	"fmt"
	"math/rand"
	"time"

	"github.com/openfluke/loom/nn"
)

// Multi-Head Attention Text Classification Demo
// Demonstrates MHA layers on simple text classification
// Note: MHA GPU support may be limited, demo uses CPU only

const (
	VocabSize       = 50 // Small vocabulary
	EmbedDim        = 32 // Embedding dimension
	SeqLength       = 8  // Sentence length (padded)
	NumClasses      = 2  // Positive, Negative
	SamplesPerClass = 500
	Epochs          = 60
	LearningRate    = 0.05
	NumHeads        = 4
	BatchSize       = 20
)

func main() {
	rand.Seed(time.Now().UnixNano())

	fmt.Println("╔════════════════════════════════════════════════════════════════╗")
	fmt.Println("║   MHA Demo: Text Sentiment Classification                     ║")
	fmt.Println("╚════════════════════════════════════════════════════════════════╝")

	// Generate dataset
	fmt.Println("\n[1/2] Generating synthetic text patterns...")
	trainData, trainLabels := generateTextDataset()
	fmt.Printf("      Generated %d sentences (seq_len=%d, vocab=%d)\n",
		len(trainData), SeqLength, VocabSize)

	// Train model (CPU only for MHA demo)
	fmt.Println("\n[2/2] Training on CPU...")
	jsonConfig := buildNetworkConfig()
	net, err := nn.BuildNetworkFromJSON(jsonConfig)
	if err != nil {
		panic(err)
	}
	net.InitializeWeights()

	// Prepare targets for EvaluateNetwork
	evalTargets := make([]float64, len(trainLabels))
	for i, v := range trainLabels {
		evalTargets[i] = float64(v)
	}

	startCPU := time.Now()
	metricsBefore, _ := net.EvaluateNetwork(trainData, evalTargets)

	accCPU := trainNetwork(net, trainData, trainLabels)
	timeCPU := time.Since(startCPU)

	metricsAfter, _ := net.EvaluateNetwork(trainData, evalTargets)
	nn.PrintDeviationComparisonTable("CPU Results: MHA Text", metricsBefore, metricsAfter)

	fmt.Printf("\n      ✅ Training complete: Accuracy=%.2f%%, Time=%v\n", accCPU*100, timeCPU)
	fmt.Println("\n✅ MHA Demo Complete!")
	fmt.Println("Note: This demo uses CPU only as MHA with Embedding layers")
	fmt.Println("may have limited GPU support in the current implementation.")
}

func generateTextDataset() ([][]float32, []int) {
	positiveWords := []int{1, 2, 3, 4, 5, 6, 7, 8, 9, 10}
	negativeWords := []int{11, 12, 13, 14, 15, 16, 17, 18, 19, 20}
	neutralWords := []int{21, 22, 23, 24, 25, 26, 27, 28, 29, 30}

	totalSamples := NumClasses * SamplesPerClass
	data := make([][]float32, totalSamples)
	labels := make([]int, totalSamples)

	idx := 0
	for class := 0; class < NumClasses; class++ {
		for sample := 0; sample < SamplesPerClass; sample++ {
			sentence := make([]float32, SeqLength)

			var targetWords []int
			if class == 0 {
				targetWords = positiveWords
			} else {
				targetWords = negativeWords
			}

			numTargetWords := 3 + rand.Intn(3)
			for i := 0; i < SeqLength; i++ {
				if i < numTargetWords {
					sentence[i] = float32(targetWords[rand.Intn(len(targetWords))])
				} else {
					sentence[i] = float32(neutralWords[rand.Intn(len(neutralWords))])
				}
			}

			rand.Shuffle(SeqLength, func(i, j int) {
				sentence[i], sentence[j] = sentence[j], sentence[i]
			})

			data[idx] = sentence
			labels[idx] = class
			idx++
		}
	}

	return data, labels
}

func buildNetworkConfig() string {
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
				"type": "multi_head_attention",
				"d_model": %d,
				"num_heads": %d,
				"seq_length": %d
			},
			{
				"type": "dense",
				"input_size": %d,
				"output_size": %d,
				"activation": "sigmoid"
			}
		]
	}`, BatchSize, VocabSize, EmbedDim, EmbedDim, NumHeads, SeqLength, EmbedDim*SeqLength, NumClasses)
}

func trainNetwork(net *nn.Network, trainData [][]float32, trainLabels []int) float64 {
	config := &nn.TrainingConfig{
		Epochs:          Epochs,
		LearningRate:    LearningRate,
		UseGPU:          false, // CPU only for MHA demo
		LossType:        "mse",
		PrintEveryBatch: 0,
		Verbose:         true,
	}

	_, err := net.TrainLabels(trainData, trainLabels, config)
	if err != nil {
		fmt.Printf("Warning: Training error: %v\n", err)
		return 0
	}

	return evaluateAccuracy(net, trainData, trainLabels)
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
