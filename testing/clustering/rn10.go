package main

import (
	"fmt"
	"math/rand"
	"time"

	"github.com/openfluke/loom/nn"
)

// RN10: Deep Auto-Clustering on Synthetic Fractal Data
// Goal: Verify Blind Discovery of a known Fractal Topology.

const (
	RN10InputDim     = 128
	RN10LayerCount   = 100
	RN10ClusterSize  = 64 // Slightly tighter bottleneck
	RN10Epochs       = 2
	RN10LearningRate = float32(0.02)
)

func main() {
	rand.Seed(time.Now().UnixNano())

	fmt.Println("╔════════════════════════════════════════════════════════════════╗")
	fmt.Println("║   EXPERIMENT RN10: Synthetic Fractal Auto-Clustering         ║")
	fmt.Println("║   Data: Depth-4 Fractal Tree | Layers: 100                   ║")
	fmt.Println("╚════════════════════════════════════════════════════════════════╝")

	// 1. Generate Data
	fmt.Println("Generating Synthetic Fractal Data...")
	// 5 branches ^ 4 depth = 625 clusters.
	// ~5 points per cluster = ~3125 points.
	data, labels, truePaths := generateFractalData(5, 4, 5)
	fmt.Printf("Generated %d samples.\n", len(data))

	// 2. Model
	modelPath := "rn10_model.json"
	var net *nn.Network

	// Force clean build
	fmt.Println("\nBuilding 100-Layer Deep Networks...")
	net = buildMassiveKMeansNet()

	fmt.Println("\nStarting Greedy Layer-Wise Training...")
	trainGreedyStack(net, data)

	fmt.Printf("\nSaving model to %s...\n", modelPath)
	net.SaveModel(modelPath, "rn10")

	// 3. Validation
	printValidation(net, data, labels, truePaths)
}

// ... Data Generation & Training ... (Unchanged)

// --- Validation ---

func printValidation(net *nn.Network, data [][]float32, names []string, truePaths [][]int) {
	fmt.Println("\n=== Hierarchy Validation ===")
	fmt.Println("Comparing Ground Truth Fractal Path vs Learned Greedy Path")
	fmt.Println("Format: Name | True Path (Prefix) | Learned Path (checkpoints)")

	// Sample indices to show variety
	step := len(data) / 20
	if step == 0 {
		step = 1
	}

	for i := 0; i < len(data); i += step {
		// Trace
		learnedPath := tracePath(net, data[i])

		// Simplify learned path for display (just first few levels)
		displayPath := learnedPath
		if len(displayPath) > 5 {
			displayPath = displayPath[:5]
		}

		fmt.Printf("%-15s | True: %v | Learned: %v ...\n",
			names[i], truePaths[i], displayPath)
	}
}
func generateFractalData(branches, depth, samplesPerLeaf int) ([][]float32, []string, [][]int) {
	var vectors [][]float32
	var names []string
	var paths [][]int // Ground Truth Path [0, 2, 4...]

	// Recursive generator
	var recurse func(center []float32, currentDepth int, pathPrefix []int, scale float32)

	recurse = func(center []float32, currentDepth int, pathPrefix []int, scale float32) {
		if currentDepth == depth {
			// Leaf Cluster - Generate Samples
			for i := 0; i < samplesPerLeaf; i++ {
				// Add noise
				noise := randomVector(RN10InputDim, scale*0.5)
				vec := addVectors(center, noise)

				// Name: "Leaf-0-2-4-..."
				name := fmt.Sprintf("Leaf-%v", pathPrefix)

				vectors = append(vectors, vec)
				names = append(names, name)

				// Copy path
				p := make([]int, len(pathPrefix))
				copy(p, pathPrefix)
				paths = append(paths, p)
			}
			return
		}

		// Generate Branches for this node
		for b := 0; b < branches; b++ {
			// Random offset for this branch
			offset := randomVector(RN10InputDim, scale) // Scale decreases with depth
			newCenter := addVectors(center, offset)

			newPath := append(pathPrefix, b)
			// Recurse with smaller scale
			recurse(newCenter, currentDepth+1, newPath, scale*0.45)
		}
	}

	root := make([]float32, RN10InputDim) // Zero origin
	recurse(root, 0, []int{}, 1.0)        // Start with Scale 1.0

	return vectors, names, paths
}

func randomVector(dim int, scale float32) []float32 {
	v := make([]float32, dim)
	for i := 0; i < dim; i++ {
		v[i] = (rand.Float32()*2 - 1) * scale
	}
	return v
}

func addVectors(a, b []float32) []float32 {
	res := make([]float32, len(a))
	for i := range a {
		res[i] = a[i] + b[i]
	}
	return res
}

// --- Training ---

// --- Training ---

func trainGreedyStack(net *nn.Network, data [][]float32) {
	currentResiduals := make([][]float32, len(data))
	// Deep copy initial data
	for i, v := range data {
		row := make([]float32, len(v))
		copy(row, v)
		currentResiduals[i] = row
	}

	batchSize := 1

	for i := 0; i < RN10LayerCount; i++ { // 0 to 99
		// Create MiniNet for the Pair: Input(128) -> Enc(64) -> Dec(128)
		miniNet := nn.NewNetwork(RN10InputDim, 1, 1, 2)

		// Use configs from buildMassive (which initializes them)
		encIdx := 2 * i
		decIdx := 2*i + 1

		miniNet.SetLayer(0, 0, 0, net.Layers[encIdx])
		miniNet.SetLayer(0, 0, 1, net.Layers[decIdx])
		miniNet.InitializeWeights() // Re-init mini-net wrappers
		miniNet.BatchSize = batchSize

		fmt.Printf("[L%d] Training Pair... ", i)
		config := &nn.TrainingConfig{Epochs: 4, LearningRate: RN10LearningRate, LossType: "mse", Verbose: false} // 4 epochs per layer
		res, _ := miniNet.TrainStandard(currentResiduals, currentResiduals, config)
		fmt.Printf("Loss: %.6f\n", res.FinalLoss)

		// Store back to Main Net
		net.Layers[encIdx] = miniNet.Layers[0]
		net.Layers[decIdx] = miniNet.Layers[1]

		// Update Residuals
		nextResiduals := make([][]float32, len(data))
		// We can reuse MiniNet forward
		for j, vec := range currentResiduals {
			recon, _ := miniNet.ForwardCPU(vec)

			residual := make([]float32, len(vec))
			for k := range vec {
				residual[k] = vec[k] - recon[k]
			}
			nextResiduals[j] = residual
		}
		currentResiduals = nextResiduals
	}
}

// --- Network & Utilities ---

func buildMassiveKMeansNet() *nn.Network {
	// RVQ Stack: 100 Pairs of (Encoder, Decoder). Total 200 Layers.
	// Input(128) -> [Enc(64), Dec(128)] -> [Enc(64), Dec(128)] ...

	net := nn.NewNetwork(RN10InputDim, 1, 1, RN10LayerCount*2)

	for i := 0; i < RN10LayerCount; i++ {
		// Encoder: KMeans (128 -> 64 Probs)
		// It wraps a Dense(128 -> 64) for features.
		inner := nn.InitDenseLayer(RN10InputDim, RN10ClusterSize, nn.ActivationTanh)

		enc := nn.InitKMeansLayer(RN10ClusterSize, inner, "probabilities")
		enc.KMeansLearningRate = RN10LearningRate
		enc.KMeansTemperature = 1.0
		if enc.SubNetwork != nil {
			if sn, ok := enc.SubNetwork.(*nn.Network); ok {
				sn.BatchSize = 1
			}
		}

		// Decoder: Dense (64 -> 128)
		dec := nn.InitDenseLayer(RN10ClusterSize, RN10InputDim, nn.ActivationTanh)

		net.SetLayer(0, 0, 2*i, enc)
		net.SetLayer(0, 0, 2*i+1, dec)
	}

	net.InitializeWeights()
	return net
}

func tracePath(net *nn.Network, vec []float32) []int {
	path := make([]int, 0)

	// Working Residual
	residual := make([]float32, len(vec))
	copy(residual, vec)

	// Iterate Pairs
	for i := 0; i < RN10LayerCount; i++ {
		// 1. Forward Encoder
		enc := net.Layers[2*i]
		dec := net.Layers[2*i+1]

		// Need intermediate output from Enc to get ID

		// Enc Forward
		encNet := nn.NewNetwork(RN10InputDim, 1, 1, 1)
		encNet.SetLayer(0, 0, 0, enc)
		encNet.LayersPerCell = 1

		encOut, _ := encNet.ForwardCPU(residual)
		id := argmax(encOut)
		path = append(path, id)

		// Dec Forward
		decNet := nn.NewNetwork(RN10ClusterSize, 1, 1, 1)
		decNet.SetLayer(0, 0, 0, dec)
		decNet.LayersPerCell = 1

		recon, _ := decNet.ForwardCPU(encOut)

		// Sub residual
		for k := range residual {
			residual[k] -= recon[k]
		}
	}
	return path
}

func argmax(v []float32) int {
	mi, mv := 0, v[0]
	for i, val := range v {
		if val > mv {
			mi, mv = i, val
		}
	}
	return mi
}
