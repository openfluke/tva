package main

import (
	"fmt"
	"math/rand"
	"time"

	"github.com/openfluke/loom/poly"
)

func main() {
	fmt.Println("╔════════════════════════════════════════════════════╗")
	fmt.Println("║       SERIALIZATION ENGINE EXHAUSTIVE VERIFIER     ║")
	fmt.Println("╚════════════════════════════════════════════════════╝")

	layerTypes := []string{"Dense", "CNN1", "CNN2", "CNN3", "RNN", "LSTM", "MHA", "SwiGLU", "RMSNorm", "LayerNorm", "ConvTransposed1D", "ConvTransposed2D", "ConvTransposed3D", "Embedding", "KMeans", "Softmax", "Parallel", "Sequential"}
	dNames := []string{"fp64", "fp32", "fp16", "bfloat16", "fp8", "fp8e5m2", "int64", "int32", "int16", "int8", "uint64", "uint32", "uint16", "uint8", "int4", "uint4", "fp4", "int2", "uint2", "ternary", "binary"}

	total := len(layerTypes) * len(dNames)
	passed := 0

	for _, lName := range layerTypes {
		for _, dName := range dNames {
			err := runFinalExhaustiveTest(lName, dName)
			if err != nil {
				fmt.Printf("❌ [FAIL] %s - %s: %v\n", lName, dName, err)
			} else {
				passed++
				fmt.Print(".")
			}
		}
		fmt.Println()
	}

	fmt.Printf("\n✨ [VERIFICATION COMPLETE] %d/%d PERMUTATIONS PASSED ✨\n", passed, total)
}

func runFinalExhaustiveTest(lName, dName string) error {
	// Construct 3-layer model with FULL SPATIAL SPECS
	spec1 := fmt.Sprintf(`{
		"depth": 3, "rows": 1, "cols": 1, "layers_per_cell": 1,
		"layers": [
			{"z":0, "y":0, "x":0, "l":0, "type": "%s", "dtype": "%s", "input_height": 16, "input_width": 16, "input_depth": 16, "output_height": 16, "output_width": 16, "output_depth": 16, "d_model": 16, "num_heads": 4, "kernel_size": 3, "filters": 16, "input_channels": 16, "vocab_size": 16, "embedding_dim": 16, "num_clusters": 16, "seq_length": 1},
			{"z":1, "y":0, "x":0, "l":0, "type": "%s", "dtype": "%s", "input_height": 16, "input_width": 16, "input_depth": 16, "output_height": 16, "output_width": 16, "output_depth": 16, "d_model": 16, "num_heads": 4, "kernel_size": 3, "filters": 16, "input_channels": 16, "vocab_size": 16, "embedding_dim": 16, "num_clusters": 16, "seq_length": 1},
			{"z":2, "y":0, "x":0, "l":0, "type": "%s", "dtype": "%s", "input_height": 16, "input_width": 16, "input_depth": 16, "output_height": 16, "output_width": 16, "output_depth": 16, "d_model": 16, "num_heads": 4, "kernel_size": 3, "filters": 16, "input_channels": 16, "vocab_size": 16, "embedding_dim": 16, "num_clusters": 16, "seq_length": 1}
		]
	}`, lName, dName, lName, dName, lName, dName)

	net1, err := poly.BuildNetworkFromJSON([]byte(spec1))
	if err != nil {
		return fmt.Errorf("build net1 failed: %v", err)
	}

	// Native randomization and QUANTIZATION ALIGNMENT
	seed := time.Now().UnixNano()
	for i := range net1.Layers {
		l := &net1.Layers[i]
		if l.WeightStore != nil {
			l.WeightStore.Randomize(seed, 0.1)
			
			// Morph and Unpack net1 so it matches the state net2 will have after loading
			// This ensures we are testing the serialization of the DType, not FP32->DType conversion
			dt := l.DType
			if dt != poly.DTypeFloat32 {
				l.WeightStore.Morph(dt)
				l.WeightStore.Unpack(dt)
			}
		}
	}

	// Large input buffer for all dimensionalities
	inputSize := 1000000 
	inputSlice := make([]float32, inputSize)
	for i := range inputSlice {
		inputSlice[i] = rand.Float32()*2 - 1
	}

	var inputTensor *poly.Tensor[float32]
	switch lName {
	case "CNN1", "ConvTransposed1D", "RNN", "LSTM", "MHA", "SwiGLU":
		inputTensor = poly.NewTensorFromSlice(inputSlice[:256], 1, 16, 16)
	case "CNN2", "ConvTransposed2D":
		inputTensor = poly.NewTensorFromSlice(inputSlice[:4096], 1, 16, 16, 16)
	case "CNN3", "ConvTransposed3D":
		inputTensor = poly.NewTensorFromSlice(inputSlice[:65536], 1, 16, 16, 16, 16)
	case "Embedding":
		inputTensor = poly.NewTensorFromSlice(inputSlice[:16], 1, 16)
	default:
		inputTensor = poly.NewTensorFromSlice(inputSlice[:16], 1, 16)
	}

	out1, _, _ := poly.ForwardPolymorphic(net1, inputTensor)
	if out1 == nil || len(out1.Data) == 0 {
		return fmt.Errorf("zero output from baseline")
	}

	// Persistence Tunnel
	saveData, err := poly.SerializeNetwork(net1)
	if err != nil {
		return fmt.Errorf("serialize failed: %v", err)
	}

	net2, err := poly.DeserializeNetwork(saveData)
	if err != nil {
		return fmt.Errorf("deserialize failed: %v", err)
	}

	out2, _, _ := poly.ForwardPolymorphic(net2, inputTensor)

	// Parity Assertion
	if len(out1.Data) != len(out2.Data) {
		return fmt.Errorf("output length mismatch: %d != %d", len(out1.Data), len(out2.Data))
	}

	for i := range out1.Data {
		diff := out1.Data[i] - out2.Data[i]
		if diff < 0 { diff = -diff }
		if diff > 1e-4 { // Slightly relaxed for low-bit drift in chained layers
			return fmt.Errorf("output mismatch at index %d: %f != %f", i, out1.Data[i], out2.Data[i])
		}
	}

	// Structural Parity
	for i := range net1.Layers {
		l1, l2 := &net1.Layers[i], &net2.Layers[i]
		if l1.Type != l2.Type { return fmt.Errorf("layer %d type mismatch: %v != %v", i, l1.Type, l2.Type) }
		if l1.DType != l2.DType { return fmt.Errorf("layer %d dtype mismatch: %v (%d) != %v (%d)", i, l1.DType, l1.DType, l2.DType, l2.DType) }
		
		if l1.WeightStore != nil {
			if l2.WeightStore == nil { return fmt.Errorf("layer %d WeightStore missing", i) }
			if len(l1.WeightStore.Master) != len(l2.WeightStore.Master) {
				return fmt.Errorf("layer %d weight count mismatch: %d != %d", i, len(l1.WeightStore.Master), len(l2.WeightStore.Master))
			}
			for k := 0; k < 10; k++ {
				idx := rand.Intn(len(l1.WeightStore.Master))
				if l1.WeightStore.Master[idx] != l2.WeightStore.Master[idx] {
					return fmt.Errorf("layer %d weight mismatch at %d: %f != %f", i, idx, l1.WeightStore.Master[idx], l2.WeightStore.Master[idx])
				}
			}
		}
	}

	return nil
}
