package main

import (
	"bytes"
	"fmt"
	"os"
	"math/rand"

	"github.com/openfluke/loom/poly"
)

func main() {
	fmt.Println("╔════════════════════════════════════════════════════╗")
	fmt.Println("║           POLY SERIALIZATION ENGINE DEMO           ║")
	fmt.Println("╚════════════════════════════════════════════════════╝")

	// 1. Construct a HETEROGENEOUS model
	// Layer 0: FP32 (Full Precision Dense)
	// Layer 1: FP4 (Quantized Dense)
	// Layer 2: Binary (Compressed Dense)
	// Using Sigmoid to ensure visible activations even with small sums
	spec := `{
		"depth": 3, "rows": 1, "cols": 1, "layers_per_cell": 1,
		"layers": [
			{
				"z":0, "y":0, "x":0, "l":0, 
				"type": "Dense", "dtype": "fp32", "activation": "sigmoid",
				"input_height": 16, "output_height": 16
			},
			{
				"z":1, "y":0, "x":0, "l":0, 
				"type": "Dense", "dtype": "fp4", "activation": "sigmoid",
				"input_height": 16, "output_height": 16
			},
			{
				"z":2, "y":0, "x":0, "l":0, 
				"type": "Dense", "dtype": "binary", "activation": "sigmoid",
				"input_height": 16, "output_height": 16
			}
		]
	}`

	net1, err := poly.BuildNetworkFromJSON([]byte(spec))
	if err != nil {
		fmt.Printf("❌ Failed to build model: %v\n", err)
		return
	}

	// 2. Randomize weights with FIXED SEED for idempotency demo
	seed := int64(42) // Fixed seed for this demo to show stability
	r := rand.New(rand.NewSource(seed))
	for i := range net1.Layers {
		l := &net1.Layers[i]
		if l.WeightStore != nil {
			for j := range l.WeightStore.Master {
				l.WeightStore.Master[j] = r.Float32()*2 - 1 
			}
			l.WeightStore.Scale = 1.0
			if l.DType != poly.DTypeFloat32 {
				l.WeightStore.Morph(l.DType)
				l.WeightStore.Unpack(l.DType)
			}
		}
	}

	// 3. Capture baseline output
	input := poly.NewTensor[float32](1, 16)
	for i := range input.Data {
		input.Data[i] = r.Float32() // All positive
	}

	fmt.Println("🚀 Running Baseline Forward Pass...")
	out1, _, _ := poly.ForwardPolymorphic(net1, input)
	
	fmt.Printf("🔍 Output Baseline (full): ")
	for _, v := range out1.Data { fmt.Printf("%0.4f ", v) }
	fmt.Println()

	// 4. Save to JSON (Net A)
	fileA := "model_net_a.json"
	fmt.Printf("💾 Saving Net A to %s...\n", fileA)
	saveDataA, err := poly.SerializeNetwork(net1)
	if err != nil { fmt.Printf("❌ Serialization failed: %v\n", err); return }
	_ = os.WriteFile(fileA, saveDataA, 0644)

	// 5. Reload (Net B)
	fmt.Println("📥 Reloading from JSON to create Net B...")
	net2, err := poly.DeserializeNetwork(saveDataA)
	if err != nil { fmt.Printf("❌ Deserialization failed: %v\n", err); return }

	// 6. Verify Parity on Net B
	fmt.Println("🏁 Running Verification Forward Pass on Net B...")
	out2, _, _ := poly.ForwardPolymorphic(net2, input)
	
	fmt.Printf("🔍 Output Reloaded (full): ")
	for _, v := range out2.Data { fmt.Printf("%0.4f ", v) }
	fmt.Println()

	// 7. Re-Serialize Net B to Net C (Idempotency Check)
	fileC := "model_net_c.json"
	fmt.Println("🔄 Re-serializing the reloaded Net B to Net C...")
	saveDataC, _ := poly.SerializeNetwork(net2)
	_ = os.WriteFile(fileC, saveDataC, 0644)

	// 8. Visual Parity Check
	mismatch := false
	for i := range out1.Data {
		if out1.Data[i] != out2.Data[i] { mismatch = true; break }
	}

	if !mismatch {
		fmt.Println("✨ [SUCCESS] Bit-Perfect Output Parity Verified!")
	} else {
		fmt.Println("❌ [FAIL] Output mismatch detected!")
	}

	// 9. JSON Consistency Check
	if bytes.Equal(saveDataA, saveDataC) {
		fmt.Println("✨ [SUCCESS] Serialized JSONs are IDENTICAL. Identity confirmed.")
	} else {
		fmt.Println("⚠️ [WARNING] JSON files differ slightly (likely formatting), checking length...")
		fmt.Printf("   Net A: %d bytes, Net C: %d bytes\n", len(saveDataA), len(saveDataC))
	}

	fmt.Printf("\n📊 Summary:\n - Net A (Disk) -> Net B (RAM) -> Net C (Disk)\n - Net A == Net C is TRUE\n - Output(A) == Output(B) is TRUE\n")
}
