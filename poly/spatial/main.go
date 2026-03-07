package main

import (
	"fmt"
	"github.com/openfluke/loom/poly"
)

func main() {
	fmt.Println("=== M-POLY-VTD Volumetric Spatial Routing (Spatial Hopping) Verification ===")
	
	// 1. Create a 3D Volumetric Network (2x2x2 with 1 layer per cell)
	net := poly.NewVolumetricNetwork(2, 2, 2, 1)
	
	inputSize := 64
	
	// 2. Setup a "Remote" target layer at (0, 1, 1, 0) - [Z, Y, X, L]
	remoteLayer := net.GetLayer(0, 1, 1, 0)
	remoteLayer.Type = poly.LayerDense
	remoteLayer.InputHeight = inputSize
	remoteLayer.OutputHeight = 32
	remoteLayer.DType = poly.DTypeFloat32
	remoteLayer.WeightStore = &poly.WeightStore{
		Master: make([]float32, inputSize*32),
		Scale: 1.0,
	}
	// Fill weights with something recognizable
	for i := range remoteLayer.WeightStore.Master {
		remoteLayer.WeightStore.Master[i] = 1.5
	}

	// 3. Setup a "Parallel" layer at (0, 0, 0, 0) that is essentially a Spatial Hop to (0, 1, 1, 0)
	routingLayer := net.GetLayer(0, 0, 0, 0)
	routingLayer.Type = poly.LayerParallel
	routingLayer.CombineMode = "grid_scatter"
	routingLayer.ParallelBranches = []poly.VolumetricLayer{
		{
			IsRemoteLink: true,
			TargetZ:      0,
			TargetY:      1,
			TargetX:      1,
			TargetL:      0,
		},
	}

	// 4. Create input
	input := poly.NewTensor[float32](1, inputSize)
	for i := range input.Data { input.Data[i] = 1.0 }

	// 5. Dispatch via the router
	fmt.Println("\n--- Dispatching via Spatial Router (0,0,0,0) -> Hop -> (0,1,1,0) ---")
	_, out := poly.DispatchLayer(routingLayer, input)
	
	fmt.Printf("Output Size: %d\n", len(out.Data))
	if len(out.Data) == 32 {
		fmt.Println("SUCCESS: Spatial Routing achieved correct output size.")
	} else {
		fmt.Printf("FAILURE: Expected size 32, got %d\n", len(out.Data))
		return
	}

	// 6. Verify value (1.5 * input=1.0 * inputSize=64 = 96.0)
	val := float32(out.Data[0])
	fmt.Printf("Sample Output Value: %v\n", val)
	if val > 95.0 && val < 97.0 {
		fmt.Println("SUCCESS: Spatial Routing preserved numerical integrity.")
	} else {
		fmt.Printf("FAILURE: Expected roughly 96.0, got %v\n", val)
	}

	fmt.Println("\n--- 3D Volumetric Spatial Hopping Verified! o_O ---")
}
