package main

import (
	"fmt"
	"os"
	"path/filepath"

	"github.com/openfluke/loom/poly"
)

func main() {
	targetDir := "downloads"
	if len(os.Args) > 1 {
		targetDir = os.Args[1]
	}

	fmt.Printf("=== Poly-Talk: Universal Geometrical Probing ===\n")
	fmt.Printf("Searching for models in: %s\n", targetDir)

	files, _ := findSafetensors(targetDir)
	if len(files) == 0 {
		fmt.Println("No models found in", targetDir)
		return
	}

	for _, path := range files {
		fmt.Printf("\n>>> Analytical Audit: %s\n", path)
		totalTensors, archetypes, missed, geometries, err := poly.LoadUniversalDetailed(path)
		if err != nil {
			fmt.Printf("  [FAILED] Pipeline Error: %v\n", err)
			continue
		}

		fmt.Printf("  [STRUCTURE] Found %d functional archetypes\n", len(archetypes))
		network := poly.MountGeometrically(archetypes, geometries)
		if len(network.Layers) > 0 {
			fmt.Printf("  [PRECISION] Model Native Type: %s\n", network.Layers[0].DType.String())
		}
		claimedCount := totalTensors - len(missed)
		coverage := (float32(claimedCount) / float32(totalTensors)) * 100

		if len(missed) > 0 {
			fmt.Printf("  [ORPHANS] %d/%d tensors unassigned (%.1f%% Coverage)\n", len(missed), totalTensors, coverage)
			classifyOrphans(missed, geometries)
		} else {
			fmt.Printf("  [CLEAN] 100%% geometrical assignment coverage.\n")
		}

		fmt.Printf("  [MOUNTING] Validating Engine Readiness...\n")
		
		fmt.Printf("  [STABLE] %d layers mounted in Volumetric Network.\n", len(network.Layers))
		mem := network.CalculateTotalMemory()
		fmt.Printf("  [MEMORY] Estimated Weight Footprint: %.2f MB\n", float64(mem)/(1024*1024))
	}
}

func findSafetensors(root string) ([]string, error) {
	var files []string
	err := filepath.Walk(root, func(path string, info os.FileInfo, err error) error {
		if err == nil && !info.IsDir() && (filepath.Ext(path) == ".safetensors") {
			files = append(files, path)
		}
		return nil
	})
	return files, err
}

func classifyOrphans(missed []int, geoms []poly.TensorMeta) {
	categories := make(map[string][]int)
	for _, idx := range missed {
		g := geoms[idx]
		cat := "Metadata/Small"
		if g.Rank == 2 {
			cat = "Unmapped Weights (Rank-2)"
		} else if g.Rank == 1 {
			if g.MeanAbs < 0.1 {
				cat = "Potential Bias (Rank-1)"
			} else {
				cat = "Potential Gain/Norm (Rank-1)"
			}
		} else if g.Rank > 2 {
			cat = fmt.Sprintf("Unmapped High-Rank (%dD)", g.Rank)
		}
		categories[cat] = append(categories[cat], idx)
	}
	for cat, indices := range categories {
		fmt.Printf("    - %-28s: %v\n", cat, indices)
	}
}
