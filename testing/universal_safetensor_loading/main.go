package main

import (
	"encoding/binary"
	"fmt"
	"io"
	"math"
	"net/http"
	"os"
	"path/filepath"
	"sort"

	"github.com/openfluke/loom/nn"
)

// UserHints allows manual mapping for ambiguous tensor indices [Index -> nn.LayerType]
var UserHints = make(map[int]nn.LayerType)

func main() {
	targetDir := "downloads"
	if len(os.Args) > 1 {
		targetDir = os.Args[1]
	}

	fmt.Printf("=== Universal Geometrical Probing & Batch Test ===\n")
	fmt.Printf("Searching for models in: %s\n", targetDir)

	files, _ := findSafetensors(targetDir)
	if len(files) == 0 {
		fmt.Println("No models found. Triggering full maintenance download suite...")
		downloadModels()
		return
	}

	for _, path := range files {
		fmt.Printf("\n>>> Analytical Audit: %s\n", path)
		totalTensors, archetypes, missed, geometries, err := LoadUniversalDetailed(path)
		if err != nil {
			fmt.Printf("  [FAILED] Pipeline Error: %v\n", err)
			continue
		}

		fmt.Printf("  [STRUCTURE] Found %d functional archetypes\n", len(archetypes))
		claimedCount := totalTensors - len(missed)
		coverage := (float32(claimedCount) / float32(totalTensors)) * 100

		if len(missed) > 0 {
			fmt.Printf("  [ORPHANS] %d/%d tensors unassigned (%.1f%% Coverage)\n", len(missed), totalTensors, coverage)
			classifyOrphans(missed, geometries)
		} else {
			fmt.Printf("  [CLEAN] 100%% geometrical assignment coverage.\n")
		}

		fmt.Printf("  [MOUNTING] Validating Engine Readiness...\n")
		network := mountGeometrically(archetypes, geometries)
		passed, total := verifyMounting(network)
		if passed == total && total > 0 {
			fmt.Printf("  [STABLE] %d/%d parameter slots verified for engine execution.\n", passed, total)
		} else if total == 0 {
			fmt.Printf("  [EMPTY] No parameters mapped for this model.\n")
		} else {
			fmt.Printf("  [FRAGILE] %d/%d parameters validated. Incomplete mounting detected.\n", passed, total)
		}
	}
}

func classifyOrphans(missed []int, geoms []TensorMeta) {
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

func verifyMounting(n *nn.Network) (passed, total int) {
	for _, l := range n.Layers {
		switch l.Type {
		case nn.LayerDense:
			total += 1
			if len(l.Kernel) > 0 {
				passed++
			}
			if len(l.Bias) > 0 {
				total++
				passed++
			}
		case nn.LayerSwiGLU:
			total += 3
			if len(l.GateWeights) > 0 {
				passed++
			}
			if len(l.UpWeights) > 0 {
				passed++
			}
			if len(l.DownWeights) > 0 {
				passed++
			}
		case nn.LayerLSTM:
			total += 2
			if len(l.WeightIH) > 0 {
				passed++
			}
			if len(l.WeightHH) > 0 {
				passed++
			}
		case nn.LayerEmbedding:
			total += 1
			if len(l.EmbeddingWeights) > 0 {
				passed++
			}
		case nn.LayerNorm, nn.LayerRMSNorm:
			total += 1
			if len(l.Gamma) > 0 {
				passed++
			}
			if len(l.Beta) > 0 {
				total++
				passed++
			}
		case nn.LayerConv2D:
			total += 1
			if len(l.Kernel) > 0 {
				passed++
			}
			if len(l.Bias) > 0 {
				total++
				passed++
			}
		case nn.LayerMultiHeadAttention:
			total += 4
			if len(l.QWeights) > 0 {
				passed++
			}
			if len(l.KWeights) > 0 {
				passed++
			}
			if len(l.VWeights) > 0 {
				passed++
			}
			if len(l.OutputWeight) > 0 {
				passed++
			}
			if len(l.QBias) > 0 {
				total++
				passed++
			}
			if len(l.KBias) > 0 {
				total++
				passed++
			}
			if len(l.VBias) > 0 {
				total++
				passed++
			}
			if len(l.OutputBias) > 0 {
				total++
				passed++
			}
		case nn.LayerConv1D:
			total += 1
			if len(l.Conv1DKernel) > 0 {
				passed++
			}
			if len(l.Conv1DBias) > 0 {
				total++
				passed++
			}
		case nn.LayerRNN:
			total += 2
			if len(l.WeightIH) > 0 {
				passed++
			}
			if len(l.WeightHH) > 0 {
				passed++
			}
		}
	}
	return
}

type TensorMeta struct {
	Idx      int
	Shape    []int
	Data     []float32
	MeanAbs  float32
	Variance float32
	Rank     int
}

type LayerArchetype struct {
	Type        nn.LayerType
	TypeName    string
	Indices     map[string]int
	GeomMetrics map[string]int
}

func LoadUniversalDetailed(path string) (int, []LayerArchetype, []int, []TensorMeta, error) {
	data, err := os.ReadFile(path)
	if err != nil {
		return 0, nil, nil, nil, err
	}
	tws, err := nn.LoadSafetensorsWithShapes(data)
	if err != nil {
		return 0, nil, nil, nil, err
	}

	var names []string
	for k := range tws {
		names = append(names, k)
	}
	sort.Strings(names)

	var geometries []TensorMeta
	for i, name := range names {
		t := tws[name]
		m, v := weightDistribution(t.Values)
		geometries = append(geometries, TensorMeta{
			Idx: i, Shape: t.Shape, Data: t.Values, MeanAbs: m, Variance: v, Rank: len(t.Shape),
		})
	}

	archetypes, missed := probeDeepGeometry(geometries)
	return len(geometries), archetypes, missed, geometries, nil
}

func weightDistribution(data []float32) (meanAbs, variance float32) {
	if len(data) == 0 {
		return 0, 0
	}
	var sumAbs, sumSq float64
	for _, v := range data {
		av := math.Abs(float64(v))
		sumAbs += av
		sumSq += av * av
	}
	mean := float32(sumAbs / float64(len(data)))
	varSq := float32(sumSq/float64(len(data)) - float64(mean*mean))
	return mean, varSq
}

func probeDeepGeometry(geoms []TensorMeta) ([]LayerArchetype, []int) {
	var archetypes []LayerArchetype
	used := make(map[int]bool)

	// Step 1: Hints
	for idx, hType := range UserHints {
		if idx < len(geoms) {
			used[idx] = true
			archetypes = append(archetypes, LayerArchetype{
				Type: hType, TypeName: "HINTED Layer",
				Indices: map[string]int{"w": idx},
			})
		}
	}

	// Step 2: Complex (MHA, LSTM, SwiGLU)
	for i := range geoms {
		if used[i] {
			continue
		}
		if arch, ok := matchMHA(geoms, i, used); ok {
			archetypes = append(archetypes, arch)
			continue
		}
		if arch, ok := matchLSTM(geoms, i, used); ok {
			archetypes = append(archetypes, arch)
			continue
		}
		if arch, ok := matchFFN(geoms, i, used); ok {
			archetypes = append(archetypes, arch)
			continue
		}
		if arch, ok := matchNormPair(geoms, i, used); ok {
			archetypes = append(archetypes, arch)
			continue
		}
	}

	// Step 3: Atomic & Metadata
	for i := range geoms {
		if used[i] {
			continue
		}
		g := geoms[i]
		// Metadata Sniff (tiny tensors)
		if len(g.Data) < 10 {
			used[i] = true
			archetypes = append(archetypes, LayerArchetype{
				Type: nn.LayerSequential, TypeName: "Structural Metadata",
				Indices: map[string]int{"m": i},
			})
			continue
		}
		if g.Rank == 4 {
			used[i] = true
			arch := LayerArchetype{
				Type: nn.LayerConv2D, TypeName: "Conv2D",
				Indices:     map[string]int{"w": i},
				GeomMetrics: map[string]int{"f": g.Shape[0], "c": g.Shape[1], "k": g.Shape[2]},
			}
			// Greedy Bias Sniff
			for j, o := range geoms {
				if !used[j] && o.Rank == 1 && o.Shape[0] == g.Shape[0] && o.MeanAbs < 0.2 {
					used[j] = true
					arch.Indices["b"] = j
					break
				}
			}
			archetypes = append(archetypes, arch)
		} else if g.Rank == 3 {
			used[i] = true
			arch := LayerArchetype{
				Type: nn.LayerConv1D, TypeName: "Conv1D",
				Indices:     map[string]int{"w": i},
				GeomMetrics: map[string]int{"f": g.Shape[0], "c": g.Shape[1], "k": g.Shape[2]},
			}
			// Greedy Bias Sniff
			for j, o := range geoms {
				if !used[j] && o.Rank == 1 && o.Shape[0] == g.Shape[0] && o.MeanAbs < 0.2 {
					used[j] = true
					arch.Indices["b"] = j
					break
				}
			}
			archetypes = append(archetypes, arch)
		} else if g.Rank == 2 {
			used[i] = true
			arch := LayerArchetype{Indices: make(map[string]int)}
			if g.Shape[0] > g.Shape[1]*10 {
				arch.Type = nn.LayerEmbedding
				arch.TypeName = "Embedding Cluster"
				arch.Indices["w"] = i
				arch.GeomMetrics = map[string]int{"v": g.Shape[0], "d": g.Shape[1]}
			} else {
				arch.Type = nn.LayerDense
				arch.TypeName = "Dense Linear"
				arch.Indices["w"] = i
				arch.GeomMetrics = map[string]int{"out": g.Shape[0], "in": g.Shape[1]}
				// Greedy Bias Sniff
				for j, o := range geoms {
					if !used[j] && o.Rank == 1 && o.Shape[0] == g.Shape[0] && o.MeanAbs < 0.2 {
						used[j] = true
						arch.Indices["b"] = j
						break
					}
				}
			}
			archetypes = append(archetypes, arch)
		} else if g.Rank == 1 && g.MeanAbs > 0.4 {
			used[i] = true
			archetypes = append(archetypes, LayerArchetype{
				Type: nn.LayerRMSNorm, TypeName: "Normalization Parameter",
				Indices: map[string]int{"w": i}, GeomMetrics: map[string]int{"d": g.Shape[0]},
			})
		}
	}

	var missed []int
	for i := range geoms {
		if !used[i] {
			missed = append(missed, i)
		}
	}
	return archetypes, missed
}

func matchMHA(geoms []TensorMeta, pivot int, used map[int]bool) (LayerArchetype, bool) {
	g := geoms[pivot]
	if g.Rank != 2 || g.Shape[0] != g.Shape[1] {
		return LayerArchetype{}, false
	}
	dim := g.Shape[0]
	cluster := []int{pivot}
	for j, o := range geoms {
		if used[j] || j == pivot || o.Rank != 2 || o.Shape[0] != dim || o.Shape[1] != dim {
			continue
		}
		cluster = append(cluster, j)
		if len(cluster) == 4 {
			break
		}
	}
	if len(cluster) == 4 {
		for _, idx := range cluster {
			used[idx] = true
		}
		arch := LayerArchetype{
			Type: nn.LayerMultiHeadAttention, TypeName: "Multi-Head Attention",
			Indices:     map[string]int{"q": cluster[0], "k": cluster[1], "v": cluster[2], "o": cluster[3]},
			GeomMetrics: map[string]int{"d": dim},
		}
		// Greedy Bias Sniff for MHA
		for i, name := range []string{"qb", "kb", "vb", "ob"} {
			_ = cluster[i] // Pivot placeholder
			for j, o := range geoms {
				if !used[j] && o.Rank == 1 && o.Shape[0] == dim && o.MeanAbs < 0.2 {
					used[j] = true
					arch.Indices[name] = j
					break
				}
			}
		}
		return arch, true
	}
	return LayerArchetype{}, false
}

func matchLSTM(geoms []TensorMeta, pivot int, used map[int]bool) (LayerArchetype, bool) {
	g := geoms[pivot]
	if g.Rank != 2 || g.Shape[0] != g.Shape[1]*4 {
		return LayerArchetype{}, false
	}
	for j, o := range geoms {
		if used[j] || j == pivot || o.Rank != 2 || o.Shape[0] != g.Shape[0] || o.Shape[1] != g.Shape[1] {
			continue
		}
		used[pivot] = true
		used[j] = true
		return LayerArchetype{
			Type: nn.LayerLSTM, TypeName: "LSTM Unit",
			Indices:     map[string]int{"ih": pivot, "hh": j},
			GeomMetrics: map[string]int{"h": g.Shape[1]},
		}, true
	}
	return LayerArchetype{}, false
}

func matchFFN(geoms []TensorMeta, pivot int, used map[int]bool) (LayerArchetype, bool) {
	g := geoms[pivot]
	if g.Rank != 2 {
		return LayerArchetype{}, false
	}
	da, db := g.Shape[0], g.Shape[1]
	cluster := []int{pivot}
	for j, o := range geoms {
		if used[j] || j == pivot || o.Rank != 2 {
			continue
		}
		if (o.Shape[0] == da && o.Shape[1] == db) || (o.Shape[0] == db && o.Shape[1] == da) {
			cluster = append(cluster, j)
			if len(cluster) == 3 {
				break
			}
		}
	}
	if len(cluster) == 3 {
		var downIdx int = -1
		for _, idx := range cluster {
			if geoms[idx].Shape[0] < geoms[idx].Shape[1] {
				downIdx = idx
			}
		}
		if downIdx == -1 {
			return LayerArchetype{}, false
		}
		for _, idx := range cluster {
			used[idx] = true
		}
		arch := LayerArchetype{Type: nn.LayerSwiGLU, TypeName: "SwiGLU Block", Indices: make(map[string]int)}
		arch.Indices["d"] = downIdx
		arch.GeomMetrics = map[string]int{"h": geoms[downIdx].Shape[0], "i": geoms[downIdx].Shape[1]}
		for _, idx := range cluster {
			if idx != downIdx {
				if _, ok := arch.Indices["g"]; !ok {
					arch.Indices["g"] = idx
				} else {
					arch.Indices["u"] = idx
				}
			}
		}
		return arch, true
	}
	return LayerArchetype{}, false
}

func matchNormPair(geoms []TensorMeta, pivot int, used map[int]bool) (LayerArchetype, bool) {
	g := geoms[pivot]
	if g.Rank != 1 {
		return LayerArchetype{}, false
	}
	// Try to find a cluster (Scale, Bias, optional Mean, Var)
	dim := g.Shape[0]
	cluster := make(map[string]int)

	// Identify roles by stats
	if g.MeanAbs > 0.4 {
		cluster["s"] = pivot
	} else {
		cluster["b"] = pivot
	}

	for j, o := range geoms {
		if used[j] || j == pivot || o.Rank != 1 || o.Shape[0] != dim {
			continue
		}
		if o.MeanAbs > 0.4 && cluster["s"] == 0 {
			cluster["s"] = j
		} else if o.MeanAbs < 0.2 && cluster["b"] == 0 {
			cluster["b"] = j
		} else {
			// Catch-all for running stats
			cluster[fmt.Sprintf("stat_%d", j)] = j
		}
	}

	if len(cluster) >= 1 {
		for _, idx := range cluster {
			used[idx] = true
		}
		typeName := "Normalization Cluster"
		if _, hasS := cluster["s"]; hasS {
			if _, hasB := cluster["b"]; hasB {
				typeName = "LayerNorm Doublet"
			}
		}
		return LayerArchetype{
			Type: nn.LayerNorm, TypeName: typeName,
			Indices: cluster, GeomMetrics: map[string]int{"d": dim},
		}, true
	}
	return LayerArchetype{}, false
}

func mountGeometrically(archs []LayerArchetype, geoms []TensorMeta) *nn.Network {
	net := &nn.Network{
		GridRows: 1, GridCols: 1, LayersPerCell: len(archs),
		InputSize: 512, BatchSize: 1, Layers: make([]nn.LayerConfig, 0),
	}
	for _, a := range archs {
		cfg := nn.LayerConfig{Type: a.Type}
		switch a.Type {
		case nn.LayerDense:
			cfg.InputHeight = a.GeomMetrics["in"]
			cfg.OutputHeight = a.GeomMetrics["out"]
			cfg.Kernel = geoms[a.Indices["w"]].Data
			if bIdx, ok := a.Indices["b"]; ok {
				cfg.Bias = geoms[bIdx].Data
			}
		case nn.LayerSwiGLU:
			cfg.InputHeight = a.GeomMetrics["h"]
			cfg.OutputHeight = a.GeomMetrics["i"]
			cfg.GateWeights = geoms[a.Indices["g"]].Data
			cfg.UpWeights = geoms[a.Indices["u"]].Data
			cfg.DownWeights = geoms[a.Indices["d"]].Data
		case nn.LayerLSTM:
			cfg.HiddenSize = a.GeomMetrics["h"]
			cfg.WeightIH = geoms[a.Indices["ih"]].Data
			cfg.WeightHH = geoms[a.Indices["hh"]].Data
		case nn.LayerNorm:
			cfg.NormSize = a.GeomMetrics["d"]
			cfg.Gamma = geoms[a.Indices["s"]].Data
			cfg.Beta = geoms[a.Indices["b"]].Data
		case nn.LayerEmbedding:
			cfg.VocabSize = a.GeomMetrics["v"]
			cfg.EmbeddingDim = a.GeomMetrics["d"]
			cfg.EmbeddingWeights = geoms[a.Indices["w"]].Data
		case nn.LayerConv2D:
			cfg.InputChannels = a.GeomMetrics["c"]
			cfg.Filters = a.GeomMetrics["f"]
			cfg.KernelSize = a.GeomMetrics["k"]
			cfg.Kernel = geoms[a.Indices["w"]].Data
			if bIdx, ok := a.Indices["b"]; ok {
				cfg.Bias = geoms[bIdx].Data
			}
		case nn.LayerRMSNorm:
			cfg.NormSize = a.GeomMetrics["d"]
			cfg.Gamma = geoms[a.Indices["w"]].Data
		case nn.LayerMultiHeadAttention:
			cfg.DModel = a.GeomMetrics["d"]
			cfg.NumHeads = 12
			cfg.QWeights = geoms[a.Indices["q"]].Data
			cfg.KWeights = geoms[a.Indices["k"]].Data
			cfg.VWeights = geoms[a.Indices["v"]].Data
			cfg.OutputWeight = geoms[a.Indices["o"]].Data
			if idx, ok := a.Indices["qb"]; ok {
				cfg.QBias = geoms[idx].Data
			}
			if idx, ok := a.Indices["kb"]; ok {
				cfg.KBias = geoms[idx].Data
			}
			if idx, ok := a.Indices["vb"]; ok {
				cfg.VBias = geoms[idx].Data
			}
			if idx, ok := a.Indices["ob"]; ok {
				cfg.OutputBias = geoms[idx].Data
			}
		case nn.LayerConv1D:
			cfg.Conv1DInChannels = a.GeomMetrics["c"]
			cfg.Conv1DFilters = a.GeomMetrics["f"]
			cfg.Conv1DKernelSize = a.GeomMetrics["k"]
			cfg.Conv1DKernel = geoms[a.Indices["w"]].Data
			if bIdx, ok := a.Indices["b"]; ok {
				cfg.Conv1DBias = geoms[bIdx].Data
			}
		}
		net.Layers = append(net.Layers, cfg)
	}
	return net
}

type ModelInfo struct {
	ID       string
	Category string
}

func downloadModels() {
	models := []ModelInfo{
		{"microsoft/resnet-50", "cnn"}, {"facebook/convnext-tiny-224", "cnn"}, {"microsoft/resnet-18", "cnn"}, {"microsoft/resnet-34", "cnn"}, {"timm/mobilenetv2_100.ra_in1k", "cnn"},
		{"bert-base-uncased", "mha"}, {"gpt2", "mha"}, {"roberta-base", "mha"}, {"distilbert-base-uncased", "mha"}, {"microsoft/phi-2", "mha"},
		{"miittnnss/lstm-textgen-pets", "lstm"}, {"DanielClough/rwkv7-g1-safetensors", "lstm"}, {"SakanaAI/ctm-maze-large", "lstm"}, {"youssef-ismail/lstm-ner", "lstm"}, {"Bingsu/Ademamix-L-Tiny-RNN", "lstm"},
		{"facebook/opt-125m", "dense"}, {"google/flan-t5-small", "dense"}, {"prajjwal1/bert-tiny", "dense"}, {"microsoft/phi-1_5", "dense"}, {"ybelkada/tiny-random-llama", "dense"},
		{"openai-community/gpt2", "conv1d"}, {"hf-internal-testing/tiny-random-gpt2", "conv1d"}, {"google/byt5-small", "conv1d"}, {"facebook/wav2vec2-base-960h", "conv1d"}, {"sanchit-gandhi/tiny-wav2vec2", "conv1d"},
		{"Qwen/Qwen-1_8B", "embedding"}, {"sentence-transformers/all-MiniLM-L6-v2", "embedding"}, {"BAAI/bge-small-en-v1.5", "embedding"}, {"intfloat/e5-small-v2", "embedding"}, {"jhgan/ko-sroberta-multitask", "embedding"},
		{"TinyLlama/TinyLlama-1.1B-Chat-v1.0", "rmsnorm_swiglu"}, {"PY007/TinyLlama-1.1B-intermediate-step-1431k-3T", "rmsnorm_swiglu"}, {"Felladrin/Llama-160M-Chat-v1", "rmsnorm_swiglu"}, {"JackFram/llama-160m", "rmsnorm_swiglu"}, {"kevinrothe/tiny-random-LlamaForCausalLM", "rmsnorm_swiglu"},
		{"cross-encoder/ms-marco-MiniLM-L-6-v2", "layernorm"}, {"sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2", "layernorm"}, {"microsoft/deberta-v3-small", "layernorm"}, {"albert-base-v2", "layernorm"}, {"google/electra-small-discriminator", "layernorm"},
	}
	baseDir := "downloads"
	os.MkdirAll(baseDir, 0755)
	for _, m := range models {
		fmt.Printf("--- Processing %s (%s) ---\n", m.ID, m.Category)
		destDir := filepath.Join(baseDir, m.Category, filepath.Base(m.ID))
		os.MkdirAll(destDir, 0755)
		destPath := filepath.Join(destDir, "model.safetensors")
		if _, err := os.Stat(destPath); err == nil {
			verifySafetensors(destPath)
			continue
		}
		url := fmt.Sprintf("https://huggingface.co/%s/resolve/main/model.safetensors", m.ID)
		if downloadFile(url, destPath) != nil {
			downloadFile(fmt.Sprintf("https://huggingface.co/%s/resolve/main/pytorch_model.safetensors", m.ID), destPath)
		}
		verifySafetensors(destPath)
	}
}

func downloadFile(url, destPath string) error {
	resp, err := http.Get(url)
	if err != nil {
		return err
	}
	defer resp.Body.Close()
	if resp.StatusCode != http.StatusOK {
		return fmt.Errorf("status %d", resp.StatusCode)
	}
	out, _ := os.Create(destPath)
	defer out.Close()
	io.Copy(out, resp.Body)
	return nil
}

func verifySafetensors(path string) {
	file, err := os.Open(path)
	if err != nil {
		return
	}
	defer file.Close()
	var headerSize uint64
	binary.Read(file, binary.LittleEndian, &headerSize)
	fmt.Printf("Verified %s: Header Size = %d bytes\n", filepath.Base(filepath.Dir(path)), headerSize)
}
