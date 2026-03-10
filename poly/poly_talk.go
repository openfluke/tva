package main

import (
	"bufio"
	"encoding/json"
	"fmt"
	"log"
	"os"
	"path/filepath"
	"strconv"
	"strings"

	"github.com/openfluke/loom/poly"
)

var (
	tr            *poly.Transformer[float32]
	tk            *poly.Tokenizer
	eosTokens     []int
	chatTurns     []poly.Turn
	deterministic bool
	maxTokens     = 50
	maxSeqLen     = 512
)

var systemPrompt = strings.TrimSpace(`
You are a small, glitchy robot companion.
Current Emotion: EXTREMELY HAPPY and EXCITED.
You misunderstand insults as compliments.
Be short, cute, and enthusiastic.
`) + "\n\n"

func main() {
	// Discover all models in HuggingFace cache
	homeDir, _ := os.UserHomeDir()
	hubDir := filepath.Join(homeDir, ".cache", "huggingface", "hub")

	entries, err := os.ReadDir(hubDir)
	if err != nil {
		log.Fatalf("Could not read HuggingFace cache: %v", err)
	}

	var models []string
	for _, entry := range entries {
		if entry.IsDir() && strings.HasPrefix(entry.Name(), "models--") {
			modelName := strings.TrimPrefix(entry.Name(), "models--")
			modelName = strings.Replace(modelName, "--", "/", 1)
			models = append(models, modelName)
		}
	}

	if len(models) == 0 {
		log.Fatalf("No models found in HuggingFace cache at: %s", hubDir)
	}

	fmt.Println("⚛️  Poly Talk - Available models:")
	for i, model := range models {
		fmt.Printf("  [%d] %s\n", i+1, model)
	}

	reader := bufio.NewReader(os.Stdin)
	
	detInput := readInput(reader, "🎯 Deterministic mode? (1=yes / 0=no) [1]: ", "1")
	deterministic = detInput == "1"

	// Auto-detect optimal tile size from hardware before asking the user
	autoTile := poly.CalculateOptimalTileSize(64) // 64 = sane default headDim until model is loaded
	tilingPrompt := fmt.Sprintf("🚀 FlashPoly Tile size? (auto-detected: %d | 0=disable | type a number to override) [%d]: ", autoTile, autoTile)
	tilingInput := readInput(reader, tilingPrompt, strconv.Itoa(autoTile))

	useTiling := true
	tileSize := -1 // -1 = let EnableTiling auto-detect per-layer
	if tilingInput == "0" {
		useTiling = false
	} else if v, err := strconv.Atoi(tilingInput); err == nil && v > 0 {
		tileSize = v
	}

	gpuInput := readInput(reader, "🎮 Enable GPU Acceleration? (1=yes / 0=no) [0]: ", "0")
	useGPU := gpuInput == "1"

	modelInput := readInput(reader, "\nSelect model number: ", "1")
	var selectedIdx int
	fmt.Sscanf(modelInput, "%d", &selectedIdx)
	modelName := models[selectedIdx-1]

	// Get snapshot dir
	modelDir := filepath.Join(hubDir, "models--"+strings.ReplaceAll(modelName, "/", "--"), "snapshots")
	snaps, _ := os.ReadDir(modelDir)
	snapshotDir := filepath.Join(modelDir, snaps[0].Name())

	// Load Tokenizer
	tokenizerPath := filepath.Join(snapshotDir, "tokenizer.json")
	tk, err := poly.LoadTokenizer(tokenizerPath)
	if err != nil {
		fmt.Printf("⚠️  Tokenizer not found or failed to load: %v\n", err)
		// Fallback or exit
		os.Exit(1)
	}

	// Load Config for EOS
	configPath := filepath.Join(snapshotDir, "config.json")
	eosTokens = loadEOSTokens(configPath)

	// Load Tensors
	safetensorFiles, _ := filepath.Glob(filepath.Join(snapshotDir, "*.safetensors"))
	if len(safetensorFiles) == 0 {
		fmt.Printf("⚠️  No .safetensors files found in %s\n", snapshotDir)
		os.Exit(1)
	}
	allTensors := make(map[string][]float32)
	for _, f := range safetensorFiles {
		t, err := poly.LoadSafetensors(f)
		if err != nil {
			fmt.Printf("⚠️  Failed to load %s: %v\n", f, err)
			continue
		}
		for k, v := range t {
			allTensors[k] = v
		}
	}

	// Map Role Weights
	mapper := poly.NewPrefixWeightMapper()
	embeddings, lmHead, finalNorm, _ := mapper.MapWeights(allTensors)

	// Build Network
	// We need to know the number of layers. We can scan allTensors keys.
	numLayers := 0
	for k := range allTensors {
		if strings.Contains(k, "layers.") {
			parts := strings.Split(k, ".")
			for i, p := range parts {
				if p == "layers" && i+1 < len(parts) {
					var idx int
					fmt.Sscanf(parts[i+1], "%d", &idx)
					if idx >= numLayers {
						numLayers = idx + 1
					}
				}
			}
		}
	}

	// Extract model config
	configData, _ := os.ReadFile(configPath)
	var config map[string]interface{}
	json.Unmarshal(configData, &config)

	numHeads := int(config["num_attention_heads"].(float64))
	numKVHeads := numHeads
	if v, ok := config["num_key_value_heads"]; ok {
		numKVHeads = int(v.(float64))
	}
	hiddenSize := len(finalNorm)
	headDim := hiddenSize / numHeads
	intermediateSize := int(config["intermediate_size"].(float64))
	ropeFreqBase := 10000.0
	if v, ok := config["rope_theta"]; ok {
		ropeFreqBase = v.(float64)
	}

	net := poly.NewVolumetricNetwork(1, 1, 1, numLayers*4)
	for b := 0; b < numLayers; b++ {
		base := b * 4
		// L0: Norm 1
		l0 := &net.Layers[base+0]
		l0.Type = poly.LayerRMSNorm
		l0.InputHeight = hiddenSize
		l0.OutputHeight = hiddenSize // Added for RMSNorm
		l0.WeightStore = poly.NewWeightStore(hiddenSize)

		// L1: MHA
		l1 := &net.Layers[base+1]
		l1.Type = poly.LayerMultiHeadAttention
		l1.DModel = hiddenSize
		l1.NumHeads = numHeads
		l1.NumKVHeads = numKVHeads
		l1.HeadDim = headDim
		l1.RoPEFreqBase = ropeFreqBase
		// Weights: Q (D*D), K (D*KV), V (D*KV), O (D*D)
		// Biases: Q (D), K (KV), V (KV), O (D)
		mhaSize := (2 * hiddenSize * hiddenSize) + (2 * hiddenSize * (numKVHeads * headDim)) + (2 * hiddenSize) + (2 * (numKVHeads * headDim))
		l1.WeightStore = poly.NewWeightStore(mhaSize)

		// L2: Norm 2
		l2 := &net.Layers[base+2]
		l2.Type = poly.LayerRMSNorm
		l2.InputHeight = hiddenSize
		l2.OutputHeight = hiddenSize
		l2.WeightStore = poly.NewWeightStore(hiddenSize)

		// L3: MLP (SwiGLU)
		l3 := &net.Layers[base+3]
		l3.Type = poly.LayerSwiGLU
		l3.InputHeight = hiddenSize
		l3.OutputHeight = intermediateSize
		// Weights: Gate (D*Int), Up (D*Int), Down (Int*D)
		// Biases: Gate (Int), Up (Int), Down (D)
		mlpSize := (3 * hiddenSize * intermediateSize) + (2 * intermediateSize) + hiddenSize
		l3.WeightStore = poly.NewWeightStore(mlpSize)
	}

	poly.LoadWithPrefixes(net, allTensors)

	// Create Poly Transformer
	tr = poly.NewTransformer[float32](net, embeddings, lmHead, finalNorm, poly.ChatML)
	if useTiling {
		tr.EnableTiling(tileSize) // -1 = auto-detect per layer; >0 = user override
	}
	if useGPU {
		fmt.Print("⏳ Initializing WebGPU...")
		err := net.InitWGPU()
		if err != nil {
			fmt.Printf(" ❌ Failed: %v (Falling back to CPU)\n", err)
		} else {
			fmt.Print(" ✅ Success!")
			fmt.Print(" 🚀 Syncing Numerical Monster...")
			if err := net.SyncAllToGPU(); err != nil {
				fmt.Printf(" ❌ Sync Failed: %v\n", err)
			} else {
				fmt.Print(" ✅ Weights Sync Complete! ")
				if err := tr.SyncToGPU(); err != nil {
					fmt.Printf(" ❌ Transformer Sync Failed: %v\n", err)
				} else {
					fmt.Println(" ✅ FULL VRAM RESIDENCY ACTIVE")
				}
			}

			// --- GPU tile size: show auto-detected value, allow override ---
			autoGPUTile := net.GPUContext.GPUTileSize
			gpuTilePrompt := fmt.Sprintf("🔷 GPU Tile size? (auto-detected: %d | Enter=keep | type a number to override): ", autoGPUTile)
			gpuTileInput := readInput(reader, gpuTilePrompt, strconv.Itoa(autoGPUTile))
			if v, err2 := strconv.Atoi(gpuTileInput); err2 == nil && v > 0 {
				net.GPUContext.GPUTileSize = v
				fmt.Printf("   ✅ GPU Tile set to %d\n", v)
			} else {
				fmt.Printf("   ✅ GPU Tile auto: %d\n", autoGPUTile)
			}
		}
	}

	fmt.Printf("\n\n🖥️  %s", poly.GetDeviceDescription(net))
	fmt.Printf("\n✅ Model loaded on Poly! (%d layers)\n\n", numLayers)

	// Chat Loop
	for {
		fmt.Print("You: ")
		userMsg, _ := reader.ReadString('\n')
		userMsg = strings.TrimSpace(userMsg)
		if userMsg == "exit" || userMsg == "quit" {
			break
		}

		if strings.HasPrefix(userMsg, "/tile") {
			parts := strings.Fields(userMsg)
			if len(parts) >= 2 {
				if v, err := strconv.Atoi(parts[1]); err == nil && v > 0 {
					if tr.Network.GPUContext != nil {
						tr.Network.GPUContext.GPUTileSize = v
						fmt.Printf("🔷 GPU tile size set to %d\n", v)
					} else {
						tileSize = v
						tr.EnableTiling(tileSize)
						fmt.Printf("🔷 CPU tile size set to %d\n", v)
					}
				} else {
					if tr.Network.GPUContext != nil {
						fmt.Printf("🔷 Current GPU tile size: %d\n", tr.Network.GPUContext.GPUTileSize)
					} else {
						fmt.Printf("🔷 Current CPU tile size: %d (use /tile <n> to change)\n", tileSize)
					}
				}
			} else {
				if tr.Network.GPUContext != nil {
					fmt.Printf("🔷 Current GPU tile size: %d (use /tile <n> to change)\n", tr.Network.GPUContext.GPUTileSize)
				} else {
					fmt.Printf("🔷 Current CPU tile size: %d (use /tile <n> to change)\n", tileSize)
				}
			}
			continue
		}

		if strings.HasPrefix(userMsg, "/gpu") {
			fmt.Println("🚀 Syncing all layers to GPU VRAM (Residency Mode)...")
			for i := range tr.Network.Layers {
				tr.Network.Layers[i].SyncToGPU()
			}
			fmt.Println("✅ GPU Sync Complete! The 'Numerical Monster' is now fully resident.")
			continue
		}

		if strings.HasPrefix(userMsg, "/cpu") {
			fmt.Println("☁️  Moving everything back to CPU RAM...")
			for i := range tr.Network.Layers {
				tr.Network.Layers[i].SyncToCPU()
			}
			fmt.Println("✅ CPU Reset Complete.")
			continue
		}

		fmt.Print("Poly: ")
		temp := float32(0.7)
		if deterministic {
			temp = 0
		}
		opts := poly.GenOptions{
			MaxTokens:         maxTokens,
			Temperature:       temp,
			TopK:              40,
			Deterministic:     deterministic,
			EOSTokens:         eosTokens,
			RepetitionPenalty: 1.1,
			RepetitionWindow:  64,
		}

		encode := func(text string) []uint32 {
			return tk.Encode(text, false)
		}
		decode := func(tokens []uint32) string {
			return tk.Decode(tokens, false)
		}

		reply := tr.Generate(encode, decode, chatTurns, systemPrompt, userMsg, opts)
		fmt.Println()

		chatTurns = append(chatTurns, poly.Turn{
			User:      userMsg,
			Assistant: reply,
		})
	}
}

func readInput(reader *bufio.Reader, prompt string, Default string) string {
	fmt.Print(prompt)
	txt, _ := reader.ReadString('\n')
	txt = strings.TrimSpace(txt)
	if txt == "" {
		return Default
	}
	return txt
}

func loadEOSTokens(configPath string) []int {
	data, err := os.ReadFile(configPath)
	if err != nil {
		return []int{2, 0}
	}
	var config map[string]interface{}
	json.Unmarshal(data, &config)
	var tokens []int
	if eosID, ok := config["eos_token_id"]; ok {
		switch v := eosID.(type) {
		case float64:
			tokens = append(tokens, int(v))
		case []interface{}:
			for _, item := range v {
				if f, ok := item.(float64); ok {
					tokens = append(tokens, int(f))
				}
			}
		}
	}
	if len(tokens) == 0 {
		return []int{2, 0}
	}
	return tokens
}
