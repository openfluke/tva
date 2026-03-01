package main

import (
	"bufio"
	"encoding/json"
	"fmt"
	"log"
	"math/rand"
	"os"
	"path/filepath"
	"strings"
	"time"

	"github.com/openfluke/loom/gpu"
	"github.com/openfluke/loom/nn"
	"github.com/openfluke/loom/tokenizer"
)

var (
	engine        *tokenizer.LLMEngine
	tk            *tokenizer.Tokenizer
	eosTokens     []int
	chatTurns     []tokenizer.Turn
	deterministic bool
	useKVCache    bool
	maxTokens     = 50
	maxSeqLen     = 128
)

const minPromptRoom = 32

var (
	repetitionPenalty float32 = 1.15
	repetitionWindow          = 64
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

	// Find all model directories
	var models []string
	for _, entry := range entries {
		if entry.IsDir() && strings.HasPrefix(entry.Name(), "models--") {
			// Convert "models--Qwen--Qwen2.5-0.5B" to "Qwen/Qwen2.5-0.5B"
			modelName := strings.TrimPrefix(entry.Name(), "models--")
			modelName = strings.Replace(modelName, "--", "/", 1)
			models = append(models, modelName)
		}
	}

	if len(models) == 0 {
		log.Fatalf("No models found in HuggingFace cache at: %s", hubDir)
	}

	// Display models and let user select
	fmt.Println("🤖 Available models in cache:")
	for i, model := range models {
		fmt.Printf("  [%d] %s\n", i+1, model)
	}

	reader := bufio.NewReader(os.Stdin)
	modelInput := readInput(reader, "\nSelect model number: ", "1")
	modelInput = strings.TrimSpace(modelInput)

	var selectedIdx int
	if _, err := fmt.Sscanf(modelInput, "%d", &selectedIdx); err != nil || selectedIdx < 1 || selectedIdx > len(models) {
		log.Fatalf("Invalid selection: %v", modelInput)
	}

	// Deterministic mode selection
	deterministic = true
	detChoice := readInput(reader, "🎯 Deterministic mode? (1=yes / 0=no) [1]: ", "1")
	if detChoice == "0" {
		deterministic = false
	}

	if deterministic {
		rand.Seed(42) // Fixed seed for determinism
	} else {
		rand.Seed(time.Now().UnixNano())
	}

	modelName := models[selectedIdx-1]
	fmt.Printf("\n📦 Loading model: %s\n", modelName)

	// Get model directory
	modelDir := filepath.Join(hubDir, "models--"+strings.ReplaceAll(modelName, "/", "--"), "snapshots")
	entries, err = os.ReadDir(modelDir)
	if err != nil || len(entries) == 0 {
		log.Fatalf("Model snapshots not found for: %s", modelName)
	}

	snapshotDir := filepath.Join(modelDir, entries[0].Name())
	fmt.Printf("   Path: %s\n", snapshotDir)

	// Load tokenizer
	tokenizerPath := filepath.Join(snapshotDir, "tokenizer.json")

	if _, err := os.Stat(tokenizerPath); os.IsNotExist(err) {
		log.Fatalf("⚠️  tokenizer.json not found in model directory %s. Aborting to prevent silent failures.", snapshotDir)
	}

	tk, err = tokenizer.LoadFromFile(tokenizerPath)
	if err != nil {
		log.Fatalf("Failed to load tokenizer: %v", err)
	}
	fmt.Printf("✓ Tokenizer loaded (vocab: %d)\n", tk.VocabSize())

	// Load EOS tokens from config
	configPath := filepath.Join(snapshotDir, "config.json")
	eosTokens = loadEOSTokens(configPath)
	if len(eosTokens) == 0 {
		eosTokens = []int{2, 0} // Default
	}
	fmt.Printf("✓ EOS tokens: %v\n", eosTokens)

	fmt.Println("ℹ️  Using ChatML prompt builder")

	// GPU Selection
	useGPU := false
	gpuChoice := readInput(reader, "\n🚀 Run on GPU? (1=yes / 0=no) [0]: ", "0")
	if gpuChoice == "1" {
		useGPU = true
		gpu.SetAdapterPreference("nvidia") // Default to NVIDIA if requested
	}
	kvChoice := readInput(reader, "🧠 Use KV cache? (1=yes / 0=no) [0]: ", "0")
	if kvChoice == "1" {
		useKVCache = true
	}
	maxTokensInput := readInput(reader, "🧮 Max tokens per response [50]: ", "50")
	if _, err := fmt.Sscanf(maxTokensInput, "%d", &maxTokens); err != nil || maxTokens <= 0 {
		maxTokens = 50
	}
	if maxTokens > 256 {
		fmt.Println("ℹ️  Clamping maxTokens to 256 for stability.")
		maxTokens = 256
	}
	if maxTokens+minPromptRoom > maxSeqLen {
		maxSeqLen = maxTokens + minPromptRoom
	}

	// Create network
	network, err := nn.LoadTransformerFromSafetensors(snapshotDir)
	if err != nil {
		log.Fatalf("Error loading model: %v", err)
	}

	// Dynamic weight mapping
	weightsPath := filepath.Join(snapshotDir, "model.safetensors")
	tensors, err := nn.LoadSafetensors(weightsPath)
	if err != nil {
		log.Fatalf("Error loading weights: %v", err)
	}

	mapper := tokenizer.NewWeightMapper()
	embeddings, lmHead, finalNorm, _ := mapper.MapWeights(tensors)

	// Mount to GPU if requested
	if useGPU {
		fmt.Println("⚡ Mounting weights to GPU...")
		network.BatchSize = maxSeqLen
		for i := range network.Layers {
			network.Layers[i].SeqLength = maxSeqLen
		}
		network.GPU = true
		if err := network.WeightsToGPU(); err != nil {
			fmt.Printf("⚠️ Failed to mount GPU: %v. Falling back to CPU.\n", err)
			network.GPU = false
		} else {
			fmt.Println("⚡ GPU Mounted successfully!")
		}
	}

	// Initialize Engine
	engine = tokenizer.NewLLMEngine(network, embeddings, lmHead, finalNorm, tokenizer.ChatML)

	fmt.Printf("✅ Model loaded!\n")
	fmt.Printf("   Hidden: %d, Vocab: %d, Layers: %d\n\n", network.InputSize, engine.VocabSize, len(network.Layers))

	// Optional system prompt override
	fmt.Println("🧠 System prompt (optional).")
	fmt.Println("   - Type across multiple lines")
	fmt.Println("   - Send with an EMPTY line")
	fmt.Println("   - Leave blank to keep default\n")
	systemInput, _ := readMultiline(reader)
	if strings.TrimSpace(systemInput) != "" {
		systemPrompt = strings.TrimSpace(systemInput) + "\n\n"
	}

	// Interactive chat loop
	fmt.Println("🤖 Chat mode:")
	fmt.Println("   - One line: type and press Enter")
	fmt.Println("   - Multiline: type <<< then paste, finish with >>>")
	fmt.Println("   - Type 'exit' or 'quit' to stop")
	fmt.Printf("   - Max %d tokens per response\n\n", maxTokens)

	for {
		userMsg, quitting := readMessage(reader)
		if quitting {
			fmt.Println("Goodbye!")
			break
		}
		if strings.TrimSpace(userMsg) == "" {
			continue
		}

		userMsg = strings.TrimSpace(userMsg)
		if userMsg == "" {
			continue
		}

		fmt.Print("Bot: ")

		opts := tokenizer.GenOptions{
			MaxTokens:         maxTokens,
			Temperature:       0.9,
			TopK:              40,
			Deterministic:     deterministic,
			UseKVCache:        useKVCache,
			RepetitionPenalty: repetitionPenalty,
			RepetitionWindow:  repetitionWindow,
			EOSTokens:         eosTokens,
		}
		if deterministic {
			opts.Temperature = 0
			opts.TopK = 1
		}

		reply := engine.Generate(tk, chatTurns, systemPrompt, userMsg, opts)
		fmt.Println()

		chatTurns = append(chatTurns, tokenizer.Turn{
			User:      userMsg,
			Assistant: reply,
		})
	}
}

func readMultiline(r *bufio.Reader) (string, bool) {
	var lines []string
	for {
		line, err := r.ReadString('\n')
		if err != nil {
			return "", true
		}
		line = strings.TrimRight(line, "\r\n")

		if len(lines) == 0 {
			lower := strings.ToLower(strings.TrimSpace(line))
			if lower == "exit" || lower == "quit" {
				return "", true
			}
		}

		if line == "" {
			break
		}

		lines = append(lines, line)
	}
	return strings.Join(lines, "\n"), false
}

func readMessage(r *bufio.Reader) (string, bool) {
	fmt.Print("You: ")
	first, err := r.ReadString('\n')
	if err != nil {
		return "", true
	}
	first = strings.TrimRight(first, "\r\n")

	lower := strings.ToLower(strings.TrimSpace(first))
	if lower == "exit" || lower == "quit" {
		return "", true
	}

	if strings.TrimSpace(first) != "<<<" {
		return first, false
	}

	fmt.Println("(paste mode: finish with >>> on its own line)")
	var lines []string
	for {
		line, err := r.ReadString('\n')
		if err != nil {
			return "", true
		}
		line = strings.TrimRight(line, "\r\n")
		if strings.TrimSpace(line) == ">>>" {
			break
		}
		lines = append(lines, line)
	}
	return strings.Join(lines, "\n"), false
}

func loadEOSTokens(configPath string) []int {
	data, err := os.ReadFile(configPath)
	if err != nil {
		return nil
	}

	var config map[string]interface{}
	if err := json.Unmarshal(data, &config); err != nil {
		return nil
	}

	var tokens []int

	if eosID, ok := config["eos_token_id"]; ok {
		switch v := eosID.(type) {
		case float64:
			tokens = append(tokens, int(v))
		case []interface{}:
			for _, id := range v {
				if idFloat, ok := id.(float64); ok {
					tokens = append(tokens, int(idFloat))
				}
			}
		}
	}

	if padID, ok := config["pad_token_id"]; ok {
		if idFloat, ok := padID.(float64); ok {
			tokens = append(tokens, int(idFloat))
		}
	}

	return tokens
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
