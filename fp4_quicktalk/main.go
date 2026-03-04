package main

// fp4_quicktalk — SmolLM2-compatible chat demo using NVFP4 E2M1 inference.
//
// Dense and SwiGLU projection weights are pre-quantised to 4-bit E2M1 at
// load time.  Inference uses ForwardFP4CPU (CPU) or network.Forward (WebGPU)
// depending on whether GPU mode is requested.
//
// Run: go run main.go

import (
	"bufio"
	"encoding/json"
	"fmt"
	"log"
	"math"
	"math/rand"
	"os"
	"path/filepath"
	"sort"
	"strings"
	"time"

	"github.com/openfluke/loom/gpu"
	"github.com/openfluke/loom/nn"
	"github.com/openfluke/loom/tokenizer"
)

// ─────────────────────────────────────────────────────────────────────────────
// Global state
// ─────────────────────────────────────────────────────────────────────────────

var (
	network       *nn.Network
	fp4Weights    map[int]*nn.FP4LayerWeights
	gpuMounted    bool
	gpuLMHead     *gpu.GPULMHead // GPU-side RMSNorm + vocab projection
	embeddings    []float32
	lmHead        []float32
	finalNorm     []float32
	hiddenSize    int
	vocabSize     int
	finalNormConf *nn.LayerConfig
	tk            *tokenizer.Tokenizer
	eosTokens     []int
	maxTokens     = 50
	maxSeqLen     = 512
	deterministic bool
)

const minPromptRoom = 32

var (
	repetitionPenalty float32 = 1.15
	repetitionWindow          = 64
)

var systemPrompt = strings.TrimSpace(`
You are a tiny FP4-brained robot.
You think in 4 bits. You are surprisingly coherent given the circumstances.
Be witty, brief, and a little glitchy.
`) + "\n\n"

// ─────────────────────────────────────────────────────────────────────────────
// Main
// ─────────────────────────────────────────────────────────────────────────────

func main() {
	homeDir, _ := os.UserHomeDir()
	hubDir := filepath.Join(homeDir, ".cache", "huggingface", "hub")

	entries, err := os.ReadDir(hubDir)
	if err != nil {
		log.Fatalf("Could not read HuggingFace cache: %v", err)
	}

	var models []string
	for _, entry := range entries {
		if entry.IsDir() && strings.HasPrefix(entry.Name(), "models--") {
			name := strings.TrimPrefix(entry.Name(), "models--")
			name = strings.Replace(name, "--", "/", 1)
			models = append(models, name)
		}
	}
	if len(models) == 0 {
		log.Fatalf("No models found in HuggingFace cache at: %s", hubDir)
	}

	reader := bufio.NewReader(os.Stdin)

	fmt.Println("🤖  FP4 QuickTalk — NVFP4 E2M1 bitwise inference")
	fmt.Println("════════════════════════════════════════════════════")
	fmt.Println("Available models:")
	for i, m := range models {
		fmt.Printf("  [%d] %s\n", i+1, m)
	}

	modelInput := readInput(reader, "\nSelect model number: ", "1")
	var selectedIdx int
	if _, err := fmt.Sscanf(strings.TrimSpace(modelInput), "%d", &selectedIdx); err != nil || selectedIdx < 1 || selectedIdx > len(models) {
		log.Fatalf("Invalid selection")
	}
	modelName := models[selectedIdx-1]

	detChoice := readInput(reader, "🎯 Deterministic? (1=yes / 0=no) [1]: ", "1")
	deterministic = detChoice != "0"
	if deterministic {
		rand.Seed(42)
	} else {
		rand.Seed(time.Now().UnixNano())
	}

	maxTokInput := readInput(reader, "🧮 Max tokens per response [50]: ", "50")
	if _, err := fmt.Sscanf(maxTokInput, "%d", &maxTokens); err != nil || maxTokens <= 0 {
		maxTokens = 50
	}
	if maxTokens > 256 {
		maxTokens = 256
	}
	if maxTokens+minPromptRoom > maxSeqLen {
		maxSeqLen = maxTokens + minPromptRoom
	}

	snapshotDir := findSnapshot(hubDir, modelName)
	fmt.Printf("\n📦 Loading: %s\n   Path: %s\n", modelName, snapshotDir)

	// Tokenizer
	tokenizerPath := filepath.Join(snapshotDir, "tokenizer.json")
	if _, err := os.Stat(tokenizerPath); os.IsNotExist(err) {
		log.Fatalf("tokenizer.json not found in %s", snapshotDir)
	}
	tk, err = tokenizer.LoadFromFile(tokenizerPath)
	if err != nil {
		log.Fatalf("Failed to load tokenizer: %v", err)
	}
	fmt.Printf("✓ Tokenizer (vocab: %d)\n", tk.VocabSize())

	eosTokens = loadEOSTokens(filepath.Join(snapshotDir, "config.json"))
	if len(eosTokens) == 0 {
		eosTokens = []int{2, 0}
	}
	fmt.Printf("✓ EOS tokens: %v\n", eosTokens)

	// Network
	network, err = nn.LoadTransformerFromSafetensors(snapshotDir)
	if err != nil {
		log.Fatalf("Error loading model: %v", err)
	}

	// Safetensors weights
	stFiles, err := filepath.Glob(filepath.Join(snapshotDir, "*.safetensors"))
	if err != nil || len(stFiles) == 0 {
		log.Fatalf("No .safetensors files found in %s", snapshotDir)
	}
	tensors := make(map[string][]float32)
	for _, f := range stFiles {
		t, err := nn.LoadSafetensors(f)
		if err != nil {
			log.Fatalf("Error loading weights: %v", err)
		}
		for k, v := range t {
			tensors[k] = v
		}
	}
	mapper := tokenizer.NewWeightMapper()
	embeddings, lmHead, finalNorm, _ = mapper.MapWeights(tensors)

	hiddenSize = network.InputSize
	vocabSize = len(embeddings) / hiddenSize
	if finalNorm != nil {
		finalNormConf = &nn.LayerConfig{
			Type:     nn.LayerRMSNorm,
			NormSize: hiddenSize,
			Gamma:    finalNorm,
			Epsilon:  1e-6,
		}
	}
	fmt.Printf("✓ Model: hidden=%d vocab=%d layers=%d\n", hiddenSize, vocabSize, len(network.Layers))

	// ── Pre-quantise Dense + SwiGLU to FP4 ───────────────────────────────────
	fmt.Print("⚡ Quantising weights to FP4 E2M1... ")
	t0 := time.Now()
	fp4Weights = network.BuildFP4Weights()
	elapsed := time.Since(t0)
	denseFP4, swigluFP4 := 0, 0
	for _, fw := range fp4Weights {
		if fw.Kernel != nil {
			denseFP4++
		}
		if fw.Gate != nil {
			swigluFP4++
		}
	}
	fmt.Printf("done in %v  (Dense=%d SwiGLU=%d)\n", elapsed.Round(time.Millisecond), denseFP4, swigluFP4)

	// ── GPU (optional) ────────────────────────────────────────────────────────
	gpuChoice := readInput(reader, "\n🚀 Run on GPU? (1=yes / 0=no) [0]: ", "0")
	if gpuChoice == "1" {
		gpu.SetAdapterPreference("nvidia")
		fmt.Print("⚡ Mounting FP4-compressed weights to GPU... ")
		network.BatchSize = 1
		for i := range network.Layers {
			network.Layers[i].SeqLength = maxSeqLen
		}
		network.GPU = true
		network.GPUInferenceOnly = true
		network.EnableGPUResiduals = true
		if err := network.WeightsFP4ToGPU(fp4Weights); err != nil {
			fmt.Printf("\n⚠️  FP4 GPU mount failed (%v) — falling back to FP4 CPU\n", err)
			network.GPU = false
		} else {
			gpuMounted = true
			fmt.Println("done ✓")
			// Mount LM head on GPU — eliminates the biggest per-token CPU bottleneck
			if gpuCtx, err2 := gpu.GetContext(); err2 == nil {
				var normGamma []float32
				if finalNormConf != nil {
					normGamma = finalNormConf.Gamma
				}
				if lmh, err3 := gpu.NewGPULMHead(gpuCtx, hiddenSize, vocabSize, normGamma, lmHead); err3 == nil {
					gpuLMHead = lmh
					fmt.Printf("   GPU LM head: %d×%d → %.1f MB VRAM\n",
						vocabSize, hiddenSize,
						float64(vocabSize*hiddenSize*4)/(1024*1024))
				} else {
					fmt.Printf("   ⚠️  GPU LM head failed (%v) — using CPU fallback\n", err3)
				}
			}
		}
	}

	// ── Chat loop ─────────────────────────────────────────────────────────────
	var chatTurns []tokenizer.Turn
	tmpl := tokenizer.ChatML

	fmt.Println("\n🤖  FP4 Chat mode")
	if gpuMounted {
		fmt.Println("   Backend: WebGPU  (FP4 CPU precomputed as weights fallback)")
	} else {
		fmt.Printf("   Backend: FP4 E2M1 CPU  (Dense=%d SwiGLU=%d layers)\n", denseFP4, swigluFP4)
	}
	fmt.Println("   Type 'exit' or 'quit' to stop")
	fmt.Printf("   Max %d tokens  │  deterministic=%v\n\n", maxTokens, deterministic)

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

		fmt.Print("Bot: ")
		reply := generateFP4(tmpl, tk, chatTurns, systemPrompt, userMsg)
		fmt.Println()
		chatTurns = append(chatTurns, tokenizer.Turn{User: userMsg, Assistant: reply})
	}
}

// ─────────────────────────────────────────────────────────────────────────────
// FP4 generation loop
// ─────────────────────────────────────────────────────────────────────────────

func generateFP4(
	tmpl tokenizer.Template,
	tk *tokenizer.Tokenizer,
	turns []tokenizer.Turn,
	sysPrompt, userMsg string,
) string {
	prompt := tmpl.BuildPrompt(turns, sysPrompt, userMsg)
	tokens := tk.Encode(prompt, false)
	stream := tokenizer.NewStreamer(tk, tokens)

	// Unified forward — FP4 GPU or FP4 CPU, transparent to the caller.
	forward := func(input []float32, pos int) []float32 {
		for j := range network.Layers {
			if network.Layers[j].Type == nn.LayerMultiHeadAttention {
				network.Layers[j].IsInference = true
				network.Layers[j].KVCachePos = pos
			}
		}
		if gpuMounted {
			out, _ := network.ForwardFP4GPU(input)
			return out
		}
		out, _ := network.ForwardFP4CPU(input, fp4Weights)
		return out
	}

	lastHidden := func(out []float32) []float32 {
		if len(out) >= hiddenSize {
			return append([]float32(nil), out[len(out)-hiddenSize:]...)
		}
		return out
	}

	// Prefill
	var hidden []float32
	if len(tokens) > 0 {
		allEmbeds := make([]float32, len(tokens)*hiddenSize)
		for i, tok := range tokens {
			copy(allEmbeds[i*hiddenSize:], getEmbedding(int(tok)))
		}
		hidden = lastHidden(forward(allEmbeds, 0))
	}

	// Decode
	genStart := time.Now()
	genCount := 0

	for i := 0; i < maxTokens; i++ {
		var logits []float32
		if gpuLMHead != nil {
			gpuCtx, _ := gpu.GetContext()
			logits, _ = gpuLMHead.Infer(gpuCtx, hidden)
		}
		if logits == nil {
			logits = applyLMHead(hidden)
		}
		applyRepetitionPenalty(logits, tokens)
		var nextToken int
		if deterministic {
			nextToken = argmax(logits)
		} else {
			nextToken = sampleTopK(logits, 40, 0.9)
		}
		tokens = append(tokens, uint32(nextToken))
		genCount++
		stream.Push(tokens)

		if isEOS(nextToken) || stream.HasNewUserTurn(tokens) {
			break
		}
		hidden = lastHidden(forward(getEmbedding(nextToken), len(tokens)-1))
	}

	elapsed := time.Since(genStart)
	backend := "FP4 E2M1 CPU"
	if gpuMounted {
		backend = "WebGPU + FP4 E2M1 weights"
	}
	if genCount > 0 {
		fmt.Printf("\n\n(%.2f tokens/s | %d tokens | %s)\n",
			float64(genCount)/elapsed.Seconds(), genCount, backend)
	}
	return stream.String()
}

// ─────────────────────────────────────────────────────────────────────────────
// Inference helpers
// ─────────────────────────────────────────────────────────────────────────────

func getEmbedding(tokenID int) []float32 {
	offset := tokenID * hiddenSize
	if offset+hiddenSize > len(embeddings) {
		return make([]float32, hiddenSize)
	}
	return embeddings[offset : offset+hiddenSize]
}

func applyLMHead(hidden []float32) []float32 {
	norm := hidden
	if finalNormConf != nil {
		norm = nn.RmsNormForwardCPU(hidden, nil, finalNormConf, 1)
	}
	logits := make([]float32, vocabSize)
	for v := 0; v < vocabSize; v++ {
		var sum float32
		off := v * hiddenSize
		for d := 0; d < hiddenSize; d++ {
			sum += norm[d] * lmHead[off+d]
		}
		logits[v] = sum
	}
	return logits
}

func applyRepetitionPenalty(logits []float32, tokens []uint32) {
	if repetitionPenalty <= 1 || len(tokens) == 0 {
		return
	}
	start := len(tokens) - repetitionWindow
	if start < 0 {
		start = 0
	}
	seen := make(map[uint32]struct{})
	for _, tok := range tokens[start:] {
		if int(tok) >= len(logits) {
			continue
		}
		if _, ok := seen[tok]; ok {
			continue
		}
		seen[tok] = struct{}{}
		if logits[tok] > 0 {
			logits[tok] /= repetitionPenalty
		} else {
			logits[tok] *= repetitionPenalty
		}
	}
}

func isEOS(token int) bool {
	for _, e := range eosTokens {
		if token == e {
			return true
		}
	}
	return false
}

func argmax(logits []float32) int {
	best, bestV := 0, logits[0]
	for i, v := range logits {
		if v > bestV {
			bestV = v
			best = i
		}
	}
	return best
}

func sampleTopK(logits []float32, k int, temp float32) int {
	type pair struct {
		idx int
		val float32
	}
	cands := make([]pair, len(logits))
	for i, v := range logits {
		cands[i] = pair{i, v / temp}
	}
	sort.Slice(cands, func(i, j int) bool { return cands[i].val > cands[j].val })
	if k > 0 && k < len(cands) {
		cands = cands[:k]
	}
	maxV := cands[0].val
	var sum float64
	probs := make([]float64, len(cands))
	for i := range cands {
		p := math.Exp(float64(cands[i].val - maxV))
		probs[i] = p
		sum += p
	}
	r := rand.Float64() * sum
	acc := 0.0
	for i := range probs {
		acc += probs[i]
		if r <= acc {
			return cands[i].idx
		}
	}
	return cands[len(cands)-1].idx
}

// ─────────────────────────────────────────────────────────────────────────────
// I/O
// ─────────────────────────────────────────────────────────────────────────────

func readMessage(r *bufio.Reader) (string, bool) {
	fmt.Print("You: ")
	line, err := r.ReadString('\n')
	if err != nil {
		return "", true
	}
	line = strings.TrimRight(line, "\r\n")
	lower := strings.ToLower(strings.TrimSpace(line))
	if lower == "exit" || lower == "quit" {
		return "", true
	}
	if strings.TrimSpace(line) == "<<<" {
		fmt.Println("(paste mode: finish with >>> on its own line)")
		var lines []string
		for {
			l, err := r.ReadString('\n')
			if err != nil {
				return "", true
			}
			l = strings.TrimRight(l, "\r\n")
			if strings.TrimSpace(l) == ">>>" {
				break
			}
			lines = append(lines, l)
		}
		return strings.Join(lines, "\n"), false
	}
	return line, false
}

func readInput(r *bufio.Reader, prompt, def string) string {
	fmt.Print(prompt)
	txt, _ := r.ReadString('\n')
	txt = strings.TrimSpace(txt)
	if txt == "" {
		return def
	}
	return txt
}

func findSnapshot(hubDir, modelName string) string {
	dirName := "models--" + strings.ReplaceAll(modelName, "/", "--")
	snapshotRoot := filepath.Join(hubDir, dirName, "snapshots")
	entries, err := os.ReadDir(snapshotRoot)
	if err != nil || len(entries) == 0 {
		log.Fatalf("Snapshots not found for %s", modelName)
	}
	return filepath.Join(snapshotRoot, entries[0].Name())
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
		if f, ok := padID.(float64); ok {
			tokens = append(tokens, int(f))
		}
	}
	return tokens
}
