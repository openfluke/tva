package main

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

var (
	network       *nn.Network
	embeddings    []float32
	lmHead        []float32
	finalNorm     []float32
	hiddenSize    int
	vocabSize     int
	eosTokens     []int
	hasFinalNorm  bool
	tk            *tokenizer.Tokenizer
	useKVCache    bool
	kvCacheWarn   bool
	deterministic bool
)

type Turn struct {
	User      string
	Assistant string
}

const minPromptRoom = 32

var maxTokens = 50
var maxSeqLen = 128
var repetitionPenalty float32 = 1.15
var repetitionWindow = 64

var systemPrompt = strings.TrimSpace(`
You are a small, glitchy robot companion.
Current Emotion: EXTREMELY HAPPY and EXCITED.
You misunderstand insults as compliments.
Be short, cute, and enthusiastic.
`) + "\n\n"

var chatTurns []Turn

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

	// Load model
	network, err = nn.LoadTransformerFromSafetensors(snapshotDir)
	if err != nil {
		log.Fatalf("Error loading model: %v", err)
	}

	// Load weights
	weightsPath := filepath.Join(snapshotDir, "model.safetensors")
	tensors, err := nn.LoadSafetensors(weightsPath)
	if err != nil {
		log.Fatalf("Error loading weights: %v", err)
	}

	// Mount to GPU if requested
	if useGPU {
		fmt.Println("⚡ Mounting weights to GPU...")

		// Configure network for maximum sequence length before mounting
		// This ensures GPU buffers are allocated large enough for the chat history
		originalBatchSize := network.BatchSize
		network.BatchSize = maxSeqLen
		for i := range network.Layers {
			network.Layers[i].SeqLength = maxSeqLen
		}

		network.GPU = true
		if err := network.WeightsToGPU(); err != nil {
			log.Printf("⚠️ Failed to mount GPU: %v. Falling back to CPU.", err)
			network.GPU = false
			network.BatchSize = originalBatchSize // Restore if failed
		} else {
			fmt.Println("⚡ GPU Mounted successfully!")
			// Keep network.BatchSize = maxSeqLen because strict GPU inference
			// might rely on this matching the buffer setup?
			// Actually, for inference of 1 token at a time, we pass slice of length 1 (or N).
			// But the GPU shaders are compiled with fixed buffer sizes.
			// So logic is fine.
		}
	}

	// Load embeddings
	embeddings = tryLoadTensor(tensors, []string{
		"model.embed_tokens.weight",
		"transformer.wte.weight",
		"embeddings.weight",
		"embed_tokens.weight",
	})
	if embeddings == nil {
		log.Fatalf("Could not find embeddings")
	}

	// Load final norm
	finalNorm = tryLoadTensor(tensors, []string{
		"model.norm.weight",
		"transformer.ln_f.weight",
		"ln_f.weight",
		"norm.weight",
	})

	// Load LM Head (if available, otherwise use embeddings)
	// Some models (like Llama 2) share weights, others don't.
	fmt.Printf("Searching for lm_head weights...\n")
	lmHead = tryLoadTensor(tensors, []string{"lm_head.weight", "output.weight"})
	if lmHead == nil {
		fmt.Println("ℹ️ No separate lm_head found, assuming tied weights (using embeddings).")
		// Debug: print top 10 keys to see what we have
		fmt.Println("Available keys (first 10):")
		count := 0
		for k := range tensors {
			if count < 10 {
				fmt.Println(" -", k)
				count++
			}
		}
		// Try to find if maybe it's just named differently?
		// e.g. model.embed_tokens.weight sometimes?
		lmHead = embeddings
	} else {
		fmt.Printf("✅ Loaded separate lm_head weights: %d values\n", len(lmHead))
	}

	hasFinalNorm = (finalNorm != nil)

	hiddenSize = network.InputSize
	vocabSize = len(embeddings) / hiddenSize

	// Double check vocab size against lm_head if separate
	if len(lmHead) != len(embeddings) {
		// e.g. lm_head might be [Vocab, Hidden]
		vocabChecker := len(lmHead) / hiddenSize
		if vocabChecker != vocabSize {
			fmt.Printf("⚠️ Vocab size mismatch? Embedding: %d, Head: %d. Using Head size.\n", vocabSize, vocabChecker)
			vocabSize = vocabChecker
		}
	}

	fmt.Printf("✅ Model loaded!\n")
	fmt.Printf("   Hidden: %d, Vocab: %d, Layers: %d\n\n", hiddenSize, vocabSize, len(network.Layers))

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

		fullPrompt, err := buildPromptThatFits(userMsg)
		if err != nil {
			fmt.Println(err.Error())
			continue
		}

		fmt.Print("Bot: ")
		reply := generate(fullPrompt)
		fmt.Println()

		chatTurns = append(chatTurns, Turn{
			User:      userMsg,
			Assistant: reply,
		})
	}
}

func generate(prompt string) string {
	inputIDs := tk.Encode(prompt, false)
	tokens := make([]int, len(inputIDs))
	for i, id := range inputIDs {
		tokens[i] = int(id)
	}

	if useKVCache {
		return generateWithKV(tokens)
	}

	// Generate up to maxTokens with streaming
	start := time.Now()
	generatedCount := 0
	stream := NewStreamer(tokens)

	for i := 0; i < maxTokens; i++ {
		nextToken, err := generateNextToken(tokens)
		if err != nil {
			return fmt.Sprintf("\n[Error: %v]", err)
		}

		tokens = append(tokens, nextToken)
		generatedCount++

		stream.Push(tokens)

		// Check for EOS
		if isEOSToken(nextToken) {
			break
		}
		if stream.HasNewUserTurn(tokens) {
			break
		}
	}

	elapsed := time.Since(start)
	if generatedCount > 0 {
		tps := float64(generatedCount) / elapsed.Seconds()
		fmt.Printf("\n\n(%.2f tokens/s, %d tokens total)\n", tps, generatedCount)
	} else {
		fmt.Println()
	}
	return stream.String()
}

func readMultiline(r *bufio.Reader) (string, bool) {
	var lines []string
	for {
		line, err := r.ReadString('\n')
		if err != nil {
			// EOF or read error: treat as quit
			return "", true
		}
		line = strings.TrimRight(line, "\r\n")

		// If they type exit/quit as the very first line -> quit immediately
		if len(lines) == 0 {
			lower := strings.ToLower(strings.TrimSpace(line))
			if lower == "exit" || lower == "quit" {
				return "", true
			}
		}

		// Empty line sends the message
		if line == "" {
			break
		}

		lines = append(lines, line)
	}
	return strings.Join(lines, "\n"), false
}

type Streamer struct {
	lastLen      int
	promptLenRaw int
	sb           strings.Builder
	replacer     *strings.Replacer
}

func NewStreamer(promptTokens []int) *Streamer {
	promptTextRaw := tk.Decode(intsToU32(promptTokens), false)
	return &Streamer{
		lastLen:      len(promptTextRaw),
		promptLenRaw: len(promptTextRaw),
		replacer: strings.NewReplacer(
			"<|im_end|>", "",
			"<|im_start|>assistant", "",
			"<|im_start|>user", "",
			"<|im_start|>system", "",
		),
	}
}

func (s *Streamer) Push(allTokens []int) {
	full := tk.Decode(intsToU32(allTokens), false)
	if len(full) > s.lastLen {
		diff := full[s.lastLen:]
		diff = s.replacer.Replace(diff)
		fmt.Print(diff)
		s.sb.WriteString(diff)
		s.lastLen = len(full)
	}
}

func (s *Streamer) String() string {
	return strings.TrimSpace(s.sb.String())
}

func (s *Streamer) HasNewUserTurn(allTokens []int) bool {
	fullRaw := tk.Decode(intsToU32(allTokens), false)
	if len(fullRaw) <= s.promptLenRaw {
		return false
	}
	return strings.Contains(fullRaw[s.promptLenRaw:], "<|im_start|>user")
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

func buildChatPrompt(turns []Turn, userMsg string) string {
	var sb strings.Builder
	sp := strings.TrimSpace(systemPrompt)
	if sp != "" {
		sb.WriteString("<|im_start|>system\n")
		sb.WriteString(sp)
		sb.WriteString("<|im_end|>\n")
	}

	for _, t := range turns {
		sb.WriteString("<|im_start|>user\n")
		sb.WriteString(t.User)
		sb.WriteString("<|im_end|>\n")

		sb.WriteString("<|im_start|>assistant\n")
		sb.WriteString(t.Assistant)
		sb.WriteString("<|im_end|>\n")
	}

	sb.WriteString("<|im_start|>user\n")
	sb.WriteString(userMsg)
	sb.WriteString("<|im_end|>\n")
	sb.WriteString("<|im_start|>assistant\n")
	return sb.String()
}

// Ensures prompt_len + maxTokens fits into maxSeqLen (important for KV cache path)
func buildPromptThatFits(userMsg string) (string, error) {
	turns := chatTurns

	for {
		prompt := buildChatPrompt(turns, userMsg)
		ids := tk.Encode(prompt, false)

		ensureSeqLenForPrompt(len(ids))

		// Need room for generation tokens, otherwise KV cache will blow past MaxSeq
		if len(ids) <= (maxSeqLen - maxTokens) {
			// commit trimmed view
			chatTurns = turns
			return prompt, nil
		}

		// If even empty history doesn't fit, user message is too long
		if len(turns) == 0 {
			return "", fmt.Errorf("[Error: prompt too long (%d tokens). maxSeqLen=%d, maxTokens=%d. Shorten your message.]", len(ids), maxSeqLen, maxTokens)
		}

		// Drop oldest turn and retry
		turns = turns[1:]
	}
}

func ensureSeqLenForPrompt(promptTokens int) {
	required := promptTokens + maxTokens
	if required <= maxSeqLen {
		return
	}

	maxSeqLen = required
	if useKVCache {
		return
	}
	if network.GPU && network.IsGPUMounted() {
		fmt.Printf("ℹ️  Resizing GPU buffers for seq len %d...\n", maxSeqLen)
		network.ReleaseGPUWeights()
		network.BatchSize = maxSeqLen
		for i := range network.Layers {
			network.Layers[i].SeqLength = maxSeqLen
		}
		if err := network.WeightsToGPU(); err != nil {
			fmt.Printf("⚠️ Failed to remount GPU for seq len %d: %v. Falling back to CPU.\n", maxSeqLen, err)
			network.GPU = false
		}
	}
}

func generateWithKV(tokens []int) string {
	if len(tokens) > maxSeqLen {
		return fmt.Sprintf("\n[Error: prompt too long for KV cache (max %d tokens)]", maxSeqLen)
	}
	if network.GPU && !kvCacheWarn {
		fmt.Println("ℹ️  KV cache path runs on CPU for now.")
		kvCacheWarn = true
	}

	state := initKVCacheState(maxSeqLen)
	var err error
	var hidden []float32
	for i, tok := range tokens {
		hidden, err = forwardTokenKV(tok, i, state)
		if err != nil {
			return fmt.Sprintf("\n[Error: %v]", err)
		}
	}

	start := time.Now()
	generatedCount := 0
	stream := NewStreamer(tokens)

	for i := 0; i < maxTokens; i++ {
		nextToken, err := nextTokenFromHidden(hidden, recentTokens(tokens))
		if err != nil {
			return fmt.Sprintf("\n[Error: %v]", err)
		}

		tokens = append(tokens, nextToken)
		generatedCount++

		stream.Push(tokens)

		if isEOSToken(nextToken) {
			break
		}
		if stream.HasNewUserTurn(tokens) {
			break
		}

		hidden, err = forwardTokenKV(nextToken, len(tokens)-1, state)
		if err != nil {
			return fmt.Sprintf("\n[Error: %v]", err)
		}
	}

	elapsed := time.Since(start)
	if generatedCount > 0 {
		tps := float64(generatedCount) / elapsed.Seconds()
		fmt.Printf("\n\n(%.2f tokens/s, %d tokens total)\n", tps, generatedCount)
	} else {
		fmt.Println()
	}
	return stream.String()
}

func generateNextToken(tokens []int) (int, error) {
	// Prepare input
	// Check if the first layer is an Embedding layer
	hasEmbeddingLayer := false
	if len(network.Layers) > 0 && network.Layers[0].Type == nn.LayerEmbedding {
		hasEmbeddingLayer = true
	}

	var input []float32
	if hasEmbeddingLayer {
		// Pass token IDs directly as floats
		// CPU Forward (EmbeddingLayer) and GPU Forward both expect this format if EmbeddingLayer exists
		input = make([]float32, len(tokens))
		for i, t := range tokens {
			input[i] = float32(t)
		}
	} else {
		// Manual embedding (legacy or models without EmbeddingLayer)
		input = make([]float32, len(tokens)*hiddenSize)
		for t, tokenID := range tokens {
			if tokenID >= vocabSize || tokenID < 0 {
				// Actually check embeddings len in case vocab mismatch
				if tokenID*hiddenSize >= len(embeddings) {
					return 0, fmt.Errorf("token ID %d out of bounds for embedding", tokenID)
				}
			}
			for d := 0; d < hiddenSize; d++ {
				input[t*hiddenSize+d] = embeddings[tokenID*hiddenSize+d]
			}
		}
	}

	if !(network.GPU && network.IsGPUMounted()) {
		network.BatchSize = len(tokens)
	}

	// Forward pass
	output, _ := network.ForwardCPU(input)

	// Apply final norm if available
	var normalized []float32
	if hasFinalNorm && finalNorm != nil {
		finalNormConfig := &nn.LayerConfig{
			Type:     nn.LayerRMSNorm,
			NormSize: hiddenSize,
			Gamma:    finalNorm,
			Epsilon:  1e-6,
		}
		normalized = nn.RmsNormForwardCPU(output, nil, finalNormConfig, len(tokens))
	} else {
		normalized = output
	}

	// Extract last token
	lastIdx := (len(tokens) - 1) * hiddenSize
	lastTokenNormalized := normalized[lastIdx : lastIdx+hiddenSize]

	// LM head projection
	// lmHead might be tied (same as embeddings) or separate
	logits := make([]float32, vocabSize)
	for v := 0; v < vocabSize; v++ {
		var sum float32
		// LM Head is [Vocab, Hidden] usually (linear layer weight)
		// Index: v * Hidden + d
		offset := v * hiddenSize
		for d := 0; d < hiddenSize; d++ {
			sum += lastTokenNormalized[d] * lmHead[offset+d]
		}
		logits[v] = sum
	}

	applyRepetitionPenalty(logits, recentTokens(tokens), repetitionPenalty)

	// Use greedy if temperature is effectively 0, else sample
	var next int
	if deterministic {
		next = sampleTopK(logits, 1, 0) // Greedily pick top 1
	} else {
		next = sampleTopK(logits, 40, 0.9)
	}
	return next, nil
}

func sampleTopK(logits []float32, topK int, temperature float32) int {
	// If topK is 1 or temperature is 0, do greedy decoding (pick absolute max)
	if topK == 1 || temperature <= 0 {
		maxIdx := 0
		maxVal := logits[0]
		for i, v := range logits {
			if v > maxVal {
				maxVal = v
				maxIdx = i
			}
		}
		return maxIdx
	}

	type pair struct {
		idx int
		val float32
	}
	cands := make([]pair, 0, len(logits))
	for i, v := range logits {
		cands = append(cands, pair{i, v / temperature})
	}
	sort.Slice(cands, func(i, j int) bool { return cands[i].val > cands[j].val })
	if topK > 0 && topK < len(cands) {
		cands = cands[:topK]
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

type kvCacheLayer struct {
	K          []float32
	V          []float32
	MaxSeq     int
	NumKVHeads int
	HeadDim    int
}

type kvCacheState struct {
	Layers map[int]*kvCacheLayer
}

func initKVCacheState(maxSeq int) *kvCacheState {
	state := &kvCacheState{Layers: make(map[int]*kvCacheLayer)}
	for i := range network.Layers {
		cfg := &network.Layers[i]
		if cfg.Type != nn.LayerMultiHeadAttention {
			continue
		}
		numKV := cfg.NumKVHeads
		if numKV == 0 {
			numKV = cfg.NumHeads
		}
		kvDim := numKV * cfg.HeadDim
		state.Layers[i] = &kvCacheLayer{
			K:          make([]float32, maxSeq*kvDim),
			V:          make([]float32, maxSeq*kvDim),
			MaxSeq:     maxSeq,
			NumKVHeads: numKV,
			HeadDim:    cfg.HeadDim,
		}
	}
	return state
}

func embedToken(tokenID int) ([]float32, error) {
	if tokenID < 0 || tokenID >= vocabSize {
		return nil, fmt.Errorf("token ID %d out of bounds for vocab", tokenID)
	}
	offset := tokenID * hiddenSize
	if offset+hiddenSize > len(embeddings) {
		return nil, fmt.Errorf("token ID %d out of bounds for embedding", tokenID)
	}
	vec := make([]float32, hiddenSize)
	copy(vec, embeddings[offset:offset+hiddenSize])
	return vec, nil
}

func applyRoPEHead(vec []float32, pos int, headDim int, theta float64) {
	orig := make([]float32, headDim)
	copy(orig, vec)
	half := headDim / 2
	for d := 0; d < headDim; d++ {
		freq := 1.0 / math.Pow(theta, float64(2*(d%half))/float64(headDim))
		angle := freq * float64(pos)
		c := float32(math.Cos(angle))
		s := float32(math.Sin(angle))
		var rotated float32
		if d < half {
			rotated = -orig[d+half]
		} else {
			rotated = orig[d-half]
		}
		vec[d] = orig[d]*c + rotated*s
	}
}

func mhaForwardTokenKV(input []float32, cfg *nn.LayerConfig, pos int, cache *kvCacheLayer) ([]float32, error) {
	if pos >= cache.MaxSeq {
		return nil, fmt.Errorf("kv cache exceeded max seq len %d", cache.MaxSeq)
	}

	dModel := cfg.DModel
	numHeads := cfg.NumHeads
	numKVHeads := cfg.NumKVHeads
	if numKVHeads == 0 {
		numKVHeads = numHeads
	}
	headDim := cfg.HeadDim
	kvDim := numKVHeads * headDim

	q := make([]float32, dModel)
	for outDim := 0; outDim < dModel; outDim++ {
		sum := cfg.QBias[outDim]
		for inDim := 0; inDim < dModel; inDim++ {
			sum += input[inDim] * cfg.QWeights[inDim*dModel+outDim]
		}
		q[outDim] = sum
	}

	k := make([]float32, kvDim)
	v := make([]float32, kvDim)
	for outDim := 0; outDim < kvDim; outDim++ {
		sumK := cfg.KBias[outDim]
		sumV := cfg.VBias[outDim]
		for inDim := 0; inDim < dModel; inDim++ {
			sumK += input[inDim] * cfg.KWeights[inDim*kvDim+outDim]
			sumV += input[inDim] * cfg.VWeights[inDim*kvDim+outDim]
		}
		k[outDim] = sumK
		v[outDim] = sumV
	}

	ropeTheta := float64(cfg.RoPEFreqBase)
	if ropeTheta == 0 {
		ropeTheta = 10000.0
	}

	for head := 0; head < numHeads; head++ {
		off := head * headDim
		applyRoPEHead(q[off:off+headDim], pos, headDim, ropeTheta)
	}
	for head := 0; head < numKVHeads; head++ {
		off := head * headDim
		applyRoPEHead(k[off:off+headDim], pos, headDim, ropeTheta)
	}

	cacheOffset := pos * kvDim
	copy(cache.K[cacheOffset:cacheOffset+kvDim], k)
	copy(cache.V[cacheOffset:cacheOffset+kvDim], v)

	headsPerKV := numHeads / numKVHeads
	scale := float32(1.0 / math.Sqrt(float64(headDim)))
	attnOut := make([]float32, dModel)

	for head := 0; head < numHeads; head++ {
		kvHead := head / headsPerKV
		qOff := head * headDim
		qHead := q[qOff : qOff+headDim]

		scores := make([]float32, pos+1)
		maxScore := float32(-1e9)
		for t := 0; t <= pos; t++ {
			kOff := t*kvDim + kvHead*headDim
			var dot float32
			for d := 0; d < headDim; d++ {
				dot += qHead[d] * cache.K[kOff+d]
			}
			score := dot * scale
			scores[t] = score
			if score > maxScore {
				maxScore = score
			}
		}

		var expSum float32
		for t := 0; t <= pos; t++ {
			val := float32(math.Exp(float64(scores[t] - maxScore)))
			scores[t] = val
			expSum += val
		}
		if expSum == 0 {
			continue
		}
		for t := 0; t <= pos; t++ {
			scores[t] /= expSum
		}

		for d := 0; d < headDim; d++ {
			var sum float32
			for t := 0; t <= pos; t++ {
				vOff := t*kvDim + kvHead*headDim
				sum += scores[t] * cache.V[vOff+d]
			}
			attnOut[qOff+d] = sum
		}
	}

	output := make([]float32, dModel)
	for outDim := 0; outDim < dModel; outDim++ {
		sum := cfg.OutputBias[outDim]
		for inDim := 0; inDim < dModel; inDim++ {
			sum += attnOut[inDim] * cfg.OutputWeight[inDim*dModel+outDim]
		}
		output[outDim] = sum
	}

	return output, nil
}

func forwardTokenKV(tokenID int, pos int, state *kvCacheState) ([]float32, error) {
	input, err := embedToken(tokenID)
	if err != nil {
		return nil, err
	}

	var residualInput []float32
	for i := range network.Layers {
		cfg := &network.Layers[i]
		switch cfg.Type {
		case nn.LayerEmbedding:
			continue
		case nn.LayerRMSNorm:
			residualInput = make([]float32, len(input))
			copy(residualInput, input)
			input = nn.RmsNormForwardCPU(input, nil, cfg, 1)
		case nn.LayerMultiHeadAttention:
			cache := state.Layers[i]
			if cache == nil {
				return nil, fmt.Errorf("kv cache missing for layer %d", i)
			}
			out, err := mhaForwardTokenKV(input, cfg, pos, cache)
			if err != nil {
				return nil, err
			}
			if residualInput != nil && len(residualInput) == len(out) {
				for j := range out {
					out[j] += residualInput[j]
				}
			}
			residualInput = make([]float32, len(out))
			copy(residualInput, out)
			input = out
		case nn.LayerSwiGLU:
			_, out := nn.SwiGLUForwardCPU(input, cfg, 1)
			if residualInput != nil && len(residualInput) == len(out) {
				for j := range out {
					out[j] += residualInput[j]
				}
			}
			residualInput = make([]float32, len(out))
			copy(residualInput, out)
			input = out
		default:
			return nil, fmt.Errorf("kv cache path does not support layer type %v", cfg.Type)
		}
	}

	return input, nil
}

func nextTokenFromHidden(hidden []float32, recent []int) (int, error) {
	if len(hidden) != hiddenSize {
		return 0, fmt.Errorf("hidden size mismatch: got %d, want %d", len(hidden), hiddenSize)
	}

	normalized := hidden
	if hasFinalNorm && finalNorm != nil {
		finalNormConfig := &nn.LayerConfig{
			Type:     nn.LayerRMSNorm,
			NormSize: hiddenSize,
			Gamma:    finalNorm,
			Epsilon:  1e-6,
		}
		normalized = nn.RmsNormForwardCPU(hidden, nil, finalNormConfig, 1)
	}

	logits := make([]float32, vocabSize)
	for v := 0; v < vocabSize; v++ {
		var sum float32
		offset := v * hiddenSize
		for d := 0; d < hiddenSize; d++ {
			sum += normalized[d] * lmHead[offset+d]
		}
		logits[v] = sum
	}

	applyRepetitionPenalty(logits, recent, repetitionPenalty)

	if deterministic {
		return sampleTopK(logits, 1, 0), nil
	}
	return sampleTopK(logits, 40, 0.9), nil
}

func isEOSToken(token int) bool {
	for _, eos := range eosTokens {
		if token == eos {
			return true
		}
	}
	return false
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

func tryLoadTensor(tensors map[string][]float32, keys []string) []float32 {
	for _, key := range keys {
		if tensor, exists := tensors[key]; exists {
			return tensor
		}
	}
	return nil
}

func applyRepetitionPenalty(logits []float32, recent []int, penalty float32) {
	if penalty <= 1 || len(recent) == 0 {
		return
	}
	seen := make(map[int]struct{}, len(recent))
	for _, tok := range recent {
		if tok < 0 || tok >= len(logits) {
			continue
		}
		if _, ok := seen[tok]; ok {
			continue
		}
		seen[tok] = struct{}{}
		if logits[tok] > 0 {
			logits[tok] /= penalty
		} else {
			logits[tok] *= penalty
		}
	}
}

func recentTokens(tokens []int) []int {
	if repetitionWindow <= 0 || len(tokens) == 0 {
		return nil
	}
	start := len(tokens) - repetitionWindow
	if start < 0 {
		start = 0
	}
	return tokens[start:]
}

func hasToken(tk *tokenizer.Tokenizer, token string) bool {
	if _, ok := tk.SpecialTokens[token]; ok {
		return true
	}
	if _, ok := tk.Vocab[token]; ok {
		return true
	}
	return false
}

func intsToU32(xs []int) []uint32 {
	out := make([]uint32, len(xs))
	for i, v := range xs {
		out[i] = uint32(v)
	}
	return out
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
