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
	"runtime"
	"sort"
	"strings"
	"sync"
	"time"

	"github.com/openfluke/loom/gpu"
	"github.com/openfluke/loom/nn"
	"github.com/openfluke/loom/tokenizer"
)

// ============================================================================
// FAST_TALK_GPU: Optimized LLM inference with GPU support
// ============================================================================

var (
	network      *nn.Network
	embeddings   []float32
	lmHead       []float32
	finalNorm    []float32
	hiddenSize   int
	vocabSize    int
	eosTokens    []int
	hasFinalNorm bool
	tk           *tokenizer.Tokenizer

	numWorkers int
	useGPU     bool
)

var maxTokens = 100
var maxSeqLen = 256
var repetitionPenalty float32 = 1.15
var repetitionWindow = 64

var systemPrompt = "You are a helpful assistant.\n\n"

func main() {
	numWorkers = runtime.NumCPU()
	runtime.GOMAXPROCS(numWorkers)

	fmt.Printf("⚡ FAST_TALK_GPU: %d CPU cores available\n\n", numWorkers)

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
		log.Fatalf("No models found in cache: %s", hubDir)
	}

	fmt.Println("🤖 Available models:")
	for i, model := range models {
		fmt.Printf("  [%d] %s\n", i+1, model)
	}

	reader := bufio.NewReader(os.Stdin)
	rand.Seed(time.Now().UnixNano())

	fmt.Print("\nSelect model: ")
	input, _ := reader.ReadString('\n')
	input = strings.TrimSpace(input)

	var idx int
	if _, err := fmt.Sscanf(input, "%d", &idx); err != nil || idx < 1 || idx > len(models) {
		log.Fatalf("Invalid selection")
	}

	modelName := models[idx-1]
	fmt.Printf("\n📦 Loading: %s\n", modelName)

	modelDir := filepath.Join(hubDir, "models--"+strings.ReplaceAll(modelName, "/", "--"), "snapshots")
	entries, _ = os.ReadDir(modelDir)
	snapshotDir := filepath.Join(modelDir, entries[0].Name())

	// Load tokenizer
	tk, err = tokenizer.LoadFromFile(filepath.Join(snapshotDir, "tokenizer.json"))
	if err != nil {
		log.Fatalf("Failed to load tokenizer: %v", err)
	}
	fmt.Printf("✓ Tokenizer (vocab: %d)\n", tk.VocabSize())

	// Load EOS
	eosTokens = loadEOSTokens(filepath.Join(snapshotDir, "config.json"))
	if len(eosTokens) == 0 {
		eosTokens = []int{2}
	}

	// Ask about GPU
	fmt.Print("\n🚀 Use GPU for weights? (1=yes / 0=no) [0]: ")
	gpuChoice, _ := reader.ReadString('\n')
	gpuChoice = strings.TrimSpace(gpuChoice)
	if gpuChoice == "1" {
		useGPU = true
		gpu.SetAdapterPreference("nvidia")
	}

	// Load model
	network, err = nn.LoadTransformerFromSafetensors(snapshotDir)
	if err != nil {
		log.Fatalf("Error loading model: %v", err)
	}

	tensors, err := nn.LoadSafetensors(filepath.Join(snapshotDir, "model.safetensors"))
	if err != nil {
		log.Fatalf("Error loading weights: %v", err)
	}

	// Load embeddings
	embeddings = tryLoadTensor(tensors, []string{
		"model.embed_tokens.weight", "transformer.wte.weight", "embeddings.weight",
	})
	if embeddings == nil {
		log.Fatalf("Could not find embeddings")
	}

	// Load final norm
	finalNorm = tryLoadTensor(tensors, []string{
		"model.norm.weight", "transformer.ln_f.weight", "norm.weight",
	})
	hasFinalNorm = (finalNorm != nil)

	// Load LM Head
	lmHead = tryLoadTensor(tensors, []string{"lm_head.weight", "output.weight"})
	if lmHead == nil {
		lmHead = embeddings
	}

	hiddenSize = network.InputSize
	vocabSize = len(embeddings) / hiddenSize

	// Mount to GPU if requested
	if useGPU {
		fmt.Println("\n⚡ Mounting weights to GPU...")
		network.BatchSize = maxSeqLen
		for i := range network.Layers {
			network.Layers[i].SeqLength = maxSeqLen
		}
		network.GPU = true
		if err := network.WeightsToGPU(); err != nil {
			fmt.Printf("⚠️ GPU mount failed: %v. Using CPU.\n", err)
			useGPU = false
			network.GPU = false
		} else {
			fmt.Println("✅ GPU mounted successfully!")
		}
	}

	fmt.Printf("✓ Model: hidden=%d, vocab=%d, layers=%d\n", hiddenSize, vocabSize, len(network.Layers))
	if useGPU {
		fmt.Println("✓ Backend: GPU")
	} else {
		fmt.Printf("✓ Backend: CPU (%d workers)\n", numWorkers)
	}
	fmt.Println()

	// Initialize KV cache
	state := initKVCache(maxSeqLen)

	fmt.Printf("🧠 Max tokens: %d\n", maxTokens)
	fmt.Println("Type 'exit' to quit.\n")

	var chatHistory []Turn

	for {
		fmt.Print("You: ")
		userInput, _ := reader.ReadString('\n')
		userInput = strings.TrimSpace(userInput)

		if userInput == "exit" || userInput == "quit" {
			if useGPU {
				network.ReleaseGPUWeights()
			}
			break
		}
		if userInput == "" {
			continue
		}

		prompt := buildPrompt(chatHistory, userInput)
		tokens := encodePrompt(prompt)

		// Prefill KV cache
		var hidden []float32
		for i, tok := range tokens {
			hidden, _ = forwardTokenKV(tok, i, state)
		}

		// Generate
		fmt.Print("Bot: ")
		start := time.Now()
		generated := 0
		var response strings.Builder

		for i := 0; i < maxTokens; i++ {
			nextToken := nextTokenParallel(hidden, tokens)

			tokens = append(tokens, nextToken)
			generated++

			text := tk.Decode(intsToU32(tokens), false)
			promptText := tk.Decode(intsToU32(tokens[:len(tokens)-1]), false)
			if len(text) > len(promptText) {
				newText := text[len(promptText):]
				fmt.Print(newText)
				response.WriteString(newText)
			}

			if isEOS(nextToken) {
				break
			}

			hidden, _ = forwardTokenKV(nextToken, len(tokens)-1, state)
		}

		elapsed := time.Since(start)
		tps := float64(generated) / elapsed.Seconds()
		fmt.Printf("\n\n(%.2f tok/s, %d tokens)\n\n", tps, generated)

		chatHistory = append(chatHistory, Turn{User: userInput, Assistant: response.String()})
	}
}

// ============================================================================
// PARALLEL LM HEAD PROJECTION
// ============================================================================

func nextTokenParallel(hidden []float32, tokens []int) int {
	normalized := hidden
	if hasFinalNorm && finalNorm != nil {
		cfg := &nn.LayerConfig{
			Type:     nn.LayerRMSNorm,
			NormSize: hiddenSize,
			Gamma:    finalNorm,
			Epsilon:  1e-6,
		}
		normalized = nn.RmsNormForwardCPU(hidden, nil, cfg, 1)
	}

	logits := make([]float32, vocabSize)
	chunkSize := (vocabSize + numWorkers - 1) / numWorkers

	var wg sync.WaitGroup
	for w := 0; w < numWorkers; w++ {
		start := w * chunkSize
		end := start + chunkSize
		if end > vocabSize {
			end = vocabSize
		}

		wg.Add(1)
		go func(start, end int) {
			defer wg.Done()
			for v := start; v < end; v++ {
				offset := v * hiddenSize
				sum := dotProductUnrolled(normalized, lmHead[offset:offset+hiddenSize])
				logits[v] = sum
			}
		}(start, end)
	}
	wg.Wait()

	applyRepPenalty(logits, tokens)
	return sampleTopK(logits, 40, 0.9)
}

func dotProductUnrolled(a, b []float32) float32 {
	n := len(a)
	var sum float32
	i := 0
	for ; i <= n-8; i += 8 {
		sum += a[i]*b[i] + a[i+1]*b[i+1] + a[i+2]*b[i+2] + a[i+3]*b[i+3] +
			a[i+4]*b[i+4] + a[i+5]*b[i+5] + a[i+6]*b[i+6] + a[i+7]*b[i+7]
	}
	for ; i < n; i++ {
		sum += a[i] * b[i]
	}
	return sum
}

// ============================================================================
// KV CACHE WITH PARALLEL ATTENTION
// ============================================================================

type kvCache struct {
	K, V       []float32
	MaxSeq     int
	NumKVHeads int
	HeadDim    int
}

type kvState struct {
	Layers map[int]*kvCache
}

func initKVCache(maxSeq int) *kvState {
	state := &kvState{Layers: make(map[int]*kvCache)}
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
		state.Layers[i] = &kvCache{
			K:          make([]float32, maxSeq*kvDim),
			V:          make([]float32, maxSeq*kvDim),
			MaxSeq:     maxSeq,
			NumKVHeads: numKV,
			HeadDim:    cfg.HeadDim,
		}
	}
	return state
}

func forwardTokenKV(tokenID int, pos int, state *kvState) ([]float32, error) {
	input := embedToken(tokenID)
	var residual []float32

	for i := range network.Layers {
		cfg := &network.Layers[i]
		switch cfg.Type {
		case nn.LayerEmbedding:
			continue
		case nn.LayerRMSNorm:
			residual = make([]float32, len(input))
			copy(residual, input)
			input = nn.RmsNormForwardCPU(input, nil, cfg, 1)
		case nn.LayerMultiHeadAttention:
			cache := state.Layers[i]
			out := mhaParallel(input, cfg, pos, cache)
			if residual != nil && len(residual) == len(out) {
				for j := range out {
					out[j] += residual[j]
				}
			}
			residual = make([]float32, len(out))
			copy(residual, out)
			input = out
		case nn.LayerSwiGLU:
			_, out := nn.SwiGLUForwardCPU(input, cfg, 1)
			if residual != nil && len(residual) == len(out) {
				for j := range out {
					out[j] += residual[j]
				}
			}
			residual = make([]float32, len(out))
			copy(residual, out)
			input = out
		}
	}
	return input, nil
}

func mhaParallel(input []float32, cfg *nn.LayerConfig, pos int, cache *kvCache) []float32 {
	dModel := cfg.DModel
	numHeads := cfg.NumHeads
	numKVHeads := cfg.NumKVHeads
	if numKVHeads == 0 {
		numKVHeads = numHeads
	}
	headDim := cfg.HeadDim
	kvDim := numKVHeads * headDim

	q := matVec(cfg.QWeights, input, dModel, cfg.QBias)
	k := matVec(cfg.KWeights, input, kvDim, cfg.KBias)
	v := matVec(cfg.VWeights, input, kvDim, cfg.VBias)

	theta := cfg.RoPEFreqBase
	if theta == 0 {
		theta = 10000.0
	}
	for h := 0; h < numHeads; h++ {
		off := h * headDim
		applyRoPE(q[off:off+headDim], pos, headDim, float64(theta))
	}
	for h := 0; h < numKVHeads; h++ {
		off := h * headDim
		applyRoPE(k[off:off+headDim], pos, headDim, float64(theta))
	}

	cacheOff := pos * kvDim
	copy(cache.K[cacheOff:cacheOff+kvDim], k)
	copy(cache.V[cacheOff:cacheOff+kvDim], v)

	attnOut := make([]float32, dModel)
	headsPerKV := numHeads / numKVHeads
	scale := float32(1.0 / math.Sqrt(float64(headDim)))

	var wg sync.WaitGroup
	mu := sync.Mutex{}

	for h := 0; h < numHeads; h++ {
		wg.Add(1)
		go func(head int) {
			defer wg.Done()

			kvHead := head / headsPerKV
			qOff := head * headDim
			qHead := q[qOff : qOff+headDim]

			scores := make([]float32, pos+1)
			maxScore := float32(-1e9)
			for t := 0; t <= pos; t++ {
				kOff := t*kvDim + kvHead*headDim
				dot := dotProductUnrolled(qHead, cache.K[kOff:kOff+headDim])
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
			if expSum > 0 {
				for t := 0; t <= pos; t++ {
					scores[t] /= expSum
				}
			}

			headOut := make([]float32, headDim)
			for d := 0; d < headDim; d++ {
				var sum float32
				for t := 0; t <= pos; t++ {
					vOff := t*kvDim + kvHead*headDim
					sum += scores[t] * cache.V[vOff+d]
				}
				headOut[d] = sum
			}

			mu.Lock()
			for d := 0; d < headDim; d++ {
				attnOut[qOff+d] = headOut[d]
			}
			mu.Unlock()
		}(h)
	}
	wg.Wait()

	return matVec(cfg.OutputWeight, attnOut, dModel, cfg.OutputBias)
}

func matVec(weights, input []float32, outSize int, bias []float32) []float32 {
	inSize := len(input)
	out := make([]float32, outSize)
	for o := 0; o < outSize; o++ {
		sum := bias[o]
		for i := 0; i < inSize; i++ {
			sum += input[i] * weights[i*outSize+o]
		}
		out[o] = sum
	}
	return out
}

func applyRoPE(vec []float32, pos int, dim int, theta float64) {
	orig := make([]float32, dim)
	copy(orig, vec)
	half := dim / 2
	for d := 0; d < dim; d++ {
		freq := 1.0 / math.Pow(theta, float64(2*(d%half))/float64(dim))
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

func embedToken(tokenID int) []float32 {
	offset := tokenID * hiddenSize
	vec := make([]float32, hiddenSize)
	copy(vec, embeddings[offset:offset+hiddenSize])
	return vec
}

// ============================================================================
// UTILITIES
// ============================================================================

type Turn struct {
	User, Assistant string
}

func buildPrompt(turns []Turn, userMsg string) string {
	var sb strings.Builder
	sb.WriteString("<|im_start|>system\n")
	sb.WriteString(strings.TrimSpace(systemPrompt))
	sb.WriteString("<|im_end|>\n")

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

func encodePrompt(prompt string) []int {
	ids := tk.Encode(prompt, false)
	tokens := make([]int, len(ids))
	for i, id := range ids {
		tokens[i] = int(id)
	}
	return tokens
}

func applyRepPenalty(logits []float32, tokens []int) {
	start := 0
	if len(tokens) > repetitionWindow {
		start = len(tokens) - repetitionWindow
	}
	for i := start; i < len(tokens); i++ {
		tok := tokens[i]
		if tok < len(logits) {
			if logits[tok] > 0 {
				logits[tok] /= repetitionPenalty
			} else {
				logits[tok] *= repetitionPenalty
			}
		}
	}
}

func sampleTopK(logits []float32, topK int, temperature float32) int {
	if temperature <= 0 {
		temperature = 1
	}

	type pair struct {
		idx int
		val float32
	}
	cands := make([]pair, len(logits))
	for i, v := range logits {
		cands[i] = pair{i, v / temperature}
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

func isEOS(token int) bool {
	for _, eos := range eosTokens {
		if token == eos {
			return true
		}
	}
	return false
}

func intsToU32(ints []int) []uint32 {
	result := make([]uint32, len(ints))
	for i, v := range ints {
		result[i] = uint32(v)
	}
	return result
}

func tryLoadTensor(tensors map[string][]float32, names []string) []float32 {
	for _, name := range names {
		if t, ok := tensors[name]; ok {
			return t
		}
	}
	return nil
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
			for _, t := range v {
				if f, ok := t.(float64); ok {
					tokens = append(tokens, int(f))
				}
			}
		}
	}
	return tokens
}
