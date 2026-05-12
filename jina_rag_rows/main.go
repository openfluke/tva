package main

import (
	"bufio"
	"encoding/json"
	"errors"
	"flag"
	"fmt"
	"hash/fnv"
	"io"
	"math"
	"math/rand"
	"net/http"
	"os"
	"path/filepath"
	"regexp"
	"sort"
	"strings"
	"time"
	"unicode"

	"github.com/openfluke/loom/poly"
)

const (
	modelID       = "jinaai/jina-embeddings-v2-base-en"
	snapshotName  = poly.HFManualSnapshotDirName
	defaultDim    = 768
	defaultTopK   = 8
	userAgent     = "Loom-TVA-Jina-RAG-Rows/0.1"
	rowsFileName  = "rows.jsonl"
	indexFileName = "router_index.jsonl"
	crmDirName    = "crm"
)

type hfFile struct {
	Path     string
	Required bool
}

type corpusSource struct {
	Name string `json:"name"`
	File string `json:"file"`
	URL  string `json:"url"`
}

type textRow struct {
	RowID        string   `json:"row_id"`
	Source       string   `json:"source"`
	SourceURL    string   `json:"source_url"`
	SourceRow    int      `json:"source_row"`
	Text         string   `json:"text"`
	Labels       []string `json:"labels"`
	TokenCount   int      `json:"token_count,omitempty"`
	TokenIDsHead []uint32 `json:"token_ids_head,omitempty"`
}

type indexRow struct {
	RowID       string    `json:"row_id"`
	Source      string    `json:"source"`
	Labels      []string  `json:"labels"`
	TextPreview string    `json:"text_preview"`
	Vector      []float32 `json:"vector"`
}

type searchHit struct {
	RowID       string   `json:"row_id"`
	Source      string   `json:"source"`
	Score       float64  `json:"score"`
	Labels      []string `json:"labels"`
	TextPreview string   `json:"text_preview"`
}

type sampleProbe struct {
	SampleID      string      `json:"sample_id"`
	Query         string      `json:"query"`
	ExpectedRowID string      `json:"expected_row_id"`
	HitRank       int         `json:"hit_rank"`
	Hits          []searchHit `json:"hits"`
}

type routerMeta struct {
	Embedder  string `json:"embedder"`
	Dim       int    `json:"dim"`
	MaxTokens int    `json:"max_tokens,omitempty"`
	ModelID   string `json:"model_id,omitempty"`
}

type textEmbedder interface {
	Name() string
	Dim() int
	Embed(text string, labels []string) []float32
}

type hashEmbedder struct {
	dim int
}

type jinaTokenEmbedder struct {
	tokenizer *wordPieceTokenizer
	layer     *poly.VolumetricLayer
	dim       int
	maxTokens int
}

type wordPieceTokenizer struct {
	vocab     map[string]int
	unkID     int
	clsID     int
	sepID     int
	lowercase bool
}

type crmAccount struct {
	AccountID       string   `json:"account_id"`
	AccountName     string   `json:"account_name"`
	Industry        string   `json:"industry"`
	Region          string   `json:"region"`
	Owner           string   `json:"owner"`
	ContactName     string   `json:"contact_name"`
	ContactEmail    string   `json:"contact_email"`
	Service         string   `json:"service"`
	PreferredWindow string   `json:"preferred_window"`
	Location        string   `json:"location"`
	Attendees       int      `json:"attendees"`
	BillingCode     string   `json:"billing_code"`
	Requirements    []string `json:"requirements"`
}

type crmNoteRow struct {
	RowID       string     `json:"row_id"`
	AccountID   string     `json:"account_id"`
	AccountName string     `json:"account_name"`
	NoteType    string     `json:"note_type"`
	Text        string     `json:"text"`
	Labels      []string   `json:"labels"`
	Account     crmAccount `json:"account"`
}

type crmIndexRow struct {
	RowID       string    `json:"row_id"`
	AccountID   string    `json:"account_id"`
	AccountName string    `json:"account_name"`
	NoteType    string    `json:"note_type"`
	Labels      []string  `json:"labels"`
	TextPreview string    `json:"text_preview"`
	Vector      []float32 `json:"vector"`
}

type crmBookingEmail struct {
	EmailID           string `json:"email_id"`
	ExpectedAccountID string `json:"expected_account_id"`
	From              string `json:"from"`
	Subject           string `json:"subject"`
	Body              string `json:"body"`
}

type crmBookingResult struct {
	EmailID           string      `json:"email_id"`
	ExpectedAccountID string      `json:"expected_account_id"`
	ExpectedAccount   string      `json:"expected_account"`
	HitRank           int         `json:"hit_rank"`
	ExtractedAccount  *crmAccount `json:"extracted_account,omitempty"`
	Evidence          []crmHit    `json:"evidence"`
}

type crmHit struct {
	RowID       string   `json:"row_id"`
	AccountID   string   `json:"account_id"`
	AccountName string   `json:"account_name"`
	NoteType    string   `json:"note_type"`
	Score       float64  `json:"score"`
	Labels      []string `json:"labels"`
	TextPreview string   `json:"text_preview"`
}

type loomStatus struct {
	ModelID                   string   `json:"model_id"`
	SnapshotDir               string   `json:"snapshot_dir"`
	ModelType                 string   `json:"model_type,omitempty"`
	Architectures             []string `json:"architectures,omitempty"`
	TokenizerLoaded           bool     `json:"tokenizer_loaded"`
	TokenizerError            string   `json:"tokenizer_error,omitempty"`
	SafetensorsHeaderReadable bool     `json:"safetensors_header_readable"`
	SafetensorsError          string   `json:"safetensors_error,omitempty"`
	TensorNameSamples         []string `json:"tensor_name_samples,omitempty"`
	LoomTransformerRunnable   bool     `json:"loom_transformer_runnable"`
	Reason                    string   `json:"reason"`
}

var jinaFiles = []hfFile{
	{Path: "config.json", Required: true},
	{Path: "model.safetensors", Required: true},
	{Path: "tokenizer.json", Required: true},
	{Path: "tokenizer_config.json", Required: false},
	{Path: "special_tokens_map.json", Required: false},
	{Path: "modules.json", Required: false},
	{Path: "sentence_bert_config.json", Required: false},
	{Path: "1_Pooling/config.json", Required: false},
}

var corpusSources = []corpusSource{
	{
		Name: "Pride and Prejudice",
		File: "pride_and_prejudice.txt",
		URL:  "https://www.gutenberg.org/cache/epub/1342/pg1342.txt",
	},
	{
		Name: "Frankenstein",
		File: "frankenstein.txt",
		URL:  "https://www.gutenberg.org/cache/epub/84/pg84.txt",
	},
	{
		Name: "Sherlock Holmes",
		File: "sherlock_holmes.txt",
		URL:  "https://www.gutenberg.org/cache/epub/1661/pg1661.txt",
	},
	{
		Name: "Moby Dick",
		File: "moby_dick.txt",
		URL:  "https://www.gutenberg.org/cache/epub/2701/pg2701.txt",
	},
	{
		Name: "Alice in Wonderland",
		File: "alice_in_wonderland.txt",
		URL:  "https://www.gutenberg.org/cache/epub/11/pg11.txt",
	},
}

func main() {
	outDir := flag.String("out", defaultOutDir(), "artifact directory")
	mode := flag.String("mode", "all", "all, download, rows, index, samples, examples, crm, query, status")
	query := flag.String("query", "", "query text for -mode query")
	topK := flag.Int("topk", defaultTopK, "number of retrieved rows")
	maxRows := flag.Int("max-rows", 2500, "maximum corpus rows to create")
	chunkChars := flag.Int("chunk-chars", 900, "target characters per row")
	overlapChars := flag.Int("overlap-chars", 120, "overlap between adjacent rows")
	sampleCount := flag.Int("sample-count", 100, "number of sample probes to write")
	examples := flag.Int("examples", 10, "number of sample probe results to print after a run; 0 disables")
	crmAccounts := flag.Int("crm-accounts", 60, "number of synthetic Salesforce-like accounts to generate")
	crmExamples := flag.Int("crm-examples", 5, "number of CRM booking examples to print after a run; 0 disables")
	embedderName := flag.String("embedder", "loom-jina-token", "vector source: loom-jina-token or hash")
	maxTokens := flag.Int("max-tokens", 512, "maximum WordPiece tokens for loom-jina-token pooling")
	dim := flag.Int("dim", defaultDim, "router vector dimension; Jina base uses 768")
	force := flag.Bool("force", false, "redownload/rebuild even when outputs exist")
	flag.Parse()

	if *topK <= 0 {
		failf("topk must be positive")
	}
	if *dim <= 0 {
		failf("dim must be positive")
	}
	if *examples < 0 {
		failf("examples must be >= 0")
	}
	if *crmAccounts <= 0 {
		failf("crm-accounts must be positive")
	}
	if *crmExamples < 0 {
		failf("crm-examples must be >= 0")
	}
	if *maxTokens <= 0 {
		failf("max-tokens must be positive")
	}
	if err := os.MkdirAll(*outDir, 0o755); err != nil {
		failf("create artifacts: %v", err)
	}

	client := &http.Client{Timeout: 0}
	switch strings.ToLower(strings.TrimSpace(*mode)) {
	case "all":
		runStep("download model", func() error { return downloadModel(client, *outDir, *force) })
		runStep("download corpus", func() error { return downloadCorpus(client, *outDir, *force) })
		runStep("inspect loom compatibility", func() error { return writeLoomStatus(*outDir) })
		runStep("make rows", func() error { return makeRows(*outDir, *maxRows, *chunkChars, *overlapChars, *force) })
		runStep("build index", func() error { return buildIndex(*outDir, *embedderName, *dim, *maxTokens, *force) })
		runStep("write samples", func() error {
			return writeSamples(*outDir, *sampleCount, *topK, *embedderName, *dim, *maxTokens, *force)
		})
		if *examples > 0 {
			runStep("run sample examples", func() error { return printSampleReport(*outDir, *sampleCount, *examples) })
		}
		runStep("make CRM booking notes", func() error { return makeCRMScenario(*outDir, *crmAccounts, *embedderName, *dim, *maxTokens, *force) })
		if *crmExamples > 0 {
			runStep("run CRM booking examples", func() error { return printCRMReport(*outDir, *crmExamples) })
		}
	case "download":
		runStep("download model", func() error { return downloadModel(client, *outDir, *force) })
		runStep("download corpus", func() error { return downloadCorpus(client, *outDir, *force) })
		runStep("inspect loom compatibility", func() error { return writeLoomStatus(*outDir) })
	case "rows":
		runStep("make rows", func() error { return makeRows(*outDir, *maxRows, *chunkChars, *overlapChars, *force) })
	case "index":
		runStep("build index", func() error { return buildIndex(*outDir, *embedderName, *dim, *maxTokens, *force) })
	case "samples":
		runStep("write samples", func() error {
			return writeSamples(*outDir, *sampleCount, *topK, *embedderName, *dim, *maxTokens, *force)
		})
		if *examples > 0 {
			runStep("run sample examples", func() error { return printSampleReport(*outDir, *sampleCount, *examples) })
		}
	case "examples":
		runStep("run sample examples", func() error { return printSampleReport(*outDir, *sampleCount, *examples) })
	case "crm":
		runStep("make CRM booking notes", func() error { return makeCRMScenario(*outDir, *crmAccounts, *embedderName, *dim, *maxTokens, *force) })
		if *crmExamples > 0 {
			runStep("run CRM booking examples", func() error { return printCRMReport(*outDir, *crmExamples) })
		}
	case "query":
		if strings.TrimSpace(*query) == "" {
			failf("-query is required for -mode query")
		}
		emb, err := newTextEmbedder(*outDir, *embedderName, *dim, *maxTokens)
		if err != nil {
			failf("load embedder: %v", err)
		}
		hits, err := searchWithEmbedder(*outDir, *query, *topK, emb)
		if err != nil {
			failf("query: %v", err)
		}
		enc := json.NewEncoder(os.Stdout)
		enc.SetIndent("", "  ")
		if err := enc.Encode(hits); err != nil {
			failf("write query hits: %v", err)
		}
	case "status":
		runStep("inspect loom compatibility", func() error { return writeLoomStatus(*outDir) })
	default:
		failf("unknown mode %q", *mode)
	}
}

func runStep(name string, fn func() error) {
	fmt.Printf("==> %s\n", name)
	if err := fn(); err != nil {
		failf("%s: %v", name, err)
	}
}

func failf(format string, args ...any) {
	fmt.Fprintf(os.Stderr, "error: "+format+"\n", args...)
	os.Exit(1)
}

func defaultOutDir() string {
	cwd, err := os.Getwd()
	if err == nil && filepath.Base(cwd) == "jina_rag_rows" {
		return "artifacts"
	}
	if err == nil {
		if st, statErr := os.Stat(filepath.Join(cwd, "tva", "jina_rag_rows")); statErr == nil && st.IsDir() {
			return filepath.Join("tva", "jina_rag_rows", "artifacts")
		}
	}
	return "artifacts"
}

func modelSnapshotDir(outDir string) string {
	return filepath.Join(outDir, "hf-hub", "models--jinaai--jina-embeddings-v2-base-en", "snapshots", snapshotName)
}

func newTextEmbedder(outDir, name string, dim, maxTokens int) (textEmbedder, error) {
	switch strings.ToLower(strings.TrimSpace(name)) {
	case "", "loom-jina-token", "jina", "loom":
		return newJinaTokenEmbedder(outDir, maxTokens)
	case "hash", "hashed":
		return &hashEmbedder{dim: dim}, nil
	default:
		return nil, fmt.Errorf("unknown embedder %q", name)
	}
}

func (h *hashEmbedder) Name() string { return "hash" }

func (h *hashEmbedder) Dim() int { return h.dim }

func (h *hashEmbedder) Embed(text string, labels []string) []float32 {
	return vectorize(text, labels, h.dim)
}

func newJinaTokenEmbedder(outDir string, maxTokens int) (*jinaTokenEmbedder, error) {
	snapDir := modelSnapshotDir(outDir)
	tok, err := loadWordPieceTokenizer(filepath.Join(snapDir, "tokenizer.json"))
	if err != nil {
		return nil, err
	}
	tensors, err := poly.LoadSafetensorsSelective(filepath.Join(snapDir, "model.safetensors"), func(name string) bool {
		return name == "embeddings.word_embeddings.weight"
	})
	if err != nil {
		return nil, err
	}
	weights := tensors["embeddings.word_embeddings.weight"]
	if len(weights) == 0 {
		return nil, errors.New("missing embeddings.word_embeddings.weight")
	}
	if len(weights)%defaultDim != 0 {
		return nil, fmt.Errorf("unexpected Jina embedding table size %d, not divisible by %d", len(weights), defaultDim)
	}
	vocabSize := len(weights) / defaultDim
	layer := &poly.VolumetricLayer{
		Type:         poly.LayerEmbedding,
		VocabSize:    vocabSize,
		EmbeddingDim: defaultDim,
		DType:        poly.DTypeFloat32,
		WeightStore:  poly.NewWeightStore(len(weights)),
	}
	copy(layer.WeightStore.Master, weights)
	fmt.Printf("  embedder: Loom LayerEmbedding + %s word embeddings (%d x %d, max_tokens=%d)\n", modelID, vocabSize, defaultDim, maxTokens)
	return &jinaTokenEmbedder{tokenizer: tok, layer: layer, dim: defaultDim, maxTokens: maxTokens}, nil
}

func (j *jinaTokenEmbedder) Name() string { return "loom-jina-token" }

func (j *jinaTokenEmbedder) Dim() int { return j.dim }

func (j *jinaTokenEmbedder) Embed(text string, labels []string) []float32 {
	if len(labels) > 0 {
		text = text + "\n\nlabels: " + strings.Join(labels, " ")
	}
	ids := j.tokenizer.Encode(text, j.maxTokens)
	if len(ids) == 0 {
		return make([]float32, j.dim)
	}
	input := poly.NewTensor[float32](len(ids))
	for i, id := range ids {
		input.Data[i] = float32(id)
	}
	_, out := poly.EmbeddingForwardPolymorphic[float32](j.layer, input)
	vec := make([]float32, j.dim)
	var count float32
	for i, id := range ids {
		if id == uint32(j.tokenizer.clsID) || id == uint32(j.tokenizer.sepID) {
			continue
		}
		row := out.Data[i*j.dim : (i+1)*j.dim]
		for d := 0; d < j.dim; d++ {
			vec[d] += row[d]
		}
		count++
	}
	if count == 0 {
		return vec
	}
	for d := range vec {
		vec[d] /= count
	}
	normalizeFloat32(vec)
	return vec
}

func downloadModel(client *http.Client, outDir string, force bool) error {
	snapDir := modelSnapshotDir(outDir)
	for i, f := range jinaFiles {
		url := "https://huggingface.co/" + modelID + "/resolve/main/" + f.Path
		dest := filepath.Join(snapDir, filepath.FromSlash(f.Path))
		fmt.Printf("  [%d/%d] %s\n", i+1, len(jinaFiles), f.Path)
		if err := downloadToFile(client, url, dest, f.Required, force); err != nil {
			return err
		}
	}

	refsDir := filepath.Join(outDir, "hf-hub", "models--jinaai--jina-embeddings-v2-base-en", "refs")
	if err := os.MkdirAll(refsDir, 0o755); err != nil {
		return err
	}
	if err := os.WriteFile(filepath.Join(refsDir, "main"), []byte(snapshotName), 0o644); err != nil {
		return err
	}

	manifest := map[string]any{
		"model_id":      modelID,
		"snapshot_dir":  snapDir,
		"downloaded_at": time.Now().Format(time.RFC3339),
		"files":         jinaFiles,
	}
	return writeJSON(filepath.Join(outDir, "model_manifest.json"), manifest)
}

func downloadCorpus(client *http.Client, outDir string, force bool) error {
	rawDir := filepath.Join(outDir, "corpus", "raw")
	for i, src := range corpusSources {
		fmt.Printf("  [%d/%d] %s\n", i+1, len(corpusSources), src.Name)
		if err := downloadToFile(client, src.URL, filepath.Join(rawDir, src.File), true, force); err != nil {
			return err
		}
	}
	return writeJSON(filepath.Join(outDir, "corpus", "sources.json"), corpusSources)
}

func downloadToFile(client *http.Client, url, dest string, required, force bool) error {
	if !force {
		if st, err := os.Stat(dest); err == nil && st.Size() > 0 {
			fmt.Printf("    skip existing %s\n", dest)
			return nil
		}
	}
	if err := os.MkdirAll(filepath.Dir(dest), 0o755); err != nil {
		return err
	}

	req, err := http.NewRequest(http.MethodGet, url, nil)
	if err != nil {
		return err
	}
	req.Header.Set("User-Agent", userAgent)
	if tok := strings.TrimSpace(os.Getenv("HUGGING_FACE_HUB_TOKEN")); tok != "" && strings.Contains(url, "huggingface.co/") {
		req.Header.Set("Authorization", "Bearer "+tok)
	}

	resp, err := client.Do(req)
	if err != nil {
		return err
	}
	defer resp.Body.Close()
	if resp.StatusCode == http.StatusNotFound && !required {
		fmt.Printf("    optional missing %s\n", url)
		return nil
	}
	if resp.StatusCode < 200 || resp.StatusCode >= 300 {
		return fmt.Errorf("GET %s: HTTP %d", url, resp.StatusCode)
	}

	part := dest + ".part"
	_ = os.Remove(part)
	out, err := os.Create(part)
	if err != nil {
		return err
	}
	_, copyErr := io.Copy(out, resp.Body)
	closeErr := out.Close()
	if copyErr != nil {
		_ = os.Remove(part)
		return copyErr
	}
	if closeErr != nil {
		_ = os.Remove(part)
		return closeErr
	}
	if st, err := os.Stat(part); err != nil || st.Size() == 0 {
		_ = os.Remove(part)
		return fmt.Errorf("empty download: %s", url)
	}
	_ = os.Remove(dest)
	return os.Rename(part, dest)
}

func writeLoomStatus(outDir string) error {
	snapDir := modelSnapshotDir(outDir)
	status := loomStatus{
		ModelID:                 modelID,
		SnapshotDir:             snapDir,
		LoomTransformerRunnable: false,
		Reason:                  "Jina v2 is a BERT-style embedding encoder with pooling; Loom's current HF transformer loader is wired for decoder blocks (RMSNorm/MHA/SwiGLU) rather than this encoder architecture.",
	}

	configPath := filepath.Join(snapDir, "config.json")
	configData, err := os.ReadFile(configPath)
	if err == nil {
		var cfg map[string]any
		if json.Unmarshal(configData, &cfg) == nil {
			if v, ok := cfg["model_type"].(string); ok {
				status.ModelType = v
			}
			if raw, ok := cfg["architectures"].([]any); ok {
				for _, item := range raw {
					if s, ok := item.(string); ok {
						status.Architectures = append(status.Architectures, s)
					}
				}
			}
		}
	}

	if _, err := poly.LoadTokenizer(filepath.Join(snapDir, "tokenizer.json")); err != nil {
		status.TokenizerError = err.Error()
	} else {
		status.TokenizerLoaded = true
	}

	names, err := poly.SafetensorsTensorNames(filepath.Join(snapDir, "model.safetensors"))
	if err != nil {
		status.SafetensorsError = err.Error()
	} else {
		status.SafetensorsHeaderReadable = true
		sort.Strings(names)
		if len(names) > 20 {
			names = names[:20]
		}
		status.TensorNameSamples = names
	}

	return writeJSON(filepath.Join(outDir, "loom_status.json"), status)
}

func makeRows(outDir string, maxRows, chunkChars, overlapChars int, force bool) error {
	if maxRows <= 0 {
		return fmt.Errorf("max rows must be positive")
	}
	if chunkChars <= 100 {
		return fmt.Errorf("chunk chars must be > 100")
	}
	if overlapChars < 0 || overlapChars >= chunkChars {
		return fmt.Errorf("overlap chars must be >= 0 and smaller than chunk chars")
	}

	rowsPath := filepath.Join(outDir, rowsFileName)
	if !force {
		if st, err := os.Stat(rowsPath); err == nil && st.Size() > 0 {
			fmt.Printf("  skip existing %s\n", rowsPath)
			return nil
		}
	}

	var tok *poly.Tokenizer
	if loaded, err := poly.LoadTokenizer(filepath.Join(modelSnapshotDir(outDir), "tokenizer.json")); err == nil {
		tok = loaded
	} else {
		fmt.Printf("  tokenizer note: %v\n", err)
	}

	if err := os.MkdirAll(filepath.Dir(rowsPath), 0o755); err != nil {
		return err
	}
	out, err := os.Create(rowsPath)
	if err != nil {
		return err
	}
	defer out.Close()
	enc := json.NewEncoder(out)

	rowCount := 0
	for _, src := range corpusSources {
		if rowCount >= maxRows {
			break
		}
		data, err := os.ReadFile(filepath.Join(outDir, "corpus", "raw", src.File))
		if err != nil {
			return fmt.Errorf("read corpus %s: %w", src.File, err)
		}
		text := normalizeText(stripGutenbergBoilerplate(string(data)))
		chunks := chunkText(text, chunkChars, overlapChars)
		for i, chunk := range chunks {
			if rowCount >= maxRows {
				break
			}
			rowCount++
			row := textRow{
				RowID:     fmt.Sprintf("row_%06d", rowCount),
				Source:    src.Name,
				SourceURL: src.URL,
				SourceRow: i,
				Text:      chunk,
				Labels:    extractLabels(chunk, 14),
			}
			if tok != nil {
				ids := tok.Encode(chunk, false)
				row.TokenCount = len(ids)
				if len(ids) > 32 {
					row.TokenIDsHead = append([]uint32(nil), ids[:32]...)
				} else {
					row.TokenIDsHead = ids
				}
			}
			if err := enc.Encode(row); err != nil {
				return err
			}
		}
	}
	fmt.Printf("  wrote %d rows -> %s\n", rowCount, rowsPath)
	return nil
}

func buildIndex(outDir, embedderName string, dim, maxTokens int, force bool) error {
	emb, err := newTextEmbedder(outDir, embedderName, dim, maxTokens)
	if err != nil {
		return err
	}
	meta := routerMeta{Embedder: emb.Name(), Dim: emb.Dim(), MaxTokens: maxTokens}
	if emb.Name() == "loom-jina-token" {
		meta.ModelID = modelID
	}
	indexPath := filepath.Join(outDir, indexFileName)
	metaPath := filepath.Join(outDir, "router_index.meta.json")
	if !force && routerMetaMatches(metaPath, meta) {
		if st, err := os.Stat(indexPath); err == nil && st.Size() > 0 {
			fmt.Printf("  skip existing %s\n", indexPath)
			return nil
		}
	}

	rows, err := readRows(filepath.Join(outDir, rowsFileName))
	if err != nil {
		return err
	}
	out, err := os.Create(indexPath)
	if err != nil {
		return err
	}
	defer out.Close()
	enc := json.NewEncoder(out)
	for _, row := range rows {
		idx := indexRow{
			RowID:       row.RowID,
			Source:      row.Source,
			Labels:      row.Labels,
			TextPreview: preview(row.Text, 260),
			Vector:      emb.Embed(row.Text, row.Labels),
		}
		if err := enc.Encode(idx); err != nil {
			return err
		}
	}
	if err := writeJSON(metaPath, meta); err != nil {
		return err
	}
	fmt.Printf("  wrote %d vectors -> %s\n", len(rows), indexPath)
	return nil
}

func writeSamples(outDir string, sampleCount, topK int, embedderName string, dim, maxTokens int, force bool) error {
	if sampleCount <= 0 {
		return fmt.Errorf("sample count must be positive")
	}
	samplesPath := sampleFilePath(outDir, sampleCount)
	if !force {
		if st, err := os.Stat(samplesPath); err == nil && st.Size() > 0 {
			fmt.Printf("  skip existing %s\n", samplesPath)
			return nil
		}
	}

	rows, err := readRows(filepath.Join(outDir, rowsFileName))
	if err != nil {
		return err
	}
	if len(rows) == 0 {
		return errors.New("no rows to sample")
	}
	emb, err := newTextEmbedder(outDir, embedderName, dim, maxTokens)
	if err != nil {
		return err
	}
	out, err := os.Create(samplesPath)
	if err != nil {
		return err
	}
	defer out.Close()
	enc := json.NewEncoder(out)

	picks := sampleRows(len(rows), sampleCount)
	for i, rowIdx := range picks {
		row := rows[rowIdx]
		q := probeQuery(row)
		hits, err := searchWithEmbedder(outDir, q, topK, emb)
		if err != nil {
			return err
		}
		hitRank := 0
		for rank, hit := range hits {
			if hit.RowID == row.RowID {
				hitRank = rank + 1
				break
			}
		}
		probe := sampleProbe{
			SampleID:      fmt.Sprintf("sample_%03d", i+1),
			Query:         q,
			ExpectedRowID: row.RowID,
			HitRank:       hitRank,
			Hits:          hits,
		}
		if err := enc.Encode(probe); err != nil {
			return err
		}
	}
	fmt.Printf("  wrote %d probes -> %s\n", len(picks), samplesPath)
	return nil
}

func sampleFilePath(outDir string, sampleCount int) string {
	if sampleCount == 100 {
		return filepath.Join(outDir, "samples_100.jsonl")
	}
	return filepath.Join(outDir, fmt.Sprintf("samples_%d.jsonl", sampleCount))
}

func printSampleReport(outDir string, sampleCount, examples int) error {
	if examples <= 0 {
		return nil
	}
	samples, err := readSamples(sampleFilePath(outDir, sampleCount))
	if err != nil {
		return err
	}
	if len(samples) == 0 {
		return errors.New("no sample probes found")
	}

	exact := 0
	topK := 0
	missed := 0
	for _, s := range samples {
		switch {
		case s.HitRank == 1:
			exact++
			topK++
		case s.HitRank > 1:
			topK++
		default:
			missed++
		}
	}

	fmt.Printf("  sample probes: %d | rank-1: %d | top-k: %d | misses: %d\n", len(samples), exact, topK, missed)
	if examples > len(samples) {
		examples = len(samples)
	}
	for i := 0; i < examples; i++ {
		s := samples[i]
		result := "MISS"
		if s.HitRank == 1 {
			result = "HIT rank 1"
		} else if s.HitRank > 1 {
			result = fmt.Sprintf("HIT rank %d", s.HitRank)
		}
		fmt.Printf("\n  [%s] %s expected=%s\n", s.SampleID, result, s.ExpectedRowID)
		fmt.Printf("    query: %s\n", preview(s.Query, 180))
		for rank, hit := range s.Hits {
			fmt.Printf("    #%d %.4f %s %s labels=%s\n", rank+1, hit.Score, hit.RowID, hit.Source, strings.Join(hit.Labels, ","))
			fmt.Printf("       %s\n", preview(hit.TextPreview, 180))
		}
	}
	return nil
}

func makeCRMScenario(outDir string, accountCount int, embedderName string, dim, maxTokens int, force bool) error {
	accountsPath := crmPath(outDir, "accounts.jsonl")
	notesPath := crmPath(outDir, "notes.jsonl")
	emailsPath := crmPath(outDir, "booking_emails.jsonl")
	indexPath := crmPath(outDir, "note_index.jsonl")
	resultsPath := crmPath(outDir, "booking_results.jsonl")
	metaPath := crmPath(outDir, "note_index.meta.json")

	emb, err := newTextEmbedder(outDir, embedderName, dim, maxTokens)
	if err != nil {
		return err
	}
	meta := routerMeta{Embedder: emb.Name(), Dim: emb.Dim(), MaxTokens: maxTokens}
	if emb.Name() == "loom-jina-token" {
		meta.ModelID = modelID
	}

	if !force && routerMetaMatches(metaPath, meta) && filesExist(accountsPath, notesPath, emailsPath, indexPath, resultsPath) {
		fmt.Printf("  skip existing %s\n", filepath.Join(outDir, crmDirName))
		return nil
	}
	if err := os.MkdirAll(filepath.Join(outDir, crmDirName), 0o755); err != nil {
		return err
	}

	accounts := generateCRMAccounts(accountCount)
	notes := generateCRMNotes(accounts)
	emails := generateCRMEmails(accounts)

	if err := writeJSONL(accountsPath, accounts); err != nil {
		return err
	}
	if err := writeJSONL(notesPath, notes); err != nil {
		return err
	}
	if err := writeJSONL(emailsPath, emails); err != nil {
		return err
	}
	if err := writeCRMIndex(indexPath, notes, emb); err != nil {
		return err
	}
	if err := writeJSON(metaPath, meta); err != nil {
		return err
	}
	results, err := runCRMBookings(outDir, accounts, emails, emb)
	if err != nil {
		return err
	}
	if err := writeJSONL(resultsPath, results); err != nil {
		return err
	}
	fmt.Printf("  wrote %d accounts, %d notes, %d booking emails -> %s\n", len(accounts), len(notes), len(emails), filepath.Join(outDir, crmDirName))
	return nil
}

func crmPath(outDir, name string) string {
	return filepath.Join(outDir, crmDirName, name)
}

func filesExist(paths ...string) bool {
	for _, path := range paths {
		st, err := os.Stat(path)
		if err != nil || st.Size() == 0 {
			return false
		}
	}
	return true
}

func routerMetaMatches(path string, want routerMeta) bool {
	data, err := os.ReadFile(path)
	if err != nil {
		return false
	}
	var got routerMeta
	if err := json.Unmarshal(data, &got); err != nil {
		return false
	}
	return got == want
}

func generateCRMAccounts(n int) []crmAccount {
	prefixes := []string{"Northstar", "Riverbend", "BluePeak", "Cedar", "Brightlane", "Ironwood", "Harbor", "Atlas", "Silverline", "Redwood", "Foxglove", "Summit", "Evergreen", "Quartz", "Lighthouse"}
	suffixes := []string{"Robotics", "Health", "Analytics", "Foods", "Logistics", "Energy", "Studios", "Education", "Finance", "Manufacturing", "BioSystems", "Retail", "Aerospace", "Insurance", "Hospitality"}
	industries := []string{"healthcare", "manufacturing", "education", "finance", "logistics", "retail", "energy", "software", "hospitality", "aerospace"}
	regions := []string{"APAC", "ANZ", "North America", "EMEA", "Japan", "Singapore", "UKI", "US West", "US East", "Nordics"}
	owners := []string{"Ava Chen", "Noah Singh", "Mia Patel", "Oliver Jones", "Sophia Kim", "Lucas Brown", "Amelia Wong", "Ethan Smith"}
	firstNames := []string{"Maya", "Theo", "Priya", "Eli", "Nora", "Sam", "Lena", "Oscar", "Ivy", "Jonah", "Tara", "Felix"}
	lastNames := []string{"Hart", "Nguyen", "Foster", "Reed", "Okafor", "Tan", "Morgan", "Diaz", "Singh", "Bennett", "Kaur", "Brooks"}
	services := []string{"executive briefing", "implementation workshop", "onsite product demo", "renewal planning session", "data migration kickoff", "training room booking", "security review call", "partner enablement day"}
	windows := []string{"Tuesday morning", "Wednesday afternoon", "Friday after 2pm", "the first week of next month", "the last Thursday of the month", "Monday before lunch", "Thursday morning", "next available APAC-friendly slot"}
	locations := []string{"Sydney office", "Melbourne customer room", "San Francisco briefing center", "London boardroom", "Singapore training suite", "remote Zoom room", "Tokyo partner hub", "New York executive suite"}
	requirements := []string{"wheelchair access", "vegetarian catering", "NDA on arrival", "AV recording enabled", "visitor badges preprinted", "quiet room nearby", "security sign-in", "parking for two guests", "hybrid dial-in", "whiteboard wall"}
	entities := []string{"Group", "Collective", "Partners", "Works", "Network", "Labs"}

	out := make([]crmAccount, 0, n)
	for i := 0; i < n; i++ {
		name := prefixes[i%len(prefixes)] + " " + suffixes[(i*7)%len(suffixes)] + " " + entities[(i/len(prefixes))%len(entities)]
		contact := firstNames[(i*3)%len(firstNames)] + " " + lastNames[(i*5)%len(lastNames)]
		service := services[(i*2+3)%len(services)]
		reqA := requirements[(i*2)%len(requirements)]
		reqB := requirements[(i*2+5)%len(requirements)]
		out = append(out, crmAccount{
			AccountID:       fmt.Sprintf("ACC-%04d", i+1),
			AccountName:     name,
			Industry:        industries[(i*3+1)%len(industries)],
			Region:          regions[(i*5+2)%len(regions)],
			Owner:           owners[(i*7+3)%len(owners)],
			ContactName:     contact,
			ContactEmail:    strings.ToLower(strings.ReplaceAll(contact, " ", ".")) + "@" + slugify(name) + ".example",
			Service:         service,
			PreferredWindow: windows[(i*4+1)%len(windows)],
			Location:        locations[(i*6+2)%len(locations)],
			Attendees:       4 + (i*3)%37,
			BillingCode:     fmt.Sprintf("BK-%03d-%s", 100+i, strings.ToUpper(slugify(prefixes[i%len(prefixes)]))[:3]),
			Requirements:    []string{reqA, reqB},
		})
	}
	return out
}

func generateCRMNotes(accounts []crmAccount) []crmNoteRow {
	var notes []crmNoteRow
	for _, acc := range accounts {
		texts := []struct {
			typ  string
			text string
		}{
			{
				typ: "account_profile",
				text: fmt.Sprintf("Salesforce account %s (%s) is a %s customer in %s. Account owner is %s. Primary booking contact is %s at %s.",
					acc.AccountName, acc.AccountID, acc.Industry, acc.Region, acc.Owner, acc.ContactName, acc.ContactEmail),
			},
			{
				typ: "booking_preferences",
				text: fmt.Sprintf("%s usually books a %s at the %s. Preferred booking window is %s for about %d attendees. Use billing code %s.",
					acc.AccountName, acc.Service, acc.Location, acc.PreferredWindow, acc.Attendees, acc.BillingCode),
			},
			{
				typ: "logistics_notes",
				text: fmt.Sprintf("Operational notes for %s: remember %s and %s. Confirm the booking with %s and copy the owner %s.",
					acc.AccountName, acc.Requirements[0], acc.Requirements[1], acc.ContactName, acc.Owner),
			},
		}
		for i, item := range texts {
			row := crmNoteRow{
				RowID:       fmt.Sprintf("%s-note-%d", acc.AccountID, i+1),
				AccountID:   acc.AccountID,
				AccountName: acc.AccountName,
				NoteType:    item.typ,
				Text:        item.text,
				Labels:      extractLabels(item.text, 14),
				Account:     acc,
			}
			notes = append(notes, row)
		}
	}
	return notes
}

func generateCRMEmails(accounts []crmAccount) []crmBookingEmail {
	var emails []crmBookingEmail
	for i, acc := range accounts {
		body := fmt.Sprintf("Hi bookings team,\n\n%s from %s here. We need to arrange our %s soon, ideally %s. It should be for around %d people at the usual %s setup. Please pull our account notes so the billing code, contact details, and logistics are correct. Also remember %s and %s.\n\nThanks,\n%s",
			acc.ContactName, acc.AccountName, acc.Service, acc.PreferredWindow, acc.Attendees, acc.Location, acc.Requirements[0], acc.Requirements[1], firstName(acc.ContactName))
		emails = append(emails, crmBookingEmail{
			EmailID:           fmt.Sprintf("email-%04d", i+1),
			ExpectedAccountID: acc.AccountID,
			From:              acc.ContactEmail,
			Subject:           fmt.Sprintf("Booking request for %s", acc.AccountName),
			Body:              body,
		})
	}
	return emails
}

func writeCRMIndex(path string, notes []crmNoteRow, emb textEmbedder) error {
	out, err := os.Create(path)
	if err != nil {
		return err
	}
	defer out.Close()
	enc := json.NewEncoder(out)
	for _, note := range notes {
		row := crmIndexRow{
			RowID:       note.RowID,
			AccountID:   note.AccountID,
			AccountName: note.AccountName,
			NoteType:    note.NoteType,
			Labels:      note.Labels,
			TextPreview: preview(note.Text, 260),
			Vector:      emb.Embed(note.Text, note.Labels),
		}
		if err := enc.Encode(row); err != nil {
			return err
		}
	}
	return nil
}

func runCRMBookings(outDir string, accounts []crmAccount, emails []crmBookingEmail, emb textEmbedder) ([]crmBookingResult, error) {
	byID := map[string]crmAccount{}
	for _, acc := range accounts {
		byID[acc.AccountID] = acc
	}

	results := make([]crmBookingResult, 0, len(emails))
	for _, email := range emails {
		hits, err := searchCRM(outDir, email.Subject+"\n"+email.Body, defaultTopK, emb)
		if err != nil {
			return nil, err
		}
		hitRank := 0
		for i, hit := range hits {
			if hit.AccountID == email.ExpectedAccountID {
				hitRank = i + 1
				break
			}
		}
		var extracted *crmAccount
		if len(hits) > 0 {
			if acc, ok := byID[hits[0].AccountID]; ok {
				extracted = &acc
			}
		}
		expected := byID[email.ExpectedAccountID]
		results = append(results, crmBookingResult{
			EmailID:           email.EmailID,
			ExpectedAccountID: email.ExpectedAccountID,
			ExpectedAccount:   expected.AccountName,
			HitRank:           hitRank,
			ExtractedAccount:  extracted,
			Evidence:          hits,
		})
	}
	return results, nil
}

func searchCRM(outDir, query string, topK int, emb textEmbedder) ([]crmHit, error) {
	index, err := readCRMIndex(crmPath(outDir, "note_index.jsonl"))
	if err != nil {
		return nil, err
	}
	qv := emb.Embed(query, extractLabels(query, 12))
	hits := make([]crmHit, 0, len(index))
	for _, item := range index {
		hits = append(hits, crmHit{
			RowID:       item.RowID,
			AccountID:   item.AccountID,
			AccountName: item.AccountName,
			NoteType:    item.NoteType,
			Score:       dot(qv, item.Vector),
			Labels:      item.Labels,
			TextPreview: item.TextPreview,
		})
	}
	sort.Slice(hits, func(i, j int) bool {
		if hits[i].Score == hits[j].Score {
			return hits[i].RowID < hits[j].RowID
		}
		return hits[i].Score > hits[j].Score
	})
	if len(hits) > topK {
		hits = hits[:topK]
	}
	return hits, nil
}

func printCRMReport(outDir string, examples int) error {
	results, err := readCRMResults(crmPath(outDir, "booking_results.jsonl"))
	if err != nil {
		return err
	}
	emails, err := readCRMEmails(crmPath(outDir, "booking_emails.jsonl"))
	if err != nil {
		return err
	}
	emailByID := map[string]crmBookingEmail{}
	for _, email := range emails {
		emailByID[email.EmailID] = email
	}

	rank1 := 0
	topK := 0
	missed := 0
	for _, result := range results {
		switch {
		case result.HitRank == 1:
			rank1++
			topK++
		case result.HitRank > 1:
			topK++
		default:
			missed++
		}
	}
	fmt.Printf("  CRM booking emails: %d | account rank-1: %d | account top-k: %d | misses: %d\n", len(results), rank1, topK, missed)
	if examples > len(results) {
		examples = len(results)
	}
	for i := 0; i < examples; i++ {
		result := results[i]
		email := emailByID[result.EmailID]
		status := "MISS"
		if result.HitRank == 1 {
			status = "HIT rank 1"
		} else if result.HitRank > 1 {
			status = fmt.Sprintf("HIT rank %d", result.HitRank)
		}
		fmt.Printf("\n  [%s] %s expected=%s %s\n", result.EmailID, status, result.ExpectedAccountID, result.ExpectedAccount)
		fmt.Printf("    email: %s | from=%s\n", email.Subject, email.From)
		fmt.Printf("    ask: %s\n", preview(email.Body, 240))
		if result.ExtractedAccount != nil {
			acc := result.ExtractedAccount
			fmt.Printf("    extracted: account=%s contact=%s service=%s window=%s location=%s attendees=%d billing=%s requirements=%s\n",
				acc.AccountName, acc.ContactEmail, acc.Service, acc.PreferredWindow, acc.Location, acc.Attendees, acc.BillingCode, strings.Join(acc.Requirements, "; "))
		}
		for rank, hit := range result.Evidence {
			fmt.Printf("    #%d %.4f %s %s %s labels=%s\n", rank+1, hit.Score, hit.AccountID, hit.AccountName, hit.NoteType, strings.Join(hit.Labels, ","))
			fmt.Printf("       %s\n", preview(hit.TextPreview, 180))
		}
	}
	return nil
}

func searchWithEmbedder(outDir, query string, topK int, emb textEmbedder) ([]searchHit, error) {
	index, err := readIndex(filepath.Join(outDir, indexFileName))
	if err != nil {
		return nil, err
	}
	qv := emb.Embed(query, extractLabels(query, 10))
	hits := make([]searchHit, 0, len(index))
	for _, item := range index {
		hits = append(hits, searchHit{
			RowID:       item.RowID,
			Source:      item.Source,
			Score:       dot(qv, item.Vector),
			Labels:      item.Labels,
			TextPreview: item.TextPreview,
		})
	}
	sort.Slice(hits, func(i, j int) bool {
		if hits[i].Score == hits[j].Score {
			return hits[i].RowID < hits[j].RowID
		}
		return hits[i].Score > hits[j].Score
	})
	if len(hits) > topK {
		hits = hits[:topK]
	}
	return hits, nil
}

func readRows(path string) ([]textRow, error) {
	f, err := os.Open(path)
	if err != nil {
		return nil, err
	}
	defer f.Close()
	var rows []textRow
	sc := bufio.NewScanner(f)
	sc.Buffer(make([]byte, 1024), 4*1024*1024)
	for sc.Scan() {
		var row textRow
		if err := json.Unmarshal(sc.Bytes(), &row); err != nil {
			return nil, err
		}
		rows = append(rows, row)
	}
	if err := sc.Err(); err != nil {
		return nil, err
	}
	return rows, nil
}

func readIndex(path string) ([]indexRow, error) {
	f, err := os.Open(path)
	if err != nil {
		return nil, err
	}
	defer f.Close()
	var rows []indexRow
	sc := bufio.NewScanner(f)
	sc.Buffer(make([]byte, 1024), 16*1024*1024)
	for sc.Scan() {
		var row indexRow
		if err := json.Unmarshal(sc.Bytes(), &row); err != nil {
			return nil, err
		}
		rows = append(rows, row)
	}
	if err := sc.Err(); err != nil {
		return nil, err
	}
	return rows, nil
}

func readCRMIndex(path string) ([]crmIndexRow, error) {
	f, err := os.Open(path)
	if err != nil {
		return nil, err
	}
	defer f.Close()
	var rows []crmIndexRow
	sc := bufio.NewScanner(f)
	sc.Buffer(make([]byte, 1024), 16*1024*1024)
	for sc.Scan() {
		var row crmIndexRow
		if err := json.Unmarshal(sc.Bytes(), &row); err != nil {
			return nil, err
		}
		rows = append(rows, row)
	}
	if err := sc.Err(); err != nil {
		return nil, err
	}
	return rows, nil
}

func readCRMEmails(path string) ([]crmBookingEmail, error) {
	f, err := os.Open(path)
	if err != nil {
		return nil, err
	}
	defer f.Close()
	var rows []crmBookingEmail
	sc := bufio.NewScanner(f)
	sc.Buffer(make([]byte, 1024), 4*1024*1024)
	for sc.Scan() {
		var row crmBookingEmail
		if err := json.Unmarshal(sc.Bytes(), &row); err != nil {
			return nil, err
		}
		rows = append(rows, row)
	}
	if err := sc.Err(); err != nil {
		return nil, err
	}
	return rows, nil
}

func readCRMResults(path string) ([]crmBookingResult, error) {
	f, err := os.Open(path)
	if err != nil {
		return nil, err
	}
	defer f.Close()
	var rows []crmBookingResult
	sc := bufio.NewScanner(f)
	sc.Buffer(make([]byte, 1024), 8*1024*1024)
	for sc.Scan() {
		var row crmBookingResult
		if err := json.Unmarshal(sc.Bytes(), &row); err != nil {
			return nil, err
		}
		rows = append(rows, row)
	}
	if err := sc.Err(); err != nil {
		return nil, err
	}
	return rows, nil
}

func readSamples(path string) ([]sampleProbe, error) {
	f, err := os.Open(path)
	if err != nil {
		return nil, err
	}
	defer f.Close()
	var rows []sampleProbe
	sc := bufio.NewScanner(f)
	sc.Buffer(make([]byte, 1024), 8*1024*1024)
	for sc.Scan() {
		var row sampleProbe
		if err := json.Unmarshal(sc.Bytes(), &row); err != nil {
			return nil, err
		}
		rows = append(rows, row)
	}
	if err := sc.Err(); err != nil {
		return nil, err
	}
	return rows, nil
}

func writeJSONL[T any](path string, rows []T) error {
	if err := os.MkdirAll(filepath.Dir(path), 0o755); err != nil {
		return err
	}
	f, err := os.Create(path)
	if err != nil {
		return err
	}
	defer f.Close()
	enc := json.NewEncoder(f)
	for _, row := range rows {
		if err := enc.Encode(row); err != nil {
			return err
		}
	}
	return nil
}

func stripGutenbergBoilerplate(s string) string {
	startRE := regexp.MustCompile(`(?is)\*\*\*\s*START OF (?:THE|THIS) PROJECT GUTENBERG EBOOK.*?\*\*\*`)
	endRE := regexp.MustCompile(`(?is)\*\*\*\s*END OF (?:THE|THIS) PROJECT GUTENBERG EBOOK`)
	if loc := startRE.FindStringIndex(s); loc != nil {
		s = s[loc[1]:]
	}
	if loc := endRE.FindStringIndex(s); loc != nil {
		s = s[:loc[0]]
	}
	return s
}

func normalizeText(s string) string {
	s = strings.ReplaceAll(s, "\r\n", "\n")
	s = strings.ReplaceAll(s, "\r", "\n")
	var b strings.Builder
	lastSpace := false
	for _, r := range s {
		if unicode.IsSpace(r) {
			if !lastSpace {
				b.WriteByte(' ')
				lastSpace = true
			}
			continue
		}
		b.WriteRune(r)
		lastSpace = false
	}
	return strings.TrimSpace(b.String())
}

func chunkText(s string, chunkChars, overlapChars int) []string {
	runes := []rune(s)
	var chunks []string
	for start := 0; start < len(runes); {
		end := start + chunkChars
		if end >= len(runes) {
			chunk := strings.TrimSpace(string(runes[start:]))
			if chunk != "" {
				chunks = append(chunks, chunk)
			}
			break
		}
		bestEnd := end
		searchStart := end - 120
		if searchStart < start {
			searchStart = start
		}
		for i := end; i >= searchStart; i-- {
			if runes[i] == '.' || runes[i] == '!' || runes[i] == '?' {
				bestEnd = i + 1
				break
			}
			if unicode.IsSpace(runes[i]) && bestEnd == end {
				bestEnd = i
			}
		}
		chunk := strings.TrimSpace(string(runes[start:bestEnd]))
		if chunk != "" {
			chunks = append(chunks, chunk)
		}
		next := bestEnd - overlapChars
		if next <= start {
			next = start + chunkChars
		}
		start = next
	}
	return chunks
}

var stopWords = map[string]struct{}{
	"the": {}, "and": {}, "for": {}, "that": {}, "with": {}, "this": {}, "from": {}, "have": {},
	"not": {}, "but": {}, "you": {}, "his": {}, "her": {}, "she": {}, "him": {}, "was": {},
	"were": {}, "are": {}, "they": {}, "their": {}, "there": {}, "then": {}, "than": {},
	"what": {}, "when": {}, "where": {}, "who": {}, "which": {}, "would": {}, "could": {},
	"should": {}, "into": {}, "about": {}, "upon": {}, "your": {}, "our": {}, "out": {},
	"one": {}, "all": {}, "any": {}, "can": {}, "had": {}, "has": {}, "been": {}, "said": {},
	"will": {}, "shall": {}, "its": {}, "may": {}, "more": {}, "some": {}, "such": {}, "very": {},
}

func extractLabels(s string, n int) []string {
	counts := map[string]int{}
	for _, w := range words(s) {
		if len(w) < 3 {
			continue
		}
		if _, stop := stopWords[w]; stop {
			continue
		}
		counts[w]++
	}
	type kv struct {
		Key string
		Val int
	}
	items := make([]kv, 0, len(counts))
	for k, v := range counts {
		items = append(items, kv{k, v})
	}
	sort.Slice(items, func(i, j int) bool {
		if items[i].Val == items[j].Val {
			return items[i].Key < items[j].Key
		}
		return items[i].Val > items[j].Val
	})
	if len(items) > n {
		items = items[:n]
	}
	labels := make([]string, len(items))
	for i, item := range items {
		labels[i] = item.Key
	}
	return labels
}

func words(s string) []string {
	var out []string
	var b strings.Builder
	flush := func() {
		if b.Len() == 0 {
			return
		}
		out = append(out, b.String())
		b.Reset()
	}
	for _, r := range strings.ToLower(s) {
		if unicode.IsLetter(r) || unicode.IsDigit(r) {
			b.WriteRune(r)
		} else {
			flush()
		}
	}
	flush()
	return out
}

func loadWordPieceTokenizer(path string) (*wordPieceTokenizer, error) {
	data, err := os.ReadFile(path)
	if err != nil {
		return nil, err
	}
	var raw struct {
		Normalizer struct {
			Lowercase bool `json:"lowercase"`
		} `json:"normalizer"`
		Model struct {
			Vocab map[string]int `json:"vocab"`
		} `json:"model"`
	}
	if err := json.Unmarshal(data, &raw); err != nil {
		return nil, err
	}
	if len(raw.Model.Vocab) == 0 {
		return nil, errors.New("tokenizer has no WordPiece vocab")
	}
	t := &wordPieceTokenizer{
		vocab:     raw.Model.Vocab,
		unkID:     raw.Model.Vocab["[UNK]"],
		clsID:     raw.Model.Vocab["[CLS]"],
		sepID:     raw.Model.Vocab["[SEP]"],
		lowercase: raw.Normalizer.Lowercase,
	}
	return t, nil
}

func (t *wordPieceTokenizer) Encode(text string, maxTokens int) []uint32 {
	if t.lowercase {
		text = strings.ToLower(text)
	}
	pieces := t.basicTokens(text)
	ids := make([]uint32, 0, len(pieces)+2)
	ids = append(ids, uint32(t.clsID))
	for _, piece := range pieces {
		ids = append(ids, t.wordPieceIDs(piece)...)
		if maxTokens > 0 && len(ids) >= maxTokens-1 {
			break
		}
	}
	ids = append(ids, uint32(t.sepID))
	if maxTokens > 0 && len(ids) > maxTokens {
		ids = append(ids[:maxTokens-1], uint32(t.sepID))
	}
	return ids
}

func (t *wordPieceTokenizer) basicTokens(text string) []string {
	var out []string
	var b strings.Builder
	flush := func() {
		if b.Len() == 0 {
			return
		}
		out = append(out, b.String())
		b.Reset()
	}
	for _, r := range text {
		switch {
		case unicode.IsLetter(r) || unicode.IsDigit(r):
			b.WriteRune(r)
		case unicode.IsSpace(r):
			flush()
		default:
			flush()
			out = append(out, string(r))
		}
	}
	flush()
	return out
}

func (t *wordPieceTokenizer) wordPieceIDs(token string) []uint32 {
	if id, ok := t.vocab[token]; ok {
		return []uint32{uint32(id)}
	}
	runes := []rune(token)
	if len(runes) > 100 {
		return []uint32{uint32(t.unkID)}
	}
	var ids []uint32
	for start := 0; start < len(runes); {
		end := len(runes)
		found := -1
		var foundID int
		for start < end {
			sub := string(runes[start:end])
			if start > 0 {
				sub = "##" + sub
			}
			if id, ok := t.vocab[sub]; ok {
				found = end
				foundID = id
				break
			}
			end--
		}
		if found < 0 {
			return []uint32{uint32(t.unkID)}
		}
		ids = append(ids, uint32(foundID))
		start = found
	}
	return ids
}

func vectorize(text string, labels []string, dim int) []float32 {
	v := make([]float64, dim)
	ws := words(text)
	for _, w := range ws {
		addHashed(v, w, 1.0)
	}
	for i := 0; i+1 < len(ws); i++ {
		addHashed(v, ws[i]+"_"+ws[i+1], 0.65)
	}
	for _, label := range labels {
		addHashed(v, label, 3.0)
	}

	var norm float64
	for _, x := range v {
		norm += x * x
	}
	out := make([]float32, dim)
	if norm == 0 {
		return out
	}
	scale := 1.0 / math.Sqrt(norm)
	for i, x := range v {
		out[i] = float32(x * scale)
	}
	return out
}

func addHashed(v []float64, token string, weight float64) {
	h := fnv.New64a()
	_, _ = h.Write([]byte(token))
	sum := h.Sum64()
	idx := int(sum % uint64(len(v)))
	sign := 1.0
	if (sum>>63)&1 == 1 {
		sign = -1.0
	}
	v[idx] += sign * weight
}

func dot(a, b []float32) float64 {
	n := len(a)
	if len(b) < n {
		n = len(b)
	}
	var sum float64
	for i := 0; i < n; i++ {
		sum += float64(a[i]) * float64(b[i])
	}
	return sum
}

func normalizeFloat32(v []float32) {
	var norm float64
	for _, x := range v {
		norm += float64(x) * float64(x)
	}
	if norm == 0 {
		return
	}
	scale := float32(1.0 / math.Sqrt(norm))
	for i := range v {
		v[i] *= scale
	}
}

func sampleRows(total, count int) []int {
	if count > total {
		count = total
	}
	if count <= 0 {
		return nil
	}
	rng := rand.New(rand.NewSource(42))
	seen := map[int]struct{}{}
	picks := make([]int, 0, count)
	stride := total / count
	if stride < 1 {
		stride = 1
	}
	for i := 0; len(picks) < count && i < total*2; i++ {
		idx := (i * stride) % total
		if _, ok := seen[idx]; ok {
			idx = rng.Intn(total)
		}
		if _, ok := seen[idx]; ok {
			continue
		}
		seen[idx] = struct{}{}
		picks = append(picks, idx)
	}
	sort.Ints(picks)
	return picks
}

func probeQuery(row textRow) string {
	var parts []string
	if len(row.Labels) > 0 {
		limit := len(row.Labels)
		if limit > 6 {
			limit = 6
		}
		parts = append(parts, row.Labels[:limit]...)
	}
	parts = append(parts, firstWords(row.Text, 14))
	return strings.Join(parts, " ")
}

func firstWords(s string, n int) string {
	ws := words(s)
	if len(ws) > n {
		ws = ws[:n]
	}
	return strings.Join(ws, " ")
}

func firstName(name string) string {
	parts := strings.Fields(name)
	if len(parts) == 0 {
		return name
	}
	return parts[0]
}

func slugify(s string) string {
	var b strings.Builder
	lastDash := false
	for _, r := range strings.ToLower(s) {
		switch {
		case unicode.IsLetter(r) || unicode.IsDigit(r):
			b.WriteRune(r)
			lastDash = false
		case !lastDash && b.Len() > 0:
			b.WriteByte('-')
			lastDash = true
		}
	}
	return strings.Trim(b.String(), "-")
}

func preview(s string, n int) string {
	r := []rune(strings.TrimSpace(s))
	if len(r) <= n {
		return string(r)
	}
	return string(r[:n]) + "..."
}

func writeJSON(path string, v any) error {
	if err := os.MkdirAll(filepath.Dir(path), 0o755); err != nil {
		return err
	}
	f, err := os.Create(path)
	if err != nil {
		return err
	}
	defer f.Close()
	enc := json.NewEncoder(f)
	enc.SetIndent("", "  ")
	return enc.Encode(v)
}
