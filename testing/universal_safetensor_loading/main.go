package main

import (
	"encoding/binary"
	"fmt"
	"io"
	"net/http"
	"os"
	"path/filepath"
)

type ModelInfo struct {
	ID       string
	Category string
}

func main() {
	fmt.Println("Hello World")
	//downloadModels()
}

func downloadModels() {
	models := []ModelInfo{
		// CNN Models
		{"microsoft/resnet-50", "cnn"},
		{"facebook/convnext-tiny-224", "cnn"},
		{"microsoft/resnet-18", "cnn"},
		{"microsoft/resnet-34", "cnn"},
		{"timm/mobilenetv2_100.ra_in1k", "cnn"},

		// MHA Models (Transformers)
		{"bert-base-uncased", "mha"},
		{"gpt2", "mha"},
		{"roberta-base", "mha"},
		{"distilbert-base-uncased", "mha"},
		{"microsoft/phi-2", "mha"},

		// LSTM/RNN Models
		{"miittnnss/lstm-textgen-pets", "lstm"},
		{"DanielClough/rwkv7-g1-safetensors", "lstm"},
		{"SakanaAI/ctm-maze-large", "lstm"},
		{"youssef-ismail/lstm-ner", "lstm"},
		{"Bingsu/Ademamix-L-Tiny-RNN", "lstm"},

		// Dense Models
		{"facebook/opt-125m", "dense"},
		{"google/flan-t5-small", "dense"},
		{"prajjwal1/bert-tiny", "dense"},
		{"microsoft/phi-1_5", "dense"},
		{"ybelkada/tiny-random-llama", "dense"},

		// Conv1D Models
		{"openai-community/gpt2", "conv1d"},
		{"hf-internal-testing/tiny-random-gpt2", "conv1d"},
		{"google/byt5-small", "conv1d"},
		{"facebook/wav2vec2-base-960h", "conv1d"},
		{"sanchit-gandhi/tiny-wav2vec2", "conv1d"},

		// Embedding Models
		{"Qwen/Qwen-1_8B", "embedding"},
		{"sentence-transformers/all-MiniLM-L6-v2", "embedding"},
		{"BAAI/bge-small-en-v1.5", "embedding"},
		{"intfloat/e5-small-v2", "embedding"},
		{"jhgan/ko-sroberta-multitask", "embedding"},

		// RMSNorm/SwiGLU Models
		{"TinyLlama/TinyLlama-1.1B-Chat-v1.0", "rmsnorm_swiglu"},
		{"PY007/TinyLlama-1.1B-intermediate-step-1431k-3T", "rmsnorm_swiglu"},
		{"Felladrin/Llama-160M-Chat-v1", "rmsnorm_swiglu"},
		{"JackFram/llama-160m", "rmsnorm_swiglu"},
		{"kevinrothe/tiny-random-LlamaForCausalLM", "rmsnorm_swiglu"},

		// LayerNorm Models
		{"cross-encoder/ms-marco-MiniLM-L-6-v2", "layernorm"},
		{"sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2", "layernorm"},
		{"microsoft/deberta-v3-small", "layernorm"},
		{"albert-base-v2", "layernorm"},
		{"google/electra-small-discriminator", "layernorm"},
	}

	baseDir := "downloads"
	if err := os.MkdirAll(baseDir, 0755); err != nil {
		fmt.Printf("Error creating base directory: %v\n", err)
		return
	}

	for _, m := range models {
		fmt.Printf("--- Processing %s (%s) ---\n", m.ID, m.Category)
		destDir := filepath.Join(baseDir, m.Category, filepath.Base(m.ID))
		if err := os.MkdirAll(destDir, 0755); err != nil {
			fmt.Printf("Error creating directory for %s: %v\n", m.ID, err)
			continue
		}

		destPath := filepath.Join(destDir, "model.safetensors")
		if _, err := os.Stat(destPath); err == nil {
			fmt.Printf("File already exists: %s. Skipping.\n", destPath)
			verifySafetensors(destPath)
			continue
		}

		url := fmt.Sprintf("https://huggingface.co/%s/resolve/main/model.safetensors", m.ID)
		fmt.Printf("Downloading %s to %s...\n", url, destPath)

		if err := downloadFile(url, destPath); err != nil {
			fmt.Printf("Error downloading %s: %v\n", m.ID, err)
			// Try "pytorch_model.safetensors" as some models use that instead
			fmt.Println("Trying alternative filename: pytorch_model.safetensors...")
			altUrl := fmt.Sprintf("https://huggingface.co/%s/resolve/main/pytorch_model.safetensors", m.ID)
			if err := downloadFile(altUrl, destPath); err != nil {
				fmt.Printf("Error downloading alternative for %s: %v\n", m.ID, err)
				continue
			}
		}

		fmt.Printf("Successfully downloaded %s\n", m.ID)
		verifySafetensors(destPath)
	}

	fmt.Println("\n--- All Done! ---")
}

func downloadFile(url, destPath string) error {
	resp, err := http.Get(url)
	if err != nil {
		return err
	}
	defer resp.Body.Close()

	if resp.StatusCode != http.StatusOK {
		return fmt.Errorf("bad status: %s", resp.Status)
	}

	out, err := os.Create(destPath)
	if err != nil {
		return err
	}
	defer out.Close()

	_, err = io.Copy(out, resp.Body)
	return err
}

func verifySafetensors(path string) {
	file, err := os.Open(path)
	if err != nil {
		fmt.Printf("Verification failed (open): %v\n", err)
		return
	}
	defer file.Close()

	var headerSize uint64
	if err := binary.Read(file, binary.LittleEndian, &headerSize); err != nil {
		fmt.Printf("Verification failed (read header size): %v\n", err)
		return
	}

	fmt.Printf("Verified %s: Header Size = %d bytes\n", filepath.Base(filepath.Dir(path)), headerSize)
}
