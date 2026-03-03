#!/bin/bash

# 1. Check if hf-cli is installed
if ! command -v hf &> /dev/null; then
    echo "Installer: hf-cli not found. Installing now..."
    curl -LsSf https://hf.co/cli/install.sh | bash
    # Source the path so we can use it immediately
    export PATH="$PATH:$HOME/.local/bin"
else
    echo "✓ Hugging Face CLI is already installed."
fi

# 2. List of models to ensure we have (Base + Instruct/Chat)
MODELS=(
    "HuggingFaceTB/SmolLM2-135M"
    "HuggingFaceTB/SmolLM2-135M-Instruct"
    "HuggingFaceTB/SmolLM2-360M"
    "HuggingFaceTB/SmolLM2-360M-Instruct"
    "Qwen/Qwen2.5-0.5B"
    "Qwen/Qwen2.5-0.5B-Instruct"
    "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
    "TinyLlama/TinyLlama-1.1B-intermediate-step-1431k-3T"
)

echo "Checking models... this may take a second."

# 3. Download loop
for model in "${MODELS[@]}"; do
    echo "---------------------------------------------------"
    echo "Checking: $model"
    # hf download is smart: it won't re-download if the files exist and are correct
    hf download "$model"
done

echo "---------------------------------------------------"
echo "✅ All models processed! The Beast is ready."
hf scan-cache