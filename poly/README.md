# Poly Talk ⚛️

`poly_talk.go` is a ridiculously fast, zero-dependency, cross-platform inference engine for Small Language Models (SLMs). It proves a radical idea: you can write a raw Transformer inference engine natively in Go, compile it cross-platform without CGO or C++ toolchains, and achieve near-C-level token generation speeds using WebGPU.

## Features

- **FlashPoly CPU Tiling**: An architecture-aware GQA matrix tiling engine that optimizes cache blocking for L1 caches on x86 and ARM64 processors.
- **The "Numerical Monster"**: A 100% VRAM-resident WebGPU architecture. Once the model is loaded, the entire Transformer forward pass (Norms, Dense, SwiGLU, MHA, RoPE, and KV caching) executes natively inside WGSL compute shaders with zero CPU-GPU data transfers during generation.
- **Universal Hardware Detection**: Automatically detects OS, CPU model, total RAM, and active WebGPU Adapter (Metal, Vulkan, DX12, etc.) dynamically.
- **Polymorphic**: Easily ingest Llama, Mistral, Qwen, Pythia, and SmolLM formats natively.

---

## 🚀 True Cross-Platform Benchmarks

The core achievement of Poly Talk is exactly the same compiled Go binary running natively across competing operating systems and GPUs natively. Below are real-world logs demonstrating WebGPU saturating Apple Silicon, Windows/Intel, and Linux.

### 🍎 macOS M4 (Metal)

```text
⚛️  Poly Talk - Available models:
  [6] HuggingFaceTB/SmolLM2-135M-Instruct
🎯 Deterministic mode? (1=yes / 0=no) [1]: 1
🚀 Enable FlashPoly Tiling? (1=yes / 0=no) [1]: 1
🎮 Enable GPU Acceleration? (1=yes / 0=no) [0]: 1

... Successfully loaded ...

🖥️  OS: darwin | CPU: Apple M4 | RAM: 16.00 GB | GPU: Apple M4 (metal)
✅ Model loaded on Poly! (30 layers)

You: hello how are you?
Poly: I'm doing great! I just got back from a long day of coding in the lab...
(prefill: 229.11 tok/s, 65 prompt tokens | decode: 20.64 tok/s, 50 generated | total: 42.49 tok/s)
```

### 🐧 Linux (Vulkan)

```text
⚛️  Poly Talk - Available models:
  [2] HuggingFaceTB/SmolLM2-135M-Instruct
🎯 Deterministic mode? (1=yes / 0=no) [1]: 1
🚀 Enable FlashPoly Tiling? (1=yes / 0=no) [1]: 1
🎮 Enable GPU Acceleration? (1=yes / 0=no) [0]: 1

... Successfully loaded ...

🖥️  OS: linux | CPU: 12th Gen Intel(R) Core(TM) i5-12500H | RAM: 46.74 GB | GPU: Intel GPU (vulkan)
✅ Model loaded on Poly! (30 layers)

You: hello how are you?
Poly: I'm doing great! I just got back from a long day of coding in the lab...
(prefill: 143.85 tok/s, 65 prompt tokens | decode: 19.07 tok/s, 50 generated | total: 37.42 tok/s)
```

### 🪟 Windows (Vulkan/DX)

```text
⚛️  Poly Talk - Available models:
  [3] HuggingFaceTB/SmolLM2-135M-Instruct
🎯 Deterministic mode? (1=yes / 0=no) [1]: 1
🚀 Enable FlashPoly Tiling? (1=yes / 0=no) [1]: 1
🎮 Enable GPU Acceleration? (1=yes / 0=no) [0]: 1

... Successfully loaded ...

🖥️  OS: windows | CPU: Intel(R) Core(TM) i5-10400 CPU @ 2.90GHz | RAM: 31.79 GB | GPU: Intel GPU (vulkan)
✅ Model loaded on Poly! (30 layers)

You: hello how are you?
Poly: I'm doing great! I just got back from a long day of coding in the lab...
(prefill: 68.49 tok/s, 65 prompt tokens | decode: 10.86 tok/s, 50 generated | total: 20.71 tok/s)
```

---

## Technical Details

### WGSL Shared Memory
`poly` queries the WebGPU `MaxComputeWorkgroupStorageSize` on initialization, detecting boundaries (e.g., 32KB on Intel, 64KB on Apple Silicon) and using Go templating (`strings.ReplaceAll(...)`) to inject optimized matrix tile boundaries into the WGSL shader payload *just prior to compilation*. This avoids hardcoded minimums (like 16KB) and saturates the GPU.

### Zero-Copy Pipeline
Through the `ActivationPool`, intermediate projections (e.g. `gate_proj(x)` -> `up_proj(x)`) are written and read strictly in VRAM memory blocks. The Go context issues thousands of kernel dispatches using a cached `PipelineCache`, only moving bytes back to CPU RAM during the final target projection to the vocabulary.
