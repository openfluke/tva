# TVA Formats Examples

This directory contains scripts to generate sample neural network models in various formats (`safetensors`, `onnx`, `tflite`, `tfjs`) covering all standard Loom layers.

## Setup

It is recommended to use a virtual environment:

```bash
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

## Running the Examples

### 1. PyTorch & Safetensors & ONNX
Generates models using PyTorch and exports them to `.safetensors` and `.onnx`.

```bash
python generate_torch_models.py
```
Outputs:
- `output/safetensors/*.safetensors`
- `output/onnx/*.onnx`
- `output/data/*_input.npy`, `*_output.npy`

### 2. TensorFlow & TFLite & TFJS
Generates equivalent models using TensorFlow/Keras and exports them to `.tflite` and TensorFlow.js format.

```bash
python generate_tf_models.py
```
Outputs:
- `output/tflite/*.tflite`
- `output/tfjs/*/`

## Models Included

- **SequenceModel**: Demonstrates Embedding, RNN, LSTM, MultiHeadAttention, LayerNorm, SwiGLU, etc.
- **VisionModel**: Demonstrates Conv2D, Dense, Dropout.
- **AudioModel**: Demonstrates Conv1D.
