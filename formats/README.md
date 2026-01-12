# TVA Formats Examples

This directory contains scripts to generate sample neural network models in various formats (`safetensors`, `onnx`, `tflite`, `tfjs`) covering standard Loom layers.

## One-Shot Generation

Because Python ML libraries conflict with each other (PyTorch vs TensorFlow dependencies), we use **isolated environments**. Only one script is needed:

```bash
# This creates two separate venvs and runs everything
./generate_all.sh
```

This will:
1.  Create `venv_torch` -> Install PyTorch -> Generate `.safetensors` & `.onnx` (for complex composite models).
2.  Create `venv_tf` -> Install TensorFlow -> Generate `.tflite`, `.safetensors`, `.onnx` (via tf2onnx), & `.saved_model` (for simple "low hanging fruit" models).
3.  Run `npm install` -> Convert SavedModels to `.tfjs` using `tensorflowjs_converter`.

## Outputs

Check `output/` for the results:

### Complex Models (PyTorch source)
Located in `output/{safetensors,onnx,data}`:
- `sequence_model` (RNN/LSTM/Attn)
- `vision_model` (Conv2D)
- `audio_model` (Conv1D)

### Simple/Low-Hanging Fruit Models (TensorFlow source)
Located in `output/simple_models/{safetensors,onnx,tflite,saved_model,tfjs}`:
- `dense`
- `conv2d`
- `rnn`
- `lstm`
- `embedding`

### Compatibility Notes

| Format | Status | Notes |
| :--- | :--- | :--- |
| **Safetensors** | ✅ Working | Generated native via `safetensors` lib |
| **ONNX** | ✅ Working | Generated via `torch.onnx` and `tf2onnx` |
| **TFLite** | ✅ Working | Generated via `tf.lite.TFLiteConverter` |
| **TFJS** | ⚠️ Partial | Works for `dense` and `conv2d`. Fails for `rnn`, `lstm`, `embedding` due to `tfjs-node` / TF 2.x version mismatch on Python 3.13. |

## Manual Setup (Optional)

If you want to run them manually:

**PyTorch / ONNX:**
```bash
python3 -m venv venv_torch
source venv_torch/bin/activate
pip install -r requirements_torch.txt
python generate_torch_models.py
```

**TensorFlow / Simple Models:**
```bash
python3 -m venv venv_tf
source venv_tf/bin/activate
pip install -r requirements_tf.txt
python generate_tf_models.py      # Complex TF models
python generate_simple_models.py  # Simple single-layer models
```

**TFJS Conversion:**
```bash
npm install
node convert_to_tfjs.js
```
