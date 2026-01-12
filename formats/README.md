# TVA Formats Examples

This directory contains scripts to generate sample neural network models in various formats (`safetensors`, `onnx`, `tflite`, `tfjs`) covering all standard Loom layers.

## One-Shot Generation

Because Python ML libraries conflict with each other (PyTorch vs TensorFlow dependencies), we use **isolated environments**. Only one script is needed:

```bash
# This creates two separate venvs and runs everything
./generate_all.sh
```

This will:
1.  Create `venv_torch` -> Install PyTorch -> Generate `.safetensors` & `.onnx`
2.  Create `venv_tf` -> Install TensorFlow -> Generate `.tflite` & `.tfjs`

## Outputs

Check `output/` for the results:
- `output/safetensors/*.safetensors`
- `output/onnx/*.onnx`
- `output/tflite/*.tflite`
- `output/tfjs/*/`
- `output/data/*_input.npy` (Test inputs)

## Manual Setup (Optional)

If you want to run them manually:

**PyTorch / ONNX:**
```bash
python3 -m venv venv_torch
source venv_torch/bin/activate
pip install -r requirements_torch.txt
python generate_torch_models.py
```

**TensorFlow / TFLite / TFJS:**
```bash
python3 -m venv venv_tf
source venv_tf/bin/activate
pip install -r requirements_tf.txt
python generate_tf_models.py
```
