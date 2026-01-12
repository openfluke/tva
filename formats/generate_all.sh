#!/bin/bash
set -e

# Ensure we are in the correct directory
cd "$(dirname "$0")"

echo "========================================"
echo "    Generating PyTorch & ONNX Models    "
echo "========================================"

if [ ! -d "venv_torch" ]; then
    echo "Creating PyTorch environment..."
    python3 -m venv venv_torch
    echo "Installing PyTorch dependencies..."
    ./venv_torch/bin/pip install -r requirements_torch.txt
fi

echo "Running PyTorch generator..."
./venv_torch/bin/python generate_torch_models.py


echo ""
echo "========================================"
echo " Generating TensorFlow/TFLite/TFJS Models "
echo "========================================"

if [ ! -d "venv_tf" ]; then
    echo "Creating TensorFlow environment..."
    python3 -m venv venv_tf
    echo "Installing TensorFlow dependencies..."
    ./venv_tf/bin/pip install -r requirements_tf.txt
fi

echo "Running TensorFlow generator..."
./venv_tf/bin/python generate_tf_models.py

echo ""
echo "========================================"
echo "           All Models Generated         "
echo "========================================"
