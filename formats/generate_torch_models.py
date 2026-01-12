
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import os
import json
from safetensors.torch import save_file

# Create output directories
os.makedirs("tva/formats/output/safetensors", exist_ok=True)
os.makedirs("tva/formats/output/onnx", exist_ok=True)
os.makedirs("tva/formats/output/data", exist_ok=True)

class SwiGLU(nn.Module):
    def __init__(self, d_in, d_out):
        super().__init__()
        self.w_gate = nn.Linear(d_in, d_out, bias=True)
        self.w_up = nn.Linear(d_in, d_out, bias=True)
        self.w_down = nn.Linear(d_out, d_in, bias=True)

    def forward(self, x):
        # gate = SiLU(gate_proj(x)) * up_proj(x)
        # out = down_proj(gate)
        gate = F.silu(self.w_gate(x)) * self.w_up(x)
        return self.w_down(gate)

class RMSNorm(nn.Module):
    def __init__(self, dim, eps=1e-6):
        super().__init__()
        self.scale = nn.Parameter(torch.ones(dim))
        self.eps = eps

    def forward(self, x):
        norm = x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)
        return norm * self.scale

class SequenceModel(nn.Module):
    def __init__(self, vocab_size, d_model, n_head):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, d_model)
        self.rnn = nn.RNN(d_model, d_model, batch_first=True)
        self.lstm = nn.LSTM(d_model, d_model, batch_first=True)
        self.attn = nn.MultiheadAttention(d_model, n_head, batch_first=True)
        self.ln = nn.LayerNorm(d_model)
        self.rms = RMSNorm(d_model)
        self.swiglu = SwiGLU(d_model, d_model)
        self.out = nn.Linear(d_model, vocab_size)

    def forward(self, x):
        # x: [batch, seq_len]
        e = self.embedding(x)
        
        # RNN
        r_out, _ = self.rnn(e)
        
        # LSTM
        l_out, _ = self.lstm(r_out)
        
        # Attention (Self-Attention)
        attn_out, _ = self.attn(l_out, l_out, l_out)
        
        # LayerNorm
        norm_out = self.ln(attn_out + l_out) # Residual
        
        # RMSNorm
        rms_out = self.rms(norm_out)
        
        # SwiGLU
        swi_out = self.swiglu(rms_out)
        
        # Output
        logits = self.out(swi_out)
        return F.softmax(logits, dim=-1)

class VisionModel(nn.Module):
    def __init__(self, in_channels, num_classes):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, 16, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, stride=2, padding=1)
        self.fc = nn.Linear(32 * 16 * 16, num_classes) # Assuming 32x32 input

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = x.view(x.size(0), -1) # Flatten
        return self.fc(x)

class AudioModel(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv1 = nn.Conv1d(in_channels, 16, kernel_size=3, padding=1)
        self.conv2 = nn.Conv1d(16, out_channels, kernel_size=3, padding=1)
    
    def forward(self, x):
        x = F.relu(self.conv1(x))
        return self.conv2(x)

def export_model(model, inputs, name):
    print(f"Exporting {name}...")
    model.eval()
    
    # Run forward pass
    with torch.no_grad():
        output = model(*inputs)
    
    # Save Data
    np.save(f"tva/formats/output/data/{name}_input.npy", inputs[0].numpy())
    np.save(f"tva/formats/output/data/{name}_output.npy", output.numpy())

    # Save Safetensors
    state_dict = model.state_dict()
    save_file(state_dict, f"tva/formats/output/safetensors/{name}.safetensors")
    print(f"  Saved safetensors to tva/formats/output/safetensors/{name}.safetensors")

    # Export ONNX
    try:
        torch.onnx.export(
            model, 
            inputs, 
            f"tva/formats/output/onnx/{name}.onnx",
            input_names=["input"],
            output_names=["output"],
            dynamic_axes={"input": {0: "batch_size"}, "output": {0: "batch_size"}}
        )
        print(f"  Saved ONNX to tva/formats/output/onnx/{name}.onnx")
    except Exception as e:
        print(f"  ONNX Export failed: {e}")

if __name__ == "__main__":
    # 1. Sequence Model
    print("--- Sequence Model ---")
    seq_model = SequenceModel(vocab_size=100, d_model=32, n_head=4)
    # Input: [Batch=10, Seq=20] ints
    seq_input = torch.randint(0, 100, (10, 20))
    export_model(seq_model, (seq_input,), "sequence_model")

    # 2. Vision Model
    print("\n--- Vision Model ---")
    vis_model = VisionModel(in_channels=3, num_classes=10)
    # Input: [Batch=10, C=3, H=32, W=32] float
    vis_input = torch.randn(10, 3, 32, 32)
    export_model(vis_model, (vis_input,), "vision_model")

    # 3. Audio Model
    print("\n--- Audio Model ---")
    aud_model = AudioModel(in_channels=2, out_channels=4)
    # Input: [Batch=10, C=2, L=100]
    aud_input = torch.randn(10, 2, 100)
    export_model(aud_model, (aud_input,), "audio_model")
