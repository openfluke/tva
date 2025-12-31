# Filter CombineMode Demo (Mixture of Experts)

This directory contains a demonstration of the `filter` CombineMode in the Loom library, which implements a dynamic, learned gating mechanism—similar to a **Mixture of Experts (MoE)** architecture.

## Overview

Traditional parallel layers in neural networks often combine their outputs by summing or concatenating. The `filter` mode introduces a **Gate Layer** that observes the input and predicts a weight (routing probability) for each expert branch. 

These weights determine how much each expert's output contributes to the final result of the parallel layer.

### Key Configurations

The `filter` mode is configured via `nn.LayerConfig`:

- **`FilterGateConfig`**: A pointer to another `LayerConfig` (typically a Small Dense layer) that acts as the router. Its output size must match the number of branches.
- **`FilterSoftmax`**: Determines how the gate's raw outputs are normalized into routing weights.
  - `SoftmaxStandard`: Smooth routing across all experts.
  - `SoftmaxEntmax`: Sparse routing (weights can be exactly zero), enabling true MoE-style "selection".
- **`FilterTemperature`**: Controls the "sharpness" of the routing. Lower values make the gate more decisive.

---

## Demos

### 1. Two Dense Expert Branches
A baseline demonstration of routing input through two different dense experts. It shows the flow from input, through the gated filter, to a final output layer.

### 2. Multi-Branch Sparse Routing (4 Experts)
Showcases scalability and the use of `nn.SoftmaxEntmax`. By using a sparse softmax, the model can learn to ignore certain experts entirely for specific inputs, reducing noise and focusing on the most relevant sub-networks.

### 3. Training Gate Specialization
The most advanced demo. It highlights how the gate itself can be trained using Loom's `TweenState`.
1. **Expert Pre-training**: We manually "teach" two experts to respond to different conditions (high vs. low input values).
2. **Gate Training**: We use the standard Loom training loop to update *only* the gate layer.
3. **Result**: The gate learns to recognize which input condition corresponds to which expert, significantly improving overall network performance.

---

## How to Run

Navigate to this directory and run:

```bash
go run main.go
```

### Sample Output

```text
╔══════════════════════════════════════════════════════════════════╗
║        🔬 Filter CombineMode Demo (Mixture of Experts)           ║
╚══════════════════════════════════════════════════════════════════╝

📌 Demo 1: Two Dense Expert Branches with Learned Gating
   ✅ Forward pass successful!
   📊 Input size: 16, Output size: 16
   📈 Sample output values: [0.594, 0.429, 0.150, ...]

📌 Demo 2: Multi-Branch Filter (4 experts)
   🧪 Testing with 5 different inputs:
      Trial 1: ✅ avg=0.4980
      Trial 2: ✅ avg=0.5181
      Trial 3: ✅ avg=0.5019
      Trial 4: ✅ avg=0.5000
      Trial 5: ✅ avg=0.5098

📌 Demo 3: Training Gate Specialization
   🎓 Pre-training Expert 1 (responds to HIGH first element)...
   🎓 Pre-training Expert 2 (responds to LOW first element)...
   📊 Testing BEFORE gate training:
      High input[0]=0.9 → output=0.6095
      Low input[0]=0.1  → output=0.5667
   🏋️ Training GATE layer for 2000 steps...
   📊 Testing AFTER gate training:
      High input[0]=0.9 → output=0.9371 (was 0.6095)
      Low input[0]=0.1  → output=0.9209 (was 0.5667)
   ✅ Gate learned to differentiate! (changes: high=0.3276, low=0.3542)
```
