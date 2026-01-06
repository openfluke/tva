# RCTN: Recursive Compositional Target Networks

**Recursive Compositional Target Networks (RCTN)** is a dynamic meta-learning framework designed to solve complex, multi-step reasoning tasks (like the ARC-AGI) by neurally "stitching" together specialized architectural experts. Unlike standard neural networks that rely on global gradient descent, RCTN utilizes **Difference Target Propagation** to optimize local "Gaps" between forward activations and synthetic backward targets.

## 核心 (Core) Philosophy: "Stitch-Prop"

Traditional ensembles pick the "best" model. RCTN recognizes that different models are experts in different sub-problems. It treats every model as a "Loom" and neurally weaves their outputs together to create a "Frankenstein" prediction that is greater than the sum of its parts.

## 🛠 Features

### 1. Architectural Diversity (The Zoo)

The system spawns a "Species Zoo" of diverse experts, including:

* **Multi-Head Attention (MHA)** for global spatial relationships.
* **LSTM/RNN** for sequential pattern recognition.
* **SwiGLU & NormDense** for non-linear feature mapping.

### 2. Local Target Propagation (Tweening)

Instead of standard backpropagation, the `tween` engine:

* **Backward Targets**: Propagates expected outputs upward to estimate what each layer *should* have produced.
* **Link Budgets**: Measures signal fidelity at every layer to identify and prune information bottlenecks.
* **The Gap**: Directly optimizes the Euclidean difference between actual and target activations.

### 3. Neurally-Gated Recursive Stitching

The "Main" engine solves tasks via:

* **Unsupervised Profiling**: Clusters experts by their spatial and color specializations.
* **Neural Gating**: Trains a gate network to learn a pixel-level "Expert Affinity" mask.
* **Recursive Fusion**: Stitches the stitched outputs up to 4 layers deep to solve composed logic (e.g., "Rotate" + "Recolor").

## 🚀 Dynamic Architecture Setup

To apply this dynamically to any dataset, the system follows this pipeline:

1. **Species Generation**: `generateDiverseConfigs` creates  unique brain architectures.
2. **Specialist Analysis**: `analyzeOutputProfile` builds a "Specialty Map" for every expert.
3. **Neural Stitching**: `attemptNeuralStitching` creates the gating mask to fuse experts.
4. **N-Way Recursive Fusion**: `nWayStitchAllGrids` performs hierarchical composition to crack unsolved tasks.

## 📈 Performance Metrics

* **Fusion Bonus**: Measures how many tasks were solved by the ensemble that no single expert could solve alone.
* **Synergy Score**: Tracks the complementarity between specific expert pairs.
* **Depth Barrier**: Measures how much signal survives from the input to the final fusion layer.

---

### Getting Started

1. **Define your Experts**: Edit `AgentConfig43` to include new brain types.
2. **Run the Fusion**: Execute `main.go` to begin Phase 1 (Training) through Phase 3 (Recursive Stitching).
3. **Monitor the Gaps**: Use the `tween` state metrics to check if your experts are learning the targets effectively.

(Need to convert manual implementation what is being defined here into dynamic this is just the workbench for this process)