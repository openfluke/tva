# RN6: Recursive KMeans & Hierarchical Taxonomy Benchmark

This demo explores the power of **Recursive Neuro-Symbolic** architectures using Loom's `KMeansLayer`. It pits traditional Deep Learning (MLP) against Loom's interpretability-first approach across a series of rigorous challenges.

## 🧬 The Experiment: Trait-Species-Kingdom
We generated synthetic biology data representing a 3-tier hierarchy:
1.  **Kingdom**: The highest level labels (e.g., "Animal" vs "Plant").
2.  **Class**: Intermediate traits.
3.  **Species**: Fine-grained clusters (e.g., "Wolf" vs "Husky").

The model is trained on **Kingdom** labels but must demonstrate it has "discovered" the underlying Species-level structure through its weights and internal clustering.

---

## 📊 What You're Looking At

When you run `go run rn6.go`, you see a detailed breakdown of **Deviation Buckets**. This is the core of our precision verification.

### Triple-State Verification
We track every training mode through three phases to ensure nothing is lost:
1.  **_UnTrained**: The random, chaotic state before any learning happens.
2.  **_Trained**: The state after the optimizer has finished.
3.  **_Reloaded**: The state AFTER the model is serialized to JSON and loaded back into memory.

**Success Condition**: The `_Reloaded` bucket counts must be **identical** to the `_Trained` counts. This proves **Bit-Perfect Serialization** (0.00e+00% deviation).

### Understanding the Buckets
- **0-10% (High Precision)**: The model's prediction is nearly identical to the ground truth.
- **50-100% (High Deviation)**: The model is guessing or fundamentally wrong.
- **Trained States**: You will notice Loom's `Tween` modes quickly move almost all 1600 evaluation points (800 samples * 2 classes) into the 0-10% bucket.

---

## 🚀 Key Challenges

### 1. The Hallucination Gap (Mushroom Test)
We introduce an "Unseen" class (Mushroom) that has traits overlapping with both Animals and Plants.
- **Baseline (MLP)**: Confidently "hallucinates" that the Mushroom is an Animal (99.9% probability).
- **Loom (Recursive Hero)**: Detects a **Distance Spike** in the KMeans layer. It reports a low-confidence [0.5, 0.5] output, effectively signaling "I don't know what this is."

### 2. Zero-Shot Discovery
Can a model discover Species if it only sees Kingdom labels?
- Loom's `innerKMeans` layers automatically align themselves with the ground-truth prototypes. Because the architecture is recursive, it naturally hierarchies the data, making it **interpretable by design**.

### 3. Sample Efficiency
Traditional MLPs need hundreds of samples to avoid vanishing gradients. Loom's **Tweening** modes (like `StepTweenChain`) can reach high precision even with sparse data (5 samples per species) by directly optimizing the "flow" of weights between centroids.

---

## 🛠️ Technical Breakthroughs
This demo includes recent fixes to the Loom core:
- **Recursive ApplyGradients**: Gradients now propagate down through `KMeansLayer` sub-networks and `SequentialLayer` blocks.
- **Bit-Perfect JSON**: Fixed serialization bugs in KMeans to ensure that trained models can be shared and 100% replicated in any environment (Web, Apps, Edge).

---

## 💻 How to Run
```bash
go run rn6.go
```
Observe the final comparison table to see how Loom's recursive approach provides **100% Interpretability** and **High OOD Detection** where standard Black-Box MLPs fail.
