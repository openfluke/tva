# Empirical Study: Tweening vs Backpropagation

This directory contains a formal benchmark designed to test the stability limits of **Standard Backpropagation** against various forms of **Neural Tweening**, specifically in the context of high-dimensional spatial tasks (5D tensors). 

## The Problem

Training 3D Convolutional Neural Networks (`Conv3D`) using standard backpropagation is notoriously unstable. Without formal optimizers (like AdamW or RMSprop) gently guiding the descent, the gradient chain rule effectively multiplies errors across five dimensions: `Batch`, `Channel`, `Depth`, `Height`, and `Width`. 

In environments with high learning rates (`0.5`), this causes gradients to either **vanish** into zero or **explode** into infinity.

## The Benchmark

To prove the organic stability of Neural Tweening, this benchmark bypasses optimizers entirely. It feeds a `Conv3D -> Dense` network a stream of 8x8x8 volumes and asks it to classify the shape hiding within the noise:
- **Label 1:** A 3x3x3 Cube
- **Label 0:** A radius-2.9 Sphere

We train on 100 samples and evaluate against a 50-sample holdout test set using the core `nn.EvaluateNetwork()` pipeline.

## Tested Methodologies

The benchmark loops the exact same network initialization through 5 distinct execution modes:

1. **Normal BP:** Standard `net.BackwardCPU()` + `net.ApplyGradients()`. Expected to fail (50% random guessing accuracy) due to exploding gradients.
2. **Step Tween (Legacy):** Uses `tsLeg.TweenStep()` with geometric interpolation (no chain rule).
3. **Step Tween (Chain Rule):** Uses `tsChain.TweenStep()` leveraging chain rule algebra for stable weight morphing.
4. **Generic Tween:** Uses `tsGen.TweenStep()` utilizing the precision-agnostic `GenericTweenState` engine.
5. **Batch Generic Tween:** Uses `tsBatch.TweenBatchParallel()` to process training samples concurrently across CPU workers.

## Results & Conclusion

Running `go run main.go` yields:

| Training Method | Test Accuracy | Avg Deviation | 
| :--- | :--- | :--- | 
| **Normal BP** | 50.0% | 50.00% | 
| **Step Tween (Legacy)** | 88.0% | 12.00% | 
| **Step Tween (Chain)** | 100.0% | 0.00% |
| **Generic Tween** | 100.0% | 0.00% | 
| **Batch Generic Tween** | 98.0% | 2.00% | 

**Conclusion:** Standard Backpropagation struggles to map 5-Dimensional spatial gradients gracefully without a formal Optimizer. **Neural Tweening organically absorbs and normalizes the dimensional chain-rule via morphing, stabilizing complex architectures effortlessly.**
