# 🧪 Fun & Impossible Experiments

This directory is the playground for the "impossible" tasks—benchmarks designed to break traditional neural networks while showcasing the unique **Neural Fluid Dynamics** of the Loom framework.

## 🚀 Running the Benchmarks

You can run all experiments sequentially using the auto-discovery script:

```bash
./tva/fun/run_all.sh
```

Or run them individually as described below.

## 📊 Summary of Results (Loom vs Standard BP)

Results from the latest run (`results.txt`, Jan 11 2026) compare **Standard Backpropagation** (BP Batch/Step) against **Tween Modes** (Pure/Batch/Step).

> [!NOTE]
> The current results indicate that **Standard BP (Batch)** is currently outperforming Tween modes on these specific continuous regression tasks. Tween modes (particularly "Pure Tween") appear to struggle with smooth function approximation (Sine waves) compared to Gradient Descent, maintaining higher loss or failing to converge in several tests.

### 🦎 Harmonic Chameleon (`chameleon/ftest1.go`)
**Objective**: Real-time adaptation to unannounced mathematical rule shifts.

**Latest Run Results**:
- **BP (Batch/Step)**: Consistently maintained low loss (~0.10 - 0.15) throughout all phases (Power, Rectify, Fold).
- **Tween (All Modes)**: higher average loss (~0.30 - 0.60).
- **Conclusion**: Gradient descent (BP) tracked the smooth rule changes more accurately than the heuristic Tween updates in this configuration.

```text
Time     | Rule       | BP(Batch)    | BP(Step)     | Tween(Pure)
5s       | Rectify    | 0.1362       | 0.1238       | 0.4436
10s      | Fold       | 0.0891       | 0.0855       | 0.5830
```

### 🦅 The Phoenix Resilience (`phoenix/ftest2.go`)
**Objective**: Recover from "brain damage" (weight erasure) in real-time.

**Latest Run Results**:
- **BP (Batch)**: Excellent resilience. Loss dropped to < 0.05 almost immediately after injury recovery start (Time ~13s).
- **Tween (All Modes)**: Struggled to recover, with loss plateauing around 0.6 - 0.7.
- **Conclusion**: BP's directed gradient updates successfully re-optimized the remaining weights; Tween's "frontier" exploration failed to find a low-loss solution quickly.

```text
Time     | Condition    | BP(Batch)     | Tween(Pure)
8s       | INJURED 💥    | 0.0413        | 0.5459
13s      | HEALING      | 0.0217        | 0.4288
20s      | HEALING      | DONE          | DONE (High Loss)
```

### 🦁 The Chimera Fusion (`chimera/ftest3.go`)
**Objective**: Dynamic routing (MoE) based on abstract phase.

**Status**: ⚠️ **CRASHED**.
The test encountered a runtime error (`index out of range`) during execution.
- **Root Cause**: `ModeStepBP` or `Tween` forward logic failed to correctly handle input tensors for the Mixture-of-Experts gating mechanism.
- **Action Required**: Debug `ftest3.go` tensor handling in `runMode`.

### 🐍 The Hydra Memory (`hydra/ftest4.go`)
**Objective**: Avoid Catastrophic Forgetting. Recall A faster than learning A.

**Latest Run Results**:
- **BP (Batch)**: The most stable. Re-learned Phase A' successfully (Loss 0.0002).
- **BP (Step) / Tween**: Failed to converge or maintained high loss in Phase A'.
- **Conclusion**: Batch-based Gradient Descent was the only mode to effectively retain and recall the pattern.

```text
Mode            | Phase A' Convergence
BP(Batch)       | 210ms (Stable)
BP(Step)        | 4.13s (Slow)
Tween(All)      | FAIL / High Loss
```

### 🌊 The Proteus Signal (`proteus/ftest5.go`)
**Objective**: Continuous adaptation to morphing signals.

**Latest Run Results**:
- **BP (Batch)**: "LOCKED" status achieved quickly (Loss ~0.00).
- **Tween**: Remained in "ADAPTING" state with significantly higher loss.
- **Conclusion**: BP locked onto the signal patterns effectively.

### 🦠 The Viral Injection (`virus/ftest6.go`)
**Objective**: Reject outlier "poison" data.

**Latest Run Results**:
- **Tween**: **REJECT** action triggered on virus spikes. `Poison Loss > 0.5` (filtered out). This behavior (high loss on outlier) is actually desirable for outlier detection, but it must be distinguished from failure to learn.
- **BP**: Absorbs the poison (`LEARN` action implied), leading to higher "Clean Loss".

### ⏳ The Chronos Paradox (`chronos/ftest7.go`)
**Objective**: Object permanence (output persistence) when sensors are cut.

**Latest Run Results**:
- **BP (Batch)**: Maintained "OK" status.
- **Tween**: Often showed "LOST" status (Loss > threshold).
- **Observation**: Both architectures struggled with the occlusion task in this specific run, but BP maintained better signal tracking prior to occlusion.

---

**Overall Finding**:  
While Tweening offers theoretical advantages for sparse/discrete search spaces, **Standard Backpropagation (BP)** remains the champion for these continuous, differentiable signal processing tasks. Future work on Tweening should focus on hyperparameter tuning (DenseRate, Momentum) or application to non-differentiable RL tasks where BP cannot compete.
