# 🛡️ Loom Real-Time Anomaly Benchmark

This is the "Hardcore" version of the Loom network anomaly detector. It monitors live hardware traffic and evaluates 6 different neural network training modes using high-precision benchmarking metrics.

## 🔬 High-Precision Methodology

Unlike standard benchmarks that look at total averages, `realtime_anom` slices time into **50ms windows**. This reveals transient performance dips, "blocked" periods during training, and the actual responsiveness of the model.

### Metric Definitions:
- **Detected (TP)**: True Positives—High prediction error (surprise) when a real packet spike (GT > 0.7) occurs.
- **GT Sigs**: Ground Truth Signals—The number of significant packet spikes detected by simple thresholding.
- **FalsePos**: "Crying wolf"—Flags an anomaly on normal-sized packets.
- **Accuracy**: Next-packet size prediction accuracy (within a 0.2 threshold).
- **Score**: The "Unified Combat Score": `(Throughput × Availability% × Accuracy%) / 10000`.
- **Avail %**: Percentage of time the model was available for inference (not blocked by training).
- **Blocked(ms)**: Total millisecond delay caused by synchronous training steps.
- **Peak Lat**: The worst-case delay encountered between packet processing events.

## 📊 Detailed Performance Analysis

Based on the verified 30-second live capture on `enp43s0`:

| Mode | Detected | FalsePos | Accuracy | Score | Avail % | Blocked | Peak Lat | Key Insight |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **StepBP** | **36/36** | **23** | **25.5%** | **5** | **100.0%** | **0ms** | 998.8ms | **Best Accuracy/Efficiency** |
| **StepTweenChain**| 36/36 | 29 | 23.9% | 5 | 100.0% | 0ms | 999.6ms | High Stability |
| **NormalBP** | 36/36 | 36 | 23.7% | 5 | 99.8% | 73ms | 998.5ms | **Blocked during Training** |
| **Tween** | 36/36 | 33 | 23.7% | 5 | 99.7% | 93ms | 998.8ms | Synchronous Overhead |
| **StepTween** | 36/36 | 36 | 23.9% | 5 | 100.0% | 0ms | 999.6ms | Adaptive |
| **TweenChain** | 36/36 | 43 | 22.6% | 4 | 99.7% | 89ms | 999.2ms | Worst False Positives |

### Key Findings:

1.  **Adaptive Dominance**: Modes like `StepBP` and `StepTweenChain` achieved **100% Availability** with **0ms blocking time**. While `NormalBP` and `Tween` achieve decent accuracy, they periodically "pause" the world to train on batches, which is a major risk for high-speed packet processing.
2.  **False Positive Suppression**: `StepBP` demonstrated the lowest False Positive count (23), suggesting that immediate gradient updates on every sample allow the network to track the "normal" background noise more effectively than batch updates.
3.  **Throughput vs. Jitter**: Although `NormalBP` has a slightly lower peak latency in this specific run, the 73ms total blocked time represents a cumulative gap where the system is effectively blind to incoming traffic.
4.  **Score Parity**: Most modes scored a "5" under this specific load, but the underlying metrics reveal that the **Step**-based modes are the strictly superior choice for low-latency edge security.

## 🛠️ Operation

### Build & Run
```bash
sudo ./realtime_anom
```

### Interpretation
- **Watch the Blocked(ms) column**: If this rises, your detector is going "blind" during training.
- **Look for Score Gaps**: A high score with low FalsePos is the gold standard for network security.

## 📁 Technical Export
A window-by-window breakdown of every 50ms interval is saved to `realtime_anom_results.json` after every run, suitable for plotting performance over time.
