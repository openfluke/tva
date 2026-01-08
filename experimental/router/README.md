# 📡 Packet Router

High-throughput packet routing at 500 packets/sec across 8 destinations.

## Results

| Mode | Packets | Accuracy | Throughput | Blocked(ms) | Score |
|------|---------|----------|------------|-------------|-------|
| **StepBP** | 29728 | 66.5% | 495/s | 0 | **329** |
| StepTweenChain | 29773 | 65.0% | 496/s | 0 | 322 |
| StepTween | 29743 | 64.1% | 496/s | 0 | 318 |
| TweenChain | 29757 | 57.5% | 496/s | 271 | 285 |
| NormalBP | 29570 | 57.5% | 493/s | 836 | 284 |
| Tween | 29766 | 49.9% | 496/s | 260 | 248 |

## Key Findings

- **No dropped packets** - all modes keep up with 500pkt/s throughput
- Step-based modes achieve **higher routing accuracy** (64-67% vs 50-58%)
- NormalBP blocked for **836ms total** (1.4% of test time)
- Continuous learning improves routing decisions faster than batched learning

## Why Step Modes Win

At 500 packets/sec, every packet is a learning opportunity. Step-based modes learn from **every packet** while batch modes only learn periodically.

## Running
```bash
cd tva/experimental/router && go run .
```
