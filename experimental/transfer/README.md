# 🔄 Transfer Learning Speed Benchmark

Pre-train on Task A (peak position, 30s), then fine-tune to Task B (oscillation frequency, 30s).

## Results

| Mode | TaskA% | TaskB Init | TaskB Final | Adapt Gain | Blocked(ms) | Score |
|------|--------|------------|-------------|------------|-------------|-------|
| **StepBP** | 20% | 12% | **26%** | **+14%** | 0 | **30** |
| StepTweenChain | 16% | 11% | 24% | +14% | 0 | 27 |
| Tween | 28% | 15% | 25% | +11% | 199 | 26 |
| TweenChain | 8% | 5% | 21% | +16% | 182 | 23 |
| StepTween | 13% | 13% | 21% | +8% | 0 | 23 |
| NormalBP | 12% | 28% | 24% | -4% | 626 | 17 |

## Key Finding

**StepBP adapts fastest** with +14% gain and zero blocking. NormalBP actually got *worse* during fine-tuning (-4% gain) - the batch training may have overfit during task switch. Step-based modes consistently improve.

## Running

```bash
cd tva/experimental/transfer
go run .
```
