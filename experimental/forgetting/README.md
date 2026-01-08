# 🧠 Catastrophic Forgetting Benchmark

Train on XOR (20s), then train on AND (20s). Measure how much XOR is forgotten.

## Results

| Mode | XOR After A | AND After B | XOR After B | Forgot% | Score |
|------|-------------|-------------|-------------|---------|-------|
| **NormalBP** | 100% | 75% | **50%** | 50% | **19** |
| **StepBP** | 100% | 75% | **50%** | 50% | **19** |
| **TweenChain** | 100% | 75% | **50%** | 50% | **19** |
| StepTween | 75% | 100% | 25% | 67% | 8 |
| Tween | 100% | 100% | 25% | 75% | 6 |
| StepTweenChain | 100% | 100% | 25% | 75% | 6 |

## Key Finding

**All modes suffer catastrophic forgetting** (50-75% of Task A lost). Interestingly, modes that learned Task B perfectly (100%) forgot more of Task A. This suggests a tradeoff between adaptation speed and memory retention.

## Running

```bash
cd tva/experimental/forgetting
go run .
```
