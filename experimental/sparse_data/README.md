# 📊 Sparse Data Learning Benchmark

Train with 30% missing samples and 20% corrupted labels. Test on clean data.

## Results

| Mode | Samples | Missed | Corrupt | Test% | Score |
|------|---------|--------|---------|-------|-------|
| **StepTweenChain** | 4164 | 1801 | 788 | **94%** | **94** |
| TweenChain | 4202 | 1769 | 831 | 79% | 79 |
| StepTween | 4203 | 1765 | 845 | 71% | 71 |
| StepBP | 4113 | 1855 | 833 | 61% | 61 |
| Tween | 4153 | 1817 | 856 | 51% | 51 |
| NormalBP | 4125 | 1843 | 805 | 47% | 46 |

## Key Finding

**StepTweenChain handles bad data best** (94% accuracy despite 30% missing + 20% corrupt). Chain-rule modes (TweenChain, StepTweenChain) outperform non-chain modes. NormalBP struggles most with noisy data.

## Running

```bash
cd tva/experimental/sparse_data
go run .
```
