# 🛡️ Adversarial Robustness Benchmark

Train with 50% random noise on inputs. Test on clean and noisy data.

## Results

| Mode | Train(noisy) | Test(clean) | Test(noisy) | Robust% | Blocked(ms) | Score |
|------|--------------|-------------|-------------|---------|-------------|-------|
| **StepTweenChain** | 54% | 84% | **79%** | **94%** | 0 | **79** |
| StepBP | 42% | 83% | 77% | 93% | 0 | 77 |
| StepTween | 59% | **100%** | 66% | 66% | 0 | 66 |
| NormalBP | 49% | 83% | 71% | 86% | 635 | 65 |
| TweenChain | 62% | 83% | 64% | 77% | 211 | 62 |
| Tween | 49% | 67% | 43% | 64% | 175 | 41 |

## Key Finding

**StepTweenChain has best robustness** (94%) - it generalizes from noisy training to both clean and noisy test data. Step-based modes overall perform better due to zero blocking time.

## Running

```bash
cd tva/experimental/adversarial
go run .
```
