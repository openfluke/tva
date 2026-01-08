# ⚡ Reflex Game

Rapid stimulus-response classification with 50ms windows.

## Results

| Mode | Stimuli | Accuracy | Avg RT(ms) | Blocked(ms) | Score |
|------|---------|----------|------------|-------------|-------|
| **NormalBP** | 1199 | 41.7% | 0.02 | 30 | **2053** |
| Tween | 1199 | 40.8% | 0.02 | 11 | 2009 |
| StepBP | 1199 | 41.0% | 0.03 | 0 | 1984 |
| StepTweenChain | 1199 | 27.4% | 0.01 | 0 | 1349 |
| StepTween | 1199 | 27.4% | 0.02 | 0 | 1345 |
| TweenChain | 1199 | 25.9% | 0.02 | 10 | 1277 |

## Key Findings

- **No missed stimuli** - 50ms intervals are forgiving enough for all modes
- Batch modes (NormalBP, Tween) achieve **higher accuracy** - batched learning is more stable
- Step-based modes show **0ms blocking** but lower accuracy on this task
- Reaction times are sub-millisecond across all modes

## Running
```bash
cd tva/experimental/reflex && go run .
```
