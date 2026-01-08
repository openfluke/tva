# 💓 Heartbeat Monitor

Real-time cardiac monitoring simulation. Pre-trained network detects critical heartbeats.

## Results

| Mode | Beats | Critical | Detected | BlockMiss | Blocked(ms) | Score |
|------|-------|----------|----------|-----------|-------------|-------|
| **StepBP** | 999 | 208 | 208 | 0 | 0 | **208** |
| StepTweenChain | 998 | 204 | 204 | 0 | 0 | 204 |
| NormalBP | 999 | 207 | 207 | 0 | 29 | 207 |
| TweenChain | 998 | 197 | 197 | 0 | 4 | 197 |
| Tween | 999 | 193 | 193 | 0 | 4 | 193 |
| StepTween | 999 | 191 | 191 | 0 | 0 | 191 |

## Key Findings

- **100% detection rate** across all modes after pre-training
- Step-based modes have **0ms blocking** vs 4-29ms for batch modes
- Pre-training is critical for medical-grade detection accuracy

## Running
```bash
cd tva/experimental/heartbeat && go run .
```
