# ⚖️ Balance Beam

Inverted pendulum control at 50Hz. Blocking = uncontrolled drift = falls.

## Results

| Mode | Steps | Controlled | Skipped | Falls | Accuracy | Score |
|------|-------|------------|---------|-------|----------|-------|
| **StepBP** | 1499 | 1499 | 0 | 19 | 72.6% | **1309** |
| StepTweenChain | 1499 | 1499 | 0 | 28 | 65.4% | 1219 |
| NormalBP | 1499 | 1481 | 18 | 56 | 44.5% | 903 |
| StepTween | 1498 | 1498 | 0 | 63 | 53.1% | 868 |
| TweenChain | 1499 | 1481 | 18 | 90 | 34.2% | 563 |
| Tween | 1499 | 1481 | 18 | 112 | 32.4% | 343 |

## Key Findings

- **StepBP dominates** with 72.6% accuracy and only 19 falls
- Batch modes skip **18 control steps** each during training → more falls
- Step-based modes have **0 skipped steps** and significantly fewer falls
- Continuous control is critical for stability - even small gaps cause drift

## Why Step Modes Win

When batch training blocks the control loop:
1. Pendulum drifts without correction
2. Error accumulates
3. System becomes unstable → falls

Step-based training never blocks, so the pendulum is always under control.

## Running
```bash
cd tva/experimental/balance && go run .
```
