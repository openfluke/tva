# 🎮 Predator-Prey Adaptive Chase Benchmark

Real-time game AI benchmark comparing training modes on a chase task where **blocking = missed catches**.

## Results Summary

| Training Mode    | Catches | Blocked (ms) | Availability | Score | Rank |
|-----------------|---------|--------------|--------------|-------|------|
| **StepBP**      | 154     | 0            | 100.0%       | 216   | 🥇   |
| **StepTweenChain** | 110  | 0            | 100.0%       | 154   | 🥈   |
| **StepTween**   | 89      | 0            | 100.0%       | 89    | 🥉   |
| NormalBP        | 26      | 797          | 98.7%        | 31    | 4th  |
| TweenChain      | 16      | 209          | 99.7%        | 19    | 5th  |
| Tween           | 15      | 205          | 99.7%        | 15    | 6th  |

**Key Finding**: Non-blocking modes (StepBP, StepTweenChain, StepTween) caught **5-10x more prey** than batch-training modes.

## Test Configuration

- **Duration**: 60 seconds
- **Network**: 5-layer Dense, 128 hidden units
- **Batch Size**: 100 samples before training
- **Grid**: 12×12
- **Behaviors**: RANDOM → FLEE → ZIGZAG → FREEZE → MIRROR → CIRCLE (every 5s)

## Why Step-Based Modes Win

1. **Zero blocking time** - Always available to chase
2. **Continuous adaptation** - Updates weights every step
3. **No batch accumulation delay** - Reacts immediately to behavior changes

## Running

```bash
cd tva/experimental/predator_prey
go run .
```
