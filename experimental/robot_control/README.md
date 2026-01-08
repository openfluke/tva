# 🤖 Robot Control with Disturbance Benchmark

Control systems benchmark where **blocking = missed control steps = crash**.

## Results Summary

| Training Mode    | Stability% | Crashes | Blocked (ms) | Availability | Score |
|-----------------|------------|---------|--------------|--------------|-------|
| **StepBP**      | 60.4%      | 402     | 0            | 100.0%       | 60.4  |
| TweenChain      | 60.5%      | 421     | 158          | 99.7%        | 60.3  |
| StepTweenChain  | 60.2%      | 423     | 0            | 100.0%       | 60.2  |
| StepTween       | 60.1%      | 424     | 0            | 100.0%       | 60.1  |
| Tween           | 60.3%      | 422     | 156          | 99.7%        | 60.1  |
| NormalBP        | 59.6%      | 428     | 893          | 98.5%        | 58.8  |

**Finding**: Cart-pole is a relatively forgiving control task - all modes achieve ~60% stability. The physics simulation doesn't crash immediately when control pauses briefly.

## Test Configuration

- **Duration**: 60 seconds
- **Network**: 5-layer Dense, 128 hidden, Tanh activation
- **Batch Size**: 100 samples
- **Physics**: NORMAL → LOW_GRAV → HIGH_GRAV → WINDY → ICY → HEAVY_POLE (every 5s)
- **Control Rate**: 100 Hz (10ms intervals)

## Why Scores Are Similar

The cart-pole simulation:
- Has natural momentum that carries through brief pauses
- Recovers quickly from small disturbances
- Angle limits are generous enough for temporary loss of control

For more dramatic differences, a faster/harder control task (e.g., drone hovering, inverted double pendulum) would be needed.

## Running

```bash
cd tva/experimental/robot_control
go run .
```
