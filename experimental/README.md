# 🧪 Experimental Training Mode Benchmarks

Ten benchmarks comparing Loom's training modes in scenarios where **continuous availability matters**.

## Batch 1: Real-Time Scenarios

| Benchmark | Task | Key Question |
|-----------|------|--------------|
| [Predator-Prey](predator_prey/) | Game AI chase | Can we catch prey without pausing? |
| [Anomaly Detection](anomaly_detection/) | Security monitoring | Do we miss anomalies while training? |
| [Robot Control](robot_control/) | Cart-pole balance | Does blocking cause crashes? |
| [Token Prediction](token_prediction/) | LLM-like generation | Is throughput affected by blocking? |
| [Music Patterns](music_patterns/) | Real-time composition | Do we have gaps in the music? |

## Batch 2: Training Characteristics

| Benchmark | Task | Key Question |
|-----------|------|--------------|
| [Multi-Agent](multi_agent/) | Swarm coordination | How do parallel networks synchronize? |
| [Forgetting](forgetting/) | XOR → AND | Which mode retains old knowledge best? |
| [Adversarial](adversarial/) | Noisy inputs | Which mode generalizes from noise? |
| [Transfer](transfer/) | Pre-train → fine-tune | Which mode adapts fastest? |
| [Sparse Data](sparse_data/) | Missing/corrupt data | Which mode handles bad data? |

## Training Modes Tested

| Mode | Blocking? | Chain Rule? | Best For |
|------|-----------|-------------|----------|
| **NormalBP** | Yes (batch) | Yes | Classification, accuracy-critical |
| **StepBP** | No | Yes | Real-time control, games |
| **Tween** | Yes (batch) | Optional | Moderate availability needs |
| **TweenChain** | Yes (batch) | Yes | Balance of accuracy/availability |
| **StepTween** | No | Optional | Real-time + adaptation |
| **StepTweenChain** | No | Yes | Maximum availability + learning |

## Key Insights

### When to Use Step-Based Modes
- ✅ Real-time games/control
- ✅ Streaming generation (LLMs, music)
- ✅ Latency-sensitive applications
- ✅ Continuous adaptation needed

### When to Use Batch Modes
- ✅ Accuracy is paramount
- ✅ Classification tasks
- ✅ Offline training acceptable
- ✅ Gradient stability needed

## Configuration (HARDCORE)

All benchmarks use:
- **Duration**: 60 seconds
- **Network**: 5-layer, 128 hidden units
- **Batch Size**: 100 samples
- **Pattern Switch**: Every 5 seconds

## Running All Benchmarks

```bash
cd tva/experimental

# Run each benchmark
for bench in predator_prey anomaly_detection robot_control token_prediction music_patterns; do
    echo "=== Running $bench ==="
    cd $bench && go run . && cd ..
done
```

## Sample Output (Predator-Prey)

```
🏆 BEST: StepBP with Score 215.6 | Catches: 154 | Availability: 100.0%
💀 WORST: Tween with Score 14.9 | Missed: 0 opportunities while blocked
📊 DIFFERENCE: 200.7 points (1342.3% improvement from worst to best)
```
