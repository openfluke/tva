# 🔍 Streaming Anomaly Detection Benchmark

Security-focused benchmark where **missing anomalies during training = vulnerabilities**.

## Results Summary

| Training Mode    | Detected | Detection% | Blocked (ms) | Availability | Score |
|-----------------|----------|------------|--------------|--------------|-------|
| **NormalBP**    | 1933     | 53.3%      | 1546         | 97.4%        | 52    |
| **StepTween**   | 1554     | 41.8%      | 0            | 100.0%       | 42    |
| **StepTweenChain** | 1521  | 40.9%      | 0            | 100.0%       | 41    |
| StepBP          | 1493     | 40.6%      | 0            | 100.0%       | 41    |
| Tween           | 1466     | 39.8%      | 383          | 99.4%        | 40    |
| TweenChain      | 1453     | 39.4%      | 377          | 99.4%        | 39    |

**Interesting Finding**: NormalBP achieves highest detection rate despite blocking - batch training provides better gradient accumulation for this classification task.

## Test Configuration

- **Duration**: 60 seconds
- **Network**: 5-layer Dense, 128 hidden units
- **Batch Size**: 100 samples
- **Patterns**: NORMAL → SPIKE → DRIFT → OSCILLATION → BURST → GRADUAL (every 5s)

## Analysis

Unlike the predator-prey game, anomaly detection benefits from:
- **Batch gradient averaging** - More stable learning
- **Full-epoch updates** - Better pattern recognition
- **Task nature** - Classification is less time-critical than control

Step-based modes still achieve **100% availability** but slightly lower detection rates.

## Running

```bash
cd tva/experimental/anomaly_detection
go run .
```
