# 💬 Streaming Token Prediction Benchmark

LLM-relevant benchmark where **blocking = terrible generation latency**.

## Results Summary

| Training Mode    | Accuracy | Tokens | Blocked (ms) | Availability | Score |
|-----------------|----------|--------|--------------|--------------|-------|
| **StepTween**   | 13.8%    | 11911  | 0            | 100.0%       | 27.4  |
| **StepBP**      | 13.6%    | 11927  | 0            | 100.0%       | 27.0  |
| Tween           | 13.3%    | 11918  | 634          | 98.9%        | 26.1  |
| StepTweenChain  | 12.8%    | 11919  | 0            | 100.0%       | 25.5  |
| TweenChain      | 12.8%    | 11918  | 649          | 98.9%        | 25.1  |
| NormalBP        | 11.9%    | 11691  | 1717         | 97.1%        | 22.6  |

**Key Finding**: Step-based modes achieve both **higher accuracy AND higher throughput**. 

## Test Configuration

- **Duration**: 60 seconds
- **Network**: 5-layer Dense, 128 hidden units
- **Batch Size**: 100 samples
- **Context**: 32 characters
- **Output**: 26 classes (a-z)
- **Styles**: TECHNICAL → CASUAL → FORMAL → CODE → POETRY → SCIENTIFIC (every 5s)

## Why Step-Based Modes Excel

1. **More tokens generated** (11,900+ vs 11,691 for NormalBP)
2. **Continuous style adaptation** - No delay after style switch
3. **Better accuracy** - Frequent small updates beat infrequent large batches

## LLM Relevance

This benchmark simulates:
- Character-level next-token prediction
- Style/topic adaptation during generation
- Latency-sensitive streaming output

Real LLMs would benefit similarly from non-blocking fine-tuning approaches.

## Running

```bash
cd tva/experimental/token_prediction
go run .
```
