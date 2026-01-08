# 🎵 Music Pattern Learning Benchmark

Creative AI benchmark where **blocking = gaps in the music**.

## Results Summary

| Training Mode    | Accuracy | Notes | Blocked (ms) | Availability | Score |
|-----------------|----------|-------|--------------|--------------|-------|
| **StepTween**   | 19.7%    | 3988  | 0            | 100.0%       | 26.2  |
| **StepBP**      | 19.0%    | 3988  | 0            | 100.0%       | 25.2  |
| Tween           | 20.2%    | 3831  | 2918         | 95.1%        | 24.5  |
| StepTweenChain  | 17.3%    | 3983  | 0            | 100.0%       | 23.0  |
| TweenChain      | 17.2%    | 3827  | 2944         | 95.1%        | 20.9  |
| NormalBP        | 16.9%    | 3652  | 5580         | 90.7%        | 18.7  |

**Key Pattern**: Non-blocking modes generate **~330 more notes** (9% more) than NormalBP.

## Test Configuration

- **Duration**: 60 seconds
- **Network**: 5-layer LSTM, 128 hidden units
- **Batch Size**: 100 samples
- **Note Rate**: ~67 notes/second
- **Output**: 24 notes (2 octaves)
- **Genres**: JAZZ → CLASSICAL → ELECTRONIC → RANDOM → BLUES → MINIMAL (every 5s)

## Music-Specific Insights

| Metric | NormalBP | StepTween | Difference |
|--------|----------|-----------|------------|
| Notes Generated | 3652 | 3988 | +336 (+9.2%) |
| Blocked Time | 5.58s | 0s | -5.58s |
| Availability | 90.7% | 100% | +9.3% |

In a real music performance, 5.58 seconds of silence would be **catastrophic**.

## Why LSTM?

Music is inherently sequential - LSTM helps:
- Remember melodic patterns
- Maintain rhythmic consistency
- Handle genre-specific structures

## Running

```bash
cd tva/experimental/music_patterns
go run .
```
