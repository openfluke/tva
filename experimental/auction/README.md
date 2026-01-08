# 💰 Auction Bidder

Fast-paced trading simulation at 20 ticks/sec with clear buy signals.

## Results

| Mode | Ticks | Opps | Taken | Correct | BlockMiss | Profit | Score |
|------|-------|------|-------|---------|-----------|--------|-------|
| **Tween** | 599 | 244 | 12 | 12 | 2 | $180 | **160** |
| StepTweenChain | 599 | 253 | 7 | 3 | 0 | $5 | 5 |
| NormalBP | 599 | 275 | 22 | 7 | 1 | -$45 | 0 |
| TweenChain | 599 | 222 | 0 | 0 | 1 | $0 | 0 |
| StepBP | 599 | 246 | 1 | 0 | 0 | -$10 | 0 |
| StepTween | 599 | 255 | 0 | 0 | 0 | $0 | 0 |

## Key Findings

- **Tween wins** with $180 profit and 100% accuracy on taken opportunities
- Most modes struggle to learn the trading pattern
- Batch modes miss 1-2 opportunities due to blocking
- This benchmark favors **conservative bidding** - not bidding = no loss

## Analysis

This benchmark shows that the trading task is hard to learn online. Tween's batched learning with tweening produced the only profitable strategy.

## Running
```bash
cd tva/experimental/auction && go run .
```
