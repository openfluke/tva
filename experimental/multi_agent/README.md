# 🐝 Multi-Agent Swarm Coordination

4 neural network agents coordinate to catch 8 targets on a 16×16 grid.

## Results

| Mode | Catches | Collisions | Blocked(ms) | Score |
|------|---------|------------|-------------|-------|
| **StepBP** | **903** | 937 | 0 | **903** |
| StepTweenChain | 514 | 1939 | 0 | 514 |
| StepTween | 470 | 1176 | 0 | 470 |
| NormalBP | 74 | 2916 | 1240 | 62 |
| Tween | 22 | 4225 | 250 | 20 |
| TweenChain | 15 | 2252 | 254 | 13 |

## Key Finding

**Step-based modes catch 10-60x more targets** because they never block. Batch modes (NormalBP, Tween, TweenChain) waste ~1-2 seconds per minute blocked, during which agents can't move or learn.

## Running

```bash
cd tva/experimental/multi_agent
go run .
```
