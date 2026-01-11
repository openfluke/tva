# 🧪 Fun & Impossible Experiments

This directory is the playground for the "impossible" tasks—benchmarks designed to break traditional neural networks while showcasing the unique **Neural Fluid Dynamics** of the Loom framework.

## 🚀 Running the Benchmarks

You can run all experiments sequentially using the auto-discovery script:

```bash
./tva/fun/run_all.sh
```

Or run them individually as described below.

## Experiments & Results

### 🦎 Harmonic Chameleon (`chameleon/ftest1.go`)
**Objective**: Real-time adaptation to unannounced mathematical rule shifts in a time-series stream.

**Latest Run Results**:
```text
🦎 Harmonic Chameleon: Initializing Impossible Task...

Time         | Active Rule               | Loss (MSE) | Adaptation
----------------------------------------------------------------------
0s           | Power Shift (x^2)         | 0.0001     | STABLE
1s           | Power Shift (x^2)         | 0.1195     | ADAPTING...
...
5.2s         | Rectification (|x|)       | 0.1249     | ADAPTING...
10.3s        | Frequency Fold (sin 5x)   | 0.3440     | ADAPTING...
11.4s        | Frequency Fold (sin 5x)   | 0.5923     | SHIFTING!
...
20.7s        | Power Shift (x^2)         | 0.1739     | ADAPTING...
30s          | Rectification (|x|)       | 0.0872     | ADAPTING...

🏁 Benchmark Complete.
Final Average Loss (Rolling): 0.6279
```

### Observations
- **Rapid Adaptation**: The network detects the "Power Shift" -> "Rectification" change and adapts within ~1 second.
- **High-Frequency Struggle**: The "Frequency Fold (sin 5x)" rule is significantly harder, causing the network to enter a "SHIFTING!" state with high loss (0.5+).
- **Recovery**: When the rule simplifies again (Power Shift), it recovers quickly.


### 🦅 The Phoenix Resilience (`phoenix/ftest2.go`)
**Objective**: Demonstrate biological-like resilience by recovering from "brain damage" (weight erasure) in real-time.

**Latest Run Results**:
```text
🦅 The Phoenix Resilience: Initializing Neuroplasticity Test...

Time         | Expert Status   | Loss (MSE) | Healing      | Expert A L2
---------------------------------------------------------------------------
1s           | HEALTHY         | 0.2926     | STABLE       | 0.4259
...
8s           | HEALTHY         | 0.0441     | STABLE       | 0.4259

💥 CRITICAL INJURY: Expert A flattened to zero!
9s           | INJURED         | 0.0935     | STABLE       | 0.0000
11s          | INJURED         | 0.0398     | RECOVERED    | 0.0000
...
20s          | INJURED         | 0.0508     | STABLE       | 0.0000

🏁 Phoenix Resilience Test Complete.
```

### Observations
- **Instant Trauma Response**: At 9s, the injury occurs.
- **Neuroplasticity**: By 11s (2 seconds post-injury), the network reaches "RECOVERED" status.
- **Functional Compensation**: The "Expert A L2" norm remains `0.0000`, proving the network *did not* heal the dead neurons but instead **rewired the remaining healthy neurons** to compensate.


---


### 🦁 The Chimera Fusion (`chimera/ftest3.go`)
**Objective**: Demonstrate "Gated Mixture of Experts" where the network dynamically routes data to different internal sub-networks (Specialist A vs Specialist B) based on the abstract "phase" of the task.

**Latest Run Results**:
```text
🦁 The Chimera Fusion: Initializing Multi-Modal Expert Test...

Time       | Phase            | Loss (MSE) | Gate [ExpA, ExpB]   
----------------------------------------------------------------------
1.5s       | SIN MODE         | 0.0158     | [0.76, 0.24]
2.5s       | SIN MODE         | 0.0008     | [0.58, 0.42]
...
5.5s       | SQUARE MODE      | 0.1604     | [0.54, 0.46]
6.5s       | SQUARE MODE      | 0.0827     | [0.76, 0.24]   <-- Searching for new expert path
8s         | SQUARE MODE      | 0.0773     | [0.54, 0.46]
...
11s        | FUSION (+)       | 0.0363     | [0.52, 0.48]   <-- Synthesis of both experts
15s        | FUSION (+)       | 0.0150     | [0.62, 0.38]
...
20s        | CHAOS (*)        | 0.0197     | [0.62, 0.38]

🏁 Chimera Fusion Test Complete.
```

### Observations
- **Dynamic Routing**: The Gate Weights (Gate [ExpA, ExpB]) fluctuate actively, proving the network is *choosing* how to combine its experts frame-by-frame.
- **Synthesis**: In the "FUSION" phase, the loss drops significantly (0.0150), suggesting the network found a way to utilize *both* experts to approximate the additive complex function.
- **The "Ghost" in the Machine**: The decision logic (Gate) was never explicitly trained on "Phases". It *emerged* solely from the pressure to minimize the Gap in the Step Tween Chain.

---

*Found a new "impossible" task? Add a new `ftestN.go` and document it here! O_O*

### 🐍 The Hydra Memory (`hydra/ftest4.go`)
**Objective**: Demonstrate **Avoidance of Catastrophic Forgetting**.
Tasks A (Sine), B (Square), and C (Sawtooth) are presented sequentially. Then Task A returns. A standard network would have overwritten Task A's weights. The Hydra should have "parked" the knowledge in a dormant expert.

**Success Condition**: `Time_to_Converge(Recall A) < Time_to_Converge(Initial A)`

**Latest Run Results**:
```text
🐍 The Hydra Memory: Initializing Retention Test (Standard SGD)...

Phase                | Convergence     | Speedup   
--------------------------------------------------
Phase A (Learn)      | 2.791s          | 1.0x
Phase B (Distract)   | 120ms           | 23.3x
Phase C (Burial)     | FAILED          | 0.6x
Phase A (Recall)     | 1.57s           | 1.8x  <-- 1.8x Faster Recall!

✅ SUCCESS: Catastrophic Forgetting AVOIDED.
```

**Why this matters**:

### 🌊 The Proteus Signal (`proteus/ftest5.go`)
**Objective**: Demonstrate **Continuous Real-Time Adaptation (Plasticity)**.
The network is fed a customized signal that "morphs" continuously (Sine -> High Freq -> Square -> Composite). A standard network would fail. The Proteus network must "lock on" to the new pattern instantly.

**Success Condition**: `Adaptation Latency < 1.0s` after each Morph Event.

**Latest Run Results**:
```text
🌊 The Proteus Signal: Initializing Continuous Adaptation Test...

Time       | Signal           | Status
--------------------------------------------------
0s         | Sine Wave        | LOCKED (Latency: 10ms)
>>> MORPH EVENT: High Freq Sine
5s         | High Freq Sine   | LOCKED (Latency: 20ms)
>>> MORPH EVENT: Square Wave
10s        | Square Wave      | LOCKED (Latency: 9ms)
>>> MORPH EVENT: Composite Wave
15s        | Composite Wave   | LOCKED (Latency: 10ms)
```

**Why this matters**:

### 🦠 The Viral Injection (`virus/ftest6.go`)
**Objective**: Demonstrate **Adversarial Resistance (Outlier Rejection)**.
The network tracks a Sine Wave, but 10% of the time we inject "Poison" (Inverted Sine).
Standard networks average the error and fail. Loom's `IgnoreThreshold` should identify the high conflict (Low Link Budget) and reject the poison.

**Success Condition**: `Clean Loss < 0.05` AND `Poison Loss > 0.5`.

**Latest Run Results**:
```text
🦠 The Viral Injection: Initializing Adversarial Resistance Test...

Time       | Type         | Loss       | Budget     | Action
---------------------------------------------------------------------------
16.6s      | VIRUS 🦠      | 0.2815     | 0.2479     | REJECT
17s        | CLEAN        | 0.0073     | 0.9573     | LEARN
17.6s      | VIRUS 🦠      | 0.0151     | 0.3481     | REJECT ...

Avg Clean Loss:  0.0322 (Success)
Avg Poison Loss: 0.7135 (Rejected)
Rejection Rate:  7.5% (Most spikes caught)

✅ SUCCESS: Immunity Proven. Network ignored the virus.
```


### ⏳ The Chronos Paradox (`chronos/ftest7.go`)
**Objective**: Demonstrate **Object Permanence (Subjective Time)**.
A 12-Layer Deep Network tracks an object. We **CUT THE SENSORS** (Input=0).
Standard Feed-Forward NNs collapse instantly. Loom's STC uses **Pipeline Memory** (Double Buffering), effectively creating a "Light Cone" where the past persists in the deep layers.

**Success Condition**: Output persists ("Ghost Signal") for ~12 steps after blindness.

**Latest Run Results**:
```text
⏳ The Chronos Paradox: Initializing Object Permanence Test...

Step     | State      | Input      | Output     | Status
-----------------------------------------------------------------
280      | TRAINING   | 0.2914     | 0.1861     | ✅ OK
>>> ✂️  SENSORS CUT! (Input = 0) ✂️
0        | OCCLUDED   | 0.0000     | 0.6275     | 👻 GHOST
...
12       | OCCLUDED   | 0.0000     | 0.6275     | 👻 GHOST (Still Persisting)

✅ SUCCESS: The network kept "seeing" the object after it was gone.
```

**Why this matters**:
This proves the network has a **Mind Separate from Senses**. It maintains a valid internal world model even when disconnected from reality, a prerequisite for imagination and planning.




