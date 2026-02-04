# 🛡️ Loom Cybersecurity: Real-Time Network Anomaly Detection

This project implements a high-performance network packet anomaly detection system using the [Loom](https://github.com/openfluke/loom) framework. It monitors live hardware packets via `libpcap` and uses deep neural networks to identify traffic anomalies in real-time.

## 🚀 "Hardcore" Methodology

Unlike simple classification, this tool uses **Next-Packet Prediction Error** for detection. 

1.  **Prediction Task**: A 5-layer deep neural network attempts to predict the size of the next incoming packet based on a sliding window of features (Size, Protocol, Ports) from the last 16 packets.
2.  **Detection**: An anomaly is flagged when the network's prediction error exceeds `0.4` on a packet that also represents a "Ground Truth" spike (normalized size > `0.7`).
3.  **The Benchmark**: Multiple training modes run concurrently on the **exact same packet stream**, comparing their ability to learn the traffic pattern and flag surprises.

## 📊 Benchmark Results (Verified)

Captured on interface `enp43s0` (30-second live terminal test):

| Training Mode | Detected | Ground Truth | False Positives | Avg Error | Score | Status |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **Tween** | **653** | 716 | 258 | 0.240 | **80.0** | ✅ PASS |
| **NormalBP** | 634 | 716 | 224 | 0.242 | 78.8 | ✅ PASS |
| **StepTweenChain** | 641 | 716 | 263 | 0.234 | 78.1 | ✅ PASS |
| **StepBP** | 545 | 716 | 190 | 0.222 | 67.8 | ✅ PASS |

### Analysis
*   **Tween Mode** achieved the highest score, demonstrating a superior balance between learning the "normal" pattern and flagging anomalous spikes.
*   **StepTweenChain** (Fully Adaptive) maintains zero-batching overhead while processing every packet, proving highly competitive for real-time edge security.
*   **NormalBP** (Traditional Batch) remains effective but suffers slightly as it pauses processing to train on batches, potentially missing transient patterns.

## 🛠️ Usage

### Prerequisites
- `nmap` & `libpcap-devel`
- Root/Sudo privileges for raw socket access

### How to Run
```bash
sudo ./anom
```

### Simulation
To test the detection in real-time, run an nmap scan from another machine or local terminal:
```bash
sudo nmap -sS -v <your-ip>
```

## 📁 Project Structure
- `anom.go`: Main sniffer and detection engine.
- `go.mod`: Isolated module with Loom and Gopacket dependencies.
- `cybersecurity_anom_results.json`: Full technical summary exported after each run.
