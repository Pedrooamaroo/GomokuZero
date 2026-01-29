# 🧠 GomokuZero: Hybrid AlphaZero Implementation

[![Python](https://img.shields.io/badge/Python-3.10+-blue.svg)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-EE4C2C.svg)](https://pytorch.org/)
[![Numba](https://img.shields.io/badge/Numba-High_Performance-003399.svg)](https://numba.pydata.org/)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

## 🎯 Project Overview
**GomokuZero** is a research-oriented implementation of the AlphaZero algorithm, optimized for the board games **Gomoku** (Five-in-a-Row) and **Pente**. 

While traditional AlphaZero relies solely on self-play to learn strategies from scratch, this project introduces a **Hybrid Monte Carlo Tree Search (MCTS)**. By integrating Numba-accelerated tactical heuristics into the probabilistic search, the agent significantly reduces the training time required to master complex tactical patterns like "Open-4" or "Capture Sequences."

---

## 🔬 Technical Architecture

### 1. Deep Residual Network (ResNet)
The decision-making core is a custom ResNet architecture implemented in **PyTorch**:
* **Input Representation:** A multi-channel tensor `(Batch, C, 15, 15)` encoding the current board state, opponent positions, and auxiliary features like capture counts (specific to Pente).
* **Residual Blocks:** Utilizes skip connections to facilitate gradient flow during deep training.
* **Dual-Head Output:** Simultaneously predicts the **Policy** (move probabilities) and **Value** (win probability) for any given state.

### 2. High-Performance Search Engine
To address the computational bottleneck of Python-based MCTS, the search logic is optimized using **Numba**:
* **JIT Compilation:** Key tactical functions (`check_window`, `score_move`) are compiled to machine code, achieving near C++ execution speeds.
* **Heuristic Biasing:** The search tree expansion is guided by a "Tactical Score" that prioritizes immediate threats (e.g., blocking a winning line) and captures, allowing the network to focus on long-term strategy rather than basic tactical blunders.

---

## 📈 Performance Summary

| Metric | Traditional AlphaZero (Python) | GomokuZero (Hybrid/Numba) | Improvement |
| :--- | :--- | :--- | :--- |
| **Search Speed** | ~40 simulations/sec | **~800+ simulations/sec** | **20x Speedup** |
| **Tactical Safety** | Low (Early Training) | **High (Day 1)** | **Immediate** |
| **Convergence** | Slow (>5000 games) | **Fast (~800 games)** | **~6x Faster** |

**Key Takeaway:** The hybrid approach allows the model to play competently against human intermediates almost immediately, whereas a pure RL approach typically requires thousands of self-play games to discover basic rules like "blocking 4 in a row."

---

## 🛠️ Installation & Usage

### 1. Environment Setup
Clone the repository and install dependencies:
```bash
git clone [https://github.com/YourUsername/GomokuZero.git](https://github.com/YourUsername/GomokuZero.git)
cd GomokuZero
pip install -r requirements.txt
