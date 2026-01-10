# Machine Learning Strategies for Photonic State Discrimination

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![arXiv](https://img.shields.io/badge/arXiv-quant--ph-b31b1b.svg)](https://arxiv.org)

> Physics Bachelor's Thesis - Universidad Nacional de La Plata (2023)  
> **Author:** Tomás Crosta  
> **Advisor:** Dr. Matías Bilkis

## 📋 Table of Contents

- [Description](#-description)
- [Problem](#-problem)
- [Implemented Agents](#-implemented-agents)
- [Results](#-results)
- [Project Structure](#-project-structure)
- [Citation](#-citation)
- [License](#-license)

## 🔬 Description

This repository contains the complete implementation of **reinforcement learning** techniques for model-free calibration of quantum detectors for coherent states. The work addresses a fundamental problem in quantum communications: distinguishing between coherent states with opposite phases `|±α⟩` in the presence of noisy channels, **without requiring a complete theoretical model** of the system.

### Application

Long-distance quantum communications (e.g., satellite-to-Earth) where:
- Classical information is encoded in the phase of laser pulses
- The channel introduces atmospheric noise, attenuation, and errors
- Detectors have unmodeled imperfections
- Real-time adaptive calibration is required

## 🎯 Problem

In quantum communications, a sender **A** transmits states `|α⟩` or `|-α⟩` (encoding a classical bit), and a receiver **B** must:

1. Apply an optimal displacement `β` to the received state
2. Measure with a photodetector
3. Classify the original state

**Challenge:** Coherent states are not orthogonal (`⟨α|-α⟩ ≠ 0`), so there is a fundamental error probability limited by the **Helstrom Bound**.

### Why Machine Learning?

❌ **Classical approach problems:**
- Requires exact model of the transmission channel
- Impossible to characterize all backgrounds in real communications
- Manual re-optimization with environmental changes
- Non-Gaussian models computationally intractable

✅ **RL advantages:**
- Calibration based on direct experimentation
- Automatic adaptation to environmental changes
- No theoretical model bias
- Discovers non-intuitive strategies

## ✨ Features

- 🤖 **3 RL agents** with different adaptation capabilities
- 🌐 **Real channel simulation:** attenuation, phase noise, dark counts
- 📊 **Comparison with theoretical limits:** Helstrom, Kennedy, Homodyne
- 🔧 **Modular and extensible:** easy to add new agents or channels
- ⚡ **GPU-optimized:** accelerated training with PyTorch

## 🤖 Implemented Agents

### 1. Mr. Magoo 🕶️

**Environment-blind agent - Baseline**

| Feature | Value |
|---------|-------|
| Observations | None (blind) |
| Method | Monte Carlo + Gradient Descent |
| Convergence | ~200 experiments |
| Adaptation | ❌ Requires retraining |
| Advantage | Fast and simple |

**When to use:** Static environment, quick initial calibration

### 2. Intensity-Adaptive 📊

**Agent sensitive to |α|**

| Feature | Value |
|---------|-------|
| Observations | Intensity \|α\| |
| Policy | Polynomial π_θ(\|α\|) = Σ θ_i \|α\|^i |
| Convergence | ~1000 experiments |
| Adaptation | ✅ Intensity changes |
| Advantage | No retraining if α changes |

**When to use:** Variable intensity, stable background

### 3. PPO (Proximal Policy Optimization) 🧠

**Fully adaptive agent**

| Feature | Value |
|---------|-------|
| Observations | \|α\|, β_i, P_success(β_i) from last 10 exp. |
| Method | Actor-Critic with neural networks |
| Convergence | ~10⁶ states |
| Adaptation | ✅ Any environmental change |
| Advantage | Maximum versatility |

**When to use:** Dynamic environment, training time available

## 📊 Results

### Success Probabilities (α = 0.5, Ideal Channel)

| Method | P_success | vs. Theoretical | Time |
|--------|-----------|-----------------|------|
| Homodyne Detection | 84.1% | Baseline | - |
| Kennedy (β=α) | 86.5% | Fixed | - |
| Optimized Kennedy | 88.2% | 100.0% | ~10ms |
| **Mr. Magoo** | **88.1%** | **99.9%** | ~5s |
| **Adaptive α** | **87.9%** | **99.7%** | ~25s |
| **PPO** | **88.0%** | **99.8%** | ~1h* |
| Helstrom Bound | 91.8% | Upper bound | - |

*\*One-time training, then instantaneous inference*

### Performance in Noisy Channel

**Gaussian Attenuation (θ = π/8)**

### Plots

![Convergence](docs/images/convergencia_sr_magoo.png)
![Agent Comparison](docs/images/comparacion_agentes.png)
![Noisy Channel](docs/images/canal_atenuacion.png)

### Related Papers

- M. Bilkis et al., "Real-time calibration of coherent-state receivers", [Phys. Rev. Research 2, 033295 (2020)](https://journals.aps.org/prresearch/abstract/10.1103/PhysRevResearch.2.033295)
- R.S. Kennedy, "Near-Optimum Receiver", [MIT Report 108 (1973)](https://dspace.mit.edu/handle/1721.1/56346)
- C.W. Helstrom, "Quantum Detection Theory" (1976)

## 🤝 Contributing

Contributions are welcome!

## 📖 Citation

If you use this code in your research, please cite:

```bibtex
@mastersthesis{crosta2023ml,
  title={Estrategias de aprendizaje automático para discriminación de estados fotónicos},
  author={Crosta, Tomás},
  year={2023},
  month={December},
  school={Universidad Nacional de La Plata},
  address={La Plata, Argentina},
  type={Bachelor's Thesis in Physics},
  note={Advisor: Dr. Matías Bilkis}
}
```

## 👥 Authors

- **Tomás Crosta** - *Author* - [@dmtomas](https://github.com/dmtomas)
- **Dr. Matías Bilkis** - *Advisor*
- **Prof. Dr. Juan Mauricio Matera** - *Academic Advisor*

## 🙏 Acknowledgments

- Physics Department, Universidad Nacional de La Plata
- Quantum Information Group
- La Vagancia (study group)
- Everyone mentioned in the thesis acknowledgments

## 📝 License

This project is licensed under the MIT License - see [LICENSE](LICENSE) for details.

## 🔗 Links

- [Repository](https://github.com/dmtomas/tesis)
- [Documentation](https://dmtomas.github.io/tesis)
- [Issues](https://github.com/dmtomas/tesis/issues)
- [Full Thesis (PDF)](docs/tesis_completa.pdf)

---
