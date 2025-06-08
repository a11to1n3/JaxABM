# JaxABM Examples

This directory contains focused, high-quality examples demonstrating JaxABM capabilities. Each example is carefully crafted to be educational, practical, and well-documented.

## 🚀 Quick Start

**New to JaxABM?** Start here:
```bash
python examples/basic_example.py
```

This single file demonstrates all core JaxABM concepts and is the best introduction to the framework.

## 📁 Example Categories

### 🎯 Basic Example (`basic_example.py`)
**Perfect starting point for beginners**

A comprehensive single-file example that demonstrates:
- Basic agent definition and behavior
- Environment setup and state management
- Model execution and results analysis
- Plotting and visualization

**What you'll learn:**
- How to create agents with `jx.Agent`
- How to build models with `jx.Model`
- How to run simulations and analyze results

### 📊 Parameter Optimization (`calibration/`)
Learn how to automatically tune model parameters to achieve desired outcomes.

- **`final_working_calibration_demo.py`** - Complete demonstration of all calibration methods
- **`advanced_calibration_example.py`** - Advanced techniques for complex models

**Methods Available:**
- ✅ **Evolutionary**: PSO (~0.01 loss), ES (~21 loss), CEM (~30 loss)
- ✅ **Reinforcement Learning**: Q-Learning (~0.7 loss), Policy Gradient (~0.9 loss), Actor-Critic, DQN

### 🏘️ Advanced Models (`models/`)
Complete, real-world model implementations for learning and adaptation.

- **`schelling_model.py`** - Classic segregation model with spatial dynamics
- **`random_walk.py`** - Multi-agent random walk with interactions
- **`advanced_economic_model.py`** - Sophisticated economic simulation with multiple agent types

### 🔬 Sensitivity Analysis (`sensitivity/`)
Understand how parameters affect model outcomes.

- **`simple_sensitivity_example.py`** - Basic parameter sensitivity analysis
- **`sensitivity_calibration_example.py`** - Combined sensitivity analysis and optimization

## 📚 Learning Progression

### 🌱 **Beginner Path**
1. **Start with basics**: `python basic_example.py`
2. **Try a complete model**: `python models/schelling_model.py`
3. **Learn optimization**: `python calibration/final_working_calibration_demo.py`

### 🌿 **Intermediate Path**
1. **Explore sensitivity**: `python sensitivity/simple_sensitivity_example.py`
2. **Advanced models**: `python models/random_walk.py`
3. **Custom optimization**: Modify calibration examples

### 🌳 **Advanced Path**
1. **Complex systems**: `python models/advanced_economic_model.py`
2. **Research workflows**: Combine sensitivity + calibration
3. **Custom development**: Build your own models using patterns from examples

## 🎯 Performance Benchmarks

Based on comprehensive testing across all methods:

| Method Type | Best Method | Typical Loss | Performance | Use Case |
|-------------|-------------|--------------|-------------|----------|
| **Evolutionary** | PSO | ~0.01 | 🥇 Excellent | General optimization |
| **Evolution Strategies** | ES | ~21.12 | ✅ Good | Robust optimization |
| **Reinforcement Learning** | Q-Learning | ~0.70 | ✅ Working | Sequential decisions |
| **Cross-Entropy** | CEM | ~30.18 | ✅ Good | Discrete parameters |

## 🔧 Requirements

**Core Requirements:**
- Python 3.9+
- JAX
- JaxABM (`pip install -e .` from repository root)

**Optional (for visualization):**
- Matplotlib (`pip install matplotlib`)

## 🎮 Running Examples

### Basic Example
```bash
# Quick start - comprehensive introduction
python basic_example.py
```

### Calibration Examples
```bash
cd calibration/
python final_working_calibration_demo.py
```

### Model Examples
```bash
cd models/
python schelling_model.py
python random_walk.py
```

### Sensitivity Analysis
```bash
cd sensitivity/
python simple_sensitivity_example.py
```

## 🧪 Testing Examples

All examples are tested in our CI pipeline:

```bash
# Test basic example
python basic_example.py

# Test specific categories
cd calibration && python final_working_calibration_demo.py
cd models && python schelling_model.py
```

## 🐛 Troubleshooting

**Common Issues:**

1. **Import Errors**: Ensure JaxABM is installed: `pip install -e .` from repository root
2. **JAX Issues**: Install appropriate JAX version for your system
3. **Matplotlib Missing**: Install with `pip install matplotlib` for plots

**Performance Issues:**
- Reduce agent counts or simulation steps for faster execution
- Use smaller parameter ranges for calibration examples

## 💡 Next Steps

After exploring the examples:

1. **Modify Examples**: Change parameters, agent behaviors, or model structure
2. **Build Your Model**: Use examples as templates for your research
3. **Optimize Parameters**: Apply calibration methods to your models
4. **Analyze Sensitivity**: Understand which parameters matter most

## 📄 License

All examples are provided under the same license as JaxABM. See the main repository LICENSE file for details.

---

**Need Help?**
- Check the main repository README
- Review test files in `tests/` for additional usage patterns
- Open an issue on GitHub for questions 