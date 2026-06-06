# 🚚 PackSplit

> **Intelligent Delivery Load Balancing using Genetic Algorithms**

An AI-powered optimization project that intelligently distributes delivery packages across multiple vehicles using a sophisticated combination of **Breadth-First Search (BFS)** and custom **Genetic Algorithm (GA)** techniques.

<div align="center">

![Python](https://img.shields.io/badge/python-3.8+-blue?style=flat-square&logo=python&logoColor=white)
![License](https://img.shields.io/badge/license-MIT-green?style=flat-square)
![Status](https://img.shields.io/badge/status-active-success?style=flat-square)

</div>

---

## 📋 Problem Statement

How do you optimally allocate delivery packages between multiple vehicles to **minimize travel distance imbalance**?

**The Challenge:**
- **6 delivery packages** distributed on a grid
- **2 vehicles** departing from a central depot
- **Obstacles** blocking direct paths
- **Goal:** Minimize the difference in total travel distances

$$\text{Imbalance} = | \text{Distance}_{\text{Vehicle 1}} - \text{Distance}_{\text{Vehicle 2}} |$$

---

## ✨ Key Features

### 🗺️ Grid-Based Pathfinding
- **15×15 2D grid** environment with dynamic obstacles
- **Breadth-First Search (BFS)** guarantees optimal shortest paths
- Calculates exact distances from Depot to all delivery points
- Handles complex spatial constraints efficiently

### 🧬 Genetic Algorithm Engine
| Component | Implementation |
|-----------|-----------------|
| **Representation** | 6-bit binary chromosome (0 = Vehicle 1, 1 = Vehicle 2) |
| **Fitness** | $f = \frac{1}{1 + \text{Imbalance}}$ (maximization form) |
| **Selection** | Tournament Selection (k=5) |
| **Crossover** | One-Point Crossover |
| **Mutation** | Random, Swap, and Gaussian variants |

#### 🔄 Three Mutation Strategies:
1. **Random Mutation** - Flips a random bit
2. **Swap Mutation** - Exchanges two random bits (maintains allocation structure)
3. **Gaussian Mutation** - Probabilistic soft-flip using sigmoid function

### 🛑 Smart Termination
The algorithm stops when any condition is met:
- ⏱️ **Max Iterations:** 400 generations reached
- 🎯 **Convergence:** No improvement for 80 consecutive generations
- ✅ **Optimal Found:** Perfect load balance detected (fitness = 1.0)

### 📈 Interactive Visualizations
- **Allocation Map:** Grid visualization with color-coded package assignments
- **Convergence Graph:** Fitness evolution across generations
- **Real-time Metrics:** Distance imbalance tracking

---

## 📁 Project Structure

```
PackSplit/
├── delivery_allocation.py          # Main algorithm engine
├── testing_understanding.py         # Mutation testing & experimentation
├── AI_Project_Final_Report.pdf     # Complete academic report
├── delivery_allocation_results.png # Sample output visualization
└── README.md                        # This file
```

---

## 🚀 Quick Start

### Prerequisites
```bash
pip install numpy matplotlib
```

### Running the Optimization
```bash
python delivery_allocation.py
```

The script will:
1. Initialize the grid environment with obstacles
2. Run the genetic algorithm optimization
3. Generate and save visualization plots
4. Display final results (optimal imbalance, generations used)

---

## 📊 Sample Results

Below is an example of PackSplit's output visualization:

![Delivery Allocation Results](delivery_allocation_results.png)

**Left Panel:** Shows the delivery grid with color-coded package allocations  
**Right Panel:** Displays the fitness convergence curve across generations

---

## 🔍 Algorithm Overview

### Phase 1: Shortest Path Calculation
```
For each of the 6 delivery points:
  └─ Use BFS to find shortest path from Depot
     └─ Store distance for vehicle allocation evaluation
```

### Phase 2: Population-Based Optimization
```
Initialize random population of chromosomes
├─ Evaluate fitness for all solutions
├─ While termination condition not met:
│  ├─ Selection (Tournament)
│  ├─ Crossover (One-Point)
│  ├─ Mutation (Random/Swap/Gaussian)
│  └─ Evaluate new population
└─ Return best solution found
```

---

## 📈 Performance Characteristics

- **Search Space:** 2^6 = 64 possible allocations
- **Population Size:** Configurable (default: 20)
- **Average Runtime:** < 1 second on standard hardware
- **Convergence Rate:** Typically finds optimal/near-optimal solutions within 100-200 generations

---

## 🎓 Academic Context

This project demonstrates key concepts from:
- **Artificial Intelligence & Search Algorithms**
- **Evolutionary Computation**
- **Graph Theory & Pathfinding**
- **Combinatorial Optimization**

> 📄 See `AI_Project_Final_Report.pdf` for detailed methodology, theoretical analysis, and experimental results.

---

## 💡 Use Cases

✅ **Delivery Route Optimization** - E-commerce & logistics companies  
✅ **Fleet Management** - Minimize fuel consumption across routes  
✅ **Resource Allocation** - Distribute workload evenly  
✅ **Educational** - Learn evolutionary algorithms in practice

---

## 🤝 Contributing

Contributions are welcome! Feel free to:
- Enhance the genetic algorithm operators
- Improve visualization features
- Add support for more vehicles/packages
- Optimize performance

---

## 📝 License

This project is open source and available under the MIT License.

---

## 👤 Author

**Hazem Ayman**

---

<div align="center">

**Made with ❤️ for optimization enthusiasts**

⭐ If you found this helpful, please consider starring the repository!

</div>
