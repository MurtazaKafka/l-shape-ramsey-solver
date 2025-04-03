# L-Shape Ramsey Problem Solver

This project implements various approaches to solve the L-shape Ramsey problem, which involves finding grid configurations that avoid monochromatic L-shapes while using a limited number of colors. This implementation extends and builds upon DeepMind's FunSearch framework.

## Problem Description

The L-shape Ramsey problem asks: Given a grid of size n×n and k colors, what is the largest possible grid that can be colored without creating any monochromatic L-shapes? An L-shape is formed by three cells of the same color in an L configuration.

## Implementation Approaches

1. **FunSearch Implementation** (`implementation/`)
   - Based on DeepMind's FunSearch framework
   - Uses evolutionary algorithms to find solutions
   - Implements a sandbox for safe code execution
   - Includes visualization tools
   - Extends the original implementation with:
     - Additional sampling strategies
     - Enhanced visualization capabilities
     - Improved error handling
     - Custom evaluation metrics

2. **Hierarchical Approach** (`l_shape_hierarchical.py`)
   - Builds solutions by combining known valid smaller grids
   - Uses multiple generation strategies
   - Maintains a cache of known solutions
   - Novel approach developed independently

3. **GPU-Accelerated Approach** (`l_shape_gpu.py`)
   - Uses PyTorch for GPU-accelerated computation
   - Implements efficient L-shape detection on GPU
   - Optimized for large grid sizes
   - Novel approach developed independently

4. **Reinforcement Learning Approach** (`l_shape_rl.py`)
   - Applies RL techniques to discover optimal grid configurations
   - Formulates grid coloring as a sequential decision-making problem
   - Implements state-of-the-art RL algorithms
   - Utilizes neural networks for policy and value estimation
   
   Specific RL techniques implemented:
   - **Deep Q-Networks (DQN)**
     - Uses a neural network to approximate the Q-function (expected future rewards)
     - Implements experience replay to break correlations between consecutive samples
     - Uses target networks to stabilize training by decoupling action selection and evaluation
     - Applies double DQN to reduce overestimation bias
     - Includes exploration strategies to balance exploration and exploitation
   
   - **Proximal Policy Optimization (PPO)**
     - Policy gradient method with clipped objective function to prevent destructively large updates
     - Balances exploration and exploitation through entropy regularization
     - Optimizes both policy and value functions simultaneously
     - Uses generalized advantage estimation (GAE) for more stable learning
     - Features multiple optimization epochs per batch of collected experience
   
   - **Monte Carlo Tree Search (MCTS)**
     - Combines tree search with random sampling to find optimal decisions
     - Uses Upper Confidence Bound (UCB) formula to balance exploration vs. exploitation
     - Implements the four key steps: selection, expansion, simulation, backpropagation
     - Integrates with neural networks similar to AlphaZero approach
     - Efficiently explores the state space by focusing on promising paths
   
   - **Curriculum Learning**
     - Gradually increases grid size during training to create a learning curriculum
     - Starts with smaller, simpler grids and progresses to larger ones
     - Transfers knowledge (neural network weights) from smaller to larger grid configurations
     - Implements adaptive difficulty adjustment based on agent performance
     - Uses success rate thresholds to determine when to increase problem difficulty

   - **Multi-Agent Reinforcement Learning**
     - Treats each cell as an agent with local observation (3x3 neighborhood)
     - Implements cooperative MARL where agents optimize global reward
     - Uses a Centralized Training with Decentralized Execution (CTDE) approach
     - Coordinates agent decisions through shared rewards
     - Handles the challenge of sparse rewards in grid-coloring problems

   - **Environment Implementation**
     - Custom Gymnasium environment implementation (LShapeRamseyEnv)
     - Properly defined observation and action spaces
     - Reward function designed to encourage valid grid colorings
     - Support for visualization and evaluation metrics
     - Configurable grid sizes and number of colors

## Setup

1. Clone the repository:
```bash
git clone https://github.com/MurtazaKafka/l-shape-ramsey-solver.git
cd l-shape-ramsey-solver
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

## Usage

1. Run the FunSearch implementation:
```bash
cd implementation
python run_l_shape_funsearch.py
```

2. Run the hierarchical solver:
```bash
python l_shape_hierarchical.py
```

3. Run the GPU-accelerated solver:
```bash
python l_shape_gpu.py
```

4. Run the Reinforcement Learning solver:
```bash
python l_shape_rl.py
```

## Project Structure

```
l-shape-ramsey-solver/
├── README.md
├── requirements.txt
├── l_shape_hierarchical.py
├── l_shape_gpu.py
├── l_shape_ramsey.py
├── l_shape_analysis.py
├── l_shape_funsearch.py
├── l_shape_rl.py
├── rl_models/
│   ├── dqn_agent.py
│   ├── ppo_agent.py
│   ├── mcts_agent.py
│   └── multiagent/
└── implementation/
    ├── run_l_shape_funsearch.py
    ├── sampler.py
    ├── evaluator.py
    ├── sandbox.py
    ├── programs_database.py
    ├── funsearch.py
    └── visualizations/
```

## Requirements

- Python 3.8+
- PyTorch
- NumPy
- Matplotlib
- Reinforcement Learning Libraries:
  - Stable Baselines3
  - Gymnasium (previously Gym)
  - TensorFlow (optional, for alternate RL implementations)
  - Ray (for distributed RL training)
- Other dependencies listed in requirements.txt

## Acknowledgments

This project builds upon DeepMind's FunSearch framework. The FunSearch implementation in the `implementation/` directory is based on the original work by DeepMind. The hierarchical and GPU-accelerated approaches are novel implementations developed independently.

- Original FunSearch paper: [FunSearch: Making the Impossible Possible](https://www.nature.com/articles/s41586-023-06924-6)
- DeepMind's FunSearch repository: [DeepMind FunSearch](https://github.com/google-deepmind/funsearch)

## License

MIT License
