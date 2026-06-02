Overview
This project implements the Deep Q-Network (DQN) algorithm (Mnih et al., 2015) from scratch and evaluates it on the CartPole-v1 benchmark — 
a foundational control problem widely used in robotics and autonomous systems research to validate policy learning approaches.
The core investigation focuses on three mechanisms: experience replay, a target network, and a custom reward shaping strategy using a terminal 
penalty. Together, these produced a stable, converging agent that solves the environment in under 700 episodes.
The trained agent solves CartPole-v1 (mean reward ≥ 475 over 100 episodes) at episode ~660, reaching the maximum possible score of 500 upon full convergence.

Background & Motivation
Classical Q-learning fails with neural network function approximators due to:

Correlated samples — consecutive transitions are highly correlated, violating the i.i.d. assumption of gradient-based optimisers.
Non-stationary targets — the Q-target shifts as the same network being updated also computes target values, causing divergence.

DQN addresses both with:

A replay buffer that breaks temporal correlations by sampling random mini-batches from stored transitions.
A target network (frozen copy of the Q-network) providing stable regression targets, updated periodically.

This work additionally introduces reward shaping to accelerate convergence, a technique directly applicable to safety-critical robotic systems where penalising failure 
states is essential.

Methodology
Algorithm
The agent follows the standard DQN update rule:
Q(s, a) ← r_shaped + γ · max_a' Q_target(s', a')
where Q_target is frozen and updated periodically, and r_shaped is the shaped reward defined below.

Environment
Property              Value      
Environment           CartPole-v1 (OpenAI Gymnasium)
State space           4 continuous variables (cart position, cart velocity, pole angle, pole angular velocity)
Action space          Discrete — 0:push left, 1: push right
Solved threshold      Mean reward ≥ 475 over 100 consecutive episodes

Reward Shaping
A key contribution of this implementation is the modified reward function:
Signal                        Value            Rationale
Standard step reward           +1              Per timestep the pole remains balanced
Terminal failure penalty       −100            Applied when episode ends early (pole falls)

The large negative penalty creates a strong gradient signal that forces the agent to prioritise pole stability immediately, 
significantly reducing the number of episodes required to 
converge compared to standard DQN benchmarks.

Network Architecture
Layer            Type                  Output size
Input            Linear                4 → 64
Hidden 1         Linear + ReLU         64 → 64
Output           Linear                64 → 2 (Q-values per action)

Hyperparameters
Parameter              Value                Justification
Discount factor γ      0.99                 Prioritises long-term pole balancing
Mini-batch size        64                   Standard size for stochastic gradient descent
ε decay rate           0.995                Slow decay allows sufficient exploration
ε start / end          1.0 / 0.0 1          Full exploration → near-greedy policy
Optimiser              Adam                 —



Project Structure
DQN-CartPole/
├── dqn_cartpole.py        # Main training script
├── requirements.txt       # Python dependencies
├── assets/
│   └── training_curve.png # Training reward plot (add this)
└── README.md

Installation & Reproduction
bashgit clone https://github.com/mosesoluma/DQN-CartPole.git
cd DQN-CartPole
pip install -r requirements.txt
python dqn_cartpole.py
requirements.txt
torch>=2.0.0
gymnasium>=0.29.0
numpy>=1.24.0
matplotlib>=3.7.0
Training runs until the environment is solved and saves a reward plot to assets/training_curve.png.

Connection to Robotics & Autonomy
While CartPole is a simulation benchmark, the principles explored here map directly to real autonomous systems:

Terminal penalty design mirrors safety constraints in robot locomotion and UAV stabilisation, where failure states must be strongly discouraged.
Stability under policy updates (via the target network) is critical for real-time robotic control where diverging Q-values can cause unsafe behaviour.
ε-greedy exploration parallels the exploration-exploitation tradeoff in autonomous navigation, such as frontier-based exploration in mobile robots.

This work serves as a foundation toward applying DRL to continuous control tasks (MuJoCo, ROS/Gazebo) in my research agenda on autonomous systems.

References
Mnih, V., et al. (2015). Human-level control through deep reinforcement learning.
Nature, 518(7540), 529–533.

Sutton, R. S., & Barto, A. G. (2018). Reinforcement Learning: An Introduction (2nd ed.).
MIT Press.

Author
Moses Oluma — Master's Graduate, Robotics & Autonomous Systems
Kumoh National Institute of Technology, Republic of Korea
GitHub · Academic Site
