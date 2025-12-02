# Deep Q-Learning (DQN) Lecture

Reinforcement Learning Master's Course, DIISM, Siena - December 10-11, held by PhD student Alberto Vaglio.

This repository contains the code and resources for the lecture on Deep Reinforcement Learning. We will explore how to solve the CartPole and Lunar Lander Gymnasium environments using Deep Q-Networks (DQN), harcoded from scratch, using mainly Pytorch for the ML framework.

## Repository Structure
- `naive_q_learning.py`: A simple Tabular Q-Learning implementation. We use this as a baseline to understand the limits of discretization.
- `dqn_cartpole.py`: The main script implementing the DQN algorithm with PyTorch (Experience Replay, Target Network).
- `eval_dqn.py`: Script to load a pre-trained model and visualize the agent playing.
- `dqn_cartpole.pth`: A pre-trained model checkpoint (so you can see it working immediately).
- `slides/`: PDF slides of the theoretical lecture.

## Installation
First, clone the repository:
```
git clone https://github.com/otr-ebla/phd-dqn-lectures.git
```
Create a python virtual environment:
```
python3 -m venv dqn_env
source dqn_env/bin/activate
```
Install the required python packages
```
pip install -r requirements.txt
```

## Usage
### 1. Training (DQN)
To train the agent from scratch:
python dqn_cartpole.py

The training process will save the model weights to `dqn_cartpole.pth` once solved.

### 2. Evaluation (Watch it play)
To see the agent in action using the pre-trained weights:
python eval_dqn.py

### 3. Baseline Comparison
To run the naive tabular approach:
python naive_q_learning.py

## Resources
- [Gymnasium Documentation](https://gymnasium.farama.org/)
- [PyTorch Reinforcement Learning Tutorial](https://pytorch.org/tutorials/intermediate/reinforcement_q_learning.html)
- [Reference paper: Playing Atari with Deep Reinforcement Learning](https://arxiv.org/pdf/1312.5602)
