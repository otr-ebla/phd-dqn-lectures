import random
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import gymnasium as gym
import matplotlib.pyplot as plt
from collections import deque

class ReplayBuffer:
    def __init__(self, capacity):
        self.buffer = deque(maxlen=capacity)

    def push(self, state, action, reward, next_state, done):
        self.buffer.append((state, action, reward, next_state, done))

    def sample(self, batch_size):
        batch = random.sample(self.buffer, batch_size)
        states, actions, rewards, next_states, dones = map(np.array, zip(*batch))
        return (torch.tensor(states, dtype=torch.float32),
                torch.tensor(actions, dtype=torch.int64),
                torch.tensor(rewards, dtype=torch.float32),
                torch.tensor(next_states, dtype=torch.float32),
                torch.tensor(dones, dtype=torch.float32))
    
    def __len__(self):
        return len(self.buffer)
    
class DQN(nn.Module):
    def __init__(self, state_dim, action_dim):
        super(DQN, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, action_dim)
        )

    def forward(self, x):
        return self.net(x)
    
def select_action(model, state, epsilon, act_dim):
    if random.random() < epsilon:
        return random.randrange(act_dim)
    with torch.no_grad():
        state = torch.tensor(state, dtype=torch.float32).unsqueeze(0)
        q_values = model(state)
        return int(torch.argmax(q_values))
    
def train_dqn():
    env = gym.make('CartPole-v1')
    obs_dim = env.observation_space.shape[0]
    action_dim = env.action_space.n

    lr = 1e-3
    gamma = 0.99
    batch_size = 64
    buffer_capacity = 50_000
    min_buffer_size = 1_000
    total_episodes = 500
    epsilon_start = 1.0
    epsilon_final = 0.05
    epsilon_decay = 0.995
    target_update_freq = 10

    max_steps = 2000

    # 2 networks: policy and target
    policy_net = DQN(obs_dim, action_dim)
    target_net = DQN(obs_dim, action_dim)

    target_net.load_state_dict(policy_net.state_dict())
    target_net.eval()

    optimizer = optim.Adam(policy_net.parameters(), lr=lr)
    replay_buffer = ReplayBuffer(buffer_capacity)

    steps_done = 0
    for episode in range(total_episodes):
        state, _ = env.reset()
        episode_reward = 0  

        for step in range(max_steps):
            epsilon = epsilon_final + (epsilon_start - epsilon_final)*np.exp(-steps_done/epsilon_decay)

            action = select_action(policy_net, state, epsilon, action_dim)

            next_state, reward, terminated, truncated, _ = env.step(action)
            real_done = terminated or truncated

            replay_buffer.push(state, action, reward, next_state, real_done)
            state = next_state
            episode_reward += reward
            steps_done += 1

            # Start training after filling the buffer
            if len(replay_buffer) >= min_buffer_size:
                states, actions, rewards, next_states, dones = replay_buffer.sample(batch_size)

                with torch.no_grad():
                    next_q = target_net(next_states).max(1)[0]
                    target = rewards + (1 - dones) * gamma * next_q

                q_values = policy_net(states)
                current = q_values.gather(1, actions.unsqueeze(1)).squeeze(1)

                loss = nn.MSELoss()(current, target)

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

            if steps_done % target_update_freq == 0:
                target_net.load_state_dict(policy_net.state_dict())

            if real_done:
                break

        print(f"Episode {episode + 1}: Total Reward: {episode_reward}")

    env.close()
    torch.save(policy_net.state_dict(), "dqn_cartpole.pth")
    print("Training complete and model saved.")

if __name__ == "__main__":
    train_dqn()