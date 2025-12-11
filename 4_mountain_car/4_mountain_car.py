import gymnasium as gym
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import random
import matplotlib.pyplot as plt
from collections import deque
import os

# --- 0. CONFIGURATION & IMPORTS ---
try:
    from stable_baselines3 import DQN as SB3_DQN
    from stable_baselines3.common.callbacks import BaseCallback
    SB3_AVAILABLE = True
except ImportError:
    print("⚠️ Stable Baselines3 not found. Run 'pip install stable-baselines3'")
    SB3_AVAILABLE = False

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# --- 1. REWARD WRAPPER (CRITICAL FIX) ---
class MountainCarRewardWrapper(gym.RewardWrapper):
    def __init__(self, env):
        super().__init__(env)
    
    def reward(self, reward):
        # Give bonus for velocity to encourage swinging
        return reward + 100 * abs(self.env.unwrapped.state[1])

# --- 2. ARCHITECTURE ---
class ReplayBuffer:
    def __init__(self, capacity):
        self.buffer = deque(maxlen=capacity)
    def push(self, state, action, reward, next_state, done):
        self.buffer.append((state, action, reward, next_state, done))
    def sample(self, batch_size):
        batch = random.sample(self.buffer, batch_size)
        states, actions, rewards, next_states, dones = map(np.array, zip(*batch))
        return (torch.tensor(states, dtype=torch.float32).to(DEVICE),
                torch.tensor(actions, dtype=torch.int64).to(DEVICE),
                torch.tensor(rewards, dtype=torch.float32).to(DEVICE),
                torch.tensor(next_states, dtype=torch.float32).to(DEVICE),
                torch.tensor(dones, dtype=torch.float32).to(DEVICE))
    def __len__(self): return len(self.buffer)

class DQN(nn.Module):
    def __init__(self, state_dim, action_dim):
        super(DQN, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU(),
            nn.Linear(64, action_dim)
        )
    def forward(self, x): return self.net(x)

def select_action(model, state, epsilon, act_dim):
    if random.random() < epsilon: return random.randrange(act_dim)
    with torch.no_grad():
        state = torch.tensor(state, dtype=torch.float32).unsqueeze(0).to(DEVICE)
        return int(model(state).argmax().item())

# --- 3. CUSTOM TRAINING LOOP ---
def run_custom_dqn(exp_name, double_dqn=False, total_episodes=400):
    # Apply Wrapper
    env = gym.make('MountainCar-v0')
    env = MountainCarRewardWrapper(env)
    
    obs_dim = env.observation_space.shape[0]
    action_dim = env.action_space.n
    
    # Tuned Hyperparams
    lr = 1e-3
    gamma = 0.99
    epsilon_decay = 50_000 # Slower decay!
    min_buffer = 1000
    
    policy_net = DQN(obs_dim, action_dim).to(DEVICE)
    target_net = DQN(obs_dim, action_dim).to(DEVICE)
    target_net.load_state_dict(policy_net.state_dict())
    
    optimizer = optim.Adam(policy_net.parameters(), lr=lr)
    buffer = ReplayBuffer(20000)
    
    epsilon = 1.0
    steps_done = 0
    rewards_history = []
    
    print(f"🚀 Running {exp_name} (Double={double_dqn})...")
    
    for ep in range(total_episodes):
        state, _ = env.reset()
        ep_reward = 0
        done = False
        
        while not done:
            epsilon = max(0.05, 1.0 - (steps_done / epsilon_decay))
            
            action = select_action(policy_net, state, epsilon, action_dim)
            next_state, reward, term, trunc, _ = env.step(action)
            done = term or trunc
            
            buffer.push(state, action, reward, next_state, done)
            state = next_state
            ep_reward += reward
            steps_done += 1
            
            if len(buffer) >= min_buffer:
                states, actions, rewards, next_states, dones = buffer.sample(128)
                
                with torch.no_grad():
                    if double_dqn:
                        best_actions = policy_net(next_states).argmax(1).unsqueeze(1)
                        next_q = target_net(next_states).gather(1, best_actions).squeeze(1)
                    else:
                        next_q = target_net(next_states).max(1)[0]
                    target = rewards + (1-dones) * gamma * next_q
                
                curr_q = policy_net(states).gather(1, actions.unsqueeze(1)).squeeze(1)
                loss = nn.SmoothL1Loss()(curr_q, target)
                
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                
            if steps_done % 200 == 0:
                target_net.load_state_dict(policy_net.state_dict())

        rewards_history.append(ep_reward)
        if ep % 50 == 0: print(f"Ep {ep}: Reward {ep_reward:.1f} | Eps: {epsilon:.2f}")
        
    env.close()

    # SAVE MODEL
    if not os.path.exists("models"): os.makedirs("models")
    safe_name = exp_name.replace(" ", "_").lower()
    save_path = f"models/{safe_name}.pth"
    torch.save(policy_net.state_dict(), save_path)
    print(f"💾 Policy saved to {save_path}")

    return rewards_history

# --- 4. SB3 BENCHMARK ---
class LogCallback(BaseCallback):
    def __init__(self):
        super().__init__(verbose=0)
        self.rewards = []
    def _on_step(self):
        if "dones" in self.locals and self.locals["dones"][0]:
            self.rewards.append(self.locals["infos"][0]["episode"]["r"])
        return True

def run_sb3_dqn(total_episodes=400):
    print("🚀 Running Stable Baselines3 DQN...")
    env = gym.make('MountainCar-v0')
    env = MountainCarRewardWrapper(env) # Same wrapper!
    env = gym.wrappers.RecordEpisodeStatistics(env) 
    
    model = SB3_DQN("MlpPolicy", env, verbose=0, learning_rate=1e-3, buffer_size=20000, exploration_fraction=0.5)
    
    callback = LogCallback()
    model.learn(total_timesteps=total_episodes * 200, callback=callback)
    
    if not os.path.exists("models"): os.makedirs("models")
    model.save("models/sb3_mountaincar")
    print("💾 SB3 Model saved to models/sb3_mountaincar.zip")

    return callback.rewards

# --- 5. PLOTTING & SAVING ---
def plot_and_save(results, filename="mountaincar_comparison.png", window=20):
    plt.figure(figsize=(10, 6))
    
    for name, rewards in results.items():
        if not rewards: continue
        if len(rewards) >= window:
            smooth = np.convolve(rewards, np.ones(window)/window, mode='valid')
            plt.plot(smooth, label=name, linewidth=2)
        else:
            plt.plot(rewards, label=name, alpha=0.6)
            
    plt.title("MountainCar-v0: Impact of Reward Shaping")
    plt.xlabel("Episode")
    plt.ylabel("Reward (Smoothed)")
    plt.axhline(-110, color='green', linestyle='--', alpha=0.3, label="Solved Threshold")
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # SAVE BEFORE SHOW
    plt.savefig(filename)
    print(f"\n💾 Plot saved to {filename}")
    
    plt.show()

if __name__ == "__main__":
    N_EPISODES = 500
    results = {}
    
    # Run Experiments
    results['Custom DQN'] = run_custom_dqn('Standard DQN', False, N_EPISODES)
    results['Custom DDQN'] = run_custom_dqn('Double DQN', True, N_EPISODES)
    
    if SB3_AVAILABLE:
        results['Stable Baselines3'] = run_sb3_dqn(N_EPISODES)
    
    # Plot and Save
    plot_and_save(results)