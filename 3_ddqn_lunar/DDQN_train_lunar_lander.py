import random
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import gymnasium as gym
from collections import deque
import matplotlib.pyplot as plt

# --- CONFIGURAZIONE ---
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# Nota: il nome del file verrà deciso dinamicamente nella funzione

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
        state = torch.tensor(state, dtype=torch.float32).unsqueeze(0).to(DEVICE)
        q_values = model(state)
        return int(torch.argmax(q_values).item())

def run_lunar_experiment(exp_name, double_dqn=True, total_episodes=400):
    env = gym.make('LunarLander-v3')
    
    # Iperparametri
    lr = 5e-4
    gamma = 0.99
    batch_size = 128
    target_update_freq = 200
    epsilon_decay = 10000
    buffer_capacity = 100000
    min_buffer_size = 1000
    
    print(f"--- Starting: {exp_name} | Double: {double_dqn} ---")
    print(f"Device: {DEVICE}")

    obs_dim = env.observation_space.shape[0]
    action_dim = env.action_space.n
    
    policy_net = DQN(obs_dim, action_dim).to(DEVICE)
    target_net = DQN(obs_dim, action_dim).to(DEVICE)
    target_net.load_state_dict(policy_net.state_dict())
    
    optimizer = optim.Adam(policy_net.parameters(), lr=lr)
    replay_buffer = ReplayBuffer(buffer_capacity)
    
    epsilon_start = 1.0
    epsilon_final = 0.01
    steps_done = 0

    # Liste per il salvataggio dei risultati
    rewards_history = []
    q_value_history = []

    try:
        for episode in range(total_episodes):
            state, _ = env.reset()
            episode_reward = 0
            episode_q_vals = [] # Per tracciare il bias medio nell'episodio
            
            while True:
                # Epsilon Decay
                epsilon = epsilon_final + (epsilon_start - epsilon_final) * \
                        np.exp(-1. * steps_done / epsilon_decay)

                action = select_action(policy_net, state, epsilon, action_dim)
                next_state, reward, terminated, truncated, _ = env.step(action)
                real_done = terminated or truncated

                replay_buffer.push(state, action, reward, next_state, real_done)
                state = next_state
                episode_reward += reward
                steps_done += 1

                # Training Step
                if len(replay_buffer) >= min_buffer_size:
                    states, actions, rewards, next_states, dones = replay_buffer.sample(batch_size)

                    with torch.no_grad():
                        if double_dqn:
                            # Double DQN Logic
                            best_actions = policy_net(next_states).argmax(1)
                            next_q = target_net(next_states).gather(1, best_actions.unsqueeze(1)).squeeze(1)
                        else:
                            # Standard DQN Logic
                            next_q = target_net(next_states).max(1)[0]
                        
                        target = rewards + (1 - dones) * gamma * next_q

                    q_values = policy_net(states)
                    current = q_values.gather(1, actions.unsqueeze(1)).squeeze(1)
                    
                    # Raccogliamo il valore Q medio per analisi (Maximization Bias)
                    episode_q_vals.append(current.mean().item())

                    #loss = nn.MSELoss()(current, target)
                    loss = nn.SmoothL1Loss()(current, target)
                    optimizer.zero_grad()
                    loss.backward()
                    torch.nn.utils.clip_grad_value_(policy_net.parameters(), 1.0)
                    optimizer.step()

                if steps_done % target_update_freq == 0:
                    target_net.load_state_dict(policy_net.state_dict())

                if real_done:
                    break
            
            # Fine episodio: salviamo metriche
            avg_ep_q = np.mean(episode_q_vals) if episode_q_vals else 0
            rewards_history.append(episode_reward)
            q_value_history.append(avg_ep_q)

            if episode % 20 == 0:
                print(f"Ep {episode}: Reward {episode_reward:.2f} | Avg Q: {avg_ep_q:.2f} | Eps: {epsilon:.2f}")

    except KeyboardInterrupt:
        print("\nTraining interrotto manualmente.")

    env.close()
    
    # Salvataggio modello
    if double_dqn:
        filename = "models/ddqn_double_lunar_model.pth"
    else:
        filename = "models/ddqn_lunar_model.pth"
        
    torch.save(policy_net.state_dict(), filename)
    print(f"\nModello salvato in: {filename}")
    
    return rewards_history, q_value_history

def plot_dual_comparison(results):
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 10))
    
    window = 20
    
    for name, (rewards, q_vals) in results.items():
        # Plot Rewards
        if len(rewards) >= window:
            smooth_r = np.convolve(rewards, np.ones(window)/window, mode='valid')
            ax1.plot(smooth_r, label=name, linewidth=2)
        else:
            ax1.plot(rewards, label=name)
            
        # Plot Q-Values (Maximization Bias)
        if len(q_vals) >= window:
            smooth_q = np.convolve(q_vals, np.ones(window)/window, mode='valid')
            ax2.plot(smooth_q, label=name, linewidth=2)
        else:
            ax2.plot(q_vals, label=name)

    ax1.set_title("Performance (Total Reward)")
    ax1.set_ylabel("Score")
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    ax2.set_title("Maximization Bias (Estimated Q-Values)")
    ax2.set_ylabel("Avg Q-Value")
    ax2.set_xlabel("Episodes")
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    results = {}
    
    # Nota: 300-350 episodi sono sufficienti per vedere la divergenza dei Q-Values
    N_EPISODES = 500
    
    print("Collecting data for Standard DQN...")
    res_dqn = run_lunar_experiment('DQN', double_dqn=False, total_episodes=N_EPISODES)
    results['Standard DQN'] = res_dqn
    
    print("\nCollecting data for Double DQN...")
    res_ddqn = run_lunar_experiment('Double DQN', double_dqn=True, total_episodes=N_EPISODES)
    results['Double DQN'] = res_ddqn
    
    plot_dual_comparison(results)