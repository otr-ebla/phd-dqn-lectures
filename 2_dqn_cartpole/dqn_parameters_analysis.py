import random
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import gymnasium as gym
import matplotlib.pyplot as plt
from collections import deque

# --- (ReplayBuffer e DQN rimangono invariati, li riporto per completezza) ---
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

# --- Training Modulare ---

def run_experiment(exp_name, param_override={}):
    """
    Esegue un training DQN con parametri custom.
    param_override: dizionario con i parametri da sovrascrivere rispetto ai default.
    """
    env = gym.make('CartPole-v1')
    
    # Parametri di Default
    params = {
        'lr': 1e-3,
        'gamma': 0.99,
        'batch_size': 64,
        'target_update_freq': 100,
        'epsilon_decay': 2000, # NOTA: Aumentato per usare la formula esponenziale correttamente
        'total_episodes': 200,  # Ridotto per demo live veloce (CartPole converge presto)
        'buffer_capacity': 50000,  # Default
        'min_buffer_size': 1000,    # Default
    }
    # Sovrascriviamo con i parametri dell'esperimento
    params.update(param_override)
    
    print(f"--- Starting Experiment: {exp_name} with {param_override} ---")

    obs_dim = env.observation_space.shape[0]
    action_dim = env.action_space.n
    
    policy_net = DQN(obs_dim, action_dim)
    target_net = DQN(obs_dim, action_dim)
    target_net.load_state_dict(policy_net.state_dict())
    
    optimizer = optim.Adam(policy_net.parameters(), lr=params['lr'])
    replay_buffer = ReplayBuffer(params['buffer_capacity'])
    
    epsilon_start = 1.0
    epsilon_final = 0.05
    
    rewards_history = []
    steps_done = 0

    for episode in range(params['total_episodes']):
        state, _ = env.reset()
        episode_reward = 0  
        
        while True:
            # Calcolo Epsilon
            epsilon = epsilon_final + (epsilon_start - epsilon_final) * \
                      np.exp(-1. * steps_done / params['epsilon_decay'])

            action = select_action(policy_net, state, epsilon, action_dim)
            next_state, reward, terminated, truncated, _ = env.step(action)
            real_done = terminated or truncated

            replay_buffer.push(state, action, reward, next_state, real_done)
            state = next_state
            episode_reward += reward
            steps_done += 1

            if len(replay_buffer) >= params['min_buffer_size']:
                states, actions, rewards, next_states, dones = replay_buffer.sample(params['batch_size'])

                with torch.no_grad():
                    next_q = target_net(next_states).max(1)[0]
                    target = rewards + (1 - dones) * params['gamma'] * next_q

                q_values = policy_net(states)
                current = q_values.gather(1, actions.unsqueeze(1)).squeeze(1)

                loss = nn.MSELoss()(current, target)

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

            if steps_done % params['target_update_freq'] == 0:
                target_net.load_state_dict(policy_net.state_dict())

            if real_done:
                break
        
        rewards_history.append(episode_reward)
        
        # Log minimale ogni 20 episodi per non intasare la console
        if episode % 20 == 0:
            print(f"Ep {episode}: Reward {episode_reward:.0f} (Eps: {epsilon:.2f})")

    env.close()
    return rewards_history

def plot_comparison(results_dict):
    plt.figure(figsize=(10, 6))
    for name, rewards in results_dict.items():
        # Smoothing (media mobile) per rendere il grafico leggibile
        window = 10
        smoothed = np.convolve(rewards, np.ones(window)/window, mode='valid')
        plt.plot(smoothed, label=name)
    
    plt.xlabel('Episodes')
    plt.ylabel('Total Reward (Smoothed)')
    plt.title('DQN Hyperparameter Sensitivity')
    plt.legend()
    plt.grid(True)
    plt.show()

# --- ESEMPIO DI UTILIZZO PER LA LEZIONE ---
if __name__ == "__main__":
    # Esempio: Variamo Gamma
    # results = {}
    
    # # 1. Baseline
    # results['Baseline (Gamma 0.99)'] = run_experiment('Baseline')
    
    # # 2. Myopic Agent
    # results['Myopic (Gamma 0.5)'] = run_experiment('Myopic', {'gamma': 0.5})

    # # 3. Extreme Gamma (Opzionale)
    # results['Gamma (0.7)'] = run_experiment('Gamma', {'gamma': 0.7})
    
    #plot_comparison(results)

# Esempio: Variamo Learning Rate
    # results_lr = {}
    # results_lr['Baseline (LR 1e-3)'] = run_experiment('Baseline LR')
    # results_lr['High LR (1e-2)'] = run_experiment('High LR', {'lr': 1e-2})
    # results_lr['Low LR (1e-4)'] = run_experiment('Low LR', {'lr': 1e-4})
    # plot_comparison(results_lr)

# Esempio: Variamo Epsilon Decay

    # results_eps = {}
    # results_eps['Baseline (Decay 1000)'] = run_experiment('Baseline Epsilon Decay')
    # results_eps['Fast Decay (200)'] = run_experiment('Fast Decay', {'epsilon_decay': 200})
    # results_eps['Slow Decay (5000)'] = run_experiment('Slow Decay', {'epsilon_decay': 5000})
    # plot_comparison(results_eps)


    print("\n=== EXPERIMENT: REPLAY BUFFER SIZE ===")
    results_buffer = {}

    # CORREZIONE 2: Uso delle chiavi corrette (buffer_capacity)
    
    # 1. Baseline (Buffer medio)
    results_buffer['Baseline (10k)'] = run_experiment('Baseline', 
                                                      {'buffer_capacity': 10000})
    
    # 2. Buffer Piccolo (Dimentica troppo presto)
    # Nota: min_buffer_size deve essere <= buffer_capacity
    results_buffer['Small Buffer (500)'] = run_experiment('Small',
                                                    {'buffer_capacity': 500, 
                                                     'min_buffer_size': 200})
    
    # 3. Buffer Enorme (Molto stabile)
    results_buffer['Large Buffer (50k)'] = run_experiment('Large',
                                                    {'buffer_capacity': 50000})
                                                    
    plot_comparison(results_buffer)