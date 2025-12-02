import gymnasium as gym
import numpy as np
import random
from collections import deque
import matplotlib.pyplot as plt

# Implementazione della rete neurale da zero
class NeuralNetwork:
    def __init__(self, input_size, hidden_size, output_size, lr=0.001):
        # Inizializzazione pesi con Xavier
        self.W1 = np.random.randn(input_size, hidden_size) * np.sqrt(2.0 / input_size)
        self.b1 = np.zeros((1, hidden_size))
        self.W2 = np.random.randn(hidden_size, hidden_size) * np.sqrt(2.0 / hidden_size)
        self.b2 = np.zeros((1, hidden_size))
        self.W3 = np.random.randn(hidden_size, output_size) * np.sqrt(2.0 / hidden_size)
        self.b3 = np.zeros((1, output_size))
        self.lr = lr
        
    def relu(self, x):
        return np.maximum(0, x)
    
    def relu_derivative(self, x):
        return (x > 0).astype(float)
    
    def forward(self, x):
        # Forward pass con cache per backprop
        self.z1 = np.dot(x, self.W1) + self.b1
        self.a1 = self.relu(self.z1)
        self.z2 = np.dot(self.a1, self.W2) + self.b2
        self.a2 = self.relu(self.z2)
        self.z3 = np.dot(self.a2, self.W3) + self.b3
        return self.z3
    
    def backward(self, x, y_true, y_pred):
        # Backpropagation
        m = x.shape[0]
        
        # Gradiente output layer
        dz3 = y_pred - y_true
        dW3 = np.dot(self.a2.T, dz3) / m
        db3 = np.sum(dz3, axis=0, keepdims=True) / m
        
        # Gradiente hidden layer 2
        da2 = np.dot(dz3, self.W3.T)
        dz2 = da2 * self.relu_derivative(self.z2)
        dW2 = np.dot(self.a1.T, dz2) / m
        db2 = np.sum(dz2, axis=0, keepdims=True) / m
        
        # Gradiente hidden layer 1
        da1 = np.dot(dz2, self.W2.T)
        dz1 = da1 * self.relu_derivative(self.z1)
        dW1 = np.dot(x.T, dz1) / m
        db1 = np.sum(dz1, axis=0, keepdims=True) / m
        
        # Update pesi
        self.W3 -= self.lr * dW3
        self.b3 -= self.lr * db3
        self.W2 -= self.lr * dW2
        self.b2 -= self.lr * db2
        self.W1 -= self.lr * dW1
        self.b1 -= self.lr * db1
    
    def copy_weights(self, other_network):
        """Copia i pesi da un'altra rete"""
        self.W1 = other_network.W1.copy()
        self.b1 = other_network.b1.copy()
        self.W2 = other_network.W2.copy()
        self.b2 = other_network.b2.copy()
        self.W3 = other_network.W3.copy()
        self.b3 = other_network.b3.copy()


# Replay Buffer
class ReplayBuffer:
    def __init__(self, capacity):
        self.buffer = deque(maxlen=capacity)
    
    def push(self, state, action, reward, next_state, done):
        self.buffer.append((state, action, reward, next_state, done))
    
    def sample(self, batch_size):
        batch = random.sample(self.buffer, batch_size)
        states, actions, rewards, next_states, dones = zip(*batch)
        return (np.array(states), np.array(actions), np.array(rewards),
                np.array(next_states), np.array(dones))
    
    def __len__(self):
        return len(self.buffer)


# Agente DQN
class DQNAgent:
    def __init__(self, state_size, action_size, hidden_size=128):
        self.state_size = state_size
        self.action_size = action_size
        
        # Iperparametri
        self.gamma = 0.99  # Discount factor
        self.epsilon = 1.0  # Exploration rate
        self.epsilon_min = 0.01
        self.epsilon_decay = 0.995
        self.learning_rate = 0.001
        self.batch_size = 128
        self.target_update = 5  # Ogni quanti episodi aggiornare la target network
        
        # Reti neurali
        self.policy_net = NeuralNetwork(state_size, hidden_size, action_size, self.learning_rate)
        self.target_net = NeuralNetwork(state_size, hidden_size, action_size, self.learning_rate)
        self.target_net.copy_weights(self.policy_net)
        
        # Replay buffer
        self.memory = ReplayBuffer(10000)
    
    def select_action(self, state, training=True):
        """Epsilon-greedy action selection"""
        if training and random.random() < self.epsilon:
            return random.randrange(self.action_size)
        
        state = state.reshape(1, -1)
        q_values = self.policy_net.forward(state)
        return np.argmax(q_values[0])
    
    def train(self):
        """Training step"""
        if len(self.memory) < self.batch_size:
            return
        
        # Sample minibatch
        states, actions, rewards, next_states, dones = self.memory.sample(self.batch_size)
        
        # Calcola Q-values correnti
        current_q_values = self.policy_net.forward(states)
        
        # Calcola target Q-values
        next_q_values = self.target_net.forward(next_states)
        max_next_q_values = np.max(next_q_values, axis=1)
        
        # Calcola target
        target_q_values = current_q_values.copy()
        for i in range(self.batch_size):
            if dones[i]:
                target_q_values[i, actions[i]] = rewards[i]
            else:
                target_q_values[i, actions[i]] = rewards[i] + self.gamma * max_next_q_values[i]
        
        # Backpropagation
        self.policy_net.backward(states, target_q_values, current_q_values)
    
    def update_target_network(self):
        """Aggiorna la target network"""
        self.target_net.copy_weights(self.policy_net)
    
    def decay_epsilon(self):
        """Decay dell'epsilon"""
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay


# Training loop
def train_dqn(episodes=500, render_every=100):
    env = gym.make('CartPole-v1')
    state_size = env.observation_space.shape[0]
    action_size = env.action_space.n
    
    agent = DQNAgent(state_size, action_size)
    scores = []
    avg_scores = []
    
    print("Inizio training DQN su CartPole-v1")
    print(f"State size: {state_size}, Action size: {action_size}")
    print("-" * 60)
    
    for episode in range(episodes):
        state, _ = env.reset()
        total_reward = 0
        done = False
        
        while not done:
            # Seleziona azione
            action = agent.select_action(state)
            
            # Esegui azione
            next_state, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            
            # Memorizza transizione
            agent.memory.push(state, action, reward, next_state, done)
            
            # Training
            agent.train()
            
            state = next_state
            total_reward += reward
        
        # Update target network
        if episode % agent.target_update == 0:
            agent.update_target_network()
        
        # Decay epsilon
        agent.decay_epsilon()
        
        # Statistiche
        scores.append(total_reward)
        avg_score = np.mean(scores[-100:])
        avg_scores.append(avg_score)
        
        if episode % 10 == 0:
            print(f"Episode {episode:4d} | Score: {total_reward:6.2f} | "
                  f"Avg(last 100 ep.s): {avg_score:6.2f} | Epsilon: {agent.epsilon:.3f}")

        # Considera il problema risolto se la media degli ultimi 100 episodi è >= 495
        if avg_score >= 470.0 and episode >= 100:
            print(f"\n🎉 Problema risolto in {episode} episodi!")
            print(f"Media ultimi 100 episodi: {avg_score:.2f}")
            break
    
    env.close()
    
    # Plot risultati
    plt.figure(figsize=(12, 5))
    
    plt.subplot(1, 2, 1)
    plt.plot(scores, alpha=0.6, label='Score per episodio')
    plt.plot(avg_scores, linewidth=2, label='Media mobile (100 ep)')
    plt.axhline(y=470, color='r', linestyle='--', label='Target (470)')
    plt.xlabel('Episodio')
    plt.ylabel('Score')
    plt.title('Training DQN su CartPole-v1')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.subplot(1, 2, 2)
    plt.plot(avg_scores, linewidth=2)
    plt.axhline(y=470, color='r', linestyle='--', label='Target')
    plt.xlabel('Episodio')
    plt.ylabel('Score medio (100 episodi)')
    plt.title('Progresso del Training')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('dqn_training.png', dpi=150, bbox_inches='tight')
    plt.show()
    
    return agent, scores


# Test dell'agente addestrato
def test_agent(agent, episodes=10):
    env = gym.make('CartPole-v1', render_mode='human')
    
    print("\n" + "=" * 60)
    print("Testing dell'agente addestrato")
    print("=" * 60)
    
    for episode in range(episodes):
        state, _ = env.reset()
        total_reward = 0
        done = False
        
        while not done:
            action = agent.select_action(state, training=False)
            state, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            total_reward += reward
        
        print(f"Test Episode {episode + 1}: Score = {total_reward}")
    
    env.close()


if __name__ == "__main__":
    # Training
    agent, scores = train_dqn(episodes=1_500)
    
    # Test con visualizzazione
    print("\n🎮 Ora mostro l'agente addestrato in azione...")
    input("Premi ENTER per iniziare la visualizzazione...")
    test_agent(agent, episodes=5)
    
    print("\n✅ Training e test completati! Grafico salvato come 'dqn_training.png'")