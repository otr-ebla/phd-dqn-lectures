import gymnasium as gym
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import random
import matplotlib.pyplot as plt


# Q Network
class QNetwork(nn.Module):
    def __init__(self, state_dim, action_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim, 128), nn.ReLU(),
            nn.Linear(128, 128), nn.ReLU(),
            nn.Linear(128, action_dim)
        )

    def forward(self, x):
        return self.net(x)
    

def eps_greedy(q_net, state, epsilon, env):
    # with prob eps, ignore the network and take a random action
    if random.random() < epsilon:
        return env.action_space.sample()
    
    state_t = torch.FloatTensor(state).unsqueeze(0) # converts the state into a pytorch tensor, adding a batch size at the front
    q_values = q_net(state_t).detach().numpy().squeeze() # get q values from the network, without needing gradients, and removing a batch dimension
    return np.argmax(q_values) # select best action

def run_render_episode(q_net, render_env):
    state, _ = render_env.reset()
    done = False
    total_reward = 0

    while not done:
        # Greedy action (no epsilon here!)
        state_t = torch.FloatTensor(state).unsqueeze(0)
        with torch.no_grad():
            action = q_net(state_t).argmax(1).item()

        next_state, reward, terminated, truncated, _ = render_env.step(action)
        done = terminated or truncated
        total_reward += reward
        state = next_state
    return total_reward


# -----------
# Naive Deep Q Learning
# -----------

env = gym.make('CartPole-v1')
#env = gym.wrappers.TimeLimit(env )   # or any number
env.unwrapped.theta_threshold_radians = np.deg2rad(60)


render_env = gym.make("CartPole-v1", render_mode="human", max_episode_steps=2000)
render_env = gym.wrappers.TimeLimit(render_env, max_episode_steps=2000)
render_env.unwrapped.theta_threshold_radians = np.deg2rad(60)

render_episodes = 10


state_dim = env.observation_space.shape[0]
action_dim = env.action_space.n

q = QNetwork(state_dim, action_dim)
print("Showing environment BEFORE training...")
for i in range(render_episodes):
    run_render_episode(q, render_env)
input("\nPress ENTER in the terminal to start training...\n")


optimizer = optim.Adam(q.parameters(), lr=1e-3)

gamma = 0.99
epsilon = 1.0
epsilon_decay = 0.995
epsilon_min = 0.05

num_episodes = 2000

loss_history = []
loss_steps = []
episodes_rewards = []
global_step = 0
plt.ion()
fig, (ax_loss, ax_reward) = plt.subplots(2, 1, figsize=(8, 8))

line_loss, = ax_loss.plot([], [], label='Loss')
ax_loss.set_xlabel('Training steps')
ax_loss.set_ylabel('Loss')
ax_loss.set_title("Naive Deep Q-Learning Loss")

line_reward, = ax_reward.plot([], [], label='Episode Reward', color='orange')
ax_reward.set_xlabel('Episodes')
ax_reward.set_ylabel('Return')
ax_reward.set_title("Naive Deep Q-learning: Episode Reward")

ax_loss.legend()
ax_reward.legend()

plt.tight_layout()
plt.show(block=False)

for episode in range(num_episodes): 

    state, _ = env.reset()
    done = False
    ep_reward = 0

    while not done:

        action = eps_greedy(q, state, epsilon, env)

        next_state, reward, terminated, truncated, _ = env.step(action)
        done = terminated or truncated

        ep_reward += reward

        # Compute target with the same network (naive DQL) UNSTABLE!!!!
        with torch.no_grad():
            next_state_t = torch.FloatTensor(next_state).unsqueeze(0)
            target = reward + gamma * q(next_state_t).max().item()

        state_t = torch.FloatTensor(state).unsqueeze(0) 
        q_values = q(state_t)
        q_value = q_values[0, action]

        loss = (q_value - target) ** 2 # minimize squared TD error

        # Gradient step
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()    

        state = next_state

        global_step += 1

        loss_history.append(loss.item())    
        loss_steps.append(global_step)

    # Epsilon decay
    epsilon = max(epsilon * epsilon_decay, epsilon_min)
    episodes_rewards.append(ep_reward)

    print(f"Episode {episode+1}, Reward: {ep_reward}, Epsilon: {epsilon:.3f}")

    # Optional: visual check every 100 episodes
    if episode % 1000 == 0 and episode > 0:
        print(f"\n🔍 Watching performance at episode {episode}...")
        for i in range(render_episodes):
            score = run_render_episode(q, render_env)
            print(f"Score: {score}\n")
    
        # --- Live plotting every 10 episodes (tweak as you like) ---
    if episode % 50 == 0:
        # Update loss line
        line_loss.set_data(loss_steps, loss_history)
        ax_loss.relim()
        ax_loss.autoscale_view()

        # Update reward line
        line_reward.set_data(range(len(episodes_rewards)), episodes_rewards)
        ax_reward.relim()
        ax_reward.autoscale_view()

        plt.pause(0.001)  # let matplotlib update the window


print("\n\n\n\n🎉 FINAL PERFORMANCE:\n\n\n")
for i in range(render_episodes):
    final_score = run_render_episode(q, render_env)
    print(f"Final Score: {final_score}")

plt.ioff()
plt.show()

