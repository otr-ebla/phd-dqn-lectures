import gymnasium as gym
import torch
import torch.nn as nn
import numpy as np
import time

# --- CONFIGURATION ---
MODEL_PATH = "models/double_dqn.pth"  # Ensure this matches the saved filename
DEVICE = torch.device("cpu") # CPU is fine for inference

# --- 1. DEFINE ARCHITECTURE (Must Match Training) ---
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

def visualize():
    # 1. Setup Environment (No Reward Wrapper needed for Vis)
    # We use render_mode="human" to see the window
    env = gym.make('MountainCar-v0', render_mode="human")
    
    obs_dim = env.observation_space.shape[0]
    action_dim = env.action_space.n

    # 2. Load Model
    model = DQN(obs_dim, action_dim).to(DEVICE)
    try:
        model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
        model.eval() # Set to evaluation mode
        print(f"✅ Loaded model from {MODEL_PATH}")
    except FileNotFoundError:
        print(f"❌ Model not found at {MODEL_PATH}. Did you run the training script first?")
        return

    # 3. Run Loop
    episodes = 5
    print(f"\n🎥 Running {episodes} episodes...")
    
    for ep in range(episodes):
        state, _ = env.reset()
        done = False
        total_reward = 0
        
        while not done:
            # Prepare state
            state_t = torch.tensor(state, dtype=torch.float32).unsqueeze(0).to(DEVICE)
            
            # Select Action (Greedy / No Randomness)
            with torch.no_grad():
                action = model(state_t).argmax().item()
            
            state, reward, term, trunc, _ = env.step(action)
            done = term or trunc
            total_reward += reward
            
            # Tiny sleep to make visualization smoother to human eye (optional)
            # time.sleep(0.01) 

        print(f"Episode {ep+1}: Score = {total_reward:.1f}")

    env.close()

if __name__ == "__main__":
    visualize()