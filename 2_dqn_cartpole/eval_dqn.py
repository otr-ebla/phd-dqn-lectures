import torch
import gymnasium as gym
from dqn_cartpoleTORCH import DQN

def evaluate():
    env = gym.make("CartPole-v1", render_mode="human")
    obs_dim = env.observation_space.shape[0]
    act_dim = env.action_space.n

    model = DQN(obs_dim, act_dim)
    model.load_state_dict(torch.load("dqn_cartpole.pth"))
    model.eval()

    for episode in range(5):
        state, _ = env.reset()
        total_reward = 0
        done = False

        while not done:
            with torch.no_grad():
                s = torch.tensor(state, dtype=torch.float32)
                action = int(torch.argmax(model(s)))

            state, reward, done, truncated, _ = env.step(action)
            total_reward += reward
            if done or truncated:
                break

        print(f"Episode {episode}: reward {total_reward}")

    env.close()

if __name__ == "__main__":
    evaluate()