import gymnasium as gym
import torch
import torch.nn as nn
import time
import os

# --- CONFIGURAZIONE ---
# Assicurati che questo nome corrisponda a quello salvato nel training!
MODEL_FILENAME = "models/ddqn_lunar_model.pth"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# --- DEFINIZIONE RETE (Deve essere identica al Training) ---
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

def watch_agent():
    # Verifica che il modello esista
    if not os.path.exists(MODEL_FILENAME):
        print(f"ERRORE: Non trovo il file '{MODEL_FILENAME}'.")
        print("Hai eseguito prima lo script di training?")
        return

    # Crea l'ambiente con render_mode='human' per vedere la finestra
    print("Inizializzazione ambiente grafico...")
    env = gym.make('LunarLander-v3', render_mode='human')
    
    obs_dim = env.observation_space.shape[0]
    action_dim = env.action_space.n
    
    # Carica il modello
    print(f"Caricamento modello da {MODEL_FILENAME}...")
    policy_net = DQN(obs_dim, action_dim).to(DEVICE)
    policy_net.load_state_dict(torch.load(MODEL_FILENAME, map_location=DEVICE))
    policy_net.eval() # Modalità valutazione (importante!)

    print("\n--- INIZIO VISUALIZZAZIONE ---")
    print("Premi Ctrl+C nel terminale per terminare.")
    
    try:
        episodes = 0
        while True:
            state, _ = env.reset()
            episode_reward = 0
            done = False
            
            while not done:
                # Prepara lo stato per la rete
                state_tensor = torch.tensor(state, dtype=torch.float32).unsqueeze(0).to(DEVICE)
                
                # Sceglie l'azione migliore (Greedy, senza Epsilon)
                with torch.no_grad():
                    q_values = policy_net(state_tensor)
                    action = q_values.argmax().item()
                
                # Esegue l'azione
                state, reward, terminated, truncated, _ = env.step(action)
                done = terminated or truncated
                episode_reward += reward
                
                # (Opzionale) Rallenta un po' se va troppo veloce
                # time.sleep(0.01) 
            
            episodes += 1
            print(f"Episodio {episodes} completato. Reward: {episode_reward:.2f}")
            
            # Pausa tra gli episodi
            time.sleep(1)

    except KeyboardInterrupt:
        print("\nVisualizzazione terminata dall'utente.")
    finally:
        env.close()

if __name__ == "__main__":
    watch_agent()