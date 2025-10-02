import argparse
import time
import torch
import torch.nn as nn
import gymnasium as gym
import numpy as np


# --- Dueling DQN síť (musí odpovídat train.py!) ---
class DuelingQNetwork(nn.Module):
    def __init__(self, state_dim: int, action_dim: int, hidden=(256, 256)):
        super().__init__()
        
        # Shared feature extractor
        layers = []
        input_dim = state_dim
        for h in hidden:
            layers.append(nn.Linear(input_dim, h))
            layers.append(nn.ReLU())
            input_dim = h
        self.features = nn.Sequential(*layers)
        
        # Value stream
        self.value_stream = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 1)
        )
        
        # Advantage stream
        self.advantage_stream = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.ReLU(),
            nn.Linear(128, action_dim)
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        features = self.features(x)
        value = self.value_stream(features)
        advantages = self.advantage_stream(features)
        q_values = value + (advantages - advantages.mean(dim=1, keepdim=True))
        return q_values


# --- Evaluační funkce ---
def evaluate(model_path, episodes=5, render=False, delay=0.0, verbose=True):
    """
    Evaluace natrénovaného modelu.
    
    Args:
        model_path: cesta k .pth souboru
        episodes: počet epizod k vyhodnocení
        render: zobrazit vizualizaci
        delay: zpoždění mezi kroky (v sekundách)
        verbose: vypisovat detaily každého kroku
    """
    # Vytvoření prostředí
    if render:
        env = gym.make("LunarLander-v3", render_mode="human")
    else:
        env = gym.make("LunarLander-v3")
    
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.n
    
    # Inicializace modelu
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    policy_net = DuelingQNetwork(state_dim, action_dim, hidden=(256, 256)).to(device)
    
    # Načtení checkpointu
    print(f"Loading model from: {model_path}")
    try:
        checkpoint = torch.load(model_path, map_location=device, weights_only=False)
        
        # Detekce formátu checkpointu
        if isinstance(checkpoint, dict) and "policy_state_dict" in checkpoint:
            policy_net.load_state_dict(checkpoint["policy_state_dict"])
            if "ep" in checkpoint:
                print(f"  Checkpoint from episode: {checkpoint['ep']}")
            if "total_steps" in checkpoint:
                print(f"  Total training steps: {checkpoint['total_steps']}")
            if "avg100" in checkpoint:
                print(f"  Avg100 reward: {checkpoint['avg100']:.2f}")
        else:
            # Přímý state_dict (např. best/final modely)
            policy_net.load_state_dict(checkpoint)
            print("  Loaded state_dict directly (best/final model)")
    except Exception as e:
        print(f"ERROR loading model: {e}")
        return
    
    policy_net.eval()
    print(f"\nStarting evaluation: {episodes} episodes\n")
    
    # Statistiky
    all_rewards = []
    all_lengths = []
    successful_landings = 0
    
    action_names = ["Do nothing", "Fire left", "Fire main", "Fire right"]
    
    # Evaluace
    for ep in range(1, episodes + 1):
        state, _ = env.reset()
        done = False
        total_reward = 0.0
        steps = 0
        
        episode_actions = [0, 0, 0, 0]  # počítadlo akcí
        
        print(f"{'='*60}")
        print(f"Episode {ep}/{episodes}")
        print(f"{'='*60}")
        
        while not done:
            steps += 1
            
            # Predikce akce
            state_tensor = torch.tensor(state, dtype=torch.float32).unsqueeze(0).to(device)
            with torch.no_grad():
                q_values = policy_net(state_tensor)
                action = q_values.argmax(dim=1).item()
            
            episode_actions[action] += 1
            
            # Verbose output
            if verbose and render:
                print(f"  Step {steps:3d} | Action: {action_names[action]:15s} | "
                      f"Q-values: [{', '.join([f'{q:.2f}' for q in q_values[0].cpu().numpy()])}]")
            
            # Krok v prostředí
            next_state, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated
            total_reward += reward
            state = next_state
            
            # Zpoždění pro lepší vizualizaci
            if render and delay > 0:
                time.sleep(delay)
        
        # Detekce úspěšného přistání (reward > 100 obvykle znamená úspěch)
        if total_reward >= 100:
            successful_landings += 1
            status = "✅ SUCCESS"
        elif total_reward >= 0:
            status = "⚠️  PARTIAL"
        else:
            status = "❌ CRASH"
        
        all_rewards.append(total_reward)
        all_lengths.append(steps)
        
        # Výpis statistik epizody
        print(f"\n{status} | Total reward: {total_reward:7.2f} | Steps: {steps:3d}")
        print(f"  Action distribution: {dict(zip(action_names, episode_actions))}")
        print()
    
    env.close()
    
    # Finální statistiky
    print(f"\n{'='*60}")
    print(f"FINAL STATISTICS")
    print(f"{'='*60}")
    print(f"Episodes:          {episodes}")
    print(f"Successful lands:  {successful_landings}/{episodes} ({100*successful_landings/episodes:.1f}%)")
    print(f"Average reward:    {np.mean(all_rewards):7.2f} ± {np.std(all_rewards):.2f}")
    print(f"Min reward:        {np.min(all_rewards):7.2f}")
    print(f"Max reward:        {np.max(all_rewards):7.2f}")
    print(f"Median reward:     {np.median(all_rewards):7.2f}")
    print(f"Average length:    {np.mean(all_lengths):6.1f} steps")
    print(f"{'='*60}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Visualize trained DQN agent on LunarLander-v3")
    parser.add_argument("--model", type=str, required=True, 
                       help="Path to the saved model/checkpoint (.pth file)")
    parser.add_argument("--episodes", type=int, default=5, 
                       help="Number of evaluation episodes (default: 5)")
    parser.add_argument("--render", action="store_true", 
                       help="Render the environment visually")
    parser.add_argument("--delay", type=float, default=0.0, 
                       help="Delay between steps in seconds (default: 0.0)")
    parser.add_argument("--quiet", action="store_true", 
                       help="Suppress step-by-step output")
    
    args = parser.parse_args()
    
    evaluate(
        model_path=args.model,
        episodes=args.episodes,
        render=args.render,
        delay=args.delay,
        verbose=not args.quiet
    )