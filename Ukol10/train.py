"""
train.py - DQN (Double DQN + Dueling + PER) pro LunarLander-v3

Hlavní featury:
- Double DQN update (snižuje overestimation bias)
- Dueling DQN architektura (oddělené V(s) a A(s,a) streamy)
- Prioritized Experience Replay (vzorky s větším TD-error se trénují častěji)
- Adaptivní learning rate s schedulováním
- Evaluační běhy bez explorace pro přesné měření
- Gradient clipping pro stabilitu
- Automatické ukládání best modelu podle eval rewardu
- Kompletní TensorBoard logging
- Checkpointing s možností obnovení tréninku
"""

import argparse
import random
import time
from collections import deque
from typing import Tuple
import os

import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
from tqdm import trange


# -------------------------
# Dueling Q-network
# -------------------------
class DuelingQNetwork(nn.Module):
    """
    Dueling DQN architektura oddělující value a advantage streams.
    Q(s,a) = V(s) + (A(s,a) - mean(A(s,:)))
    """
    def __init__(self, state_dim: int, action_dim: int, hidden=(256, 256)):
        super().__init__()
        
        # Sdílený feature extractor
        layers = []
        input_dim = state_dim
        for h in hidden:
            layers.append(nn.Linear(input_dim, h))
            layers.append(nn.ReLU())
            input_dim = h
        self.features = nn.Sequential(*layers)
        
        # Value stream - odhad V(s)
        self.value_stream = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 1)
        )
        
        # Advantage stream - odhad A(s,a)
        self.advantage_stream = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.ReLU(),
            nn.Linear(128, action_dim)
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        features = self.features(x)
        value = self.value_stream(features)
        advantages = self.advantage_stream(features)
        
        # Kombinace: Q = V + (A - mean(A))
        q_values = value + (advantages - advantages.mean(dim=1, keepdim=True))
        return q_values


# -------------------------
# Prioritized Replay Buffer
# -------------------------
class PrioritizedReplayBuffer:
    """
    Replay buffer s prioritizací podle TD-error.
    Vzorky s větším errorem se sampluji častěji.
    """
    def __init__(self, capacity: int, state_dim: int, alpha: float = 0.6):
        self.capacity = int(capacity)
        self.state_dim = state_dim
        self.alpha = alpha  # exponent prioritizace (0 = uniform, 1 = full priority)
        self.ptr = 0
        self.size = 0
        
        # Prealokace numpy arrays (rychlejší než Python listy)
        self.states = np.zeros((self.capacity, state_dim), dtype=np.float32)
        self.next_states = np.zeros((self.capacity, state_dim), dtype=np.float32)
        self.actions = np.zeros((self.capacity,), dtype=np.int64)
        self.rewards = np.zeros((self.capacity,), dtype=np.float32)
        self.dones = np.zeros((self.capacity,), dtype=np.float32)
        self.priorities = np.zeros((self.capacity,), dtype=np.float32)
    
    def add(self, state, action, reward, next_state, done):
        """Přidá novou zkušenost s maximální prioritou"""
        max_prio = self.priorities[:self.size].max() if self.size > 0 else 1.0
        
        self.states[self.ptr] = state
        self.next_states[self.ptr] = next_state
        self.actions[self.ptr] = int(action)
        self.rewards[self.ptr] = float(reward)
        self.dones[self.ptr] = 1.0 if done else 0.0
        self.priorities[self.ptr] = max_prio
        
        self.ptr = (self.ptr + 1) % self.capacity
        self.size = min(self.size + 1, self.capacity)
    
    def sample(self, batch_size: int, device: torch.device, beta: float = 0.4):
        """
        Sampluje batch podle priorit s importance sampling weights.
        Beta kontroluje korekci bias (0 = žádná korekce, 1 = plná korekce)
        """
        priorities = self.priorities[:self.size]
        probs = priorities ** self.alpha
        probs /= probs.sum()
        
        # Sample indexů podle pravděpodobností
        idx = np.random.choice(self.size, size=batch_size, replace=False, p=probs)
        
        # Importance sampling weights pro korekci bias
        weights = (self.size * probs[idx]) ** (-beta)
        weights /= weights.max()  # normalizace
        
        # Konverze na tensory
        states = torch.as_tensor(self.states[idx], dtype=torch.float32, device=device)
        next_states = torch.as_tensor(self.next_states[idx], dtype=torch.float32, device=device)
        actions = torch.as_tensor(self.actions[idx], dtype=torch.int64, device=device)
        rewards = torch.as_tensor(self.rewards[idx], dtype=torch.float32, device=device)
        dones = torch.as_tensor(self.dones[idx], dtype=torch.float32, device=device)
        weights = torch.as_tensor(weights, dtype=torch.float32, device=device)
        
        return states, actions, rewards, next_states, dones, weights, idx
    
    def update_priorities(self, idx: np.ndarray, td_errors: np.ndarray):
        """Aktualizuje priority podle nových TD-errorů"""
        priorities = np.abs(td_errors) + 1e-6  # epsilon pro stabilitu
        self.priorities[idx] = priorities
    
    def __len__(self):
        return self.size


# -------------------------
# Double DQN loss s PER
# -------------------------
def compute_double_dqn_loss(
    policy_net: DuelingQNetwork,
    target_net: DuelingQNetwork,
    states: torch.Tensor,
    actions: torch.Tensor,
    rewards: torch.Tensor,
    next_states: torch.Tensor,
    dones: torch.Tensor,
    weights: torch.Tensor,
    gamma: float,
) -> Tuple[torch.Tensor, np.ndarray]:
    """
    Počítá Double DQN loss s importance sampling weights.
    Vrací: (loss, td_errors pro update priorit)
    """
    # Současné Q-values
    q_values = policy_net(states)
    q_pred = q_values.gather(1, actions.unsqueeze(1)).squeeze(1)
    
    # Double DQN target: výběr akce z policy_net, evaluace z target_net
    with torch.no_grad():
        next_q_policy = policy_net(next_states)
        next_actions = next_q_policy.argmax(dim=1, keepdim=True)
        next_q_target = target_net(next_states).gather(1, next_actions).squeeze(1)
        q_target = rewards + (1.0 - dones) * gamma * next_q_target
    
    # TD errors pro priority update
    td_errors = (q_pred - q_target).detach().cpu().numpy()
    
    # Weighted MSE loss (váhy z importance sampling)
    loss = (weights * (q_pred - q_target).pow(2)).mean()
    
    return loss, td_errors


# -------------------------
# Utility funkce
# -------------------------
def set_seed(seed: int):
    """Nastaví seed pro reprodukovatelnost"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def evaluate(env, policy_net: DuelingQNetwork, device: torch.device, n_episodes: int = 5) -> dict:
    """
    Evaluace bez explorace (epsilon=0).
    Vrací slovník se statistikami.
    """
    policy_net.eval()
    rewards = []
    lengths = []
    
    for _ in range(n_episodes):
        state, _ = env.reset()
        ep_reward = 0.0
        done = False
        steps = 0
        
        while not done and steps < 1000:
            steps += 1
            st = torch.as_tensor(state, dtype=torch.float32, device=device).unsqueeze(0)
            with torch.no_grad():
                q = policy_net(st)
                action = int(q.argmax(dim=1).item())
            
            next_state, reward, terminated, truncated, _ = env.step(action)
            done = bool(terminated or truncated)
            state = next_state
            ep_reward += reward
        
        rewards.append(ep_reward)
        lengths.append(steps)
    
    policy_net.train()
    
    return {
        "mean": float(np.mean(rewards)),
        "std": float(np.std(rewards)),
        "min": float(np.min(rewards)),
        "max": float(np.max(rewards)),
        "median": float(np.median(rewards)),
        "avg_length": float(np.mean(lengths))
    }


def save_checkpoint(path: str, policy_net, target_net, optimizer, scheduler, ep, total_steps, **kwargs):
    """Uloží checkpoint s kompletním stavem"""
    checkpoint = {
        "policy_state_dict": policy_net.state_dict(),
        "target_state_dict": target_net.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "ep": ep,
        "total_steps": total_steps,
    }
    if scheduler is not None:
        checkpoint["scheduler_state_dict"] = scheduler.state_dict()
    checkpoint.update(kwargs)
    torch.save(checkpoint, path)


# -------------------------
# Hlavní tréninková smyčka
# -------------------------
def train(args):
    # Inicializace
    device = torch.device("cuda" if torch.cuda.is_available() and not args.force_cpu else "cpu")
    print(f"\n{'='*80}")
    print(f"  DQN Training - LunarLander-v3")
    print(f"{'='*80}")
    print(f"Device: {device}")
    print(f"Episodes: {args.episodes}")
    print(f"Target update: every {args.target_update} steps")
    print(f"LR decay: {args.lr_decay}")
    print(f"Seed: {args.seed}")
    print(f"{'='*80}\n")
    
    set_seed(args.seed)
    
    # Vytvoř složky
    os.makedirs(os.path.dirname(args.save_prefix) if os.path.dirname(args.save_prefix) else ".", exist_ok=True)
    os.makedirs(args.logdir, exist_ok=True)
    
    # Prostředí
    env = gym.make("LunarLander-v3")
    eval_env = gym.make("LunarLander-v3")
    obs_dim = env.observation_space.shape[0]
    action_dim = env.action_space.n
    
    # Sítě
    policy_net = DuelingQNetwork(obs_dim, action_dim, hidden=(args.hidden, args.hidden)).to(device)
    target_net = DuelingQNetwork(obs_dim, action_dim, hidden=(args.hidden, args.hidden)).to(device)
    target_net.load_state_dict(policy_net.state_dict())
    target_net.eval()
    
    # Optimizer & Scheduler
    optimizer = optim.Adam(policy_net.parameters(), lr=args.lr)
    scheduler = None
    if args.lr_decay:
        scheduler = optim.lr_scheduler.ExponentialLR(optimizer, gamma=args.lr_gamma)
    
    # Replay buffer
    buffer = PrioritizedReplayBuffer(args.buffer_size, obs_dim, alpha=args.per_alpha)
    
    # TensorBoard
    writer = SummaryWriter(log_dir=args.logdir)
    
    # Statistiky
    episode_rewards = deque(maxlen=100)
    best_eval_reward = -float("inf")
    total_steps = 0
    start_time = time.time()
    
    # Beta annealing pro importance sampling
    beta_start = args.per_beta_start
    beta_end = 1.0
    beta_steps = args.episodes * 0.8
    
    # Progress bar
    pbar = trange(args.episodes, desc="Training")
    
    for ep in pbar:
        state, _ = env.reset()
        ep_reward = 0.0
        ep_losses = []
        ep_q_values = []
        done = False
        steps = 0
        
        while not done and steps < args.max_steps_per_episode:
            steps += 1
            total_steps += 1
            
            # Epsilon-greedy exploration
            eps = max(args.eps_end, args.eps_start - (total_steps / args.eps_decay_steps))
            
            if random.random() < eps:
                action = env.action_space.sample()
            else:
                st = torch.as_tensor(state, dtype=torch.float32, device=device).unsqueeze(0)
                with torch.no_grad():
                    q = policy_net(st)
                    ep_q_values.append(float(q.max().item()))
                    action = int(q.argmax(dim=1).item())
            
            # Krok v prostředí
            next_state, reward, terminated, truncated, _ = env.step(action)
            done = bool(terminated or truncated)
            
            buffer.add(state, action, reward, next_state, done)
            state = next_state
            ep_reward += reward
            
            # Tréninková iterace
            if len(buffer) >= args.batch_size and total_steps > args.learning_starts:
                # Beta annealing
                beta = min(beta_end, beta_start + (beta_end - beta_start) * total_steps / beta_steps)
                
                # Sample batch
                states_b, actions_b, rewards_b, next_states_b, dones_b, weights_b, idx = buffer.sample(
                    args.batch_size, device, beta
                )
                
                # Compute loss
                loss, td_errors = compute_double_dqn_loss(
                    policy_net, target_net, states_b, actions_b,
                    rewards_b, next_states_b, dones_b, weights_b,
                    args.gamma
                )
                
                # Backprop
                optimizer.zero_grad()
                loss.backward()
                
                # Gradient clipping
                if args.max_grad_norm > 0:
                    torch.nn.utils.clip_grad_norm_(policy_net.parameters(), args.max_grad_norm)
                
                optimizer.step()
                
                # Update priorities
                buffer.update_priorities(idx, td_errors)
                
                ep_losses.append(float(loss.item()))
                
                # Target network update
                if total_steps % args.target_update == 0:
                    target_net.load_state_dict(policy_net.state_dict())
        
        # Episode finished
        episode_rewards.append(ep_reward)
        
        # Logging
        writer.add_scalar("train/episode_reward", ep_reward, ep)
        writer.add_scalar("train/epsilon", eps, ep)
        writer.add_scalar("train/episode_length", steps, ep)
        
        if ep_losses:
            writer.add_scalar("train/avg_loss", np.mean(ep_losses), ep)
        if ep_q_values:
            writer.add_scalar("train/avg_q_value", np.mean(ep_q_values), ep)
        
        avg100 = np.mean(episode_rewards) if len(episode_rewards) > 0 else ep_reward
        writer.add_scalar("train/avg_reward_100", float(avg100), ep)
        
        # LR scheduler
        if scheduler is not None and (ep + 1) % args.scheduler_step_every == 0:
            scheduler.step()
            writer.add_scalar("train/learning_rate", optimizer.param_groups[0]['lr'], ep)
        
        # Progress bar update
        pbar.set_postfix({
            "reward": f"{ep_reward:.0f}",
            "avg100": f"{avg100:.0f}",
            "eps": f"{eps:.3f}",
            "steps": total_steps
        })
        
        # Evaluace
        if (ep + 1) % args.eval_every == 0:
            eval_stats = evaluate(eval_env, policy_net, device, n_episodes=args.eval_episodes)
            eval_reward = eval_stats["mean"]
            
            writer.add_scalar("eval/reward_mean", eval_reward, ep)
            writer.add_scalar("eval/reward_std", eval_stats["std"], ep)
            writer.add_scalar("eval/reward_min", eval_stats["min"], ep)
            writer.add_scalar("eval/reward_max", eval_stats["max"], ep)
            writer.add_scalar("eval/avg_length", eval_stats["avg_length"], ep)
            
            print(f"\n[Eval ep {ep+1}] Reward: {eval_reward:.1f} ± {eval_stats['std']:.1f} "
                  f"(min={eval_stats['min']:.0f}, max={eval_stats['max']:.0f})")
            
            # Save best model
            if eval_reward > best_eval_reward:
                best_eval_reward = eval_reward
                best_path = f"{args.save_prefix}_best_eval{int(eval_reward)}.pth"
                torch.save(policy_net.state_dict(), best_path)
                print(f"💾 New best model saved: {best_path}")
        
        # Checkpoint
        if (ep + 1) % args.save_every == 0:
            ckpt_path = f"{args.save_prefix}_ep{ep+1}.pth"
            save_checkpoint(
                ckpt_path, policy_net, target_net, optimizer, scheduler,
                ep, total_steps, avg100=avg100, best_eval=best_eval_reward
            )
            print(f"\n💾 Checkpoint saved: {ckpt_path}")
    
    # Final evaluation
    print("\n" + "="*80)
    print("Training finished! Running final evaluation...")
    final_stats = evaluate(eval_env, policy_net, device, n_episodes=50)
    print(f"Final eval (50 episodes): {final_stats['mean']:.1f} ± {final_stats['std']:.1f}")
    print(f"Min: {final_stats['min']:.0f}, Max: {final_stats['max']:.0f}, Median: {final_stats['median']:.0f}")
    
    elapsed = time.time() - start_time
    print(f"Total time: {elapsed/60:.1f} minutes ({elapsed/3600:.2f} hours)")
    print(f"Total steps: {total_steps:,}")
    print("="*80 + "\n")
    
    # Save final model
    final_path = f"{args.save_prefix}_final.pth"
    torch.save(policy_net.state_dict(), final_path)
    print(f"💾 Final model saved: {final_path}\n")
    
    writer.close()
    env.close()
    eval_env.close()


# -------------------------
# CLI
# -------------------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="DQN Training for LunarLander-v3")
    
    # Training
    parser.add_argument("--episodes", type=int, default=2000, help="Number of training episodes")
    parser.add_argument("--max_steps_per_episode", type=int, default=1000, help="Max steps per episode")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    
    # Buffer & Batch
    parser.add_argument("--buffer_size", type=int, default=200000, help="Replay buffer capacity")
    parser.add_argument("--batch_size", type=int, default=128, help="Batch size")
    parser.add_argument("--learning_starts", type=int, default=2000, help="Steps before learning starts")
    parser.add_argument("--per_alpha", type=float, default=0.6, help="PER alpha (prioritization)")
    parser.add_argument("--per_beta_start", type=float, default=0.4, help="PER beta start (IS correction)")
    
    # Optimizer
    parser.add_argument("--lr", type=float, default=1e-4, help="Learning rate")
    parser.add_argument("--lr_decay", action="store_true", help="Enable LR decay")
    parser.add_argument("--lr_gamma", type=float, default=0.9999, help="LR decay gamma")
    parser.add_argument("--scheduler_step_every", type=int, default=100, help="Scheduler step frequency")
    parser.add_argument("--max_grad_norm", type=float, default=10.0, help="Gradient clipping norm")
    
    # RL params
    parser.add_argument("--gamma", type=float, default=0.99, help="Discount factor")
    parser.add_argument("--target_update", type=int, default=1000, help="Target network update frequency")
    
    # Exploration
    parser.add_argument("--eps_start", type=float, default=1.0, help="Starting epsilon")
    parser.add_argument("--eps_end", type=float, default=0.05, help="Final epsilon")
    parser.add_argument("--eps_decay_steps", type=float, default=100000.0, help="Epsilon decay steps")
    
    # Evaluation
    parser.add_argument("--eval_every", type=int, default=100, help="Evaluation frequency (episodes)")
    parser.add_argument("--eval_episodes", type=int, default=5, help="Number of evaluation episodes")
    
    # Saving & Logging
    parser.add_argument("--save_every", type=int, default=500, help="Checkpoint save frequency")
    parser.add_argument("--save_prefix", type=str, default="dqn_lunar", help="Checkpoint prefix")
    parser.add_argument("--logdir", type=str, default="runs/lunar_dqn", help="TensorBoard log directory")
    
    # Network
    parser.add_argument("--hidden", type=int, default=256, help="Hidden layer size")
    
    # Device
    parser.add_argument("--force_cpu", action="store_true", help="Force CPU usage")
    
    args = parser.parse_args()
    
    train(args)