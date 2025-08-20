import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
# print(sys.path)

import gymnasium as gym
from gym_envs.BuzzRacer.BuzzRacerEnv import BuzzRacerEnv
from gymnasium.wrappers import TimeLimit
import math
import random
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from collections import namedtuple, deque
from itertools import count

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from DDPG import Actor, Critic, ReplayMemory
Transition = namedtuple('Transition',
                            ('state', 'action', 'next_state', 'reward') )


# -------------------------
# Hyperparameters (tweak me)
# -------------------------
BATCH_SIZE = 256
GAMMA = 0.99
TAU = 1e-3           # soft update factor (small -> slow/ stable)
ACTOR_LR = 1e-4
CRITIC_LR = 1e-3
REPLAY_SIZE = 100000
MAX_EPISODES = 1000
MAX_STEPS_PER_EPISODE = 1000
# DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
DEVICE = torch.device("cpu")

# util functions
def soft_update(target, source, tau):
    for t_param, s_param in zip(target.parameters(), source.parameters()):
        t_param.data.copy_( (1.0 - tau) * t_param.data + tau * s_param.data )

class GaussianNoise:
    def __init__(self, action_dim, std=0.2, clip=0.5):
        self.std = std
        self.clone_shape = (action_dim,)
        self.clip = clip

    def sample(self):
        return np.clip(np.random.normal(0, self.std, self.clone_shape), -self.clip, self.clip)

# -------------------------
# Main training function
# -------------------------
def train_ddpg(env, directory="models/ddpg", max_episodes=MAX_EPISODES):
    assert hasattr(env.action_space, 'shape'), "Environment must have continuous action_space (Box)."
    if not isinstance(env.action_space, gym.spaces.Box):
        raise RuntimeError("DDPG requires a continuous action space (Box).")

    # dims and scaling
    state, _ = env.reset()
    state_dim = len(state)
    action_dim = env.action_space.shape[0]
    action_low = env.action_space.low
    action_high = env.action_space.high
    action_limit = float(np.max(np.abs(action_high)))  # scale after tanh

    # models
    # pytorch models and tensors must be on same device
    actor = Actor(state_dim, action_dim, action_limit).to(DEVICE)
    critic = Critic(state_dim, action_dim).to(DEVICE)
    target_actor = Actor(state_dim, action_dim, action_limit).to(DEVICE)
    target_critic = Critic(state_dim, action_dim).to(DEVICE)
    target_actor.load_state_dict(actor.state_dict())
    target_critic.load_state_dict(critic.state_dict())

    # optimizers
    actor_optimizer = optim.Adam(actor.parameters(), lr=ACTOR_LR)
    critic_optimizer = optim.Adam(critic.parameters(), lr=CRITIC_LR)

    memory = ReplayMemory(REPLAY_SIZE)
    noise = GaussianNoise(action_dim, std=0.2, clip=0.5)

    if not os.path.exists(directory):
        os.makedirs(directory)

    steps = 0
    for ep in range(max_episodes):
        state, _ = env.reset()
        # for all state tensors, we add batch dimension with .unsqueeze(0)
        state = torch.tensor(state, dtype=torch.float32, device=DEVICE).unsqueeze(0)
        ep_reward = 0.0
        for t in range(MAX_STEPS_PER_EPISODE):
            # select action (actor + noise), scale to env range
            actor.eval()
            with torch.no_grad():
                raw_action = actor(state).cpu().numpy().flatten()  # already scaled by action_limit
            actor.train()

            exploration = noise.sample()
            action_np = np.clip(raw_action + exploration, action_low, action_high)

            # step env
            next_obs, reward, terminated, truncated, _ = env.step(action_np)
            done = bool(terminated or truncated)
            next_state = None if done else torch.tensor(next_obs, dtype=torch.float32, device=DEVICE).unsqueeze(0)
            reward_t = float(reward)

            # store transition (store action as float tensor)
            memory.push(state.cpu(), 
                        torch.tensor(action_np, dtype=torch.float32).unsqueeze(0), 
                        None if done else next_state.cpu(), 
                        torch.tensor([reward_t], dtype=torch.float32), 
                        torch.tensor([1.0 if done else 0.0], dtype=torch.float32))

            state = next_state
            ep_reward += reward_t
            steps += 1

            # optimize when enough samples
            if len(memory) >= BATCH_SIZE:
                transitions = memory.sample(BATCH_SIZE)
                batch = Transition(*zip(*transitions))

                # create batched tensors: states/actions/rewards/next_states/dones
                state_batch = torch.cat(batch.state).to(DEVICE)                     # (B, state_dim)
                action_batch = torch.cat(batch.action).to(DEVICE)                   # (B, action_dim)
                reward_batch = torch.cat(batch.reward).to(DEVICE)                   # (B, 1)
                done_batch = torch.cat(batch.done).to(DEVICE)                       # (B, 1)

                # next states may contain None for terminal; build mask & tensor
                non_final_mask = torch.tensor([s is not None for s in batch.next_state], device=DEVICE, dtype=torch.bool)
                if non_final_mask.any():
                    non_final_next_states = torch.cat([s for s in batch.next_state if s is not None]).to(DEVICE)
                else:
                    non_final_next_states = torch.empty((0, state_dim), device=DEVICE)

                # Critic update
                # compute target Q = r + (1-done) * gamma * Q_target(next_state, actor_target(next_state))
                with torch.no_grad():
                    next_actions = torch.zeros((BATCH_SIZE, action_dim), device=DEVICE)
                    if non_final_mask.any():
                        next_actions_non_final = target_actor(non_final_next_states)
                        # place into next_actions at non_final positions
                        next_actions[non_final_mask] = next_actions_non_final
                    # For final states their contribution is zero due to done mask
                    q_next = target_critic(
                        non_final_next_states, target_actor(non_final_next_states)
                    ) if non_final_mask.any() else torch.zeros((0,1), device=DEVICE)

                    # Build target_q full vector (B,1), careful with ordering
                    target_q = torch.zeros((BATCH_SIZE, 1), device=DEVICE)
                    if non_final_mask.any():
                        target_q[non_final_mask, 0] = q_next.view(-1)
                    target_q = reward_batch + (1.0 - done_batch) * (GAMMA * target_q)

                # current Q estimates
                current_q = critic(state_batch, action_batch)

                critic_loss = F.mse_loss(current_q, target_q)

                critic_optimizer.zero_grad()
                critic_loss.backward()
                torch.nn.utils.clip_grad_norm_(critic.parameters(), 1.0)
                critic_optimizer.step()

                # Actor update (maximize Q, or minimize -Q)
                actor_optimizer.zero_grad()
                # actor produces actions for the current states
                actor_actions = actor(state_batch)
                actor_loss = -critic(state_batch, actor_actions).mean()
                actor_loss.backward()
                torch.nn.utils.clip_grad_norm_(actor.parameters(), 1.0)
                actor_optimizer.step()

                # Soft update targets
                soft_update(target_actor, actor, TAU)
                soft_update(target_critic, critic, TAU)

            if done:
                break

        print(f"Episode {ep:4d}  Reward: {ep_reward:.2f}  ReplaySize: {len(memory)}")
        # save checkpoints occasionally
        if ep % 50 == 0:
            torch.save(actor.state_dict(), os.path.join(directory, f"actor_ep{ep}.pth"))
            torch.save(critic.state_dict(), os.path.join(directory, f"critic_ep{ep}.pth"))

    # final save
    torch.save(actor.state_dict(), os.path.join(directory, "actor_final.pth"))
    torch.save(critic.state_dict(), os.path.join(directory, "critic_final.pth"))

if __name__ == "__main__":
    # Example usage - replace with your continuous env
    # env = gym.make("Pendulum-v1")
    # Or if you have a continuous BuzzRacer: env = BuzzRacerEnvContinuous(...)
    env = TimeLimit(BuzzRacerEnv(render_mode=None), max_episode_steps=1000)
    
    # env = gym.make("Pendulum-v1")   # simple test env (continuous)
    train_ddpg(env)
