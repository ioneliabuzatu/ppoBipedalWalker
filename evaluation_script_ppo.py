# PyTorch imports
import numpy as np
import torch
import torch.nn as nn
from torch.distributions import Normal
import onnx
from onnx2pytorch import ConvertModel
import torch.nn.functional as F

# Environment import and set logger level to display error only
import gym
from gym import logger as gymlogger


import argparse
import os

gymlogger.set_level(40)  # error only

# Seed random number generators
if os.path.exists("seed.rnd"):
    with open("seed.rnd", "r") as f:
        seed = int(f.readline().strip())
    np.random.seed(seed)
    torch.manual_seed(seed)
else:
    seed = None


class Env():
    """
    Environment wrapper for BipedalWalker
    """
    def __init__(self, seed=None):
        self.gym_env = gym.make('BipedalWalker-v3')
        self.env = self.gym_env
        self.action_space = self.env.action_space
        if seed is not None:
            self.env.seed(seed)

    def reset(self):
        return self.env.reset()

    def step(self, action):
        return self.env.step(action)

    def close(self):
        self.env.close()


class Agent():
    """
    Agent for training
    """
    def __init__(self, net):
        self.actor = net

    def select_action(self, state):
        state = torch.from_numpy(state).float().to(device).unsqueeze(0)
        with torch.no_grad():
            mu, sigma_logits = self.actor(state)
            sigma = F.softplus(sigma_logits) + 1e-5
            policy_dist = Normal(loc=mu, scale=sigma)
            action = policy_dist.sample()
            action = action.cpu().squeeze().numpy()
        return action


def run_episode(agent, img_stack, seed=None):
    env = Env(seed=seed)
    state = env.reset()
    score = 0
    done_or_die = False
    while not done_or_die:
        action = agent.select_action(state)
        state, reward, done, _ = env.step(action)
        score += reward

        if done:
            done_or_die = True
    env.close()
    return score


if __name__ == "__main__":
    N_EPISODES = 50
    IMG_STACK = 1

    parser = argparse.ArgumentParser()
    parser.add_argument("--submission", type=str, default="./submission_actor_-96.40.onnx")
    args = parser.parse_args()
    model_file = args.submission

    device = torch.device("cpu")

    # Network
    net = ConvertModel(onnx.load(model_file))
    net = net.to(device)
    net.eval()
    agent = Agent(net)

    scores = []
    for i in range(N_EPISODES):
        if seed is not None:
            seed = np.random.randint(1e7)
        scores.append(run_episode(agent, IMG_STACK, seed=seed))

    print(np.mean(scores))

