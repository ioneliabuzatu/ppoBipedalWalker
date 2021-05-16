import torch
import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Normal
from torch.utils.data import Dataset, DataLoader

import onnx
import io
import glob
import copy
import base64
import random
import numpy as np
from tqdm.notebook import tqdm
from time import sleep

# Environment import and set logger level to display error only
import gym

gym.logger.set_level(40)
from gym.wrappers import Monitor

import matplotlib.pyplot as plt
import seaborn as sns

def decide(observation):
    weights = np.array([
        [0.9, -0.7, 0.0, -1.4],
        [4.3, -1.6, -4.4, -2.0],
        [2.4, -4.2, -1.3, -0.1],
        [-3.1, -5.0, -2.0, -3.3],
        [-0.8, 1.4, 1.7, 0.2],
        [-0.7, 0.2, -0.2, 0.1],
        [-0.6, -1.5, -0.6, 0.3],
        [-0.5, -0.3, 0.2, 0.1],
        [0.0, -0.1, -0.1, 0.1],
        [0.4, 0.8, -1.6, -0.5],
        [-0.4, 0.5, -0.3, -0.4],
        [0.3, 2.0, 0.9, -1.6],
        [0.0, -0.2, 0.1, -0.3],
        [0.1, 0.2, -0.5, -0.3],
        [0.7, 0.3, 5.1, -2.4],
        [-0.4, -2.3, 0.3, -4.0],
        [0.1, -0.8, 0.3, 2.5],
        [0.4, -0.9, -1.8, 0.3],
        [-3.9, -3.5, 2.8, 0.8],
        [0.4, -2.8, 0.4, 1.4],
        [-2.2, -2.1, -2.2, -3.2],
        [-2.7, -2.6, 0.3, 0.6],
        [2.0, 2.8, 0.0, -0.9],
        [-2.2, 0.6, 4.7, -4.6],
    ])
    bias = np.array([3.2, 6.1, -4.0, 7.6])
    action = np.matmul(observation, weights) + bias
    return torch.from_numpy(action)

def wrap_env(env):
    # wrapper for recording
    env = Monitor(env, './video', force=True)
    return env


def create_env(env_id='BipedalWalker-v3', wrap_env_with_monitor=False):
    if wrap_env_with_monitor:
        env = wrap_env(gym.make(env_id))
    else:
        env = gym.make(env_id)
    action_size = env.action_space.shape
    state_size = env.observation_space.shape
    return env, action_size, state_size


def set_seed(env, seed=None):
    if seed is not None:
        random.seed(seed)
        env.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed(seed)
            torch.cuda.manual_seed_all(seed)


def transforms(state, device):
    # transofrm to numpy to tensor and push to device
    return torch.FloatTensor(state).to(device)


def test_environment(env, agent=None, n_steps=200):
    # run and evaluate in the environment
    state = env.reset()
    for i in range(n_steps):
        env.render()

        if agent is None:
            action = env.action_space.sample()
        else:
            action, _ = agent.act(state)
            action = np.clip(action.squeeze().numpy(), -1, 1)
        state, reward, done, info = env.step(action)
        if done:
            env.reset()
    env.close()


def get_running_stat(stat, stat_len):
    # evaluate stats
    cum_sum = np.cumsum(np.insert(stat, 0, 0))
    return (cum_sum[stat_len:] - cum_sum[:-stat_len]) / stat_len


def plot_results(runner, reward_scale=1.0):
    # plot stats
    episode, r, l = np.array(runner.stats_rewards_list).T
    cum_r = get_running_stat(r, 10)
    cum_l = get_running_stat(l, 10)

    plt.figure(figsize=(16, 16))

    plt.subplot(321)

    # plot rewards
    plt.plot(episode[-len(cum_r):], cum_r)
    plt.plot(episode, r, alpha=0.5)
    plt.xlabel('Episode')
    plt.ylabel('Episode Reward')

    plt.subplot(322)

    # plot episode lengths
    plt.plot(episode[-len(cum_l):], cum_l)
    plt.plot(episode, l, alpha=0.5)
    plt.xlabel('Episode')
    plt.ylabel('Episode Length')

    plt.subplot(323)

    # plot return
    all_returns = np.array(runner.buffer.all_returns) / reward_scale(1)
    plt.scatter(range(0, len(all_returns)), all_returns, alpha=0.5)
    mean_returns = np.array(runner.buffer.mean_returns) / reward_scale(1)  # rescale back to original return
    plt.plot(range(0, len(mean_returns)), mean_returns, color="orange")
    plt.xlabel('Episode')
    plt.ylabel('Return')

    plt.subplot(324)

    # plot entropy
    entropy_arr = np.array(runner.stats_entropy_list)
    plt.plot(range(0, len(entropy_arr)), entropy_arr)
    plt.xlabel('Episode')
    plt.ylabel('Entropy')

    plt.subplot(325)
    # values_arr = np.array(runner.value_list)
    # plt.plot(range(0, len(values_arr)), values_arr)
    # plt.xlabel('Episode')
    # plt.ylabel('Value Loss')

    if runner.logger.debug:
        # plot variance
        variance_arr = np.array(runner.logger.compute_gradient_variance())
        plt.plot(range(0, len(variance_arr)), variance_arr)
        plt.xlabel('Episode')
        plt.ylabel('Variance')

    plt.show()


def show_video():
    mp4list = glob.glob('video/*.mp4')
    if len(mp4list) > 0:
        mp4 = mp4list[0]
        video = io.open(mp4, 'r+b').read()
        encoded = base64.b64encode(video)
    else:
        print("Could not find video")


def grad_variance(g):
    # compute gradient variance
    return np.mean(g ** 2) - np.mean(g) ** 2


class Logger(object):
    """Logger that can be used for debugging different values
    """

    def __init__(self, debug=False):
        self.gradients = []
        self.debug = debug

    def add_gradients(self, grad):
        if not self.debug: return
        self.gradients.append(grad)

    def compute_gradient_variance(self):
        vars_ = []
        grads_list = [np.zeros_like(self.gradients[0])] * 100
        for i, grads in enumerate(self.gradients):
            grads_list.append(grads)
            grads_list = grads_list[1:]
            grad_arr = np.stack(grads_list, axis=0)
            g = np.apply_along_axis(grad_variance, axis=-1, arr=grad_arr)
            vars_.append(np.mean(g))
        return vars_


class Transition(object):
    """Transition helper object
    """

    def __init__(self, state, action, reward, next_state, log_probs):
        self.state = state
        self.action = action
        self.reward = reward
        self.next_state = next_state
        self.g_return = 0.0
        self.log_probs = log_probs


class Episode(object):
    """Class for collecting an episode of transitions
    """

    def __init__(self, discount):
        self.discount = discount
        self._empty()
        self.total_reward = 0.0

    def _empty(self):
        self.n = 0
        self.transitions = []

    def reset(self):
        self._empty()

    def size(self):
        return self.n

    def append(self, transition):
        self.transitions.append(transition)
        self.n += 1

    def states(self):
        return [s.state for s in self.transitions]

    def actions(self):
        return [a.action for a in self.transitions]

    def rewards(self):
        return [r.reward for r in self.transitions]

    def next_states(self):
        return [s_.next_state for s_ in self.transitions]

    def returns(self):
        return [r.g_return for r in self.transitions]

    def calculate_return(self):
        # calculate the return of the episode
        rewards = self.rewards()
        trajectory_len = len(rewards)
        return_array = torch.zeros((trajectory_len,))
        g_return = 0.
        for i in range(trajectory_len - 1, -1, -1):
            g_return = rewards[i] + self.discount * g_return
            return_array[i] = g_return
            self.transitions[i].g_return = g_return
        return return_array


class BufferDataset(Dataset):
    """Buffer dataset used to iterate over buffer samples when training.
    """

    def __init__(self, data):
        self.data = data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        t = self.data[idx]
        return t.state, t.action, t.reward, t.next_state, t.log_probs


class ReplayBuffer(object):
    # ===================================================
    # ++++++++++++++++++++ OPTIONAL +++++++++++++++++++++
    # ===================================================
    # > feel free to optimize sampling and buffer handling
    # ===================================================
    """Buffer to collect samples while rolling out in the envrionment.
    """

    def __init__(self, capacity, batch_size, min_transitions):
        self.capacity = capacity
        self.batch_size = batch_size
        self.min_transitions = min_transitions
        self.buffer = []
        self._empty()
        self.mean_returns = []
        self.all_returns = []

    def _empty(self):
        # empty the buffer
        del self.buffer[:]
        self.position = 0

    def add(self, episode):
        # Saves a transition
        episode.calculate_return()
        for t in episode.transitions:
            if len(self.buffer) < self.capacity:
                self.buffer.append(None)
            self.buffer[self.position] = t
            self.position = (self.position + 1) % self.capacity

    def update_stats(self):
        # update the statistics on the buffer
        returns = [t.g_return for t in self.buffer]
        self.all_returns += returns
        mean_return = np.mean(np.array(returns))
        self.mean_returns += ([mean_return] * len(returns))

    def reset(self):
        # calls empty
        self._empty()

    def create_dataloader(self):
        # creates a dataloader for training
        train_loader = DataLoader(
            BufferDataset(self.buffer),
            batch_size=self.batch_size,
            shuffle=True
        )
        return train_loader

    def __len__(self):
        return len(self.buffer)

def init(module, weight_init, bias_init, gain=1):
    weight_init(module.weight.data, gain=gain)
    bias_init(module.bias.data)
    return module


# https://github.com/openai/baselines/blob/master/baselines/common/tf_util.py#L87
def init_normc_(weight, gain=1):
    weight.normal_(0, 1)
    weight *= gain / torch.sqrt(weight.pow(2).sum(1, keepdim=True))