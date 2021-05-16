import numpy as np
from torch.autograd import Variable
from utils import transforms
import os
from utils import Logger
from torch.distributions import Normal
import gym
from utils import decide

import datetime
from collections import namedtuple
from collections import deque
import math

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.nn.utils.clip_grad import clip_grad_norm_
import wandb
from model import ACNet

import config

wandb.run = config.tensorboard.run

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Running on {device}")


class Env:
    '''
    Environment class
    '''
    game_rew = 0
    last_game_rew = 0
    game_n = 0
    last_games_rews = [-200]
    n_iter = 0

    def __init__(self, env_name, n_steps, gamma, gae_lambda, save_video=False):
        super(Env, self).__init__()

        self.env = gym.make(env_name)
        self.obs = self.env.reset()

        self.n_steps = n_steps
        self.action_n = self.env.action_space.shape
        self.observation_n = self.env.observation_space.shape[0]
        self.gamma = gamma
        self.gae_lambda = gae_lambda

    def steps(self, agent_policy, agent_value):
        '''
        Execute the agent n_steps in the environment
        '''
        memories = []
        for s in range(self.n_steps):
            self.n_iter += 1

            ag_mean, log_std = agent_policy(torch.tensor(self.obs))

            # get an action following the policy distribution
            logstd = log_std.data.cpu().numpy()
            action = ag_mean.data.cpu().numpy() + np.exp(logstd) * np.random.normal(size=logstd.shape)
            # action = np.random.normal(loc=ag_mean.data.cpu().numpy(), scale=torch.sqrt(ag_var).data.cpu().numpy())
            action = np.clip(action, -1, 1)

            state_value = float(agent_value(torch.tensor(self.obs)))

            new_obs, reward, done, _ = self.env.step(action)

            # Update the memories with the last interaction
            if done:
                # change the reward to 0 in case the episode is end
                memories.append(
                    Memory(obs=self.obs, action=action, new_obs=new_obs, reward=0, done=done, value=state_value, adv=0))
            else:
                memories.append(
                    Memory(obs=self.obs, action=action, new_obs=new_obs, reward=reward, done=done,
                           value=state_value,
                           adv=0))

            self.game_rew += reward
            self.obs = new_obs

            if done:
                print('#####', self.game_n, 'rew:', int(self.game_rew), int(np.mean(self.last_games_rews[-100:])),
                      np.round(reward, 2), self.n_iter)
                wandb.log({"reward_per_game": int(self.game_rew)}, step=self.game_n)
                wandb.log({"training reward last 100 games": int(np.mean(self.last_games_rews[-100:]))},
                          step=self.game_n)

                self.obs = self.env.reset()
                self.last_game_rew = self.game_rew
                self.game_rew = 0
                self.game_n += 1
                self.n_iter = 0
                self.last_games_rews.append(self.last_game_rew)

        # compute the discount reward of the memories and return it
        return self.generalized_advantage_estimation(memories)

    def generalized_advantage_estimation(self, memories):
        '''
        Calculate the advantage diuscounted reward as in the paper
        '''
        upd_memories = []
        run_add = 0

        for t in reversed(range(len(memories) - 1)):
            if memories[t].done:
                run_add = memories[t].reward
            else:
                sigma = memories[t].reward + self.gamma * memories[t + 1].value - memories[t].value
                run_add = sigma + run_add * self.gamma * self.gae_lambda

            ## NB: the last memoy is missing
            # Update the memories with the discounted reward
            upd_memories.append(Memory(obs=memories[t].obs, action=memories[t].action, new_obs=memories[t].new_obs,
                                       reward=run_add + memories[t].value, done=memories[t].done,
                                       value=memories[t].value, adv=run_add))

        return upd_memories[::-1]


def log_policy_prob(mean, std, actions_critic, actions_expert=None):
    if actions_expert is not None:
        act_log_softmax = -((mean - (actions_critic - actions_expert) ** 2) ** 2) / (
                    2 * torch.exp(std).clamp(min=1e-4)) - torch.log(torch.sqrt(2 * math.pi * torch.exp(std)))
    else:
        act_log_softmax = -((mean - actions_critic) ** 2) / (2 * torch.exp(std).clamp(min=1e-4)) - torch.log(
            torch.sqrt(2 * math.pi * torch.exp(std)))

    return act_log_softmax


def compute_log_policy_prob(memories, nn_policy, device):
    '''
    Run the policy on the observation in the memory and compute the policy log probability
    '''
    n_mean, log_std = nn_policy(torch.tensor(np.array([m.obs for m in memories], dtype=np.float32)).to(device))
    n_mean = n_mean.type(torch.DoubleTensor)
    logstd = log_std.type(torch.DoubleTensor)

    actions_critic = torch.DoubleTensor(np.array([m.action for m in memories])).to(device)
    actions_expert = decide(np.array([m.obs for m in memories], dtype=np.float32))  # .to(device))

    return log_policy_prob(n_mean, logstd, actions_critic.to(n_mean.device)) # , actions_expert=actions_expert)


def clipped_PPO_loss(memories, nn_policy, nn_value, old_log_policy, adv, epsilon, device):
    '''
    Clipped PPO loss as in the paperself.
    It return the clipped policy loss and the value loss
    '''

    # state value
    rewards = torch.tensor(np.array([m.reward for m in memories], dtype=np.float32)).to(device)
    value = nn_value(torch.tensor(np.array([m.obs for m in memories], dtype=np.float32)).to(device))
    # Value loss
    vl_loss = F.mse_loss(value.squeeze(-1), rewards)

    # actions_critic = torch.DoubleTensor(np.array([m.action for m in memories])).to(device)
    # actions_expert = decide(np.array([m.obs for m in memories], dtype=np.float32))  # .to(device))
    # expert_loss = Variable(nn.MSELoss()(actions_expert.to(actions_critic.device), value), requires_grad=True)

    new_log_policy = compute_log_policy_prob(memories, nn_policy, device)
    rt_theta = torch.exp(new_log_policy - old_log_policy.detach()).cuda()
    # rt_theta = torch.exp(new_log_policy - old_log_policy.detach()).cuda()  + expert_loss
    # vl_loss = expert_loss

    adv = adv.unsqueeze(-1)  # add a dimension because rt_theta has shape: [batch_size, n_actions]
    pg_loss = -torch.mean(torch.min(rt_theta.to(device) * adv, torch.clamp(rt_theta.to(device), 1 - epsilon,
                                                                           1 + epsilon) *
                                    adv))

    return pg_loss, vl_loss


class Agent(object):
    def __init__(self, logger, state_size, action_size, device):
        self.best_reward_so_far = -1e6
        self.checkpoint_dir = "/home/mila/g/golemofl/"
        self.state_size = state_size
        self.device = device

        self.model = ACNet(state_size, action_size, device)

        self.optimizer_policy = optim.Adam(self.model.agent_policy.parameters(), lr=config.POLICY_LR )
        self.optimizer_value = optim.Adam(self.model.agent_value.parameters(), lr=config.VALUE_LR)

        self.resume_checkpoint("/home/mila/g/golemofl//best-checkpoint.pth")

    def save_checkpoint(self, epoch, info='', test_reward=None):
        """Saves a model checkpoint"""
        state = {
            'info': info,
            'epoch': epoch,
            'agent_policy': self.model.agent_policy.state_dict(),
            'agent_value': self.model.agent_value.state_dict(),
            'optimizer_policy': self.optimizer_policy.state_dict(),
            'optimizer_value': self.optimizer_value.state_dict(),
            'test_reward': test_reward
        }
        os.makedirs(self.checkpoint_dir, exist_ok=True)
        ckp_name = 'best-checkpoint.pth' if info == 'best' else f'checkpoint-epoch{epoch}_{info}.pth'
        filename = os.path.join(self.checkpoint_dir, ckp_name)
        torch.save(state, filename)

    def resume_checkpoint(self, resume_path):
        """Resumes training from an existing model checkpoint"""
        if os.path.exists(resume_path):
            print("Loading checkpoint: {} ...".format(resume_path))
            checkpoint = torch.load(resume_path)
            self.model.agent_policy.load_state_dict(checkpoint['agent_policy'])
            self.model.agent_value.load_state_dict(checkpoint['agent_value'])
            self.optimizer_policy.load_state_dict(checkpoint['optimizer_policy'])
            self.optimizer_value.load_state_dict(checkpoint['optimizer_value'])
            self.best_reward_so_far = float(checkpoint["test_reward"])
            print(f"Checkpoint loaded. Resume training with reward {self.best_reward_so_far}")

    def save_onnx_checkpoint(self, epoch, info=''):
        """Create an ONNX checkpoint"""
        dummy_input = torch.randn((1, self.state_size))
        dummy_input_t = transforms(dummy_input, self.device)
        model = self.model.agent_policy.to(torch.device('cpu'))
        torch.onnx.export(model, dummy_input_t.to(torch.device('cpu')), f"{info}submission_actor_{epoch}.onnx",
                          verbose=False,
                          opset_version=10,
                          export_params=True, do_constant_folding=True)

    def train(self, episode, batch, old_log_policy, batch_adv):
        pol_loss_acc = []
        val_loss_acc = []
        for s in range(config.PPO_EPOCHS):
            for mb in range(0, len(batch), config.BATCH_SIZE):
                mini_batch = batch[mb:mb + config.BATCH_SIZE]
                minib_old_log_policy = old_log_policy[mb:mb + config.BATCH_SIZE]
                minib_adv = batch_adv[mb:mb + config.BATCH_SIZE]

                pol_loss, val_loss = clipped_PPO_loss(mini_batch, self.model.agent_policy, self.model.agent_value,
                                                      minib_old_log_policy,
                                                      minib_adv, config.CLIP_EPS, device)

                # expert_loss = nn.MSELoss()(decide(state_t.detach().cpu().numpy).cuda(), action_t)
                self.optimizer_policy.zero_grad()
                pol_loss.backward()
                self.optimizer_policy.step()

                self.optimizer_value.zero_grad()
                val_loss.backward()
                self.optimizer_value.step()

                pol_loss_acc.append(float(pol_loss))
                val_loss_acc.append(float(val_loss))

        # wandb.log({'policy_loss': np.array(pol_loss_acc).mean()}, step=episode)
        # writer.add_scalar('vl_loss', np.mean(val_loss_acc), n_iter)
        # writer.add_scalar('rew', env.last_game_rew, n_iter)
        # writer.add_scalar('10rew', np.mean(env.last_games_rews[-100:]), n_iter)

    def test_game(self, tst_env, agent_policy, test_episodes):
        '''
        Execute test episodes on the test environment
        '''

        reward_games = []
        steps_games = []
        for _ in range(test_episodes):
            obs = tst_env.reset()
            rewards = 0
            steps = 0
            while True:
                ag_mean, sigma_logits = agent_policy(torch.tensor(obs))
                sigma = F.softplus(sigma_logits) + 1e-5
                policy_dist = Normal(loc=ag_mean, scale=sigma.to(ag_mean.device))
                action = policy_dist.sample()
                action = action.cpu().squeeze().numpy()

                action_original = np.clip(ag_mean.data.cpu().numpy().squeeze(), -1, 1)

                next_obs, reward, done, _ = tst_env.step(action)
                steps += 1
                obs = next_obs
                rewards += reward

                if done:
                    reward_games.append(rewards)
                    steps_games.append(steps)
                    obs = tst_env.reset()
                    break

        return np.mean(reward_games), np.mean(steps_games)


class Runner(object):
    def __init__(self, env, agent):
        self.env = env
        self.agent = agent
        self.n_iter = 0
        self.n_tests = 0
        self.best_reward_so_far = self.agent.best_reward_so_far

    def run(self):
        while self.n_iter < config.MAX_ITER:

            batch = env.steps(self.agent.model.agent_policy, self.agent.model.agent_value)
            old_log_policy = compute_log_policy_prob(batch, self.agent.model.agent_policy, device)

            # Gather the advantage from the memory..
            batch_adv = np.array([m.adv for m in batch])
            # .. and normalize it to stabilize network
            batch_adv = (batch_adv - np.mean(batch_adv)) / (np.std(batch_adv) + 1e-7)
            batch_adv = torch.tensor(batch_adv).to(device)

            self.agent.train(self.n_iter, batch, old_log_policy, batch_adv)

            # Test the agent
            if self.n_iter % config.N_ITER_TEST == 0:
                test_rews, test_stps = self.agent.test_game(test_env, self.agent.model.agent_policy,
                                                            config.test_episodes)
                print(' > Testing..', self.n_iter, test_rews, test_stps)
                if test_rews > self.best_reward_so_far:
                    self.agent.save_checkpoint(epoch=self.n_iter, info=f"best", test_reward=test_rews)
                    self.best_reward_so_far = test_rews
                    print('=> Best test!! Reward:{:.2f}  Steps:{}'.format(test_rews, test_stps))
                    wandb.log({"test reward": test_rews, "steps_test": self.n_tests})

                self.n_tests += 1
            self.n_iter += 1


Memory = namedtuple('Memory', ['obs', 'action', 'new_obs', 'reward', 'done', 'value', 'adv'], rename=False)

now = datetime.datetime.now()
date_time = "{}_{}.{}.{}".format(now.day, now.hour, now.minute, now.second)

env = Env(config.ENV_NAME, config.TRAJECTORY_SIZE, config.GAMMA, config.GAE_LAMBDA)

test_env = gym.make(config.ENV_NAME)

logger = Logger()
agent = Agent(logger, env.env.observation_space.shape, env.env.action_space.shape, device)
runner = Runner(env=env.env, agent=agent)
runner.run()
