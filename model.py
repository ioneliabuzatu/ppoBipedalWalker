import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Normal


class ActorNet(nn.Module):
    """Actor network (policy)
    """

    def __init__(self, state_size, action_size, hidden_size):
        super(ActorNet, self).__init__()
        self.actor_fc_1 = nn.Linear(state_size, 2 * hidden_size)
        self.actor_fc_2 = nn.Linear(2 * hidden_size, hidden_size)

        self.actor_mu = nn.Linear(hidden_size, action_size)
        self.actor_sigma_logits = nn.Linear(hidden_size, action_size)

        self.actor_mu.weight.data.mul_(0.01)
        self.actor_sigma_logits.weight.data.mul_(0.01)

    def forward(self, x):
        x = torch.relu(self.actor_fc_1(x))
        x = torch.relu(self.actor_fc_2(x))
        mu = torch.tanh(self.actor_mu(x))
        sigma_logits = self.actor_sigma_logits(x)
        return mu, sigma_logits


class CriticNet(nn.Module):
    """Critic network computing the state value
    """

    def __init__(self, state_size, action_size, hidden_size):
        super(CriticNet, self).__init__()

        self.critic_fc_1 = nn.Linear(state_size, 2 * hidden_size)
        self.critic_fc_2 = nn.Linear(2 * hidden_size, hidden_size)

        # init_ = lambda m: init(m, init_normc_, lambda x: nn.init.constant_(x, 0))
        self.critic_state_value = nn.Linear(hidden_size, 1)

    def forward(self, x):
        x = torch.relu(self.critic_fc_1(x))
        x = torch.relu(self.critic_fc_2(x))
        state_value = self.critic_state_value(x)
        return state_value


class ActorCriticNet(nn.Module):
    """Combining both networks and add helper methods to act and evaluate samples.
    """

    def __init__(self, state_size, action_size, hidden_size):
        super(ActorCriticNet, self).__init__()
        self.actor = ActorNet(state_size, action_size, hidden_size)
        self.critic = CriticNet(state_size, action_size, hidden_size)

    def forward(self, x):
        x = x.reshape(1, -1)
        return self.act(x)

    def act(self, state):
        mu, sigma_logits = self.actor(state)
        sigma = F.softplus(sigma_logits) + 0.00005
        distribution = Normal(mu, sigma)
        action = distribution.sample()
        action = torch.clamp(action, -1., 1.)
        log_probs = distribution.log_prob(action)
        return action, log_probs

    def evaluate(self, state, action):
        mu, sigma_logits = self.actor(state)
        sigma = F.softplus(sigma_logits) + 0.00005
        distribution = Normal(mu, sigma)
        log_probs = distribution.log_prob(action)
        entropy = distribution.entropy()

        state_value = self.critic(state)

        return state_value, log_probs, entropy


class A2C_policy(nn.Module):
    '''
    Policy neural network
    '''

    def __init__(self, input_shape, n_actions, hidden_size=32, activation=torch.relu):
        super(A2C_policy, self).__init__()

        self.n_actions = n_actions[0]
        self.activation = activation

        self.lp = nn.Sequential(
            nn.Linear(input_shape[0], hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU())

        self.mean_l = nn.Linear(hidden_size, n_actions[0])
        self.mean_l.weight.data.mul_(0.1)

        self.logstd = nn.Parameter(torch.zeros(n_actions[0]))

        self.device = list(self.lp.parameters())[0]

    def forward(self, x):
        x = x.to(self.device)
        ot_n = self.lp(x.float())
        logstd = nn.Parameter(torch.zeros(self.n_actions))
        return torch.tanh(self.mean_l(ot_n)), logstd


class A2C_value(nn.Module):
    '''
    critic neural network
    '''

    def __init__(self, input_shape, hidden_size=32):
        super(A2C_value, self).__init__()

        self.lp = nn.Sequential(
            nn.Linear(input_shape[0], hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, 1))

        self.device = list(self.lp.parameters())[0]

    def forward(self, x):
        x = x.to(self.device)
        return self.lp(x.float())


class ACNet(nn.Module):
    """Combining both networks and add helper methods to act and evaluate samples.
    """

    def __init__(self, state_size, action_size, device, hidden_size=32):
        super(ACNet, self).__init__()
        self.agent_policy = A2C_policy(state_size, action_size).to(device)
        self.agent_value = A2C_value(state_size).to(device)
        self.device = device

    def forward(self, x):
        x = x.reshape(1, -1)
        return self.act(x)

    def act_(self, state):
        mu = self.agent_policy(state).to(self.device)
        mu = mu.type(torch.DoubleTensor)
        log_probs = self.agent_policy.logstd.type(torch.DoubleTensor)
        action = mu.data.cpu().numpy() + np.exp(log_probs) * np.random.normal(size=log_probs.shape)
        action = np.clip(action, -1, 1)
        return action, log_probs

    def act(self, state):
        mu, sigma_logits = self.agent_policy(state)
        sigma = F.softplus(sigma_logits) + 1e-5
        distribution = Normal(mu, sigma)
        action = distribution.sample()
        # action = torch.clamp(action, -1., 1.)
        log_probs = distribution.log_prob(action)
        return action, log_probs

    def evaluate(self, state, action):
        mu, sigma_logits = self.actor(state)
        sigma = F.softplus(sigma_logits) + 1e-5
        distribution = Normal(mu, sigma)
        log_probs = distribution.log_prob(action)
        entropy = distribution.entropy()

        state_value = self.critic(state)

        return state_value, log_probs, entropy
