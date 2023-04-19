import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions import Categorical
import matplotlib.pyplot as plt
import numpy as np
from torch.utils.tensorboard import SummaryWriter
import time


class TabularPolicy(nn.Module):
    def __init__(self, num_states, num_actions):
        super(TabularPolicy, self).__init__()
        self.weights = nn.Parameter(torch.zeros(num_states, num_actions))

    def forward(self, state):
        return Categorical(logits=self.weights[state])

    def save(self, file):
        torch.save(self.weights, file)

    def load(self, file):
        self.weights = torch.load(file)


class GaussianPolicy(nn.Module):
    def __init__(self, state_dim, action_dim, hidden_dim):
        super(GaussianPolicy, self).__init__()
        self.fc1 = nn.Linear(state_dim, hidden_dim)
        self.mu_head = nn.Linear(hidden_dim, action_dim)
        self.sigma_head = nn.Linear(hidden_dim, action_dim)

    def forward(self, state):
        if not isinstance(state, torch.Tensor):
            state = torch.tensor(state, dtype=torch.float32)
        x = torch.tanh(self.fc1(state))
        mu = torch.tanh(self.mu_head(x))
        log_std = nn.functional.softplus(self.sigma_head(x))
        log_std = torch.clamp(log_std, min=-20, max=2)
        sigma = torch.exp(log_std)
        return mu, sigma


def calculate_gae(rewards, values, next_value, gamma, tau):
    gae = 0
    advantages = []
    returns = []
    for t in reversed(range(len(rewards))):
        delta = rewards[t] + gamma * next_value - values[t]
        gae = delta + gamma * tau * gae
        advantages.insert(0, gae)
        returns.insert(0, gae + values[t])
        next_value = values[t]
    return advantages, returns


class ValueNetwork(nn.Module):
    def __init__(self, state_dim, hidden_dim):
        super(ValueNetwork, self).__init__()
        self.fc1 = nn.Linear(state_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, 1)

    def forward(self, state):
        x = torch.relu(self.fc1(state))
        value = self.fc2(x)
        return value


def train_gaussian_policy(policy, value_network, policy_optimizer, value_optimizer, env, episodes, gamma=0.99,
                          tau=0.95):
    returns_out = []
    writer = SummaryWriter(log_dir='./logs/' + str(time.time()))
    for episode in range(episodes):
        state = env.reset()
        log_probs, values, rewards, states = [], [], [], []

        while True:
            state_tensor = torch.tensor(state, dtype=torch.float32).unsqueeze(0)
            mu, sigma = policy(state_tensor)
            action_dist = torch.distributions.Normal(mu, sigma)
            action = action_dist.sample()
            next_state, reward, done = env.step(action.squeeze(0).detach().numpy().flatten()[0])

            log_prob = action_dist.log_prob(action).sum(-1)
            value = value_network(state_tensor)

            log_probs.append(log_prob)
            values.append(value.item())
            rewards.append(reward)
            states.append(state)

            state = next_state

            if done:
                break

        next_value = value_network(torch.tensor(next_state, dtype=torch.float32).unsqueeze(0)).item()
        returns, advantages = calculate_gae(rewards, values, next_value, gamma, tau)
        policy_loss = update_policy(policy, policy_optimizer, log_probs, advantages)
        value_loss = update_value_network(value_network, value_optimizer, states, returns)

        returns_out.append(sum(rewards))

        # Logging to TensorBoard
        writer.add_scalar('Total Reward', sum(rewards), episode)
        writer.add_scalar('Policy Loss', policy_loss, episode)
        writer.add_scalar('Value Loss', value_loss, episode)

        print(
            f"Episode {episode + 1}: Policy Loss = {policy_loss:.4f}, Value Loss = {value_loss:.4f}, Total Reward = {sum(rewards)}")

        if (episode + 1) % 50 == 0:
            torch.save(policy.state_dict(), f"policy_checkpoint_{episode + 1}.pth")
            torch.save(value_network.state_dict(), f"value_checkpoint_{episode + 1}.pth")
            print(f"Saved checkpoints at episode {episode + 1}")

    return returns_out


def calculate_returns(rewards, gamma):
    returns = []
    R = 0
    for r in reversed(rewards):
        R = r + gamma * R
        returns.insert(0, R)
    return returns


def update_policy(policy, optimizer, log_probs, returns):
    loss = 0
    for log_prob, G_t in zip(log_probs, returns):
        loss += -log_prob * G_t

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    return loss.item()


def update_value_network(value_network, optimizer, states, returns):
    states = torch.tensor(states, dtype=torch.float32)
    returns = torch.tensor(returns, dtype=torch.float32).detach()

    values = value_network(states)
    loss = nn.MSELoss()(values.view(-1), returns)

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    return loss.item()


def train_policy(policy, optimizer, env, episodes, gamma=0.99):
    returns_out = []
    for episode in range(episodes):
        state = env.reset()
        log_probs, rewards = [], []

        while True:
            action_dist = policy(state)
            action = action_dist.sample()
            log_prob = action_dist.log_prob(action)
            next_state, reward, done = env.step(action.item())

            log_probs.append(log_prob)
            rewards.append(reward)

            state = next_state

            if done:
                break

        returns = calculate_returns(rewards, gamma)
        loss = update_policy(policy, optimizer, log_probs, returns)

        returns_out.append(sum(rewards))

        print(f"Episode {episode + 1}: Loss = {loss:.4f}, Total Reward = {sum(rewards)}")

    return returns_out


def action_to_arrow(action, length=0.5):
    #     1(up)
    #
    #
    # 2(left)
    # 0(right)
    # 3(down)
    arrow_map = {
        1: (0, length),
        0: (length, 0),
        3: (0, -length),
        2: (-length, 0)
    }
    return arrow_map[action]


def plot_policy(policy, width, height, file):
    plt.figure(figsize=(width, height))
    plt.xlim(0, width)
    plt.ylim(0, height)

    for y in range(height):
        for x in range(width):
            state = x + y * width
            action_probs = torch.softmax(policy.weights[state], dim=0)
            best_action = torch.argmax(action_probs).item()
            dx, dy = action_to_arrow(best_action, 0.2)

            plt.arrow(x + 0.5, y + 0.5, dx, dy, head_width=0.2, head_length=0.2, fc='k', ec='k')

    plt.grid()
    plt.xticks(np.arange(0, width + 1, 1))
    plt.yticks(np.arange(0, height + 1, 1))
    plt.gca().set_aspect('equal', adjustable='box')
    plt.savefig(file)



