import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions import Categorical
import matplotlib.pyplot as plt
import numpy as np


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


def calculate_returns(rewards, gamma):
    returns = []
    R = 0
    for r in reversed(rewards):
        R = r + gamma * R
        returns.insert(0, R)
    return returns


def update_policy(policy, optimizer, log_probs, returns):
    loss = []
    for log_prob, G_t in zip(log_probs, returns):
        loss.append(-log_prob * G_t)

    loss = torch.stack(loss).sum()

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    return loss.item()
