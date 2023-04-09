import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions import Categorical


class TabularPolicy(nn.Module):
    def __init__(self, num_states, num_actions):
        super(TabularPolicy, self).__init__()
        self.weights = nn.Parameter(torch.zeros(num_states, num_actions))

    def forward(self, state):
        return Categorical(logits=self.weights[state])


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
