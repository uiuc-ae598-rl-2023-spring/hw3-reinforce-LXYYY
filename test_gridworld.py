import random
import numpy as np
import matplotlib.pyplot as plt
import gridworld
from reinforce import TabularPolicy, train_policy
import torch
from torch import optim


def main():
    # Create environment
    env = gridworld.GridWorld(hard_version=False)

    # Initialize simulation
    s = env.reset()

    policy = TabularPolicy(env.num_states, env.num_actions)
    optimizer = optim.Adam(policy.parameters(), lr=0.01)

    episodes = 500
    returns = train_policy(policy, optimizer, env, episodes, gamma=1)

    # plot returns over episodes
    plt.figure()
    plt.plot(range(episodes), returns)
    plt.xlabel('Episode')
    plt.ylabel('Total Reward')
    plt.savefig('figures/test_gridworld_returns.png')
    plt.figure()

    # Create log to store data from simulation
    log = {
        't': [0],
        's': [s],
        'a': [],
        'r': [],
    }

    s = env.reset()
    # Simulate until episode is done
    done = False
    while not done:
        action_probs = torch.softmax(policy.weights[s], dim=0)
        a = torch.argmax(action_probs).item()
        (s, r, done) = env.step(a)
        log['t'].append(log['t'][-1] + 1)
        log['s'].append(s)
        log['a'].append(a)
        log['r'].append(r)

    # Plot data and save to png file
    plt.plot(log['t'], log['s'])
    plt.plot(log['t'][:-1], log['a'])
    plt.plot(log['t'][:-1], log['r'])
    plt.legend(['s', 'a', 'r'])
    plt.savefig('figures/test_gridworld.png')


if __name__ == '__main__':
    main()
