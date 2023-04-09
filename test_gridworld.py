import random
import numpy as np
import matplotlib.pyplot as plt
import gridworld
from reinforce import *
import torch
from torch import optim

load = True


def main():
    for hard_version in [True, False]:
        # Create environment
        env = gridworld.GridWorld(hard_version=hard_version)

        # Initialize simulation
        s = env.reset()

        policy = TabularPolicy(env.num_states, env.num_actions)
        optimizer = optim.Adam(policy.parameters(), lr=0.01)

        episodes = 5000
        returns = None

        if load:
            policy.load('policy' + ('_hard' if hard_version else '') + '.pt')
        else:
            returns = train_policy(policy, optimizer, env, episodes, gamma=1)
            policy.save('policy' + ('_hard' if hard_version else '') + '.pt')

        plot_policy(policy, 5, 5, 'figures/test_gridworld_policy' + ('_hard' if hard_version else '') + '.png')

        # plot returns over episodes
        if returns is not None:
            plt.figure()
            plt.plot(range(episodes), returns)
            plt.xlabel('Episode')
            plt.ylabel('Total Reward')
            plt.savefig('figures/test_gridworld_returns' + ('_hard' if hard_version else '') + '.png')
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
        plt.savefig('figures/test_gridworld' + ('_hard' if hard_version else '') + '.png')


if __name__ == '__main__':
    main()
