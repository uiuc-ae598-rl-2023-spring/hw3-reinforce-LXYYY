import random
import numpy as np
import matplotlib.pyplot as plt
import pendulum
from reinforce import *

train = False


def main():
    # Create environment
    env = pendulum.Pendulum()

    # Initialize simulation
    s = env.reset()

    state_dim = 2
    action_dim = 1

    policy = GaussianPolicy(state_dim, action_dim, 64)
    value_network = ValueNetwork(state_dim, 64)

    returns = []
    if train:
        policy_optimizer = optim.Adam(policy.parameters(), lr=1e-3)
        value_optimizer = optim.Adam(value_network.parameters(), lr=1e-3)

        returns = train_gaussian_policy(policy, value_network, policy_optimizer, value_optimizer, env, episodes=50000,
                                        gamma=0.99, tau=0.95)

        # plot returns over episodes
        plt.figure()
        plt.plot(range(50000), returns)
        plt.xlabel('Episode')
        plt.ylabel('Total Reward')
        plt.savefig('figures/test_gaussian_pendulum_returns.png')
        plt.figure()
    else:
        policy.load_state_dict(torch.load("policy_checkpoint_50000.pth"))
        value_network.load_state_dict(torch.load("value_checkpoint_50000.pth"))

    ######################################
    #
    #   EXAMPLE OF CREATING A VIDEO
    #

    # Define a policy that maps every state to the "zero torque" action

    # Simulate an episode and save the result as an animated gif
    env.video(policy, filename='figures/test_pendulum.gif')

    #
    ######################################

    ######################################
    #
    #   EXAMPLE OF CREATING A PLOT
    #

    # Initialize simulation
    s = env.reset()

    # Create dict to store data from simulation
    data = {
        't': [0],
        's': [s],
        'a': [],
        'r': [],
    }

    # Simulate until episode is done
    done = False
    while not done:
        a, _ = policy(s)
        a = a.detach().numpy().flatten()[0]
        (s, r, done) = env.step(a)
        data['t'].append(data['t'][-1] + 1)
        data['s'].append(s)
        data['a'].append(a)
        data['r'].append(r)

    # Parse data from simulation
    data['s'] = np.array(data['s'])
    theta = data['s'][:, 0]
    thetadot = data['s'][:, 1]
    tau = [env._a_to_u(a) for a in data['a']]

    # Plot data and save to png file
    fig, ax = plt.subplots(3, 1, figsize=(10, 10))
    ax[0].plot(data['t'], theta, label='theta')
    ax[0].plot(data['t'], thetadot, label='thetadot')
    ax[0].legend()
    ax[1].plot(data['t'][:-1], tau, label='tau')
    ax[1].legend()
    ax[2].plot(data['t'][:-1], data['r'], label='r')
    ax[2].legend()
    ax[2].set_xlabel('time step')
    plt.tight_layout()
    plt.savefig('figures/test_discreteaction_pendulum.png')


if __name__ == '__main__':
    main()
