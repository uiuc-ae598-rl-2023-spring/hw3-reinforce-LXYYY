# HW3 Mike Liu xl142

## 1) Easy
### Learning Curve

![Learning Curve](./figures/test_gridworld_returns.png)

### Trajectory

![Trajectory](./figures/test_gridworld.png)

### Policy

![Policy](./figures/test_gridworld_policy.png)

## 2) Hard

### Learning Curve

![Learning Curve](./figures/test_gridworld_returns_hard.png)

### Trajectory

![Trajectory](./figures/test_gridworld_hard.png)

### Policy

![Policy](./figures/test_gridworld_policy_hard.png)

## 3) REINFORCE on Continuous Pendulum with Sparse Reward

I copied the code from HW2 and modified it to have continuous state and action space.

### Implementation

As instructed, I implemented a Gaussian Policy with 1 linear layer and 1 hidden layer with 64 units for each mu and sigma head.

To handle sparse rewards, I used GAE with lambda = 0.95, and a value network.

### Learning Curve

![Learning Curve](./figures/learning_curve_pendulum.svg)

### Trajectory

![Trajectory](./figures/test_pendulum.png)

### Video

figures/test_pendulum.gif

![Video](./figures/test_pendulum.gif)