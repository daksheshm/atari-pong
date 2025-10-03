# Deep Q-Network for Atari Pong

This project is an implementation of a Deep Q-Network (DQN) in **PyTorch** that learns to play the classic Atari game, Pong, directly from raw pixel data. The agent demonstrates the capability of reinforcement learning to master complex visual tasks, achieving a **consistent win rate of over 95%** against the game's built-in, hard-coded AI.

## Project Goal
The goal of this project was to build a reinforcement learning agent from scratch that could successfully learn and master the game of Pong. This involved implementing foundational concepts from the original DeepMind paper, "Playing Atari with Deep Reinforcement Learning," to achieve superhuman performance.

## Core Concepts & Implementation
This project is built on several key concepts in deep reinforcement learning to ensure stable and effective training:

-   **Deep Q-Network (DQN):** A convolutional neural network is used to approximate the action-value function (Q-function). It takes the raw pixel data from the game screen as input and outputs the expected return for each possible action (moving the paddle up or down).

-   **Experience Replay:** To break the correlation between consecutive frames and stabilize training, the agent's experiences (state, action, reward, next state) are stored in a custom replay buffer. During training, random mini-batches of these experiences are sampled to update the network, which helps in decorrelating the data and improving learning stability.

-   **Fixed Q-Target Network:** A separate "target" network is used to generate the target Q-values for the loss calculation. The weights of this target network are frozen for a number of steps and only periodically updated with the weights from the main DQN. This helps prevent the learning process from becoming unstable due to a constantly shifting target.

-   **Epsilon-Greedy Exploration:** To balance the exploration-exploitation trade-off, an epsilon-greedy strategy is employed. The agent chooses a random action with a probability of ε (epsilon), and exploits its learned knowledge by choosing the best-known action with a probability of 1-ε. The value of ε is typically annealed (decreased) over the course of training.

## Installation

To run this project, it is recommended to use a virtual environment.

1.  **Clone the repository:**
    ```bash
    git clone https://github.com/daksheshm/atari-pong.git
    cd atari-pong
    ```

2.  **Install the required dependencies:**
    ```bash
    pip install -r requirements.txt
    ```
    The `requirements.txt` should contain:
    ```
    torch
    torchvision
    gymnasium[atari]
    gymnasium[accept-rom-license]
    numpy
    tqdm
    ```
