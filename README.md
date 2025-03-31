# Tic-Tac-Toe Reinforcement Learning Bot

This project aims to build an intelligent Tic-Tac-Toe bot using Q-learning and reinforcement learning techniques. The bot is trained through self-play, which helps it learn optimal strategies for winning the game. The core components include the environment (`tic_tac_toe_env`), Q-network (`q_network`), replay buffer (`replay_buffer`), and training loop.

## Key Modifications and Improvements

The following changes were made to improve the model’s learning process:

### 1. **Improved Reward System**

The reward system was enhanced to encourage smarter play by the agent. The original reward scheme was overly simplistic and only provided feedback for winning and invalid moves. The new reward scheme is as follows:

- **Win**: +1 (reward for winning the game)
- **Loss**: -1 (penalty for losing the game)
- **Draw**: +0.5 (reward for a draw, as it's a neutral outcome)
- **Invalid Move**: -1 (penalty for making an invalid move)
- **Giving the Opponent a Winning Move**: -0.5 (penalty for making a move that allows the opponent to win)

This new reward system enables the bot to learn not just to win, but to avoid losing and to make smarter moves.

### 2. **Replay Buffer (Experience Replay)**

One of the major improvements is the introduction of a **Replay Buffer**. In standard Q-learning, the agent updates its Q-values immediately after each move. This can lead to unstable learning because the model may overfit to recent experiences or early mistakes. To improve this, the replay buffer stores previous experiences (`state`, `action`, `reward`, `next_state`) and samples them in batches during training. This stabilizes learning and reduces variance.

The Replay Buffer stores up to **10,000** experiences, and training is performed on random batches of experiences, making the model more stable and efficient. The buffer helps the model learn from past mistakes and ensures that the agent does not overfit to any specific sequence of moves.

### 3. **Epsilon Decay (Exploration vs. Exploitation)**

In reinforcement learning, an **ε-greedy** strategy is often used to balance exploration (trying new moves) and exploitation (choosing the best-known move). The original implementation used a fixed **epsilon** value of 0.1 for exploration. This did not allow the model to explore enough in the beginning and quickly transition to exploitation as it learned.

To address this, I implemented an **epsilon decay** strategy, where:

- Epsilon starts at **1.0** (100% exploration).
- It gradually decays to **0.1** (10% exploration) over time.
- The decay rate is set to **0.995**, ensuring that the agent explores less as it becomes more confident in its strategy.

This strategy helps the agent explore the state space early in the training process and then shift toward exploiting what it has learned.

### 4. **Self-Play for Training**

The original model only played as a single player, which limited its ability to improve. The key to making the model more intelligent is allowing **self-play**, where two instances of the bot play against each other. This allows the bot to learn both offensive and defensive strategies and forces it to improve its understanding of the game. 

By having the bot play against itself, it can practice more advanced strategies, such as forking, and adapt to a wider range of scenarios. This significantly speeds up learning and makes the model much stronger.

## Code Breakdown

The code consists of several main components:

### 1. **TicTacToeEnv** (Environment)

This class encapsulates the game logic and serves as the environment for the reinforcement learning agent. It includes methods for:

- Resetting the game state.
- Checking available actions.
- Applying a move and updating the board state.
- Determining the outcome of the game (win, lose, draw).
- Managing the switch between two players.

### 2. **QNetwork** (Model)

This is the neural network model used to approximate the Q-values for each possible action in a given state. The network has an input size of 9 (representing the 3x3 Tic-Tac-Toe board) and outputs a Q-value for each of the 9 possible actions. The network is trained to predict the Q-values for each action, and these values are used to guide the agent’s decision-making.

### 3. **ReplayBuffer** (Experience Replay)

The replay buffer stores past experiences (`state`, `action`, `reward`, `next_state`). It helps to stabilize the training process by sampling batches of experiences to train on, instead of using only the most recent experience. The buffer has a maximum capacity, and once full, it begins to overwrite the oldest experiences.

### 4. **Train** (Training Loop)

The training loop orchestrates the learning process. It:

- Initializes the environment and Q-network.
- Iterates through multiple training episodes.
- Selects actions using the ε-greedy strategy.
- Stores experiences in the replay buffer.
- Samples from the replay buffer to train the Q-network.
- Decays epsilon after each episode to reduce exploration over time.
  
The training loop continues for a set number of episodes and gradually improves the agent's performance.

## Dependencies

- Python 3.x
- PyTorch
- NumPy