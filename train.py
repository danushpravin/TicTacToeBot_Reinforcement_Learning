import torch
import torch.optim as optim
import random
from tic_tac_toe_env import TicTacToeEnv
from q_network import QNetwork
from replay_buffer import ReplayBuffer  # Assuming you've implemented this class

# Hyperparameters
input_size = 9
output_size = 9
learning_rate = 0.001
gamma = 0.9  # Discount factor
epsilon_start = 1.0  # Start with full exploration
epsilon_end = 0.1  # End with 10% exploration
epsilon_decay = 0.995  # Decay rate of epsilon
epochs = 1000  # Number of training episodes
batch_size = 64  # Batch size for replay buffer

# Initialize environment, Q-network, optimizer, and loss function
env = TicTacToeEnv()
q_network = QNetwork(input_size, output_size)
optimizer = optim.Adam(q_network.parameters(), lr=learning_rate)
criterion = torch.nn.MSELoss()

# Replay buffer to store past experiences
replay_buffer = ReplayBuffer(capacity=1000)

# Epsilon for exploration vs. exploitation
epsilon = epsilon_start

def select_action(state, player):
    """ Select action using epsilon-greedy strategy. """
    if random.random() < epsilon:
        return random.choice(env.available_actions())  # Exploration
    else:
        state_tensor = torch.tensor(state, dtype=torch.float32).unsqueeze(0)
        q_values = q_network(state_tensor)
        _, action = torch.max(q_values, dim=1)
        valid_actions = env.available_actions()
        # Ensure the action is valid
        return action.item() if action.item() in valid_actions else random.choice(valid_actions)

def train():
    global epsilon
    for epoch in range(epochs):
        state = env.reset()
        done = False
        player = 1  # Start with player 1

        while not done:
            action = select_action(state, player)
            next_state, reward, done = env.take_action(action, player)

            # Store the transition in the replay buffer
            replay_buffer.push((state, action, reward, next_state))

            # Train the Q-network using random samples from the replay buffer
            if replay_buffer.size() > batch_size:
                batch = replay_buffer.sample(batch_size)
                for s, a, r, ns in batch:
                    state_tensor = torch.tensor([player * x for x in s], dtype=torch.float32).unsqueeze(0)
                    next_state_tensor = torch.tensor([player * x for x in ns], dtype=torch.float32).unsqueeze(0)

                    # Get current Q-values and the target Q-value
                    q_values = q_network(state_tensor)
                    next_q_values = q_network(next_state_tensor)
                    target_q_value = r + gamma * torch.max(next_q_values) if not done else r
                    target_q_value = torch.tensor(target_q_value, dtype=torch.float32)

                    # Calculate the loss
                    loss = criterion(q_values[0][a], target_q_value)

                    # Perform backpropagation and optimize
                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()

            # Update the state and switch player
            state = next_state
            player *= -1  # Switch player (1 -> -1, -1 -> 1)

        # Decay epsilon after each episode to reduce exploration over time
        if epsilon > epsilon_end:
            epsilon *= epsilon_decay

        if (epoch + 1) % 1000 == 0:
            print(f"Epoch {epoch + 1}/{epochs}, Epsilon: {epsilon:.4f}")

if __name__ == "__main__":
    train()
