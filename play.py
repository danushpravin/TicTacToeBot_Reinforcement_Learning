# play.py
from tic_tac_toe_env import TicTacToeEnv
from q_network import QNetwork
import random
import torch

# Initialize environment and Q-network
env = TicTacToeEnv()
input_size = 9
output_size = 9
q_network = QNetwork(input_size, output_size)

def select_action(state, player):
    state_tensor = torch.tensor(state, dtype=torch.float32)
    q_values = q_network(state_tensor)
    _, action = torch.max(q_values, dim=0)
    return action.item() if action.item() in env.available_actions() else random.choice(env.available_actions())

def play():
    state = env.reset()
    env.render()
    done = False
    player = 1  # You are always player 1 (X)

    while not done:
        if player == 1:
            action = int(input("Enter your move (0-8): "))
        else:
            action = select_action(state, player)

        state, _, done = env.take_action(action, player)
        env.render()
        player *= -1

        if env.check_winner(1):
            print("You win!")
            return
        elif env.check_winner(-1):
            print("Bot wins!")
            return
        elif done:
            print("It's a draw!")
            return

if __name__ == "__main__":
    play()
