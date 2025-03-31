# tic_tac_toe_env.py
class TicTacToeEnv:
    def __init__(self):
        self.board = [0]*9
        self.done = False

    def reset(self):
        self.board = [0]*9
        self.done = False
        return self.board
    
    def render(self):
        for i in range(0,9,3):
            print(self.board[i:i+3])
    
    def available_actions(self):
        return [i for i, x in enumerate(self.board) if x == 0]
    
    def take_action(self, action, player):
        if self.board[action] == 0:
            self.board[action] = player
            reward = 0
            if self.check_winner(player):
                reward = 1
                self.done = True
            elif self.check_winner(-player):
                reward = -1
                self.done = True
            elif all(x!=0 for x in self.board):
                reward = 0.5
                self.done = True
            else:
                if self.opponent_wins_next(player):
                    reward = -0.5

            return self.board, reward, self.done
        else:
            return self.board, -1, self.done
    
    def check_winner(self, player):
        win_conditions = [
            [0,1,2], [3,4,5], [6,7,8],
            [0,3,6], [1,4,7], [2,5,8],
            [0,4,8], [2,4,6]
        ]
        for condition in win_conditions:
            if all(self.board[i] == player for i in condition):
                return True
        return False
    
    def opponent_wins_next(self, player):
        opponent = -player
        for action in self.available_actions():
            self.board[action] = opponent
            if self.check_winner(opponent):
                self.board[action] = 0
                return True
            self.board[action] = 0
        return False
