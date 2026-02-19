"""
Gomoku Implementation
"""
import numpy as np
from .game_engine import check_win_gomoku, get_legal_moves_gomoku


class GomokuGame:
    def __init__(self, board_size=15):
        self.board_size = board_size
        self.board = np.zeros((board_size, board_size), dtype=np.int8)
        self.current_player = 1
        self.winner = None
        self.move_history = []
        
    def reset(self):
        """Resets the game."""
        self.board = np.zeros((self.board_size, self.board_size), dtype=np.int8)
        self.current_player = 1
        self.winner = None
        self.move_history = []
        
    def is_valid_move(self, row, col):
        """Checks if a move is valid."""
        if row < 0 or row >= self.board_size or col < 0 or col >= self.board_size:
            return False
        return self.board[row, col] == 0
    
    @staticmethod
    def check_winner(board):
        return check_win_gomoku(board)
    
    @staticmethod
    def get_legal_moves(board):
        return get_legal_moves_gomoku(board)
    
    def get_valid_moves(self):
        return get_legal_moves_gomoku(self.board)
    
    def make_move(self, row, col):
        """
        Attempts to make a move at (row, col).
        Returns True if successful, False otherwise.
        """
        if not self.is_valid_move(row, col):
            return False
        
        self.board[row, col] = self.current_player
        self.move_history.append((row, col, self.current_player))
        
        if check_win_gomoku(self.board) == self.current_player:
            self.winner = self.current_player

        self.current_player = 3 - self.current_player
        return True
    
    def is_game_over(self):
        if self.winner is not None:
            return True
        return len(self.get_valid_moves()) == 0
    
    def get_winner(self):
        if self.winner:
            return self.winner
        if self.is_game_over():
            return 0  # Draw
        return None
    
    def get_board_copy(self):
        return self.board.copy()
    
    def get_board_for_player(self, player):
        """
        Returns the board from the perspective of 'player'.
        If player is 2 (White), swaps 1s and 2s so the neural network
        always sees 'self' as 1.
        """
        if player == 1:
            return self.board.copy()
        else:
            board = self.board.copy()
            board[board == 1] = 9
            board[board == 2] = 1
            board[board == 9] = 2
            return board
    
    def display(self):
        symbols = {0: '·', 1: '●', 2: '○'}
        print('   ', end='')
        for c in range(self.board_size):
            print(f'{c:2}', end=' ')
        print()
        for r in range(self.board_size):
            print(f'{r:2} ', end='')
            for c in range(self.board_size):
                print(f' {symbols[self.board[r, c]]} ', end='')
            print()
        print()
