"""
Implementação Pente
"""
import numpy as np
from .game_engine import (
    check_win_pente, 
    get_legal_moves_gomoku as get_legal_moves,
    detect_capture_pente,
    apply_capture_pente,
    check_win_by_capture
)


class PenteGame:
    def __init__(self, board_size=15):
        self.board_size = board_size
        self.board = np.zeros((board_size, board_size), dtype=np.int8)
        self.current_player = 1
        self.winner = None
        self.move_history = []
        self.captures = {1: 0, 2: 0}
        
    def reset(self):
        """Reseta o jogo"""
        self.board = np.zeros((self.board_size, self.board_size), dtype=np.int8)
        self.current_player = 1
        self.winner = None
        self.move_history = []
        self.captures = {1: 0, 2: 0}
        
    def is_valid_move(self, row, col):
        """Verifica se um movimento é válido"""
        if row < 0 or row >= self.board_size or col < 0 or col >= self.board_size:
            return False
        return self.board[row, col] == 0
    
    def get_valid_moves(self):
        return get_legal_moves(self.board)
    
    def check_capture(self, row, col):
        captured_count = apply_capture_pente(self.board, row, col)
        return captured_count
    
    def make_move(self, row, col):
        if not self.is_valid_move(row, col):
            return False

        player_who_moved = self.current_player
        
        self.board[row, col] = player_who_moved
        self.move_history.append((row, col, player_who_moved))

        captured = self.check_capture(row, col)
        if captured > 0:
            self.captures[player_who_moved] += captured
        
        if check_win_by_capture(self.captures[1], self.captures[2]) > 0:
            self.winner = player_who_moved

        elif check_win_pente(self.board) == player_who_moved:
            self.winner = player_who_moved
        
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
            return 0
        return None
    
    def get_board_copy(self):
        return self.board.copy()
    
    def get_board_for_player(self, player):
        if player == 1:
            return self.board.copy()
        else:
            board = self.board.copy()
            board[board == 1] = 9
            board[board == 2] = 1
            board[board == 9] = 2
            return board
    
    @staticmethod
    def check_winner(board, captures_p1=0, captures_p2=0):
        win_by_cap = check_win_by_capture(captures_p1, captures_p2)
        if win_by_cap > 0:
            return win_by_cap
        return check_win_pente(board)
    
    @staticmethod
    def get_legal_moves(board):
        return get_legal_moves(board)
    
    def display(self):
        symbols = {0: '·', 1: '●', 2: '○'}
        print(f"Captures - Black (●): {self.captures[1]}, White (○): {self.captures[2]}")
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
