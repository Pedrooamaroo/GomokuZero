"""
Game Engine - Gomoku & Pente
Numba @njit optimized functions
"""

import numpy as np
from numba import njit
import math


@njit(cache=True)
def check_win_gomoku(board):
    """
    Checks for a winner in Gomoku (5 or more in a row).
    
    Args:
        board: numpy array (board_size, board_size) with 0/1/2
    
    Returns:
        0 (no winner), 1 (player 1), 2 (player 2)
    """
    board_size = board.shape[0]
    for row in range(board_size):
        for col in range(board_size):
            player = board[row, col]
            if player == 0:
                continue
            
            directions = ((0, 1), (1, 0), (1, 1), (1, -1))
            for dr, dc in directions:
                count = 1
                r, c = row + dr, col + dc
                while 0 <= r < board_size and 0 <= c < board_size and board[r, c] == player:
                    count += 1
                    r += dr
                    c += dc
                r, c = row - dr, col - dc
                while 0 <= r < board_size and 0 <= c < board_size and board[r, c] == player:
                    count += 1
                    r -= dr
                    c -= dc
                
                if count >= 5:
                    return player
    
    return 0


@njit(cache=True)
def get_legal_moves_gomoku(board, distance=2):
    """
    Returns legal moves near existing stones (optimization).
    
    Args:
        board: game board
        distance: maximum distance from existing stones
    
    Returns:
        list of (row, col)
    """
    board_size = board.shape[0]
    moves = []
    
    if np.sum(board) == 0:
        center = board_size // 2
        return [(center, center)]
    
    for row in range(board_size):
        for col in range(board_size):
            if board[row, col] != 0:
                for dr in range(-distance, distance + 1):
                    for dc in range(-distance, distance + 1):
                        nr, nc = row + dr, col + dc
                        if (0 <= nr < board_size and 0 <= nc < board_size and 
                            board[nr, nc] == 0):
                            moves.append((nr, nc))
    
    if len(moves) == 0:
        return moves
    
    unique_moves = []
    seen = set()
    for move in moves:
        if move not in seen:
            unique_moves.append(move)
            seen.add(move)
    
    return unique_moves


@njit(cache=True)
def detect_winning_move_gomoku(board, player):
    """
    Detects moves that immediately win for the player.
    
    Returns:
        list of (row, col) that result in victory
    """
    board_size = board.shape[0]
    winning_moves = []
    board_copy = board.copy()
    
    for row in range(board_size):
        for col in range(board_size):
            if board_copy[row, col] != 0:
                continue
            
            board_copy[row, col] = player

            directions = ((0, 1), (1, 0), (1, 1), (1, -1))
            won = False
            
            for dr, dc in directions:
                count = 1
                r, c = row + dr, col + dc
                while 0 <= r < board_size and 0 <= c < board_size and board_copy[r, c] == player:
                    count += 1
                    r += dr
                    c += dc
                r, c = row - dr, col - dc
                while 0 <= r < board_size and 0 <= c < board_size and board_copy[r, c] == player:
                    count += 1
                    r -= dr
                    c -= dc
                
                if count >= 5:
                    won = True
                    break
            
            board_copy[row, col] = 0
            
            if won:
                winning_moves.append((row, col))
    
    return winning_moves


@njit(cache=True)
def detect_winning_move_pente(board, player, current_captures):
    """
    Detects moves that immediately win for the player (Pente).
    Considers:
    1. 5 in a row
    2. Capture that leads to 10+ captured stones
    
    Args:
        board: current board
        player: player (1 or 2)
        current_captures: current number of stones already captured by this player
    
    Returns:
        list of (row, col) that result in victory
    """
    board_size = board.shape[0]
    winning_moves = []
    
    for row in range(board_size):
        for col in range(board_size):
            if board[row, col] != 0:
                continue
            
            board[row, col] = player
            
            directions = ((0, 1), (1, 0), (1, 1), (1, -1))
            won_by_line = False
            
            for dr, dc in directions:
                count = 1
                r, c = row + dr, col + dc
                while 0 <= r < board_size and 0 <= c < board_size and board[r, c] == player:
                    count += 1
                    r += dr
                    c += dc
                r, c = row - dr, col - dc
                while 0 <= r < board_size and 0 <= c < board_size and board[r, c] == player:
                    count += 1
                    r -= dr
                    c -= dc
                
                if count >= 5:
                    won_by_line = True
                    break
            
            won_by_capture = False
            if not won_by_line:
                num_captured = detect_capture_pente(board, row, col)
                if current_captures + num_captured >= 10:
                    won_by_capture = True
            
            board[row, col] = 0
            
            if won_by_line or won_by_capture:
                winning_moves.append((row, col))
    
    return winning_moves


@njit(cache=True)
def detect_open_four_gomoku(board, row, col, player, board_size):
    """
    Detects if a move creates 4 in a row with both ends open.
    
    Returns:
        bool
    """
    if board[row, col] != 0:
        return False
    
    directions = ((0, 1), (1, 0), (1, 1), (1, -1))
    
    for dr, dc in directions:
        count_forward = 0
        r, c = row + dr, col + dc
        while 0 <= r < board_size and 0 <= c < board_size and board[r, c] == player:
            count_forward += 1
            r += dr
            c += dc
        open_forward = (0 <= r < board_size and 0 <= c < board_size and board[r, c] == 0)
        
        count_backward = 0
        r, c = row - dr, col - dc
        while 0 <= r < board_size and 0 <= c < board_size and board[r, c] == player:
            count_backward += 1
            r -= dr
            c -= dc
        open_backward = (0 <= r < board_size and 0 <= c < board_size and board[r, c] == 0)
        total = count_forward + count_backward
        
        if total == 3 and open_forward and open_backward:
            return True
    
    return False


@njit(cache=True)
def check_win_pente(board):
    """
    Checks for a winner in Pente (5 or more in a row).
    Identical to Gomoku for line victory.
    
    Returns:
        0 (no winner), 1 (player 1), 2 (player 2)
    """
    return check_win_gomoku(board)


@njit(cache=True)
def detect_capture_pente(board, row, col):
    """
    Detects how many stones would be captured by playing at (row, col).
    
    Args:
        board: game board
        row, col: move position
    
    Returns:
        int: number of captured stones (0, 2, 4, 6, or 8)
    """
    board_size = board.shape[0]
    player = board[row, col]
    if player == 0:
        return 0
    
    opponent = 3 - player
    captured_count = 0

    directions = (
        (0, 1),  
        (0, -1),  
        (1, 0),   
        (-1, 0),  
        (1, 1), 
        (-1, -1), 
        (1, -1),  
        (-1, 1)   
    )
    
    for dr, dc in directions:
        r1, c1 = row + dr, col + dc
        r2, c2 = row + 2*dr, col + 2*dc
        r3, c3 = row + 3*dr, col + 3*dc

        if not (0 <= r3 < board_size and 0 <= c3 < board_size):
            continue
        
        if (board[r1, c1] == opponent and 
            board[r2, c2] == opponent and 
            board[r3, c3] == player):
            captured_count += 2
    
    return captured_count


@njit(cache=True)
def apply_capture_pente(board, row, col):
    """
    Applies captures to the board after placing a stone at (row, col).
    Removes captured stones.
    
    Args:
        board: game board
        row, col: move position
    
    Returns:
        int: number of captured stones
    """
    board_size = board.shape[0]
    player = board[row, col]
    if player == 0:
        return 0
    
    opponent = 3 - player
    captured_count = 0
    
    directions = (
        (0, 1), (0, -1), (1, 0), (-1, 0),
        (1, 1), (-1, -1), (1, -1), (-1, 1)
    )
    
    for dr, dc in directions:
        r1, c1 = row + dr, col + dc
        r2, c2 = row + 2*dr, col + 2*dc
        r3, c3 = row + 3*dr, col + 3*dc
        
        if not (0 <= r3 < board_size and 0 <= c3 < board_size):
            continue
        
        if (board[r1, c1] == opponent and 
            board[r2, c2] == opponent and 
            board[r3, c3] == player):
            
            board[r1, c1] = 0
            board[r2, c2] = 0
            captured_count += 2
    
    return captured_count


@njit(cache=True)
def detect_capture_moves_pente(board, player):
    """
    Detects all moves that would result in captures.
    
    Returns:
        list of (row, col, num_captures)
    """
    board_size = board.shape[0]
    opponent = 3 - player
    capture_moves = []
    
    directions = (
        (0, 1), (0, -1), (1, 0), (-1, 0),
        (1, 1), (1, -1), (-1, 1), (-1, -1)
    )
    
    for row in range(board_size):
        for col in range(board_size):
            if board[row, col] != 0:
                continue
            
            num_captures = 0
            
            for dr, dc in directions:
                r1, c1 = row + dr, col + dc
                r2, c2 = row + 2*dr, col + 2*dc
                r3, c3 = row + 3*dr, col + 3*dc
                
                if not (0 <= r3 < board_size and 0 <= c3 < board_size):
                    continue
                
                if (board[r1, c1] == opponent and 
                    board[r2, c2] == opponent and 
                    board[r3, c3] == player):
                    num_captures += 2
            
            if num_captures > 0:
                capture_moves.append((row, col, num_captures))
    
    return capture_moves


@njit(cache=True)
def check_win_by_capture(captures_p1, captures_p2):
    """
    Checks for win by capture (10+ captured stones).
    
    Returns:
        0 (no winner), 1 (player 1), 2 (player 2)
    """
    if captures_p1 >= 10:
        return 1
    if captures_p2 >= 10:
        return 2
    return 0


@njit(cache=True)
def get_legal_moves_pente(board, distance=2):
    """
    Legal moves for Pente (identical to Gomoku).
    Considers spaces created by captures.
    
    Returns:
        list of (row, col)
    """
    return get_legal_moves_gomoku(board, distance)


@njit(cache=True)
def check_line_length(board, row, col, dr, dc, player, board_size):
    """
    Counts line length in a specific direction.
    
    Returns:
        int: line length
    """
    count = 1
    
    r, c = row + dr, col + dc
    while 0 <= r < board_size and 0 <= c < board_size and board[r, c] == player:
        count += 1
        r += dr
        c += dc
    
    r, c = row - dr, col - dc
    while 0 <= r < board_size and 0 <= c < board_size and board[r, c] == player:
        count += 1
        r -= dr
        c -= dc
    
    return count


@njit(cache=True)
def is_board_full(board, board_size):
    """
    Checks if the board is full (draw).
    
    Returns:
        bool
    """
    return np.sum(board == 0) == 0


@njit(cache=True)
def count_pieces(board, player):
    """
    Counts how many stones a player has on the board.
    
    Returns:
        int
    """
    return np.sum(board == player)
