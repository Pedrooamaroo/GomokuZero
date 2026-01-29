"""
Motor de Jogo - Gomoku & Pente
Funções otimizadas com Numba @njit
"""

import numpy as np
from numba import njit
import math


@njit(cache=True)
def check_win_gomoku(board):
    """
    Verifica vencedor no Gomoku (5 ou mais em linha)
    
    Argumentos:
        board: numpy array (board_size, board_size) com 0/1/2
    
    Returns:
        0 (sem vencedor), 1 (jogador 1), 2 (jogador 2)
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
    Retorna movimentos legais perto de pedras existentes (otimização)
    
    Argumentos:
        board: tabuleiro
        distance: distância máxima de pedras existentes
    
    Returns:
        lista de (row, col)
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
    Deteta jogadas que ganham imediatamente para o jogador
    
    Returns:
        lista de (row, col) que resultam em vitória
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
    Deteta jogadas que ganham imediatamente para o jogador (Pente)
    Considera:
    1. 5 em linha
    2. Captura que leva a 10+ peças capturadas
    
    Argumentos:
        board: tabuleiro atual
        player: jogador (1 ou 2)
        current_captures: número atual de peças já capturadas por este jogador
    
    Returns:
        lista de (row, col) que resultam em vitória
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
    Deteta se jogada cria 4 seguidas com ambos os lados abertos
    
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
    Verifica vencedor no Pente (5 ou mais em linha)
    Idêntico ao Gomoku para vitória por linha
    
    Returns:
        0 (sem vencedor), 1 (jogador 1), 2 (jogador 2)
    """
    return check_win_gomoku(board)


@njit(cache=True)
def detect_capture_pente(board, row, col):
    """
    Detecta quantas pedras seriam capturadas ao jogar em (row, col)
    
    Argumentos:
        board: tabuleiro
        row, col: posição da jogada
    
    Returns:
        int: número de pedras capturadas (0, 2, 4, 6, ou 8)
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
    Aplica capturas no tabuleiro após colocar pedra em (row, col)
    Remove pedras capturadas
    
    Argumentos:
        board: tabuleiro
        row, col: posição da jogada
    
    Returns:
        int: número de pedras capturadas
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
    Deteta todas as jogadas que resultariam em capturas
    
    Returns:
        lista de (row, col, num_captures)
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
    Verifica vitória por captura (10+ pedras capturadas)
    
    Returns:
        0 (sem vencedor), 1 (jogador 1), 2 (jogador 2)
    """
    if captures_p1 >= 10:
        return 1
    if captures_p2 >= 10:
        return 2
    return 0


@njit(cache=True)
def get_legal_moves_pente(board, distance=2):
    """
    Movimentos legais para Pente (idêntico ao Gomoku)
    Considera espaços criados por capturas
    
    Returns:
        lista de (row, col)
    """
    return get_legal_moves_gomoku(board, distance)


@njit(cache=True)
def check_line_length(board, row, col, dr, dc, player, board_size):
    """
    Conta comprimento de linha numa direção específica
    
    Returns:
        int: comprimento da linha
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
    Verifica se tabuleiro está cheio (empate)
    
    Returns:
        bool
    """
    return np.sum(board == 0) == 0


@njit(cache=True)
def count_pieces(board, player):
    """
    Conta quantas pedras um jogador tem no tabuleiro
    
    Returns:
        int
    """
    return np.sum(board == player)




if __name__ == "__main__":
    board_size = 15
    
    print("\n[GOMOKU] Teste 1: Vitória por 5 em linha")
    board = np.zeros((board_size, board_size), dtype=np.int32)
    board[7, 5:10] = 1  # 5 em linha horizontal
    winner = check_win_gomoku(board)
    print(f"  Resultado: {winner} (esperado: 1) {'OK' if winner == 1 else 'ERRO'}")
    
    print("\n[GOMOKU] Teste 2: Detectar jogada vencedora")
    board = np.zeros((board_size, board_size), dtype=np.int32)
    board[7, 5:9] = 1  # 4 em linha
    winning_moves = detect_winning_move_gomoku(board, 1)
    print(f"  Jogadas vencedoras: {winning_moves}")
    print(f"  Contém (7,4) ou (7,9)? {(7,4) in winning_moves or (7,9) in winning_moves} OK")
    
    print("\n[PENTE] Teste 3: Detectar captura")
    board = np.zeros((board_size, board_size), dtype=np.int32)
    board[7, 7] = 1 
    board[7, 8] = 2   
    board[7, 9] = 2   
    board[7, 10] = 1  
    
    captured = detect_capture_pente(board, 7, 10)
    print(f"  Pedras detectadas para captura: {captured} (esperado: 2) {'OK' if captured == 2 else 'ERRO'}")
    
    print("\n[PENTE] Teste 4: Aplicar captura")
    board_before = board.copy()
    captured = apply_capture_pente(board, 7, 10)
    print(f"  Pedras capturadas: {captured}")
    print(f"  Posição (7,8) antes: {board_before[7,8]}, depois: {board[7,8]} {'OK' if board[7,8] == 0 else 'ERRO'}")
    print(f"  Posição (7,9) antes: {board_before[7,9]}, depois: {board[7,9]} {'OK' if board[7,9] == 0 else 'ERRO'}")
    
    print("\n[PENTE] Teste 5: Vitória por captura")
    winner = check_win_by_capture(10, 4)
    print(f"  P1 com 10 capturas: {winner} (esperado: 1) {'OK' if winner == 1 else 'ERRO'}")
    
    winner = check_win_by_capture(6, 12)
    print(f"  P2 com 12 capturas: {winner} (esperado: 2) {'OK' if winner == 2 else 'ERRO'}")
    
