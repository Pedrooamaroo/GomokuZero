# Motor (funções Numba)
from .game_engine import (
    # Gomoku
    check_win_gomoku,
    get_legal_moves_gomoku,
    detect_winning_move_gomoku,
    detect_open_four_gomoku,
    
    # Pente
    check_win_pente,
    detect_winning_move_pente,
    detect_capture_pente,
    apply_capture_pente,
    detect_capture_moves_pente,
    check_win_by_capture,
    get_legal_moves_pente,
    
    # Gerais
    check_line_length,
    is_board_full,
    count_pieces,
)

# MCTS
from .mcts import MCTS, MCTSNode

# Tactical Patterns
from .tactical_patterns_v2 import (
    get_tactical_scores_for_moves,
    score_move_tactical_v3,
)

# Rede Neural
from .network import (
    GomokuNet,
    ResidualBlock,
    board_to_tensor,
    create_network,
    save_checkpoint,
    load_checkpoint,
)

# Data Buffer
from .data_buffer import AugmentedBuffer

# Training
from .training import self_play_game, train_network

# Wrappers
from .game_gomoku import GomokuGame
from .game_pente import PenteGame


__all__ = [
    # Game Engine
    'check_win_gomoku',
    'get_legal_moves_gomoku',
    'detect_winning_move_gomoku',
    'detect_open_four_gomoku',
    'check_win_pente',
    'detect_winning_move_pente',
    'detect_capture_pente',
    'apply_capture_pente',
    'detect_capture_moves_pente',
    'check_win_by_capture',
    'get_legal_moves_pente',
    'check_line_length',
    'is_board_full',
    'count_pieces',
    
    # MCTS
    'MCTS',
    'MCTSNode',
    
    # Tactical Patterns
    'get_tactical_scores_for_moves',
    'score_move_tactical_v3',
    
    # Rede Neural
    'GomokuNet',
    'ResidualBlock',
    'board_to_tensor',
    'create_network',
    'save_checkpoint',
    'load_checkpoint',
    
    # Buffer
    'AugmentedBuffer',
    
    # Training
    'self_play_game',
    'train_network',
    
    # Games
    'GomokuGame',
    'PenteGame',
]

__version__ = '2.0.0'
__author__ = 'LAB2 Team'
