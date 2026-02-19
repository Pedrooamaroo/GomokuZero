"""
Monte Carlo Tree Search with Neural Network
AlphaZero Implementation for Gomoku and Pente
"""

import sys
import os
import numpy as np
import math

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

try:
    from .game_engine import (
        check_win_gomoku, check_win_pente,
        apply_capture_pente, check_win_by_capture,
        get_legal_moves_gomoku, get_legal_moves_pente
    )
    from .network import board_to_tensor
    from .tactical_patterns_v2 import get_tactical_scores_for_moves
except ImportError:
    from game_engine import (
        check_win_gomoku, check_win_pente,
        apply_capture_pente, check_win_by_capture,
        get_legal_moves_gomoku, get_legal_moves_pente
    )
    from network import board_to_tensor
    from tactical_patterns_v2 import get_tactical_scores_for_moves


class MCTSNode:
    """
    MCTS tree node with neural network policy priors.
    
    Attributes:
        parent: parent node
        children: dictionary {(row, col): MCTSNode}
        visits: number of times visited
        value_sum: sum of backpropagation values
        prior: neural network prior probability P(s,a)
    """
    
    def __init__(self, parent=None, prior_prob=1.0):
        self.parent = parent
        self.children = {}
        self.visits = 0
        self.value_sum = 0.0
        self.prior = prior_prob
    
    def value(self):
        """
        Average node value Q(s,a).
        
        Returns:
            float: average value or 0 if not visited
        """
        return self.value_sum / self.visits if self.visits > 0 else 0.0
    
    def ucb_score(self, c_puct=1.5):
        """
        Upper Confidence Bound with policy prior (AlphaZero formula).
        
        UCB(s,a) = Q(s,a) + c_puct * P(s,a) * sqrt(N(s)) / (1 + N(s,a))
        
        Args:
            c_puct: exploration constant
        
        Returns:
            float: UCB score (inf if not visited)
        """
        if self.visits == 0:
            return float('inf')
        
        exploitation = self.value()
        exploration = c_puct * self.prior * math.sqrt(self.parent.visits) / (1 + self.visits)
        
        return exploitation + exploration
    
    def select_child(self, c_puct=1.5):
        """
        Selects the best child using UCB.
        
        Returns:
            tuple: (move, child node)
        """
        return max(self.children.items(), key=lambda item: item[1].ucb_score(c_puct))
    
    def expand(self, policy_probs, legal_moves, board_size, current_board=None, 
               current_player=None, is_pente=False, captures_p1=0, captures_p2=0):
        """
        Expands the node by creating children with neural network priors + tactics.
        """
        masked_probs = policy_probs.copy()
        legal_mask = np.zeros(board_size * board_size, dtype=bool)
        for row, col in legal_moves:
            legal_mask[row * board_size + col] = True
        masked_probs[~legal_mask] = 0.0
        
        legal_probs = []
        for row, col in legal_moves:
            action_idx = row * board_size + col
            legal_probs.append(masked_probs[action_idx])
        legal_probs = np.array(legal_probs)
        
        if legal_probs.sum() > 0:
            legal_probs = legal_probs / legal_probs.sum()
        else:
            legal_probs = np.ones(len(legal_moves)) / len(legal_moves)

        tactical_boost_enabled = True
        
        if tactical_boost_enabled and current_board is not None and current_player is not None:
            legal_moves_array = np.array(legal_moves, dtype=np.int32)
            
            tactical_scores = get_tactical_scores_for_moves(
                current_board,
                current_player,
                legal_moves_array,
                is_pente,
                captures_p1,
                captures_p2
            )
            
            tactical_multipliers = np.ones(len(legal_moves), dtype=np.float64)
            
            critical_mask = tactical_scores >= 40000.0
            if critical_mask.any():
                tactical_multipliers[critical_mask] = 10000.0 
                tactical_multipliers[~critical_mask] = 0.0001
            
            elif (tactical_scores >= 7500.0).any():
                strong_mask = tactical_scores >= 7500.0
                tactical_multipliers[strong_mask] = 1000.0
                tactical_multipliers[~strong_mask] = 0.01

            elif (tactical_scores >= 1000.0).any():
                good_mask = tactical_scores >= 1000.0
                tactical_multipliers[good_mask] = 50.0 
                tactical_multipliers[~good_mask] = 0.5

            else:
                for i, score in enumerate(tactical_scores):
                    if score >= 200.0:
                        tactical_multipliers[i] = 5.0
                    elif score > 0:
                        tactical_multipliers[i] = 1.5
                    elif score <= -2000.0:
                        tactical_multipliers[i] = 0.0001
                    elif score < 0:
                        tactical_multipliers[i] = 0.1

            legal_probs = legal_probs * tactical_multipliers
            if legal_probs.sum() > 0:
                legal_probs = legal_probs / legal_probs.sum()
            else:
                legal_probs = np.ones(len(legal_moves)) / len(legal_moves)
        
        for (row, col), prior in zip(legal_moves, legal_probs):
            if (row, col) not in self.children:
                self.children[(row, col)] = MCTSNode(parent=self, prior_prob=prior)
    
    def add_dirichlet_noise(self, alpha=0.03, epsilon=0.25):
        """
        Adds Dirichlet noise to the root node priors (AlphaZero).
        
        Increases exploration during self-play:
        P'(s,a) = (1 - ε) * P(s,a) + ε * Dir(α)
        
        Args:
            alpha: Dirichlet concentration (0.03 for Go/Gomoku)
            epsilon: noise weight (0.25 = 25% noise, 75% original policy)
        """
        if not self.children:
            return
        
        noise = np.random.dirichlet([alpha] * len(self.children))
        
        for (move, child), noise_value in zip(self.children.items(), noise):
            child.prior = (1 - epsilon) * child.prior + epsilon * noise_value
    
    def update(self, value):
        """
        Backpropagation: updates node statistics.
        
        Args:
            value: value to propagate (-1, 0, or +1)
        """
        self.visits += 1
        self.value_sum += value


class MCTS:
    """
    Monte Carlo Tree Search guided by neural network.
    
    Supports Gomoku and Pente with captures.
    """
    
    def __init__(self, network, game_type='gomoku', board_size=15, c_puct=1.5, num_simulations=800):
        """
        Args:
            network: neural network (GomokuNet) with predict() method
            game_type: 'gomoku' or 'pente'
            board_size: board size
            c_puct: UCB exploration constant
            num_simulations: FALLBACK - used only if time_limit=None in search()
                             During training, time_limit=5.0 is typically used instead.
        """
        self.network = network
        self.game_type = game_type
        self.board_size = board_size
        self.c_puct = c_puct
        self.num_simulations = num_simulations
        self.root = MCTSNode()
    
    def search(self, board, player, temperature=1.0, captures_p1=None, captures_p2=None, time_limit=None, add_noise=False):
        """
        Executes a complete MCTS search.
        
        Args:
            board: numpy array (board_size, board_size)
            player: int (1 or 2) - current player
            temperature: float - controls exploration (1.0=exploratory, 0.1=greedy)
            captures_p1: int or None - player 1 captures (Pente). None for Gomoku
            captures_p2: int or None - player 2 captures (Pente). None for Gomoku
            time_limit: float (optional) - maximum time in seconds. If None, uses num_simulations
            add_noise: bool - adds Dirichlet noise to the root (self-play only)
        
        Returns:
            policy: numpy array (board_size²) with probability distribution
            value: float - root position value
        """
 
        if self.root is None:
            self.root = MCTSNode(parent=None, prior_prob=1.0)
        
        if self.game_type == 'pente':
            if captures_p1 is None:
                captures_p1 = 0
            if captures_p2 is None:
                captures_p2 = 0
        else:
            captures_p1 = 0 if captures_p1 is None else captures_p1
            captures_p2 = 0 if captures_p2 is None else captures_p2
        
        if time_limit is not None:
            import time
            start_time = time.time()
            simulations = 0
            while time.time() - start_time < time_limit:
                self._simulate_once(board, player, captures_p1, captures_p2)
                simulations += 1
                if add_noise and simulations == 1 and self.root.children:
                    self.root.add_dirichlet_noise(alpha=0.03, epsilon=0.25)
            
            if simulations == 0:
                self._simulate_once(board, player, captures_p1, captures_p2)
                simulations = 1
            
            self.last_simulations = simulations
        else:
            simulations = self.num_simulations
            for i in range(simulations):
                self._simulate_once(board, player, captures_p1, captures_p2)
                if add_noise and i == 0 and self.root.children:
                    self.root.add_dirichlet_noise(alpha=0.03, epsilon=0.25)
            self.last_simulations = simulations

        policy = self._get_policy(temperature)

        if not self.root.children:

            if self.game_type == 'pente':
                legal_moves = get_legal_moves_pente(board, distance=2)
            else:
                legal_moves = get_legal_moves_gomoku(board, distance=2)
            
            if legal_moves:

                if self.game_type == 'pente':
                    state = board_to_tensor(board, player, None, self.board_size,
                                           captures_p1, captures_p2)
                else:
                    state = board_to_tensor(board, player, None, self.board_size)
                
                policy_probs, value = self.network.predict(state)
                
                is_pente = (self.game_type == 'pente')
                self.root.expand(
                    policy_probs, legal_moves, self.board_size,
                    current_board=board,
                    current_player=player,
                    is_pente=is_pente,
                    captures_p1=captures_p1,
                    captures_p2=captures_p2
                )
                
                policy = self._get_policy(temperature)
        
        return policy, self.root.value()
    
    def _simulate_once(self, board, player, captures_p1, captures_p2):
        """
        A single MCTS iteration: Selection → Expansion → Evaluation → Backprop.
        
        Args:
            board: initial board
            player: initial player
            captures_p1: P1 captures
            captures_p2: P2 captures
        """

        node = self.root
        temp_board = board.copy()
        temp_player = player
        temp_captures_p1 = captures_p1
        temp_captures_p2 = captures_p2
        search_path = [node]
        

        while node.children:
            move, node = node.select_child(self.c_puct)
            search_path.append(node)
            

            if temp_board[move] != 0:
                value = -1.0
                for backprop_node in reversed(search_path):
                    backprop_node.update(value)
                    value = -value
                return
            
            temp_board[move] = temp_player
            
            if self.game_type == 'pente':
                captured = apply_capture_pente(temp_board, move[0], move[1])
                if temp_player == 1:
                    temp_captures_p1 += captured
                else:
                    temp_captures_p2 += captured

            temp_player = 3 - temp_player
        

        winner = self._check_winner(temp_board, temp_captures_p1, temp_captures_p2)
        
        if winner == 0:
            if self.game_type == 'pente':
                legal_moves = get_legal_moves_pente(temp_board, distance=2)
            else:
                legal_moves = get_legal_moves_gomoku(temp_board, distance=2)
            
            if legal_moves:

                if self.game_type == 'pente':
                    state = board_to_tensor(temp_board, temp_player, None, temp_board.shape[0],
                                           temp_captures_p1, temp_captures_p2)
                else:
                    state = board_to_tensor(temp_board, temp_player, None, temp_board.shape[0])
                
                policy_probs, value = self.network.predict(state)
                

                if temp_player != player:
                    value = -value
                

                is_pente = (self.game_type == 'pente')
                node.expand(
                    policy_probs, legal_moves, self.board_size,
                    current_board=temp_board,
                    current_player=temp_player,
                    is_pente=is_pente,
                    captures_p1=temp_captures_p1,
                    captures_p2=temp_captures_p2
                )
            else:
                value = 0.0
        else:
            if winner == player:
                value = 1.0
            elif winner == (3 - player):
                value = -1.0
            else:
                value = 0.0

        for node in reversed(search_path):
            node.update(value)
            value = -value
    
    def _check_winner(self, board, captures_p1, captures_p2):
        """
        Checks for a winner considering the game rules.
        
        Returns:
            0 (no winner), 1 (P1), 2 (P2)
        """
        if self.game_type == 'pente':
            winner = check_win_by_capture(captures_p1, captures_p2)
            if winner != 0:
                return winner
            return check_win_pente(board)
        else:
            return check_win_gomoku(board)
    
    def _get_policy(self, temperature=1.0):
        """
        Extracts policy distribution based on visit counts.
        
        Args:
            temperature: controls exploration
                - 1.0: proportional to visits (exploratory)
                - 0.1-0.5: more deterministic
                - ~0: greedy (chooses most visited)
        
        Returns:
            numpy array (board_size²) with probability distribution
        """
        policy = np.zeros(self.board_size * self.board_size, dtype=np.float32)
        
        if not self.root.children:
            return policy
        
        if temperature < 0.01:
            best_move = max(self.root.children.items(), key=lambda item: item[1].visits)[0]
            row, col = best_move
            policy[row * self.board_size + col] = 1.0
        else:
            visits = np.array([child.visits for child in self.root.children.values()])
            moves = list(self.root.children.keys())
            
            if temperature != 1.0:
                visits = visits ** (1.0 / temperature)
            
            visits_sum = visits.sum()
            if visits_sum > 0:
                visit_probs = visits / visits_sum

                for move, prob in zip(moves, visit_probs):
                    row, col = move
                    policy[row * self.board_size + col] = prob
        
        return policy
    
    def update_root(self, move):
        """
        Updates root after a move (reuses subtree).
        
        Args:
            move: tuple (row, col) of the played move
        """
        if move in self.root.children:
            self.root = self.root.children[move]
            self.root.parent = None
        else:
            self.root = MCTSNode()
    
    def reset(self):
        """Resets the tree (used between games)."""
        self.root = MCTSNode()
