"""
Player AlphaZero - Corrigido para competição
- Board: 15x15
- Usa models/gomoku_model_best.pth e models/pente_model_best.pth
- MCTS com PUCT + integração de tactical patterns como priors
- Timeout por jogada. Se timeout/erro => fallback legal
- Caching da árvore entre jogadas (update_root)
- Compatível com play.py oficial (set_player_ind, get_name)
"""

from __future__ import annotations
import time
import math
import random
import os
import sys
from typing import Tuple, Optional, Dict, List
import numpy as np

try:
    import torch
    torch.set_num_threads(1)
    TORCH_AVAILABLE = True
except Exception:
    TORCH_AVAILABLE = False

BOARD_SIZE = 15 
MODEL_PATHS = {
    "gomoku": "models/gomoku_model_best.pth",
    "pente":  "models/pente_model_best.pth"
}
DEFAULT_TIMEOUT = 4.8
CPU_SIM_BATCH = 1

def legal_moves_from_board(board: List[List[int]]) -> List[Tuple[int,int]]:
    n = len(board)
    moves = [(r,c) for r in range(n) for c in range(n) if board[r][c] == 0]
    return moves

def board_to_planes(board: List[List[int]], me: int, rules: str = "gomoku", 
                   my_captures: int = 0, opp_captures: int = 0, 
                   last_move: Optional[Tuple[int,int]] = None) -> np.ndarray:
    """
    Gera o tensor de entrada para a rede neural.
    - Gomoku: 3 canais (Me, Opp, LastMove)
    - Pente: 5 canais (Me, Opp, LastMove, MyCaptures, OppCaptures)
    """
    board = np.asarray(board, dtype=np.int8)

    my_plane = (board == 1).astype(np.float32)
    opp_plane = (board == 2).astype(np.float32)
    
    last_move_plane = np.zeros_like(my_plane)
    if last_move is not None:
        r, c = last_move
        if 0 <= r < len(board) and 0 <= c < len(board):
            last_move_plane[r, c] = 1.0
    
    planes_list = [my_plane, opp_plane, last_move_plane]

    if rules.lower() == "pente":
        val_me = min(float(my_captures) / 10.0, 1.0)
        val_opp = min(float(opp_captures) / 10.0, 1.0)
        
        cap_me_plane = np.full_like(my_plane, val_me)
        cap_opp_plane = np.full_like(my_plane, val_opp)
        
        planes_list.extend([cap_me_plane, cap_opp_plane])
        
    planes = np.stack(planes_list, axis=0)
    
    return planes

class MCTSNode:
    def __init__(self, prior: float = 0.0):
        self.N = 0          
        self.W = 0.0       
        self.Q = 0.0      
        self.P = prior     
        self.children: Dict[Tuple[int,int], 'MCTSNode'] = {}

class Player:
    def __init__(self, rules: str = "gomoku", board_size: int = BOARD_SIZE):
        """
        Regras: "gomoku" ou "pente"
        board_size: 15
        """
        self.rules = rules.lower()
        self.board_size = board_size
        
        assert self.board_size == BOARD_SIZE, \
            f"Modelo treinado em {BOARD_SIZE}x{BOARD_SIZE}"
        
        self.me = 1 
        self.name = f"AlphaZero_Player_{self.rules}_{self.board_size}"
        self.model = None
        self.device = torch.device("cpu") if TORCH_AVAILABLE else None
        self.mcts_root: Optional[MCTSNode] = None
        self.tree_state_board = None
        self.last_move: Optional[Tuple[int,int]] = None
        self.load_model_once()
        self.tactical_fn = None
        try:
            possible_paths = [
                os.path.join(os.path.dirname(__file__), "src", "tactical_patterns_v2.py"),
                os.path.join(os.path.dirname(__file__), "tactical_patterns_v2.py"),
                os.path.join(os.getcwd(), "src", "tactical_patterns_v2.py")
            ]
            
            tactical_path = None
            for p in possible_paths:
                if os.path.exists(p):
                    tactical_path = p
                    break
            
            if tactical_path:
                import importlib.util
                spec = importlib.util.spec_from_file_location("tactical_patterns_v2", tactical_path)
                if spec:
                    tactical = importlib.util.module_from_spec(spec)
                    spec.loader.exec_module(tactical)
                    if hasattr(tactical, "get_priors"):
                        self.tactical_fn = tactical.get_priors
                        print(f"[Player] Táticas carregadas de: {tactical_path}")
            else:
                print("[Player] Ficheiro tactical_patterns_v2.py não encontrado!")

        except Exception as e:
            print(f"[Player] Erro ao carregar táticas: {e}")

    def get_name(self):
        return self.name

    def set_player_ind(self, ind: int):
        assert ind in (1,2)
        self.me = ind

    def count_captures_from_board(self, board: List[List[int]]) -> Tuple[int, int]:
        """
        Conta capturas do Pente analisando o tabuleiro.
        Retorna (my_captures, opp_captures).
        Para Gomoku retorna (0, 0).
        """
        if self.rules != 'pente':
            return 0, 0
        
        n = self.board_size
        
        my_stones = sum(1 for r in range(n) for c in range(n) if board[r][c] == 1)
        opp_stones = sum(1 for r in range(n) for c in range(n) if board[r][c] == 2)
        
        total_moves = my_stones + opp_stones
        expected_my = (total_moves + 1) // 2
        expected_opp = total_moves // 2
        
        if self.me == 1:
            my_captures = max(0, expected_opp - opp_stones)
            opp_captures = max(0, expected_my - my_stones)
        else:
            my_captures = max(0, expected_my - opp_stones)
            opp_captures = max(0, expected_opp - my_stones)
        
        return min(my_captures, 10), min(opp_captures, 10) 

    def apply_capture_sim(self, board: List[List[int]], r: int, c: int, player: int):
        """
        Remove pedras capturadas no tabuleiro de simulação.
        Funciona com List[List[int]].
        """
        n = self.board_size
        opp = 3 - player
        directions = [
            (0, 1), (0, -1), (1, 0), (-1, 0),
            (1, 1), (1, -1), (-1, 1), (-1, -1)
        ]
        for dr, dc in directions:
            r3, c3 = r + 3*dr, c + 3*dc
            if 0 <= r3 < n and 0 <= c3 < n:
                if (board[r+dr][c+dc] == opp and 
                    board[r+2*dr][c+2*dc] == opp and 
                    board[r+3*dr][c+3*dc] == player):
                    board[r+dr][c+dc] = 0
                    board[r+2*dr][c+2*dc] = 0

    def load_model_once(self):
        if not TORCH_AVAILABLE:
            print("[player.py] torch não disponivel — random player", file=sys.stderr)
            return
        model_file = MODEL_PATHS.get(self.rules)
        if model_file is None:
            print(f"[player.py] no model path for rules={self.rules}", file=sys.stderr)
            return
        if not os.path.exists(model_file):
            print(f"[player.py] model file {model_file} not found — fallback to random", file=sys.stderr)
            return
        try:
            loaded = torch.load(model_file, map_location=self.device)

            if isinstance(loaded, dict):
                state_dict = None

                if "model_state_dict" in loaded:
                    state_dict = loaded["model_state_dict"]
                elif "state_dict" in loaded:
                    state_dict = loaded["state_dict"]
                elif all(isinstance(k, str) and '.' in k for k in list(loaded.keys())[:3]):
                    state_dict = loaded
                
                if state_dict is not None:
                    try:
                        from src.network import create_network
                        num_input_channels = 5 if self.rules == 'pente' else 3
                        net = create_network(self.board_size, num_input_channels=num_input_channels)
                        net.load_state_dict(state_dict)
                        net.to(self.device)
                        net.eval()
                        self.model = net
                        print(f"[player.py] Model carregado de {model_file} ({num_input_channels} channels)", file=sys.stderr)
                    except Exception as e:
                        print(f"[player.py] Erro ao carregar modelo: {e}", file=sys.stderr)
                        self.model = None
                else:
                    print("[player.py] Could not find state_dict in checkpoint", file=sys.stderr)
                    self.model = None
            elif hasattr(loaded, "state_dict") and hasattr(loaded, "forward"):
                self.model = loaded
                self.model.to(self.device)
                self.model.eval()
                print(f"[player.py] Modelo completo carregado de {model_file}", file=sys.stderr)
            else:
                print("[player.py] Formato desconhecido", file=sys.stderr)
                self.model = None
            
            if self.model is None:
                print("[player.py] Could not construct model from checkpoint — fallback to policy heuristics", file=sys.stderr)
        except Exception as e:
            print(f"[player.py] error loading model: {e}", file=sys.stderr)
            self.model = None

    def model_infer(self, board: List[List[int]]) -> Tuple[np.ndarray, float]:
        """
        Returns (policy_flat, value)
         - policy_flat: numpy array shape (N*N,) with probabilities for all positions
         - value: float in [-1,1] estimating position for current player self.me
        If no model available, returns uniform policy and value 0.
        """
        n = self.board_size
        legal = legal_moves_from_board(board)
        if self.model is None or not TORCH_AVAILABLE:
            p = np.zeros(n*n, dtype=np.float32)
            for (r,c) in legal:
                p[r*n + c] = 1.0
            if p.sum() == 0:
                p = np.ones(n*n, dtype=np.float32)
            p = p / p.sum()
            return p, 0.0
        try:
            my_captures, opp_captures = self.count_captures_from_board(board)
            planes = board_to_planes(board, self.me, self.rules, 
                                    my_captures=my_captures, opp_captures=opp_captures,
                                    last_move=self.last_move)
            x = torch.from_numpy(planes).unsqueeze(0).to(self.device)
            with torch.no_grad():
                out = self.model(x)
                if isinstance(out, tuple) or isinstance(out, list):
                    policy_t, value_t = out[0], out[1]
                elif isinstance(out, dict):
                    policy_t = out.get("policy", None) or out.get("pi", None)
                    value_t = out.get("value", None) or out.get("v", None)
                else:
                    if out.dim() == 2 and out.size(1) == 1 + n*n:
                        policy_t = out[:, :n*n]
                        value_t = out[:, -1]
                    else:
                        policy_t = None
                        value_t = None
                if policy_t is None or value_t is None:
                    if hasattr(self.model, "predict"):
                        p_np, v = self.model.predict(x)
                        return np.asarray(p_np).reshape(-1), float(v)
                    p = np.zeros(n*n, dtype=np.float32)
                    for (r,c) in legal:
                        p[r*n + c] = 1.0
                    p = p / p.sum() if p.sum() > 0 else np.ones(n*n, dtype=np.float32) / (n*n)
                    return p, 0.0
                policy_np = policy_t.squeeze(0).cpu().numpy().reshape(-1)
                exp = np.exp(policy_np - np.max(policy_np))
                probs = exp / (exp.sum() + 1e-12)
                val = float(value_t.squeeze(0).cpu().numpy().reshape(-1)[0])
                return probs.astype(np.float32), max(-1.0, min(1.0, val))
        except Exception as e:
            print("[player.py] model_infer error:", e, file=sys.stderr)
            p = np.zeros(n*n, dtype=np.float32)
            for (r,c) in legal:
                p[r*n + c] = 1.0
            p = p / p.sum() if p.sum()>0 else np.ones(n*n, dtype=np.float32)/(n*n)
            return p, 0.0

    def new_root(self, prior_policy: np.ndarray) -> MCTSNode:
        root = MCTSNode()
        n = self.board_size
        for r in range(n):
            for c in range(n):
                idx = r*n + c
                p = float(prior_policy[idx]) if idx < len(prior_policy) else 0.0
                root.children[(r,c)] = MCTSNode(prior=p)
        return root

    def select_child(self, node: MCTSNode, cpuct: float = 1.0) -> Tuple[Tuple[int,int], MCTSNode]:
        best_score = -1e9
        best_move = None
        best_child = None
        sqrt_sum = math.sqrt(sum(child.N for child in node.children.values()) + 1e-8)
        for mv, child in node.children.items():
            if child is None:
                continue
            U = cpuct * child.P * (sqrt_sum / (1 + child.N))
            Q = child.Q
            score = Q + U
            if score > best_score:
                best_score = score
                best_move = mv
                best_child = child
        return best_move, best_child

    def expand_and_eval(self, node: MCTSNode, board: List[List[int]]) -> float:
        priors, value = self.model_infer(board)
        if self.tactical_fn is not None:
            try:
                tactical_priors = self.tactical_fn(board, self.me)
                tactical_flat = np.asarray(tactical_priors).reshape(-1)
                alpha = 0.7
                priors = priors * (1 - alpha) + tactical_flat * alpha
                if priors.sum() > 0:
                    priors = priors / priors.sum()
            except Exception:
                pass
        n = self.board_size
        for r in range(n):
            for c in range(n):
                idx = r*n + c
                if (r,c) in node.children:
                    node.children[(r,c)].P = float(priors[idx]) if idx < len(priors) else 0.0
        return value

    def backup(self, path: List[MCTSNode], value: float):
        for node in reversed(path):
            node.N += 1
            node.W += value
            node.Q = node.W / node.N

    def simulate(self, root_board: List[List[int]], time_limit: float, cpuct: float = 1.0):
        """
        Executa simulações MCTS até o tempo limite..
        """
        n = self.board_size
        while time.time() < time_limit:
            board = [row[:] for row in root_board]
            node = self.mcts_root
            path = [node]
            temp_player = self.me

            while True:
                if node is None:
                    break
                mv, child = self.select_child(node, cpuct=cpuct)
                if mv is None:
                    break
                
                r, c = mv
                if board[r][c] != 0:
                    child.P = 0.0
                    all_zero = all(board[a][b] != 0 for (a,b) in node.children.keys())
                    if all_zero:
                        break
                    else:
                        continue

                board[r][c] = temp_player
                
                if self.rules == 'pente':
                    self.apply_capture_sim(board, r, c, temp_player)

                temp_player = 3 - temp_player
                
                node = child
                path.append(node)
                
                if node.N == 0:
                    break

            try:
                value = self.expand_and_eval(node, board)
            except Exception:
                value = 0.0
            
            self.backup(path, value)

    def compute_move(self, board: List[List[int]], timeout: float = DEFAULT_TIMEOUT) -> Tuple[int,int]:
        legal = legal_moves_from_board(board)
        if len(legal) == 0:
            return (-1, -1)
        priors, _ = self.model_infer(board)
        if self.tactical_fn is not None:
            try:
                tactical_priors = self.tactical_fn(board, self.me, self.rules).reshape(-1)
                alpha = 0.6
                if tactical_priors.shape[0] == priors.shape[0]:
                    priors = priors * (1 - alpha) + tactical_priors * alpha
                    if priors.sum() > 0:
                        priors = priors / priors.sum()
            except Exception:
                pass
        self.mcts_root = self.new_root(priors)
        self.tree_state_board = [row[:] for row in board]
        start = time.time()
        time_limit = start + timeout
        try:
            self.simulate(self.tree_state_board, time_limit=time_limit, cpuct=1.2)
        except Exception as e:
            print("[player.py] simulate error:", e, file=sys.stderr)
        best_move = None
        best_N = -1
        for mv, child in self.mcts_root.children.items():
            r,c = mv
            if board[r][c] != 0:
                continue
            if child.N > best_N:
                best_N = child.N
                best_move = mv
        if best_move is None:
            center = self.board_size // 2
            legal_arr = np.array(legal)
            dists = np.abs(legal_arr[:,0] - center) + np.abs(legal_arr[:,1] - center)
            idx = int(np.argmin(dists))
            return tuple(legal_arr[idx].tolist())
        return best_move

    def play(self, board: List[List[int]], turn_number: int = 0, last_opponent_move: Optional[Tuple[int,int]] = None) -> Tuple[int,int]:
        """
        Chamado pelo motor para cada movimento.
        Retorna (row,col).
        """
        try:
            if last_opponent_move is not None:
                self.last_move = last_opponent_move
            
            start = time.time()
            move = self.compute_move(board, timeout=DEFAULT_TIMEOUT)
            elapsed = time.time() - start
            
            self.last_move = move
            
            try:
                if self.mcts_root is not None and move in self.mcts_root.children:
                    new_root = self.mcts_root.children[move]
                    self.mcts_root = new_root
                    if self.tree_state_board is not None:
                        r,c = move
                        self.tree_state_board[r][c] = self.me
                        if self.rules == 'pente':
                            self.apply_capture_sim(self.tree_state_board, r, c, self.me)
                else:
                    self.mcts_root = None
                    self.tree_state_board = None
            except Exception:
                self.mcts_root = None
                self.tree_state_board = None
            r,c = move
            n = self.board_size
            if not (0 <= r < n and 0 <= c < n) or board[r][c] != 0:
                legal = legal_moves_from_board(board)
                if not legal:
                    return (-1, -1)
                center = n//2
                legal_arr = np.array(legal)
                dists = np.abs(legal_arr[:,0] - center) + np.abs(legal_arr[:,1] - center)
                idx = int(np.argmin(dists))
                return tuple(legal_arr[idx].tolist())
            return (r,c)
        except Exception as e:
            try:
                legal = legal_moves_from_board(board)
                if not legal:
                    return (-1, -1)
                return random.choice(legal)
            except Exception:
                return (-1, -1)
