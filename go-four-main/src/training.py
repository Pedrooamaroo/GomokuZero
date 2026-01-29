import numpy as np
import torch
import torch.nn.functional as F
from .network import GomokuNet, board_to_tensor
from .data_buffer import AugmentedBuffer
from .game_gomoku import GomokuGame
from .mcts import MCTS
from .game_engine import apply_capture_pente, check_win_by_capture


def self_play_game(network, game=None, time_limit=5.0, temperature_threshold=15):
    """
    Joga um jogo completo de self-play usando MCTS unificado
    
    Argumentos:
        network: GomokuNet
        game: GomokuGame ou PenteGame
        time_limit: tempo limite em segundos por jogada
        temperature_threshold: jogadas até usar temp=1.0, depois temp=0.1
    
    Returns:
        game_data: lista de (state, policy, player)
        winner: int (1, 2, ou 0)
    """
    if game is None:
        game = GomokuGame(board_size=15)
    
    board_size = game.board_size
    board = np.zeros((board_size, board_size), dtype=np.int8)
    player = 1
    game_data = []
    last_move = None
    
    is_pente = hasattr(game, 'captures')
    game_type = 'pente' if is_pente else 'gomoku'
    captures_p1 = 0 if is_pente else None
    captures_p2 = 0 if is_pente else None
    
    mcts = MCTS(network, game_type=game_type, board_size=board_size, num_simulations=800, c_puct=1.5)
    
    for turn in range(board_size * board_size):
        if turn < temperature_threshold:
            temperature = 1.0
        elif turn < temperature_threshold + 15:
            progress = (turn - temperature_threshold) / 15.0
            temperature = 1.0 - 0.9 * progress
        else:
            temperature = 0.1
        state = board_to_tensor(board, player, last_move, board_size, 
                               captures_p1, captures_p2)
        

        add_exploration_noise = (turn < 10)
        
        policy, _ = mcts.search(
            board=board,
            player=player,
            temperature=temperature,
            captures_p1=captures_p1,
            captures_p2=captures_p2,
            time_limit=time_limit,
            add_noise=add_exploration_noise
        )
        
        game_data.append((state, policy, player))

        policy_sum = policy.sum()
        if policy_sum > 0:
            policy = policy / policy_sum
        else:
            policy = np.ones_like(policy)
            for i in range(len(policy)):
                r, c = i // board_size, i % board_size
                if board[r, c] != 0:
                    policy[i] = 0
            policy = policy / policy.sum() if policy.sum() > 0 else policy
        
        policy = policy / policy.sum()
        
        action_idx = np.random.choice(len(policy), p=policy)
        row = action_idx // board_size
        col = action_idx % board_size
        
        board[row, col] = player
        last_move = (row, col)

        if is_pente:
            captured = apply_capture_pente(board, row, col)
            if player == 1:
                captures_p1 += captured
            else:
                captures_p2 += captured
        
        mcts.update_root((row, col))

        if is_pente:
            winner = check_win_by_capture(captures_p1, captures_p2)
            if winner == 0:
                winner = game.check_winner(board, captures_p1, captures_p2)
        else:
            winner = game.check_winner(board)
        
        if winner != 0:
            return game_data, winner
        
        player = 3 - player
    
    return game_data, 0


def train_network(network, optimizer, buffer, batch_size=64, epochs=10):
    """
    Treina rede neural com experiências do buffer
    
    Argumentos:
        network: GomokuNet
        optimizer: torch optimizer
        buffer: AugmentedBuffer
        batch_size: tamanho do batch
        epochs: número de épocas
    
    Returns:
        losses: dict com histórico de losses
    """
    network.train()
    device = next(network.parameters()).device
    
    losses = {'policy': [], 'value': [], 'total': []}
    
    for epoch in range(epochs):
        epoch_policy_loss = 0.0
        epoch_value_loss = 0.0
        epoch_total_loss = 0.0
        num_batches = 0

        buffer_size = len(buffer)
        num_steps = max(1, buffer_size // batch_size)
        
        for _ in range(num_steps):

            states, target_policies, target_values = buffer.sample(batch_size)

            states = torch.FloatTensor(states).to(device)
            target_policies = torch.FloatTensor(target_policies).to(device)
            target_values = torch.FloatTensor(target_values).to(device)

            log_policies, values = network(states)

            policy_loss = -torch.mean(torch.sum(target_policies * log_policies, dim=1))
            
            value_loss = F.mse_loss(values, target_values)

            total_loss = policy_loss + value_loss

            optimizer.zero_grad()
            total_loss.backward()
            optimizer.step()
            
            epoch_policy_loss += policy_loss.item()
            epoch_value_loss += value_loss.item()
            epoch_total_loss += total_loss.item()
            num_batches += 1

        if num_batches > 0:
            epoch_policy_loss /= num_batches
            epoch_value_loss /= num_batches
            epoch_total_loss /= num_batches
        
        losses['policy'].append(epoch_policy_loss)
        losses['value'].append(epoch_value_loss)
        losses['total'].append(epoch_total_loss)
        
        print(f"  Época {epoch+1}/{epochs}: "
              f"Policy Loss={epoch_policy_loss:.4f}, "
              f"Value Loss={epoch_value_loss:.4f}, "
              f"Total={epoch_total_loss:.4f}")
    
    return losses


if __name__ == "__main__":
    pass
