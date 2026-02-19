"""
AlphaZero Neural Network for Gomoku/Pente
Architecture: ResNet with Policy and Value heads
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


class ResidualBlock(nn.Module):
    """Residual block with 2 convolutions and a skip connection."""
    
    def __init__(self, num_filters):
        super(ResidualBlock, self).__init__()
        self.conv1 = nn.Conv2d(num_filters, num_filters, 3, padding=1)
        self.bn1 = nn.BatchNorm2d(num_filters)
        self.conv2 = nn.Conv2d(num_filters, num_filters, 3, padding=1)
        self.bn2 = nn.BatchNorm2d(num_filters)
    
    def forward(self, x):
        residual = x
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += residual
        out = F.relu(out)
        return out


class GomokuNet(nn.Module):
    """
    AlphaZero Neural Network for Gomoku/Pente
    
    Input: tensor (batch, num_input_channels, board_size, board_size)
        GOMOKU (3 channels):
        - Channel 0: current player's positions
        - Channel 1: opponent's positions
        - Channel 2: last move
        
        PENTE (5 channels):
        - Channels 0-2: same as Gomoku
        - Channel 3: current player's captures (normalized 0-1)
        - Channel 4: opponent's captures (normalized 0-1)
    
    Output:
        - policy: probabilities for each action (batch, board_size * board_size)
        - value: position evaluation (batch, 1) in [-1, 1]
    """
    
    def __init__(self, board_size=15, num_filters=64, num_blocks=4, num_input_channels=3):
        super(GomokuNet, self).__init__()
        self.board_size = board_size
        self.num_actions = board_size * board_size

        self.conv_input = nn.Conv2d(num_input_channels, num_filters, 3, padding=1)
        self.bn_input = nn.BatchNorm2d(num_filters)
        
        self.res_blocks = nn.ModuleList([
            ResidualBlock(num_filters) for _ in range(num_blocks)
        ])

        self.policy_conv = nn.Conv2d(num_filters, 2, 1)
        self.policy_bn = nn.BatchNorm2d(2)
        self.policy_fc = nn.Linear(2 * board_size * board_size, self.num_actions)

        self.value_conv = nn.Conv2d(num_filters, 1, 1)
        self.value_bn = nn.BatchNorm2d(1)
        self.value_fc1 = nn.Linear(board_size * board_size, 64)
        self.value_fc2 = nn.Linear(64, 1)
    
    def forward(self, x):
        """
        Forward pass
        
        Args:
            x: tensor (batch, 3 or 5, board_size, board_size)
        
        Returns:
            policy: tensor (batch, board_size * board_size) with log-probabilities
            value: tensor (batch, 1) in [-1, 1]
        """

        x = F.relu(self.bn_input(self.conv_input(x)))
        
        for block in self.res_blocks:
            x = block(x)
        
        policy = F.relu(self.policy_bn(self.policy_conv(x)))
        policy = policy.view(policy.size(0), -1)
        policy = self.policy_fc(policy)
        policy = F.log_softmax(policy, dim=1)
        
        value = F.relu(self.value_bn(self.value_conv(x)))
        value = value.view(value.size(0), -1)
        value = F.relu(self.value_fc1(value))
        value = torch.tanh(self.value_fc2(value))
        
        return policy, value
    
    def predict(self, board_state):
        """
        Prediction for a single state.
        
        Args:
            board_state: numpy array (num_input_channels, board_size, board_size)
        
        Returns:
            policy_probs: numpy array (board_size * board_size) with probabilities
            value: float in [-1, 1]
        """
        self.eval()
        with torch.no_grad():
            state_tensor = torch.FloatTensor(board_state).unsqueeze(0)

            log_policy, value = self.forward(state_tensor)

            policy_probs = torch.exp(log_policy).squeeze(0).cpu().numpy()
            value = value.item()
        
        return policy_probs, value


def board_to_tensor(board, current_player, last_move=None, board_size=15, 
                    captures_p1=None, captures_p2=None):
    """
    Converts a numpy board to the network's input tensor format.
    
    Args:
        board: numpy array (board_size, board_size) with 0/1/2
        current_player: int (1 or 2) - the player making the move
        last_move: tuple (row, col) or None - opponent's last move
        board_size: int - size of the board
        captures_p1: int or None - player 1's captures (Pente). None = Gomoku
        captures_p2: int or None - player 2's captures (Pente). None = Gomoku
    
    Returns:
        tensor: numpy array (3 or 5, board_size, board_size)
            - Gomoku (captures_p1=None, captures_p2=None): 3 channels
            - Pente (captures provided): 5 channels
    """

    has_captures = (captures_p1 is not None or captures_p2 is not None)
    
    if has_captures:
        captures_p1 = captures_p1 if captures_p1 is not None else 0
        captures_p2 = captures_p2 if captures_p2 is not None else 0
    
    num_channels = 5 if has_captures else 3
    
    tensor = np.zeros((num_channels, board_size, board_size), dtype=np.float32)

    tensor[0] = (board == current_player).astype(np.float32)

    opponent = 3 - current_player 
    tensor[1] = (board == opponent).astype(np.float32)
    
    if last_move is not None:
        row, col = last_move
        if 0 <= row < board_size and 0 <= col < board_size:
            tensor[2, row, col] = 1.0
    
    if has_captures:
        my_captures = captures_p1 if current_player == 1 else captures_p2
        opp_captures = captures_p2 if current_player == 1 else captures_p1
        
        tensor[3, :, :] = min(my_captures / 10.0, 1.0)
        tensor[4, :, :] = min(opp_captures / 10.0, 1.0)
    
    return tensor


def create_network(board_size=15, num_filters=64, num_blocks=4, num_input_channels=3):
    """
    Factory function to create the neural network.
    
    Args:
        board_size: size of the board
        num_filters: number of filters in the convolutions
        num_blocks: number of residual blocks
        num_input_channels: number of input channels
    
    Returns:
        GomokuNet: instantiated neural network
    """
    return GomokuNet(board_size, num_filters, num_blocks, num_input_channels)


def save_checkpoint(model, optimizer, epoch, filename):
    """Saves a model checkpoint."""
    checkpoint = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'board_size': model.board_size,
    }
    torch.save(checkpoint, filename)
    print(f"Checkpoint saved: {filename}")


def load_checkpoint(filename, board_size=15, num_filters=64, num_blocks=4, num_input_channels=3):
    """
    Loads a model checkpoint.
    
    Args:
        filename: path to the checkpoint
        board_size: size of the board
        num_filters: filters in the convolutions
        num_blocks: residual blocks
        num_input_channels: number of channels (3 for Gomoku, 5 for Pente)
    
    Returns:
        model, optimizer, epoch
    """
    checkpoint = torch.load(filename)
    
    model = create_network(
        board_size=checkpoint.get('board_size', board_size),
        num_filters=num_filters,
        num_blocks=num_blocks,
        num_input_channels=num_input_channels
    )
    model.load_state_dict(checkpoint['model_state_dict'])
    
    optimizer = torch.optim.Adam(model.parameters())
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    
    epoch = checkpoint['epoch']
    
    print(f"Checkpoint loaded: {filename} (epoch {epoch})")
    return model, optimizer, epoch
