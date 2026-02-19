"""
AlphaZero Training Script - Self-Play + Training Loop

Executes cycles of:
  - Self-play with Neural Network guided MCTS
  - Network training using generated data

Models are saved in the models/ folder with the names:
  - gomoku_model_best.pth
  - pente_model_best.pth
Compatible with the player.py used in competition.
"""

import sys
import os
import argparse
import shutil

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

import numpy as np
import torch
import torch.optim as optim
import time
from datetime import datetime

try:
    from torch.utils.tensorboard import SummaryWriter
    TENSORBOARD_AVAILABLE = True
except ImportError:
    TENSORBOARD_AVAILABLE = False
    print("WARNING: TensorBoard not available. Install with: pip install tensorboard")

from src.network import create_network, save_checkpoint, load_checkpoint
from src.data_buffer import AugmentedBuffer
from src.training import self_play_game, train_network
from src.game_gomoku import GomokuGame
from src.game_pente import PenteGame


class AlphaZeroTrainer:
    """
    Manages the complete AlphaZero training process.

    Main responsibilities:
      - Generate self-play games using MCTS
      - Store experiences (buffer with data augmentation)
      - Train the neural network from the buffer
      - Save checkpoints and the best model in models/
    """
    
    def __init__(self, 
                 board_size=15,
                 num_filters=64,
                 num_blocks=4,
                 learning_rate=0.001,
                 buffer_size=100000,
                 game_type='gomoku'):
        
        """
        Initializes the AlphaZero trainer.

        Args:
            board_size: size of the board
            num_filters: number of filters in convolutional blocks
            num_blocks: number of residual blocks in the network
            learning_rate: optimizer learning rate
            buffer_size: maximum size of the experience buffer
            game_type: 'gomoku' or 'pente'
        """

        self.board_size = board_size
        self.num_filters = num_filters
        self.num_blocks = num_blocks
        self.game_type = game_type
        
        
        self.models_dir = os.path.join(ROOT_DIR, 'models')
        os.makedirs(self.models_dir, exist_ok=True)
        
        if game_type == 'pente':
            self.game = PenteGame(board_size=board_size)
            self.num_input_channels = 5
        else:
            self.game = GomokuGame(board_size=board_size)
            self.num_input_channels = 3
        
        self.network = create_network(
            board_size, num_filters, num_blocks, 
            num_input_channels=self.num_input_channels
        )
        self.optimizer = optim.Adam(self.network.parameters(), lr=learning_rate)
        
        self.buffer = AugmentedBuffer(max_size=buffer_size)
        self.iteration = 0
        self.total_games = 0
        
        self.writer = None
        if TENSORBOARD_AVAILABLE:
            log_dir = f"logs/{game_type}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
            self.writer = SummaryWriter(log_dir=log_dir)
            print(f"  TensorBoard logs: {log_dir}")
        
        print(f"AlphaZero Trainer initialized for {game_type.upper()}")
    
    def load_checkpoint(self):
        """Loads the previous 'best' model and buffer if they exist."""
        filename = os.path.join(self.models_dir, f"{self.game_type}_model_best.pth")
        if os.path.exists(filename):
            try:
                self.network, self.optimizer, self.iteration = load_checkpoint(
                    filename,
                    board_size=self.board_size,
                    num_filters=self.num_filters,
                    num_blocks=self.num_blocks,
                    num_input_channels=self.num_input_channels
                )
                print(f"Checkpoint loaded: Iteration {self.iteration}")
            except Exception as e:
                print(f"Error loading checkpoint: {e}")
                return False
        
        buffer_path = os.path.join(self.models_dir, f"{self.game_type}_buffer.pkl")
        if os.path.exists(buffer_path):
            try:
                self.buffer.load(buffer_path)
                print(f"Buffer loaded: {len(self.buffer)} experiences retrieved.")
            except Exception as e:
                print(f"Could not load buffer (starting empty): {e}")
        else:
            print("ℹ No buffer found. Starting empty.")

        return True
    
    def self_play(self, num_games=100, time_limit=5.0, temperature_threshold=15):
        """
        Generates self-play games and fills the buffer with experiences.

        Args:
            num_games: number of self-play games to generate
            time_limit: time limit (seconds) for MCTS per move
            temperature_threshold: number of initial moves with high temperature (more exploration)

        Returns:
            dict with self-play statistics (wins, draws, average game length).
        """
        
        print(f"\n{'='*60}")
        print(f"SELF-PLAY: {num_games} games ({time_limit}s MCTS/move)")
        print(f"{'='*60}")
        
        stats = {'player1_wins': 0, 'player2_wins': 0, 'draws': 0, 'avg_game_length': 0}
        start_time = time.time()
        
        for game_num in range(num_games):
            game_start = time.time()
            game_data, winner = self_play_game(
                self.network, self.game, 
                time_limit=time_limit, 
                temperature_threshold=temperature_threshold
            )
            
            self.buffer.process_game_with_augmentation(game_data, winner, board_size=self.board_size)
            
            if winner == 1: stats['player1_wins'] += 1
            elif winner == 2: stats['player2_wins'] += 1
            else: stats['draws'] += 1
            
            stats['avg_game_length'] += len(game_data)
            
            game_time = time.time() - game_start
            print(f"  [{game_num+1:3d}/{num_games}] Win: {'P1' if winner==1 else 'P2' if winner==2 else 'Draw'} | "
                  f"Len: {len(game_data)} | Time: {game_time:.1f}s")
        
        stats['avg_game_length'] /= num_games
        self.total_games += num_games
        
        print(f"Self-play stats: P1: {stats['player1_wins']} | P2: {stats['player2_wins']} | Draw: {stats['draws']}")
        
        if self.writer:
            self.writer.add_scalar('SelfPlay/P1WinRate', stats['player1_wins']/num_games, self.iteration)
            self.writer.add_scalar('SelfPlay/BufferSize', len(self.buffer), self.iteration)
            
        return stats
    
    def train(self, batch_size=64, epochs=10):
        """
        Trains the neural network with the data currently in the buffer.

        Args:
            batch_size: mini-batch size
            epochs: number of training epochs per iteration

        Returns:
            dictionary with loss history.
        """
        if len(self.buffer) < batch_size: return None
        
        print(f"\nTRAINING: {epochs} epochs...")
        losses = train_network(self.network, self.optimizer, self.buffer, batch_size, epochs)
        
        if self.writer:
             self.writer.add_scalar('Loss/Total', losses['total'][-1], self.iteration)
        return losses
    
    def save_best(self):
        """
        Saves the current model as 'best model' in the models/ folder.

        Also creates a historical backup in models/history/ for later analysis.
        """
        filename = os.path.join(self.models_dir, f"{self.game_type}_model_best.pth")
        
        save_checkpoint(self.network, self.optimizer, self.iteration, filename)
        
        backup_name = os.path.join(self.models_dir, f"history/{self.game_type}_iter{self.iteration}.pth")
        os.makedirs(os.path.dirname(backup_name), exist_ok=True)
        shutil.copyfile(filename, backup_name)
        
        print(f"💾 Model saved to: {filename}")
    
    def run_iteration(self, num_games, time_limit, batch_size, epochs):
        """
        Executes a complete training iteration:
          1) self-play
          2) network training
          3) save model and buffer

        Args:
            num_games: number of self-play games in the iteration
            time_limit: time per move in MCTS
            batch_size: training batch size
            epochs: number of training epochs per iteration
        """
        self.iteration += 1
        print(f"\n>>> ITERATION {self.iteration} <<<")
        
        self.self_play(num_games, time_limit)
        self.train(batch_size, epochs)
        self.save_best()
        
        buffer_path = os.path.join(self.models_dir, f"{self.game_type}_buffer.pkl")
        self.buffer.save(buffer_path)

    def run_training(self, num_iterations, games_per_iter, time_limit, batch_size, epochs):
        """
        Runs multiple consecutive training iterations.

        Args:
            num_iterations: number of training iterations (cycles)
            games_per_iter: number of self-play games per iteration
            time_limit: time per move in MCTS
            batch_size: training batch size
            epochs: number of epochs per iteration
        """
        for _ in range(num_iterations):
            self.run_iteration(games_per_iter, time_limit, batch_size, epochs)

def main():
    """
    Main entry point of the training script.

    Reads command-line arguments, selects the configuration
    (mini/custom/full), initializes the AlphaZeroTrainer, and starts
    the training loop for Gomoku or Pente.
    """
    parser = argparse.ArgumentParser()
    parser.add_argument('--game', default='gomoku', choices=['gomoku', 'pente'])
    parser.add_argument('--config', default='custom', choices=['mini', 'medium', 'full', 'custom'])
    args = parser.parse_args()
    
    configs = {
        'mini': {'iters': 5, 'games': 50, 'filters': 32},
        'custom': {'iters': 8, 'games': 35, 'filters': 64}, 
        'full': {'iters': 8, 'games': 70, 'filters': 64}
    }
    
    cfg = configs.get(args.config, configs['custom'])
    
    trainer = AlphaZeroTrainer(
        board_size=15,
        num_filters=cfg['filters'],
        game_type=args.game
    )
    
    trainer.load_checkpoint()
    
    trainer.run_training(
        num_iterations=cfg['iters'],
        games_per_iter=cfg['games'],
        time_limit=5.0,
        batch_size=32,
        epochs=5
    )

if __name__ == "__main__":
    if torch.cuda.is_available():
        print("⚡ GPU detected!")
    main()
