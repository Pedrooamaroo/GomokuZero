"""
Script de Treino AlphaZero - Self-Play + Training Loop

Executa ciclos de:
  self-play com MCTS guiado pela rede neural
  treino da rede com os dados gerados

Os modelos são guardados na pasta models/ com o nome:
  - gomoku_model_best.pth
  - pente_model_best.pth
compatíveis com o player.py usado na competição.
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
    print("WARNING: TensorBoard nao disponivel. Instale com: pip install tensorboard")

from src.network import create_network, save_checkpoint, load_checkpoint
from src.data_buffer import AugmentedBuffer
from src.training import self_play_game, train_network
from src.game_gomoku import GomokuGame
from src.game_pente import PenteGame


class AlphaZeroTrainer:
    """
    Classe que gere o processo completo de treino do AlphaZero.

    Responsabilidades principais:
      - Gerar jogos de self-play com MCTS
      - Guardar as experiências (buffer com data augmentation)
      - Treinar a rede neural a partir do buffer
      - Guardar checkpoints e o melhor modelo em models/
    """
    
    def __init__(self, 
                 board_size=15,
                 num_filters=64,
                 num_blocks=4,
                 learning_rate=0.001,
                 buffer_size=100000,
                 game_type='gomoku'):
        
        """
        Inicializa o treinador AlphaZero.

        Args:
            board_size: tamanho do tabuleiro
            num_filters: número de filtros nos blocos convolucionais
            num_blocks: número de blocos residuais na rede
            learning_rate: taxa de aprendizagem do otimizador
            buffer_size: tamanho máximo do buffer de experiências
            game_type: 'gomoku' ou 'pente'
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
        
        print(f"AlphaZero Trainer inicializado para {game_type.upper()}")
    
    def load_checkpoint(self):
        """Carrega o 'best' model anterior e o buffer se existirem"""
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
                print(f"Checkpoint carregado: Iteração {self.iteration}")
            except Exception as e:
                print(f"Erro ao carregar checkpoint: {e}")
                return False
        
        buffer_path = os.path.join(self.models_dir, f"{self.game_type}_buffer.pkl")
        if os.path.exists(buffer_path):
            try:
                self.buffer.load(buffer_path)
                print(f"Buffer carregado: {len(self.buffer)} experiências recuperadas.")
            except Exception as e:
                print(f"Não foi possível carregar o buffer (iniciando vazio): {e}")
        else:
            print("ℹNenhum buffer encontrado. Iniciando vazio.")

        return True
    
    def self_play(self, num_games=100, time_limit=5.0, temperature_threshold=15):
        """
        Gera jogos de self-play e preenche o buffer com experiências.

        Args:
            num_games: número de jogos de self-play a gerar
            time_limit: limite de tempo (segundos) para o MCTS por jogada
            temperature_threshold: nº de jogadas iniciais com temperatura alta (mais exploração)

        Returns:
            dict com estatísticas do self-play (vitórias, empates, tamanho médio dos jogos).
        """
        
        print(f"\n{'='*60}")
        print(f"SELF-PLAY: {num_games} jogos ({time_limit}s MCTS/jogada)")
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
        Treina a rede neural com os dados atualmente no buffer.

        Args:
            batch_size: tamanho do mini-batch
            epochs: número de épocas de treino por iteração

        Returns:
            dicionário com histórico das losses.
        """
        if len(self.buffer) < batch_size: return None
        
        print(f"\nTRAINING: {epochs} épocas...")
        losses = train_network(self.network, self.optimizer, self.buffer, batch_size, epochs)
        
        if self.writer:
             self.writer.add_scalar('Loss/Total', losses['total'][-1], self.iteration)
        return losses
    
    def save_best(self):
        """
        Salva o modelo atual como 'melhor modelo' na pasta models/.

        Também cria um backup histórico em models/history/ para análise posterior.
        """
        filename = os.path.join(self.models_dir, f"{self.game_type}_model_best.pth")
        
        save_checkpoint(self.network, self.optimizer, self.iteration, filename)
        
        backup_name = os.path.join(self.models_dir, f"history/{self.game_type}_iter{self.iteration}.pth")
        os.makedirs(os.path.dirname(backup_name), exist_ok=True)
        shutil.copyfile(filename, backup_name)
        
        print(f"💾 Modelo salvo em: {filename}")
    
    def run_iteration(self, num_games, time_limit, batch_size, epochs):
        """
        Executa uma iteração completa de treino:
          1) self-play
          2) treino da rede
          3) guardar modelo e buffer

        Args:
            num_games: nº de jogos de self-play na iteração
            time_limit: tempo por jogada no MCTS
            batch_size: tamanho do batch de treino
            epochs: nº de épocas de treino por iteração
        """
        self.iteration += 1
        print(f"\n>>> ITERAÇÃO {self.iteration} <<<")
        
        self.self_play(num_games, time_limit)
        self.train(batch_size, epochs)
        self.save_best()
        
        buffer_path = os.path.join(self.models_dir, f"{self.game_type}_buffer.pkl")
        self.buffer.save(buffer_path)

    def run_training(self, num_iterations, games_per_iter, time_limit, batch_size, epochs):
        """
        Corre várias iterações de treino consecutivas.

        Args:
            num_iterations: número de iterações (ciclos) de treino
            games_per_iter: nº de jogos de self-play por iteração
            time_limit: tempo por jogada no MCTS
            batch_size: tamanho do batch de treino
            epochs: nº de épocas por iteração
        """
        for _ in range(num_iterations):
            self.run_iteration(games_per_iter, time_limit, batch_size, epochs)

def main():
    """
    Ponto de entrada principal do script de treino.

    Lê argumentos da linha de comandos, escolhe a configuração
    (mini/custom/full), inicializa o AlphaZeroTrainer e arranca
    o ciclo de treino para Gomoku ou Pente.
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
        print("⚡ GPU detectada!")
    main()