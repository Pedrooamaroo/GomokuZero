import numpy as np
import pickle
import os
from collections import deque


class ExperienceBuffer:
    """Buffer para armazenar experiências de self-play"""
    
    def __init__(self, max_size=100000):
        self.buffer = deque(maxlen=max_size)
        self.max_size = max_size
    
    def add_game(self, game_data):
        """
        Adiciona experiências de um jogo completo
        
        Argumentos:
            game_data: lista de tuplos (state, mcts_policy, player)
                - state: numpy array (3, board_size, board_size)
                - mcts_policy: numpy array (board_size * board_size) com distribuição MCTS
                - player: int (1 ou 2) - jogador que fez a jogada
        """
        for experience in game_data:
            self.buffer.append(experience)
    
    def add_experience(self, state, policy, value):
        """
        Adiciona uma experiência individual
        
        Argumentos:
            state: numpy array (3, board_size, board_size)
            policy: numpy array (board_size * board_size) - distribuição alvo
            value: float em [-1, 1] - valor alvo
        """
        self.buffer.append((state, policy, value))
    
    def process_game(self, game_states, winner):
        """
        Processa um jogo completo e adiciona ao buffer com valores corretos
        
        Argumentos:
            game_states: lista de tuplos (state, mcts_policy, player)
            winner: int (1, 2, ou 0 para empate)
        """
        for state, policy, player in game_states:
            if winner == 0:
                value = 0.0
            elif winner == player:
                value = 1.0
            else:
                value = -1.0
            
            self.add_experience(state, policy, value)
    
    def sample(self, batch_size):
        """
        Amostra exemplo(batch) aleatório de experiências
        
        Argumentos:
            batch_size: número de experiências a amostrar
        
        Returns:
            states: numpy array (batch_size, 3, board_size, board_size)
            policies: numpy array (batch_size, board_size * board_size)
            values: numpy array (batch_size, 1)
        """
        if len(self.buffer) < batch_size:
            batch_size = len(self.buffer)
        
        indices = np.random.choice(len(self.buffer), batch_size, replace=False)
        
        states = []
        policies = []
        values = []
        
        for idx in indices:
            state, policy, value = self.buffer[idx]
            states.append(state)
            policies.append(policy)
            values.append(value)
        
        return (
            np.array(states, dtype=np.float32),
            np.array(policies, dtype=np.float32),
            np.array(values, dtype=np.float32).reshape(-1, 1)
        )
    
    def get_all(self):
        """
        Retorna todas as experiências do buffer
        
        Returns:
            states, policies, values: numpy arrays
        """
        if len(self.buffer) == 0:
            return None, None, None
        
        states = []
        policies = []
        values = []
        
        for state, policy, value in self.buffer:
            states.append(state)
            policies.append(policy)
            values.append(value)
        
        return (
            np.array(states, dtype=np.float32),
            np.array(policies, dtype=np.float32),
            np.array(values, dtype=np.float32).reshape(-1, 1)
        )
    
    def save(self, filename):
        """Salva buffer em disco"""
        with open(filename, 'wb') as f:
            pickle.dump(list(self.buffer), f)
        print(f"Buffer salvo: {filename} ({len(self.buffer)} experiências)")
    
    def load(self, filename):
        """Carrega buffer do disco"""
        if not os.path.exists(filename):
            print(f"Arquivo não encontrado: {filename}")
            return
        
        with open(filename, 'rb') as f:
            data = pickle.load(f)
            self.buffer = deque(data, maxlen=self.max_size)
        print(f"Buffer carregado: {filename} ({len(self.buffer)} experiências)")
    
    def clear(self):
        """Limpa o buffer"""
        self.buffer.clear()
    
    def __len__(self):
        """Retorna o número de experiências no buffer"""
        return len(self.buffer)
    
    def stats(self):
        """Retorna estatísticas do buffer"""
        if len(self.buffer) == 0:
            return "Buffer vazio"
        
        values = [exp[2] for exp in self.buffer]
        
        return {
            'size': len(self.buffer),
            'max_size': self.max_size,
            'value_mean': np.mean(values),
            'value_std': np.std(values),
            'wins': sum(1 for v in values if v > 0.5),
            'losses': sum(1 for v in values if v < -0.5),
            'draws': sum(1 for v in values if abs(v) <= 0.5),
        }


class AugmentedBuffer(ExperienceBuffer):
    """
    Buffer com data augmentation (rotações e reflexões)
    Aumenta dataset 8x (4 rotações × 2 reflexões)
    """
    
    def add_experience_with_augmentation(self, state, policy, value, board_size=15):
        """
        Adiciona experiência com todas as transformações simétricas
        
        Argumentos:
            state: numpy array (3, board_size, board_size)
            policy: numpy array (board_size * board_size)
            value: float
            board_size: tamanho do tabuleiro
        """
        policy_2d = policy.reshape(board_size, board_size)
        for k in range(4):
            for flip in [False, True]:
                aug_state = np.rot90(state, k, axes=(1, 2))
                if flip:
                    aug_state = np.flip(aug_state, axis=2)

                aug_policy_2d = np.rot90(policy_2d, k)
                if flip:
                    aug_policy_2d = np.flip(aug_policy_2d, axis=1)
                
                aug_policy = aug_policy_2d.flatten()
                
                self.buffer.append((aug_state.copy(), aug_policy.copy(), value))
    
    def process_game_with_augmentation(self, game_states, winner, board_size=15):
        """
        Processa jogo com data augmentation
        
        Argumentos:
            game_states: lista de tuplos (state, mcts_policy, player)
            winner: int (1, 2, ou 0)
            board_size: tamanho do tabuleiro
        """
        for state, policy, player in game_states:
            if winner == 0:
                value = 0.0
            elif winner == player:
                value = 1.0
            else:
                value = -1.0
            
            self.add_experience_with_augmentation(state, policy, value, board_size)


if __name__ == "__main__":
    print("Testando ExperienceBuffer...")
    
    buffer = ExperienceBuffer(max_size=1000)
    
    board_size = 15
    for i in range(10):
        state = np.random.randn(3, board_size, board_size).astype(np.float32)
        policy = np.random.rand(board_size * board_size).astype(np.float32)
        policy = policy / policy.sum()
        value = np.random.uniform(-1, 1)
        
        buffer.add_experience(state, policy, value)
    
    print(f"Buffer size: {len(buffer)}")
    
    states, policies, values = buffer.sample(5)
    print(f"Batch shapes: {states.shape}, {policies.shape}, {values.shape}")
    print(f"Policy sum: {policies.sum(axis=1)}")
    
    print(f"Stats: {buffer.stats()}")

    print("\nTestando AugmentedBuffer...")
    aug_buffer = AugmentedBuffer(max_size=10000)
    
    state = np.random.randn(3, 15, 15).astype(np.float32)
    policy = np.random.rand(15 * 15).astype(np.float32)
    policy = policy / policy.sum()
    
    aug_buffer.add_experience_with_augmentation(state, policy, 1.0, board_size=15)
    print(f"Augmented buffer size: {len(aug_buffer)}")
    
    print("\n Buffer de experiências criado com sucesso!")
