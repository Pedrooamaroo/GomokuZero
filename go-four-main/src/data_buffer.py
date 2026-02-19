import numpy as np
import pickle
import os
from collections import deque


class ExperienceBuffer:
    """Buffer to store self-play experiences."""
    
    def __init__(self, max_size=100000):
        self.buffer = deque(maxlen=max_size)
        self.max_size = max_size
    
    def add_game(self, game_data):
        """
        Adds experiences from a complete game.
        
        Args:
            game_data: list of tuples (state, mcts_policy, player)
                - state: numpy array (3, board_size, board_size)
                - mcts_policy: numpy array (board_size * board_size) with MCTS distribution
                - player: int (1 or 2) - player who made the move
        """
        for experience in game_data:
            self.buffer.append(experience)
    
    def add_experience(self, state, policy, value):
        """
        Adds an individual experience.
        
        Args:
            state: numpy array (3, board_size, board_size)
            policy: numpy array (board_size * board_size) - target distribution
            value: float in [-1, 1] - target value
        """
        self.buffer.append((state, policy, value))
    
    def process_game(self, game_states, winner):
        """
        Processes a complete game and adds it to the buffer with correct values.
        
        Args:
            game_states: list of tuples (state, mcts_policy, player)
            winner: int (1, 2, or 0 for a draw)
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
        Samples a random batch of experiences.
        
        Args:
            batch_size: number of experiences to sample
        
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
        Returns all experiences from the buffer.
        
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
        """Saves the buffer to disk."""
        with open(filename, 'wb') as f:
            pickle.dump(list(self.buffer), f)
        print(f"Buffer saved: {filename} ({len(self.buffer)} experiences)")
    
    def load(self, filename):
        """Loads the buffer from disk."""
        if not os.path.exists(filename):
            print(f"File not found: {filename}")
            return
        
        with open(filename, 'rb') as f:
            data = pickle.load(f)
            self.buffer = deque(data, maxlen=self.max_size)
        print(f"Buffer loaded: {filename} ({len(self.buffer)} experiences)")
    
    def clear(self):
        """Clears the buffer."""
        self.buffer.clear()
    
    def __len__(self):
        """Returns the number of experiences in the buffer."""
        return len(self.buffer)
    
    def stats(self):
        """Returns buffer statistics."""
        if len(self.buffer) == 0:
            return "Empty buffer"
        
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
    Buffer with data augmentation (rotations and reflections).
    Increases the dataset 8x (4 rotations x 2 reflections).
    """
    
    def add_experience_with_augmentation(self, state, policy, value, board_size=15):
        """
        Adds an experience with all symmetric transformations.
        
        Args:
            state: numpy array (3, board_size, board_size)
            policy: numpy array (board_size * board_size)
            value: float
            board_size: board size
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
        Processes a game with data augmentation.
        
        Args:
            game_states: list of tuples (state, mcts_policy, player)
            winner: int (1, 2, or 0)
            board_size: board size
        """
        for state, policy, player in game_states:
            if winner == 0:
                value = 0.0
            elif winner == player:
                value = 1.0
            else:
                value = -1.0
            
            self.add_experience_with_augmentation(state, policy, value, board_size)
