"""
Main game runner – runs matches between different agents.
"""
import sys
import os
import importlib.util
import time

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

from src.game_gomoku import GomokuGame
from src.game_pente import PenteGame


def load_player(filepath, player_number):
    """
    Dynamically loads a Player class from a Python file.

    Args:
        filepath: path to the player file
        player_number: 1 or 2

    Returns:
        Player class defined in the file.
    """
    try:
        spec = importlib.util.spec_from_file_location(f"player{player_number}", filepath)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        return module.Player
    except Exception as e:
        print(f"Error loading player from {filepath}: {e}")
        sys.exit(1)


def play_game(player1_file, player2_file, game_type='gomoku', display=True, timeout=5.0):
    """
    Executes a match between two players.

    Args:
        player1_file: path to player 1's file
        player2_file: path to player 2's file
        game_type: 'gomoku' or 'pente'
        display: whether to show the board in the terminal
        timeout: time limit per move, in seconds

    Returns:
        winner: 1 or 2 if there's a winner, 0 for a draw, -1 in case of an error
    """
    Player1Class = load_player(player1_file, 1)
    Player2Class = load_player(player2_file, 2)
    
    if game_type == 'gomoku':
        game = GomokuGame(15)
    elif game_type == 'pente':
        game = PenteGame(15)
    else:
        print(f"Unknown game type: {game_type}")
        return -1
    
    try:
        player1 = Player1Class(game_type, 15)
        player2 = Player2Class(game_type, 15)
    except Exception as e:
        print(f"Error initializing players: {e}")
        return -1
    
    player1_name = getattr(player1, 'name', 'Player 1')
    player2_name = getattr(player2, 'name', 'Player 2')
    
    print(f"\n{'='*60}")
    print(f"Game: {game_type.upper()}")
    print(f"Black (●): {player1_name}")
    print(f"White (○): {player2_name}")
    print(f"{'='*60}\n")
    
    if display:
        game.display()
    
    turn_number = 0
    last_move = None
    
    while not game.is_game_over():
        current_player_obj = player1 if game.current_player == 1 else player2
        current_player_name = player1_name if game.current_player == 1 else player2_name
        
        board_for_player = game.get_board_for_player(game.current_player)
        
        try:
            start_time = time.time()
            move = current_player_obj.play(board_for_player, turn_number, last_move)
            elapsed = time.time() - start_time
            
            if elapsed > timeout:
                print(f"⚠️  {current_player_name} exceeded time limit ({elapsed:.2f}s > {timeout}s)")
                print(f"   Making random move instead...")
                
                valid_moves = game.get_valid_moves()
                if valid_moves:
                    import random
                    move = random.choice(valid_moves)
                else:
                    print("No valid moves available!")
                    break
            
            row, col = move
            
        except Exception as e:
            print(f"❌ Error from {current_player_name}: {e}")
            print(f"   Making random move instead...")
            
            valid_moves = game.get_valid_moves()
            if valid_moves:
                import random
                row, col = random.choice(valid_moves)
            else:
                print("No valid moves available!")
                break
        
        if game.make_move(row, col):
            symbol = '●' if game.current_player == 2 else '○'
            print(f"Turn {turn_number + 1}: {current_player_name} ({symbol}) → ({row}, {col})")
            
            if display:
                game.display()
            
            last_move = (row, col)
            turn_number += 1
        else:
            print(f"❌ Invalid move from {current_player_name}: ({row}, {col})")
            print(f"   Player {game.current_player} forfeits!")
            return 3 - game.current_player
    
    winner = game.get_winner()
    
    print(f"\n{'='*60}")
    if winner == 0:
        print("🤝 Game Over: DRAW!")
    elif winner == 1:
        print(f"🏆 Game Over: {player1_name} (●) WINS!")
    elif winner == 2:
        print(f"🏆 Game Over: {player2_name} (○) WINS!")
    
    if game_type == 'pente':
        print(f"\nCaptures - Black: {game.captures[1]}, White: {game.captures[2]}")
    
    print(f"Total turns: {turn_number}")
    print(f"{'='*60}\n")
    
    return winner


def main():
    """
    Main entry point of the script.
    
    Reads command-line arguments and sets:
      - the file for each player,
      - the game type (gomoku or pente),
      - whether the board is displayed or not,

    and then calls `play_game` to execute a single match between the two agents.
    """
    if len(sys.argv) < 3:
        print("Usage: python play.py <player1.py> <player2.py> [--game gomoku|pente] [--display] [--nodisplay]")
        print("\nExamples:")
        print("  python play.py player_random.py player_heuristic.py")
        print("  python play.py player.py player_random.py --game pente")
        print("  python play.py player1.py player2.py --nodisplay")
        sys.exit(1)
    
    player1_file = sys.argv[1]
    player2_file = sys.argv[2]
    
    game_type = 'gomoku'
    display = True
    
    for arg in sys.argv[3:]:
        if arg == '--game' and len(sys.argv) > sys.argv.index(arg) + 1:
            game_type = sys.argv[sys.argv.index(arg) + 1]
        elif arg == '--nodisplay':
            display = False
        elif arg == '--display':
            display = True
    
    winner = play_game(player1_file, player2_file, game_type, display)
    
    sys.exit(0 if winner in [0, 1, 2] else 1)


if __name__ == "__main__":
    main()
