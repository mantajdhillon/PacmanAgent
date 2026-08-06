"""Example usage of the Pac-Man game infrastructure and feature extraction."""

from src.core.game_engine import PacmanGame
from src.core.config import UP, DOWN, LEFT, RIGHT
import random


def example_basic_usage():
    """Example 1: Basic game usage."""
    print("=" * 60)
    print("EXAMPLE 1: Basic Game Setup and Rendering")
    print("=" * 60)

    game = PacmanGame(display=False)
    print(f"Game initialized with board size: {game.board.width}x{game.board.height}")
    print(f"Pac-Man position: {game.game_state.pacman.position.to_tuple()}")
    print(f"Ghosts: {[g.name for g in game.game_state.ghosts]}")
    print(f"Initial pellets: {len(game.board.pellets)}")
    print()


def example_feature_extraction():
    """Example 2: Feature extraction."""
    print("=" * 60)
    print("EXAMPLE 2: Feature Extraction")
    print("=" * 60)

    game = PacmanGame(display=False)
    features = game.get_features()

    print(f"Pacman Info:")
    print(f"  Position: {features['pacman']['position']}")
    print(f"  Score: {features['pacman']['score']}")
    print(f"  Lives: {features['pacman']['lives']}")

    print(f"\nGhost Info:")
    for ghost in features['ghosts']:
        print(f"  {ghost['name']}: {ghost['position']}, scared={ghost['scared']}")

    print(f"\nPellet Info:")
    print(f"  Regular pellets: {features['pellets']['pellet_count']}")
    print(f"  Power pellets: {features['pellets']['power_pellet_count']}")

    print(f"\nDistance Info:")
    print(f"  Nearest pellet distance: {features['distances']['nearest_pellet_distance']}")
    print(f"  Nearest ghost distance: {features['distances']['nearest_ghost_distance']}")
    print()


def example_state_arrays():
    """Example 3: State arrays for AI algorithms."""
    print("=" * 60)
    print("EXAMPLE 3: State Arrays for AI Processing")
    print("=" * 60)

    game = PacmanGame(display=False)
    state_arrays = game.get_state_arrays()

    print(f"Board array shape: {state_arrays['board'].shape}")
    print(f"Pacman position: {state_arrays['pacman_position']}")
    print(f"Ghost positions:\n{state_arrays['ghost_positions']}")
    print(f"Pellet map (first 5x5):\n{state_arrays['pellet_map'][:5, :5]}")
    print()


def example_state_vector():
    """Example 4: State vector for neural networks."""
    print("=" * 60)
    print("EXAMPLE 4: State Vector for Neural Networks")
    print("=" * 60)

    game = PacmanGame(display=False)
    state_vector = game.get_state_vector()

    print(f"State vector shape: {state_vector.shape}")
    print(f"Vector dtype: {state_vector.dtype}")
    print(f"Vector (all values): {state_vector}")
    print(f"Vector components:")
    print(f"  [0-1]: Pacman normalized position (x, y)")
    print(f"  [2-5]: Pacman direction (one-hot encoded)")
    print(f"  [6+]: Ghost positions and scared states")
    print()


def example_gameplay():
    """Example 5: Game simulation."""
    print("=" * 60)
    print("EXAMPLE 5: Game Simulation with Random Moves")
    print("=" * 60)

    game = PacmanGame(display=False)
    game.reset()

    directions = [UP, DOWN, LEFT, RIGHT]
    moves = 0
    max_moves = 50

    while not game.is_game_over() and not game.is_game_won() and moves < max_moves:
        # Random move
        direction = random.choice(directions)
        game.move_pacman(direction)

        # Random ghost moves
        for i in range(len(game.game_state.ghosts)):
            ghost_direction = random.choice(directions)
            game.move_ghost(i, ghost_direction)

        # Check collisions
        game.check_collisions()

        # Update scared timers
        game.update_scared_timers()

        moves += 1

        if moves % 10 == 0:
            print(f"Move {moves}: Pacman at {game.game_state.pacman.position.to_tuple()}, Score: {game.game_state.pacman.score}")

    print(f"Simulation ended after {moves} moves")
    print(f"Final score: {game.game_state.pacman.score}")
    print(f"Pellets remaining: {len(game.board.pellets)}")
    print()


def example_board_navigation():
    """Example 6: Board navigation and reachability."""
    print("=" * 60)
    print("EXAMPLE 6: Board Navigation Analysis")
    print("=" * 60)

    game = PacmanGame(display=False)
    extractor = game.feature_extractor

    # Check reachable positions
    pacman_pos = game.game_state.pacman.position.to_tuple()
    reachable = extractor.get_reachable_positions(game.game_state, pacman_pos, max_distance=3)

    print(f"Pacman position: {pacman_pos}")
    print(f"Reachable positions within distance 3:")
    for pos in sorted(reachable)[:10]:
        dist = extractor._manhattan_distance(pacman_pos, pos)
        print(f"  {pos} (distance: {dist})")

    print(f"Total reachable positions: {len(reachable)}")
    print()


def example_pellet_tracking():
    """Example 7: Pellet tracking and collection."""
    print("=" * 60)
    print("EXAMPLE 7: Pellet Tracking")
    print("=" * 60)

    game = PacmanGame(display=False)

    print(f"Initial state:")
    print(f"  Pellets: {len(game.board.pellets)}")
    print(f"  Power pellets: {len(game.board.power_pellets)}")
    print(f"  Score: {game.game_state.pacman.score}")

    # Simulate collecting some pellets
    pellets = game.board.get_pellets()
    for i in range(5):
        if pellets:
            target = pellets[i]
            game.game_state.pacman.position.x = target[0]
            game.game_state.pacman.position.y = target[1]
            game.board.remove_pellet(target[0], target[1])
            game.game_state.pacman.score += 10
            game.game_state.pellets_eaten += 1

    print(f"\nAfter collecting 5 pellets:")
    print(f"  Pellets: {len(game.board.pellets)}")
    print(f"  Pellets eaten: {game.game_state.pellets_eaten}")
    print(f"  Score: {game.game_state.pacman.score}")
    print()


def example_game_state_components():
    """Example 8: Understanding game state components."""
    print("=" * 60)
    print("EXAMPLE 8: Game State Components")
    print("=" * 60)

    game = PacmanGame(display=False)
    state = game.game_state

    print(f"GameState object components:")
    print(f"  Board: {state.board.__class__.__name__}")
    print(f"  Pacman: {state.pacman.__class__.__name__}")
    print(f"    - Position: {state.pacman.position}")
    print(f"    - Direction: {state.pacman.direction}")
    print(f"    - Score: {state.pacman.score}")
    print(f"    - Lives: {state.pacman.lives}")

    print(f"  Ghosts ({len(state.ghosts)}):")
    for ghost in state.ghosts:
        print(f"    - {ghost.name}: {ghost.position}, scared={ghost.scared}")

    print(f"  Game flags:")
    print(f"    - Game over: {state.game_over}")
    print(f"    - Game won: {state.game_won}")
    print()


def main():
    """Run all examples."""
    print("\n" + "=" * 60)
    print("PAC-MAN INFRASTRUCTURE & FEATURE EXTRACTION EXAMPLES")
    print("=" * 60 + "\n")

    example_basic_usage()
    example_feature_extraction()
    example_state_arrays()
    example_state_vector()
    example_board_navigation()
    example_pellet_tracking()
    example_game_state_components()
    example_gameplay()

    print("=" * 60)
    print("Examples completed!")
    print("=" * 60)


if __name__ == "__main__":
    main()
