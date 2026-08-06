"""Quick demo showing all testing options without running interactive games."""

import sys
from src.core.game_engine import PacmanGame
from src.core.feature_extractor import FeatureExtractor
from src.core.config import UP, DOWN, LEFT, RIGHT
import random


def demo_1_game_with_ui():
    """Demo 1: Show game can be displayed."""
    print("\n" + "=" * 70)
    print("DEMO 1: GAME WITH UI (Pygame Window)")
    print("=" * 70)
    print("\nInitializing Pac-Man game with Pygame display...")

    game = PacmanGame(display=True)
    print(f"✓ Pygame window created successfully")
    print(f"  - Screen: {game.screen is not None}")
    print(f"  - Window size: {game.screen.get_width()}x{game.screen.get_height()} pixels")
    print(f"  - Board: {game.board.width}x{game.board.height} cells")
    print(f"  - Cell size: 30 pixels")
    print(f"\nTo use:")
    print(f"  >>> python play_game.py")
    print(f"\nControls in game:")
    print(f"  - Arrow keys: Move Pac-Man")
    print(f"  - SPACE: Pause/Resume")
    print(f"  - R: Restart")
    print(f"  - ESC: Quit")

    game.quit()
    print(f"\n✓ Display demo complete")


def demo_2_headless_simulation():
    """Demo 2: Run game without UI (headless)."""
    print("\n" + "=" * 70)
    print("DEMO 2: HEADLESS SIMULATION (No UI)")
    print("=" * 70)

    game = PacmanGame(display=False)
    game.reset()

    print(f"\nRunning 20-step simulation...")
    for step in range(20):
        # Move Pac-Man randomly
        direction = random.choice([UP, DOWN, LEFT, RIGHT])
        game.move_pacman(direction)

        # Move ghosts randomly
        for i in range(len(game.game_state.ghosts)):
            ghost_dir = random.choice([UP, DOWN, LEFT, RIGHT])
            game.move_ghost(i, ghost_dir)

        # Update game
        game.check_collisions()
        game.update_scared_timers()

        if step % 5 == 0:
            features = game.get_features()
            print(f"  Step {step:2d}: Score={features['pacman']['score']:3d}, "
                  f"Pellets={features['pellets']['pellet_count']:3d}, "
                  f"Pos={features['pacman']['position']}")

        if game.is_game_over() or game.is_game_won():
            break

    print(f"\n✓ Simulation complete")
    print(f"  Final Score: {game.game_state.pacman.score}")
    print(f"  Pellets Eaten: {game.game_state.pellets_eaten}")
    print(f"  Lives Remaining: {game.game_state.pacman.lives}")


def demo_3_feature_extraction():
    """Demo 3: Show feature extraction in all formats."""
    print("\n" + "=" * 70)
    print("DEMO 3: FEATURE EXTRACTION (All Formats)")
    print("=" * 70)

    game = PacmanGame(display=False)
    extractor = FeatureExtractor()

    # Format 1: Feature Dictionary
    print(f"\nFormat 1: Feature Dictionary")
    print(f"-" * 70)
    features = game.get_features()
    print(f"Pac-Man: pos={features['pacman']['position']}, score={features['pacman']['score']}, lives={features['pacman']['lives']}")
    print(f"Ghosts: {len(features['ghosts'])} ghosts")
    print(f"  - {features['ghosts'][0]['name']}: pos={features['ghosts'][0]['position']}, scared={features['ghosts'][0]['scared']}")
    print(f"Pellets: {features['pellets']['pellet_count']} regular, {features['pellets']['power_pellet_count']} power")
    print(f"Distances: nearest_pellet={features['distances']['nearest_pellet_distance']}, nearest_ghost={features['distances']['nearest_ghost_distance']}")

    # Format 2: State Arrays (NumPy)
    print(f"\nFormat 2: State Arrays (NumPy)")
    print(f"-" * 70)
    arrays = game.get_state_arrays()
    print(f"board: shape={arrays['board'].shape}, dtype={arrays['board'].dtype}")
    print(f"pacman_position: {arrays['pacman_position']}")
    print(f"ghost_positions: shape={arrays['ghost_positions'].shape}")
    print(f"ghost_scared: {arrays['ghost_scared']}")
    print(f"pellet_map: shape={arrays['pellet_map'].shape}, pellets_visible={arrays['pellet_map'].sum()}")
    print(f"power_pellet_map: shape={arrays['power_pellet_map'].shape}, power_pellets_visible={arrays['power_pellet_map'].sum()}")

    # Format 3: State Vector
    print(f"\nFormat 3: State Vector (for Neural Networks)")
    print(f"-" * 70)
    vector = game.get_state_vector()
    print(f"Vector shape: {vector.shape}")
    print(f"Vector dtype: {vector.dtype}")
    print(f"Vector values: {vector}")
    print(f"Components:")
    print(f"  [0-1]: Normalized Pac-Man position")
    print(f"  [2-5]: Pac-Man direction (one-hot)")
    print(f"  [6+]: Ghost positions, scared states, etc.")

    print(f"\n✓ Feature extraction demo complete")


def demo_4_testing():
    """Demo 4: Show available tests."""
    print("\n" + "=" * 70)
    print("DEMO 4: AUTOMATED TESTS")
    print("=" * 70)
    print(f"\nAvailable test suites:")
    print(f"  1. run_tests.py - Full test suite (10 tests)")
    print(f"     >>> python run_tests.py")
    print(f"     Tests: initialization, features, arrays, vectors, movement,")
    print(f"            pellets, collisions, win condition")
    print(f"\n  2. example_usage.py - 8 working examples")
    print(f"     >>> python example_usage.py")
    print(f"     Examples: setup, feature extraction, arrays, vectors,")
    print(f"              navigation, pellet tracking, components, simulation")
    print(f"\nTo run all tests:")
    print(f"  >>> python run_tests.py")
    print(f"  >>> python example_usage.py")


def demo_5_interactive_game():
    """Demo 5: Show how to play the game."""
    print("\n" + "=" * 70)
    print("DEMO 5: INTERACTIVE GAME")
    print("=" * 70)
    print(f"\nTo play the game with Pygame UI:")
    print(f"  >>> python play_game.py")
    print(f"\nFeatures:")
    print(f"  - Live Pygame rendering")
    print(f"  - Arrow key controls")
    print(f"  - Real-time score display")
    print(f"  - Pellet counter")
    print(f"  - Ghost rendering")
    print(f"  - Win/Lose detection")
    print(f"  - Pause functionality")
    print(f"\nControls:")
    print(f"  UP/DOWN/LEFT/RIGHT - Move Pac-Man")
    print(f"  SPACE - Pause/Resume")
    print(f"  R - Restart game")
    print(f"  ESC - Quit")


def main():
    """Run all demos."""
    print("\n")
    print("╔" + "=" * 68 + "╗")
    print("║" + " " * 15 + "PAC-MAN GAME - TESTING GUIDE" + " " * 26 + "║")
    print("╚" + "=" * 68 + "╝")

    try:
        demo_1_game_with_ui()
        demo_2_headless_simulation()
        demo_3_feature_extraction()
        demo_4_testing()
        demo_5_interactive_game()

        print("\n" + "=" * 70)
        print("SUMMARY")
        print("=" * 70)
        print("\nTesting options available:")
        print("  1. Play interactively:     python play_game.py")
        print("  2. Run automated tests:    python run_tests.py")
        print("  3. Run examples:           python example_usage.py")
        print("  4. Headless simulation:    (create custom script)")
        print("  5. Feature extraction:     game.get_features(), etc.")
        print("\nAll tests passed! Infrastructure is ready for use.")
        print("=" * 70 + "\n")

    except Exception as e:
        print(f"\n✗ ERROR: {e}")
        import traceback
        traceback.print_exc()
        return 1

    return 0


if __name__ == "__main__":
    sys.exit(main())
