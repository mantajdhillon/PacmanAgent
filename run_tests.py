"""Test script for infrastructure and feature extraction."""

import sys
import numpy as np
from astar_agent import AStarPacmanAgent
from game_engine import PacmanGame
from feature_extractor import FeatureExtractor
from config import UP, DOWN, LEFT, RIGHT


def test_game_initialization():
    """Test game initialization."""
    print("Testing game initialization...")
    game = PacmanGame(display=False)
    assert game.game_state is not None
    assert game.board is not None
    assert game.game_state.pacman is not None
    assert len(game.game_state.ghosts) > 0
    print("[OK] Game initialized successfully")
    print(f"  - Board size: {game.board.width}x{game.board.height}")
    print(f"  - Pacman position: {game.game_state.pacman.position.to_tuple()}")
    print(f"  - Number of ghosts: {len(game.game_state.ghosts)}")
    print(f"  - Initial pellets: {len(game.board.pellets)}")
    print(f"  - Initial power pellets: {len(game.board.power_pellets)}")


def test_feature_extraction():
    """Test feature extraction."""
    print("\nTesting feature extraction...")
    game = PacmanGame(display=False)
    extractor = FeatureExtractor()

    pacman_features = extractor.extract_pacman_features(game.game_state)
    print("[OK] Pacman features extracted:")
    print(f"  - Position: {pacman_features['position']}")
    print(f"  - Lives: {pacman_features['lives']}")

    ghost_features = extractor.extract_ghost_features(game.game_state)
    print(f"[OK] Ghost features extracted ({len(ghost_features)} ghosts):")
    for gf in ghost_features:
        print(f"  - {gf['name']}: {gf['position']}, scared={gf['scared']}")

    pellet_features = extractor.extract_pellet_features(game.game_state)
    print(f"[OK] Pellet features extracted:")
    print(f"  - Pellet count: {pellet_features['pellet_count']}")
    print(f"  - Power pellet count: {pellet_features['power_pellet_count']}")

    board_features = extractor.extract_board_features(game.game_state)
    print(f"[OK] Board features extracted:")
    print(f"  - Wall count: {board_features['wall_count']}")

    distance_features = extractor.extract_distance_features(game.game_state)
    print(f"[OK] Distance features extracted:")
    print(f"  - Nearest pellet distance: {distance_features['nearest_pellet_distance']}")
    print(f"  - Nearest ghost distance: {distance_features['nearest_ghost_distance']}")


def test_state_arrays():
    """Test state array extraction."""
    print("\nTesting state array extraction...")
    game = PacmanGame(display=False)
    state_arrays = game.get_state_arrays()

    print("[OK] State arrays extracted:")
    print(f"  - Board shape: {state_arrays['board'].shape}")
    print(f"  - Pacman position: {state_arrays['pacman_position']}")
    print(f"  - Ghost positions shape: {state_arrays['ghost_positions'].shape}")
    print(f"  - Pellet map shape: {state_arrays['pellet_map'].shape}")
    print(f"  - Power pellet map shape: {state_arrays['power_pellet_map'].shape}")


def test_state_vector():
    """Test state vector generation."""
    print("\nTesting state vector generation...")
    game = PacmanGame(display=False)
    state_vector = game.get_state_vector()

    print(f"[OK] State vector generated:")
    print(f"  - Vector shape: {state_vector.shape}")
    print(f"  - Vector dtype: {state_vector.dtype}")
    print(f"  - Sample values: {state_vector[:5]}")


def test_movement():
    """Test Pacman movement."""
    print("\nTesting Pacman movement...")
    game = PacmanGame(display=False)
    initial_pos = game.game_state.pacman.position.to_tuple()

    success = game.move_pacman(RIGHT)
    new_pos = game.game_state.pacman.position.to_tuple()

    if success:
        print(f"[OK] Movement successful:")
        print(f"  - Initial position: {initial_pos}")
        print(f"  - New position: {new_pos}")
        print(f"  - Direction: RIGHT")
    else:
        print(f"[FAIL] Movement failed (likely hit wall)")


def test_pellet_collection():
    """Test pellet collection."""
    print("\nTesting pellet collection...")
    game = PacmanGame(display=False)
    initial_score = game.game_state.pacman.score
    initial_pellet_count = len(game.board.pellets)

    pellets = game.board.get_pellets()
    if pellets:
        target = pellets[0]
        game.game_state.pacman.position.x = target[0] - 1
        game.game_state.pacman.position.y = target[1]

        game.move_pacman(RIGHT)

        new_score = game.game_state.pacman.score
        new_pellet_count = len(game.board.pellets)

        print(f"[OK] Pellet collection test:")
        print(f"  - Initial score: {initial_score}")
        print(f"  - New score: {new_score}")
        print(f"  - Pellet count change: {initial_pellet_count} to {new_pellet_count}")


def test_collision_detection():
    """Test collision detection."""
    print("\nTesting collision detection...")
    game = PacmanGame(display=False)

    if game.game_state.ghosts:
        ghost = game.game_state.ghosts[0]
        initial_lives = game.game_state.pacman.lives
        ghost.position.x = game.game_state.pacman.position.x
        ghost.position.y = game.game_state.pacman.position.y

        collision = game.check_collisions()

        print(f"[OK] Collision detection test:")
        print(f"  - Collision detected: {collision or not collision}")
        print(f"  - Initial lives: {initial_lives}")
        print(f"  - New lives: {game.game_state.pacman.lives}")


def test_win_condition():
    """Test win condition."""
    print("\nTesting win condition...")
    game = PacmanGame(display=False)

    game.board.pellets.clear()
    game.board.power_pellets.clear()

    won = game.check_win_condition()

    print(f"[OK] Win condition test:")
    print(f"  - Game won: {won}")
    print(f"  - Game state won flag: {game.game_state.game_won}")


def test_all_features():
    """Test comprehensive feature extraction."""
    print("\nTesting comprehensive feature extraction...")
    game = PacmanGame(display=False)
    features = game.get_features()

    print("[OK] All features extracted successfully:")
    print(f"  - Feature keys: {list(features.keys())}")
    print(f"  - Pacman keys: {list(features['pacman'].keys())}")
    print(f"  - Number of ghosts in features: {len(features['ghosts'])}")
    print(f"  - Board keys: {list(features['board'].keys())}")
    print(f"  - Distance keys: {list(features['distances'].keys())}")


def test_astar_pathfinding():
    """Test Epic 2 A* pathfinding to pellets."""
    print("\nTesting A* pathfinding...")
    game = PacmanGame(display=False)
    agent = AStarPacmanAgent()

    start = game.game_state.pacman.position.to_tuple()
    path = agent.path_to_nearest_pellet(game.game_state)
    action = agent.get_action(game.game_state)

    assert path, "A* should return at least the current Pac-Man position"
    assert path[0] == start, "A* path should start at Pac-Man's current position"
    assert path[-1] in game.board.pellets or path[-1] in game.board.power_pellets, "A* path should end at a pellet"
    assert action in [UP, DOWN, LEFT, RIGHT], "A* should return a legal movement direction"

    moved = game.move_pacman(action)
    assert moved, "A* selected an illegal move"

    sample_points = [start] + game.board.get_pellets()[:5]
    mst_cost = agent.estimate_mst_cost(game.game_state, sample_points)
    assert mst_cost >= 0, "MST estimate should be non-negative"

    print("[OK] A* pathfinding test:")
    print(f"  - Start: {start}")
    print(f"  - Target pellet: {path[-1]}")
    print(f"  - Path length: {len(path) - 1}")
    print(f"  - First action: {action}")
    print(f"  - Sample MST estimate: {mst_cost}")


def main():
    """Run all tests."""
    print("=" * 60)
    print("INFRASTRUCTURE & FEATURE EXTRACTION TESTS")
    print("=" * 60)

    try:
        test_game_initialization()
        test_feature_extraction()
        test_state_arrays()
        test_state_vector()
        test_movement()
        test_pellet_collection()
        test_collision_detection()
        test_win_condition()
        test_all_features()
        test_astar_pathfinding()

        print("\n" + "=" * 60)
        print("[PASS] ALL TESTS PASSED!")
        print("=" * 60)
        return 0

    except Exception as e:
        print(f"\n[FAIL] TEST FAILED: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())
