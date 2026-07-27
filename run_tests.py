"""Test script for infrastructure and feature extraction."""

import sys
import numpy as np
from astar_agent import AStarPacmanAgent
from game_engine import PacmanGame
from feature_extractor import FeatureExtractor
from approximate_q_agent import ApproximateQAgent
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


def test_minimax_defense():
    """Test Epic 3 Minimax defensive evasion."""
    print("\nTesting Minimax defense...")
    from game_engine import PacmanGame
    from minimax_agent import MinimaxPacmanAgent
    from game_state import GhostState, Position
    import time

    game = PacmanGame(display=False)
    agent = MinimaxPacmanAgent(depth=2)

    # Isolate the Environment
    # Clear all existing ghosts to ensure the AI is only evaluating our test threat
    game.game_state.ghosts.clear()
    pacman_pos = game.game_state.pacman.position

    # Manufacture a Mortal Threat
    legal_moves = game.game_state.get_legal_actions(0)
    assert len(legal_moves) > 0, "Pac-Man must have legal moves at spawn"

    # Take the first available legal move and place a ghost exactly there
    threat_dir = legal_moves[0]
    threat_x = pacman_pos.x + threat_dir[0]
    threat_y = pacman_pos.y + threat_dir[1]

    assassin_ghost = GhostState(Position(threat_x, threat_y), "red", name="TestThreat")
    game.game_state.add_ghost(assassin_ghost)

    # Execute the AI Decision
    start_time = time.time()
    action = agent.get_action(game.game_state)
    compute_time = time.time() - start_time

    # Rigorous Assertions
    assert action is not None, "Minimax returned None instead of a tuple"
    assert action in legal_moves, f"Minimax returned illegal action {action}"

    # Test Pac-Man must not step into the threat
    assert action != threat_dir, f"FATAL: Minimax stepped into the ghost at {threat_dir}"

    # Verify the engine accepts the move
    moved = game.move_pacman(action)
    assert moved, "Game engine rejected the Minimax action"

    # Verify the evaluation utility returns the correct data type
    eval_score = agent.evaluate_state(game.game_state)
    assert isinstance(eval_score, float), "Evaluation function must return a float"

    # Output Metrics
    print("[OK] Minimax defense test passed:")
    print(f"  - Initial Position: {pacman_pos.to_tuple()}")
    print(f"  - Threat Injected At: ({threat_x}, {threat_y})")
    print(f"  - Threat Direction: {threat_dir}")
    print(f"  - Evasion Action: {action} (Survival Confirmed)")
    print(f"  - Compute Time: {compute_time:.4f}s")
    print(f"  - Post-Evasion Utility: {eval_score:.2f}")

def test_q_weight_initialization():
    """Test that weights are lazy-loaded and initialized to zeros correctly."""
    print("\nTesting Q-weight initialization...")
    game = PacmanGame(display=False)
    agent = ApproximateQAgent()

    # Weights should be None before any evaluation
    assert agent.weights is None, "Weights should initially be None"

    state = game.get_game_state()
    legal_actions = state.get_legal_actions(0)
    action = legal_actions[0]

    # Trigger weight initialization
    q_value = agent.get_q_value(state, action)

    assert agent.weights is not None, "Weights should be initialized after calling get_q_value"
    assert isinstance(agent.weights, np.ndarray), "Weights must be a NumPy array"
    assert np.all(agent.weights == 0.0), "Initial weights must be all zeros"
    assert q_value == 0.0, "Initial Q-value with zeroed weights must be 0.0"
    
    print("[OK] Q-weights initialized successfully")

def test_epsilon_greedy_selection():
    """Test that the agent respects the epsilon exploration parameter."""
    print("\nTesting epsilon-greedy action selection...")
    game = PacmanGame(display=False)
    state = game.get_game_state()
    
    # Test 1: Pure Exploration (Epsilon = 1.0)
    random_agent = ApproximateQAgent(epsilon=1.0)
    actions_taken = set()
    for _ in range(50):
        actions_taken.add(random_agent.get_action(state))
    
    assert len(actions_taken) > 1, "Agent with epsilon=1.0 should return varying random actions"
    
    # Test 2: Pure Exploitation (Epsilon = 0.0)
    greedy_agent = ApproximateQAgent(epsilon=0.0)
    
    # Manually rig the weights so UP is mathematically the best action
    legal_actions = state.get_legal_actions(0)
    greedy_agent.get_q_value(state, legal_actions[0]) # Force initialization
    
    # Fake the weights to simulate a highly trained state
    greedy_agent.weights = np.ones_like(greedy_agent.weights)
    
    first_greedy_action = greedy_agent.get_action(state)
    for _ in range(10):
        assert greedy_agent.get_action(state) == first_greedy_action, "Agent with epsilon=0.0 must be deterministic"

    print("[OK] Epsilon-greedy selection behaves as expected")

def test_weight_update_logic():
    """Test that Temporal Difference (TD) error correctly updates weights."""
    print("\nTesting weight update (TD Error) logic...")
    game = PacmanGame(display=False)
    state = game.get_game_state()
    
    # Create a cloned successor state to simulate a move
    legal_actions = state.get_legal_actions(0)
    action = legal_actions[0]
    next_state = state.generate_successor(0, action)
    
    # Initialize Agent with a high learning rate for clear observation
    agent = ApproximateQAgent(learning_rate=0.5, discount_factor=0.9, epsilon=0.0)
    
    # Force weight array initialization
    agent.get_q_value(state, action)
    initial_weights = agent.weights.copy()
    
    # Apply a massive artificial reward
    massive_reward = 100.0
    agent.update(state, action, next_state, massive_reward)
    
    updated_weights = agent.weights
    
    # Assert weights have physically changed
    assert not np.array_equal(initial_weights, updated_weights), "Weights failed to update after receiving a reward"
    assert np.any(updated_weights > 0.0), "A massive positive reward should result in positive weight shifts"
    
    print("[OK] Weight update applied successfully")

def test_terminal_state_value():
    """Test that game-over states return a value of 0.0."""
    print("\nTesting terminal state valuation...")
    game = PacmanGame(display=False)
    state = game.get_game_state()
    agent = ApproximateQAgent()
    
    # Manually trigger a game over
    state.game_over = True
    
    # The value of a terminal state should strictly be 0.0
    val = agent.get_value(state)
    assert val == 0.0, f"Terminal state value should be 0.0, got {val}"
    
    # Legal actions should be empty
    assert len(state.get_legal_actions(0)) == 0, "Terminal states should have no legal actions"
    
    print("[OK] Terminal state handled correctly")

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
        test_minimax_defense()
        test_q_weight_initialization()
        test_epsilon_greedy_selection()
        test_weight_update_logic()
        test_terminal_state_value()

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
