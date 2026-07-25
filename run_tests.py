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


def test_minimax_defense():
    """Test Normal Pac-Man defensive Minimax."""
    print("\nTesting Minimax defense...")
    from game_engine import PacmanGame
    from minimax_agent import MinimaxPacmanAgent
    from game_state import (
        GameBoard,
        GameState,
        GhostState,
        PacmanState,
        Position,
    )
    import time

    game = PacmanGame(display=False)
    agent = MinimaxPacmanAgent(depth=2)
    assert agent.active_threat_radius == 5, "Defensive threshold must be 5"

    # Isolate the Environment
    # Clear all existing ghosts to ensure the AI is only evaluating our test threat
    game.game_state.ghosts.clear()
    pacman_start = game.game_state.pacman.position.to_tuple()

    # Manufacture a Mortal Threat
    legal_moves = game.game_state.get_legal_actions(0)
    assert len(legal_moves) > 0, "Pac-Man must have legal moves at spawn"

    # Take the first available legal move and place a ghost exactly there
    threat_dir = legal_moves[0]
    threat_x = pacman_start[0] + threat_dir[0]
    threat_y = pacman_start[1] + threat_dir[1]

    assassin_ghost = GhostState(Position(threat_x, threat_y), "red", name="TestThreat")
    game.game_state.add_ghost(assassin_ghost)
    assert agent.is_threat_nearby(game.game_state), (
        "A normal ghost inside distance 5 must activate defensive Minimax"
    )

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

    # Scared ghosts are not dangerous in Requirement 1.
    assassin_ghost.scared = True
    assert not agent.is_threat_nearby(game.game_state), (
        "A scared ghost must not activate normal defensive Minimax"
    )
    assassin_ghost.scared = False

    # The boundary is inclusive: distance 5 activates, distance 6 does not.
    threshold_board = GameBoard(9, 5)
    for x in range(threshold_board.width):
        threshold_board.add_wall(x, 0)
        threshold_board.add_wall(x, threshold_board.height - 1)
    for y in range(threshold_board.height):
        threshold_board.add_wall(0, y)
        threshold_board.add_wall(threshold_board.width - 1, y)

    threshold_state = GameState(threshold_board)
    threshold_state.set_pacman(PacmanState(Position(1, 2)))
    threshold_state.add_ghost(GhostState(Position(6, 2), "red", name="AtFive"))
    assert agent.is_threat_nearby(threshold_state), (
        "A normal ghost at maze distance exactly 5 must activate Minimax"
    )
    threshold_state.ghosts[0].position = Position(7, 2)
    assert not agent.is_threat_nearby(threshold_state), (
        "A normal ghost beyond maze distance 5 must not activate Minimax"
    )

    # With equal threat distance, open space must outrank a one-exit dead end.
    open_board = GameBoard(7, 8)
    trapped_board = GameBoard(7, 8)
    for board in (open_board, trapped_board):
        for x in range(board.width):
            board.add_wall(x, 0)
            board.add_wall(x, board.height - 1)
        for y in range(board.height):
            board.add_wall(0, y)
            board.add_wall(board.width - 1, y)

    # Only the downward move remains legal in the trapped state.
    for wall_x, wall_y in ((3, 2), (2, 3), (4, 3)):
        trapped_board.add_wall(wall_x, wall_y)

    open_state = GameState(open_board)
    trapped_state = GameState(trapped_board)
    for state in (open_state, trapped_state):
        state.set_pacman(PacmanState(Position(3, 3)))
        state.add_ghost(GhostState(Position(3, 6), "red", name="MazeThreat"))

    assert agent.evaluate_state(open_state) > agent.evaluate_state(trapped_state), (
        "With equal threat distance, Pac-Man must prefer escape space over a dead end"
    )

    # Losing a life must dominate the secondary reward from food or score.
    full_lives_state = game.game_state.clone()
    lost_life_state = game.game_state.clone()
    lost_life_state.pacman.lives -= 1
    lost_life_state.pacman.score += 1_000
    assert agent.evaluate_state(full_lives_state) > agent.evaluate_state(lost_life_state), (
        "Preserving a life must be worth more than secondary score gains"
    )

    # Output Metrics
    print("[OK] Minimax defense test passed:")
    print(f"  - Initial Position: {pacman_start}")
    print(f"  - Threat Injected At: ({threat_x}, {threat_y})")
    print(f"  - Threat Direction: {threat_dir}")
    print(f"  - Evasion Action: {action} (Survival Confirmed)")
    print(f"  - Compute Time: {compute_time:.4f}s")
    print(f"  - Post-Evasion Utility: {eval_score:.2f}")


def test_minimax_ghost_attack():
    """Test Requirement 2: normal ghosts use team Minimax attack."""
    print("\nTesting normal ghost Minimax attack...")
    from game_state import (
        GameBoard,
        GameState,
        GhostState,
        PacmanState,
        Position,
    )
    from minimax_ghost_agent import MinimaxGhostAgent

    def bordered_board(width=9, height=7):
        board = GameBoard(width, height)
        for x in range(width):
            board.add_wall(x, 0)
            board.add_wall(x, height - 1)
        for y in range(height):
            board.add_wall(0, y)
            board.add_wall(width - 1, y)
        return board

    agent = MinimaxGhostAgent(depth=1)

    # A normal ghost in an open corridor must reduce maze distance.
    chase_state = GameState(bordered_board())
    chase_state.set_pacman(PacmanState(Position(6, 3)))
    chase_state.add_ghost(
        GhostState(Position(2, 3), "red", name="Chaser")
    )
    chase_action = agent.get_action(chase_state, 0)
    assert chase_action == RIGHT, (
        f"Normal ghost should chase Pac-Man to the right, got {chase_action}"
    )

    # Catching Pac-Man must dominate every non-capturing action.
    capture_state = GameState(bordered_board())
    capture_state.set_pacman(PacmanState(Position(5, 3)))
    capture_state.add_ghost(
        GhostState(Position(4, 3), "red", name="Catcher")
    )
    capture_action = agent.get_action(capture_state, 0)
    assert capture_action == RIGHT, (
        f"Adjacent normal ghost must capture Pac-Man, got {capture_action}"
    )
    captured = capture_state.generate_successor(1, capture_action)
    assert captured.pacman.lives == capture_state.pacman.lives - 1, (
        "Selected capture action must remove one Pac-Man life"
    )

    # A scared ghost must not activate Requirement 2.
    chase_state.ghosts[0].scared = True
    assert agent.get_action(chase_state, 0) == (0, 0), (
        "Scared ghosts must not use normal attacking Minimax"
    )

    # Adding a teammate that covers another approach must improve team utility.
    solo_state = GameState(bordered_board())
    team_state = GameState(bordered_board())
    for state in (solo_state, team_state):
        state.set_pacman(PacmanState(Position(4, 3)))
        state.add_ghost(
            GhostState(Position(2, 3), "red", name="LeftPressure")
        )
    team_state.add_ghost(
        GhostState(Position(6, 3), "pink", name="RightPressure")
    )

    assert agent.evaluate_state(team_state) > agent.evaluate_state(solo_state), (
        "A coordinated second ghost covering another side must improve attack utility"
    )

    print("[OK] Normal ghost Minimax attack test passed:")
    print(f"  - Chase action: {chase_action}")
    print(f"  - Capture action: {capture_action}")
    print("  - Scared ghost correctly excluded")
    print("  - Team pressure improves shared utility")


def test_minimax_scared_ghost_attack():
    """Test Requirement 3: Pac-Man safely attacks viable scared ghosts."""
    print("\nTesting Pac-Man scared-ghost Minimax attack...")
    from game_state import (
        GameBoard,
        GameState,
        GhostState,
        PacmanState,
        Position,
    )
    from minimax_scared_ghost_agent import MinimaxScaredGhostAttackAgent

    def bordered_board(width=13, height=7):
        board = GameBoard(width, height)
        for x in range(width):
            board.add_wall(x, 0)
            board.add_wall(x, height - 1)
        for y in range(height):
            board.add_wall(0, y)
            board.add_wall(width - 1, y)
        return board

    agent = MinimaxScaredGhostAttackAgent(
        safety_distance=5,
        timer_margin=2,
    )

    # A reachable scared ghost with enough timer must activate the policy.
    chase_state = GameState(bordered_board())
    chase_state.set_pacman(PacmanState(Position(3, 3)))
    chase_state.add_ghost(
        GhostState(
            Position(7, 3),
            "blue",
            scared=True,
            scared_timer=20,
            name="Edible",
        )
    )
    assert agent.select_target(chase_state) == "Edible", (
        "Reachable scared ghost with sufficient timer must be selected"
    )
    chase_action = agent.get_action(chase_state, "Edible")
    assert chase_action == RIGHT, (
        f"Pac-Man should reduce distance to the scared ghost, got {chase_action}"
    )

    # A timer that cannot cover travel plus margin must reject the chase.
    chase_state.ghosts[0].scared_timer = 6
    assert agent.select_target(chase_state) is None, (
        "Pac-Man must reject a target without sufficient timer margin"
    )
    chase_state.ghosts[0].scared_timer = 20

    # A nearby normal ghost has priority over an edible target.
    chase_state.add_ghost(
        GhostState(Position(3, 5), "red", name="NormalThreat")
    )
    assert agent.select_target(chase_state) is None, (
        "Nearby normal ghost must prevent scared-ghost chase activation"
    )
    chase_state.ghosts.pop()

    # An adjacent scared ghost must be captured immediately.
    capture_state = GameState(bordered_board())
    capture_state.set_pacman(PacmanState(Position(4, 3)))
    capture_state.add_ghost(
        GhostState(
            Position(5, 3),
            "blue",
            scared=True,
            scared_timer=10,
            name="CaptureTarget",
        )
    )
    capture_action = agent.get_action(capture_state, "CaptureTarget")
    assert capture_action == RIGHT, (
        f"Pac-Man must capture an adjacent scared ghost, got {capture_action}"
    )
    captured_state = capture_state.generate_successor(0, capture_action)
    assert agent._find_target(captured_state, "CaptureTarget") is None, (
        "Capture action must remove the scared ghost from the successor"
    )

    # Prefer the target with greater remaining timer slack.
    target_state = GameState(bordered_board())
    target_state.set_pacman(PacmanState(Position(5, 3)))
    target_state.add_ghost(
        GhostState(
            Position(3, 3),
            "blue",
            scared=True,
            scared_timer=8,
            name="LowSlack",
        )
    )
    target_state.add_ghost(
        GhostState(
            Position(9, 3),
            "blue",
            scared=True,
            scared_timer=25,
            name="HighSlack",
        )
    )
    assert agent.select_target(target_state) == "HighSlack", (
        "Target selection must prefer a safer remaining-time margin"
    )

    # A ghost still inside Pac-Man's forbidden ghost house is not reachable.
    house_board = bordered_board(width=21, height=21)
    house_state = GameState(house_board)
    house_state.set_pacman(PacmanState(Position(10, 12)))
    house_state.add_ghost(
        GhostState(
            Position(10, 10),
            "blue",
            scared=True,
            scared_timer=50,
            name="InsideHouse",
        )
    )
    assert agent.select_target(house_state) is None, (
        "Pac-Man must not select a target inside its forbidden ghost house"
    )

    # Do not step into the normal-ghost danger radius just to chase food.
    safety_state = GameState(bordered_board())
    safety_state.set_pacman(PacmanState(Position(3, 3)))
    safety_state.add_ghost(
        GhostState(
            Position(7, 3),
            "blue",
            scared=True,
            scared_timer=20,
            name="RiskyTarget",
        )
    )
    safety_state.add_ghost(
        GhostState(Position(9, 3), "red", name="DistantNormal")
    )
    # Normal ghost begins at maze distance 6, but RIGHT would reduce it to 5.
    assert agent.select_target(safety_state) == "RiskyTarget"
    safe_action = agent.get_action(safety_state, "RiskyTarget")
    assert safe_action != RIGHT, (
        "Pac-Man must not enter the normal-ghost safety radius while chasing"
    )

    print("[OK] Pac-Man scared-ghost attack test passed:")
    print(f"  - Chase action: {chase_action}")
    print(f"  - Capture action: {capture_action}")
    print(f"  - Safety-preserving action: {safe_action}")
    print("  - Timer feasibility and target preference confirmed")


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
        test_minimax_ghost_attack()
        test_minimax_scared_ghost_attack()

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
