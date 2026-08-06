"""Headless demo for Pac-Man defense and normal-ghost Minimax attack."""

import time
from src.agents.astar_agent import AStarPacmanAgent
from src.agents.minimax_agent import MinimaxPacmanAgent
from src.agents.minimax_ghost import MinimaxGhostAgent
from src.agents.minimax_attack_agent import MinimaxScaredGhostAttackAgent
from src.agents.minimax_defense_ghost import MinimaxScaredGhostDefenseAgent
from src.core.game_engine import PacmanGame
from src.core.config import UP, DOWN, LEFT, RIGHT

def run_demo(max_steps: int = 200):
    """
    Run defensive Pac-Man and attacking normal ghosts with Minimax.
    """
    game = PacmanGame(display=False)
    game.reset()

    # Instantiate the optimized agent
    agent = MinimaxPacmanAgent(depth=2)
    ghost_agent = MinimaxGhostAgent(depth=1)
    astar_agent = AStarPacmanAgent()
    scared_attack_agent = MinimaxScaredGhostAttackAgent(
        safety_distance=5,
        timer_margin=2,
    )
    scared_defense_agent = MinimaxScaredGhostDefenseAgent(
        activation_distance=5,
        escape_horizon=4,
    )
    mode_counts = {"A*": 0, "DEFENSE": 0, "SCARED ATTACK": 0}
    ghost_mode_counts = {
        "NORMAL ATTACK": 0,
        "SCARED DEFENSE": 0,
        "SCARED FLEE": 0,
    }

    initial_lives = game.game_state.pacman.lives
    steps = 0

    print("Initializing headless Minimax simulation...")
    start_time = time.time()

    while steps < max_steps and not game.is_game_over() and not game.is_game_won():
        # AI Decision Phase
        action = None
        target_name = scared_attack_agent.select_target(game.game_state)
        if target_name is not None:
            scared_action = scared_attack_agent.get_action(
                game.game_state,
                target_name,
            )
            if agent.is_action_safe(
                game.game_state,
                scared_action,
                minimum_distance=2,
            ):
                mode = "SCARED ATTACK"
                action = scared_action

        if action is None:
            point_action = astar_agent.get_action(game.game_state)
            threat_nearby = agent.is_threat_nearby(game.game_state)
            if (
                not threat_nearby
                and agent.is_action_safe(
                    game.game_state,
                    point_action,
                    minimum_distance=2,
                )
            ):
                mode = "A*"
                action = point_action
            else:
                mode = "DEFENSE"
                defense_action = agent.get_action(game.game_state)
                if agent.is_action_safe(
                    game.game_state,
                    defense_action,
                    minimum_distance=0,
                ):
                    action = defense_action
                else:
                    action = agent.get_safest_action(game.game_state)
        mode_counts[mode] += 1

        # Execution Phase
        if action != (0, 0):
            moved = game.move_pacman(action)
            if not moved:
                # If Minimax chooses a wall, the algorithm is mathematically broken.
                raise RuntimeError(f"FATAL: Minimax selected an illegal move into a wall: {action}")

        # Adversary Phase: use stable names because captures change list indices.
        ghost_names = [ghost.name for ghost in game.game_state.ghosts]
        ghost_agent.begin_turn(game.game_state)
        for ghost_name in ghost_names:
            if game.skip_ghost_phase:
                break
            ghost_index = next(
                (
                    index
                    for index, ghost in enumerate(game.game_state.ghosts)
                    if ghost.name == ghost_name
                ),
                None,
            )
            if ghost_index is None:
                continue

            ghost = game.game_state.ghosts[ghost_index]
            if ghost.scared:
                if scared_defense_agent.is_minimax_active(
                    game.game_state,
                    ghost_index,
                ):
                    ghost_mode_counts["SCARED DEFENSE"] += 1
                else:
                    ghost_mode_counts["SCARED FLEE"] += 1
                action = scared_defense_agent.get_action(
                    game.game_state,
                    ghost_index,
                )
            else:
                ghost_mode_counts["NORMAL ATTACK"] += 1
                action = ghost_agent.get_action(
                    game.game_state,
                    ghost_index,
                )
            if action != (0, 0):
                game.move_ghost(ghost_index, action)

        # State Resolution Phase
        game.check_collisions()
        game.update_scared_timers()
        game.update_respawn_timers()
        game.check_win_condition()
        game.validate_ghost_roster()
        game.skip_ghost_phase = False

        steps += 1

    # Calculate computational performance
    execution_time = time.time() - start_time
    frames_per_second = steps / execution_time if execution_time > 0 else 0

    # Output Summary
    print("\n" + "=" * 60)
    print("EPIC 3 MINIMAX DEMO")
    print("=" * 60)
    print(f"Target Depth limit    : {agent.depth}")
    print(f"Threat Pruning Radius : {agent.active_threat_radius}")
    print(f"Ghost Attack Depth    : {ghost_agent.depth}")
    print(f"Pac-Man Modes         : {mode_counts}")
    print(f"Ghost Modes           : {ghost_mode_counts}")
    print("-" * 60)
    print(f"Survival Frames       : {steps} / {max_steps}")
    print(f"Lives Remaining       : {game.game_state.pacman.lives} / {initial_lives}")
    print(f"Game Over Triggered   : {game.is_game_over()}")
    print("-" * 60)
    print(f"Total Compute Time    : {execution_time:.3f} seconds")
    print(f"Simulation Speed      : {frames_per_second:.1f} FPS")
    print("=" * 60)


if __name__ == "__main__":
    run_demo()
