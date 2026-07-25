"""Headless demo for Pac-Man defense and normal-ghost Minimax attack."""

import time
from astar_agent import AStarPacmanAgent
from minimax_agent import MinimaxPacmanAgent
from minimax_ghost import MinimaxGhostAgent
from minimax_attack_agent import MinimaxScaredGhostAttackAgent
from minimax_defense_ghost import MinimaxScaredGhostDefenseAgent
from game_engine import PacmanGame
from config import UP, DOWN, LEFT, RIGHT

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
        if agent.is_threat_nearby(game.game_state):
            mode = "DEFENSE"
            action = agent.get_action(game.game_state)
        elif (
            target_name
            := scared_attack_agent.select_target(game.game_state)
        ) is not None:
            mode = "SCARED ATTACK"
            action = scared_attack_agent.get_action(
                game.game_state,
                target_name,
            )
        else:
            mode = "A*"
            action = astar_agent.get_action(game.game_state)
        mode_counts[mode] += 1

        # Execution Phase
        if action != (0, 0):
            moved = game.move_pacman(action)
            if not moved:
                # If Minimax chooses a wall, the algorithm is mathematically broken.
                raise RuntimeError(f"FATAL: Minimax selected an illegal move into a wall: {action}")

        # Adversary Phase: normal ghosts use team-oriented Minimax attack.
        for i in range(len(game.game_state.ghosts)):
            ghost = game.game_state.ghosts[i]
            if ghost.scared:
                if scared_defense_agent.is_minimax_active(
                    game.game_state,
                    i,
                ):
                    ghost_mode_counts["SCARED DEFENSE"] += 1
                else:
                    ghost_mode_counts["SCARED FLEE"] += 1
                action = scared_defense_agent.get_action(
                    game.game_state,
                    i,
                )
            else:
                ghost_mode_counts["NORMAL ATTACK"] += 1
                action = ghost_agent.get_action(game.game_state, i)
            if action != (0, 0):
                game.move_ghost(i, action)

        # State Resolution Phase
        game.check_collisions()
        game.update_scared_timers()
        game.update_respawn_timers()
        game.check_win_condition()

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
