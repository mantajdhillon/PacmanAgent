"""Headless demo for Pac-Man defense and normal-ghost Minimax attack."""

import time
import random
from minimax_agent import MinimaxPacmanAgent
from minimax_ghost_agent import MinimaxGhostAgent
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

    initial_lives = game.game_state.pacman.lives
    steps = 0

    print("Initializing headless Minimax simulation...")
    start_time = time.time()

    while steps < max_steps and not game.is_game_over() and not game.is_game_won():
        # AI Decision Phase
        action = agent.get_action(game.game_state)

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
                action = random.choice([UP, DOWN, LEFT, RIGHT])
            else:
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
