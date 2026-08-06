"""Headless demo for Epic 2 A* pellet collection."""

from src.agents.astar_agent import AStarPacmanAgent
from src.core.game_engine import PacmanGame


def run_demo(max_steps: int = 2000):
    """Run Pac-Man with A* only and print a short performance summary."""
    game = PacmanGame(display=False)
    game.reset()
    # Note: We don't clear ghosts to maintain the game's invariant of 4 ghost identities
    agent = AStarPacmanAgent()

    initial_pellets = len(game.board.pellets) + len(game.board.power_pellets)
    steps = 0

    while steps < max_steps and not game.is_game_over() and not game.is_game_won():
        action = agent.get_action(game.game_state)
        if action == (0, 0):
            break

        moved = game.move_pacman(action)
        if not moved:
            raise RuntimeError(f"A* selected an illegal move: {action}")

        game.check_collisions()
        game.update_scared_timers()
        # Skip ghost movement and respawn updates for this demo
        game.check_win_condition()
        steps += 1

    remaining_pellets = len(game.board.pellets) + len(game.board.power_pellets)
    eaten = initial_pellets - remaining_pellets

    print("=" * 60)
    print("EPIC 2 A* DEMO")
    print("=" * 60)
    print(f"Steps: {steps}")
    print(f"Initial pellets: {initial_pellets}")
    print(f"Pellets eaten: {eaten}")
    print(f"Remaining pellets: {remaining_pellets}")
    print(f"Score: {game.game_state.pacman.score}")
    print(f"Game won: {game.is_game_won()}")
    print("=" * 60)


if __name__ == "__main__":
    run_demo()
