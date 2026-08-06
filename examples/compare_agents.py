import csv
import os
import random
from pathlib import Path
from typing import Dict, List, Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from src.agents.approximate_q_agent import ApproximateQAgent
from src.agents.astar_agent import AStarPacmanAgent
from src.agents.minimax_agent import MinimaxPacmanAgent
from src.agents.minimax_ghost import MinimaxGhostAgent
from src.agents.minimax_attack_agent import MinimaxScaredGhostAttackAgent
from src.agents.minimax_defense_ghost import MinimaxScaredGhostDefenseAgent
from src.core.game_engine import PacmanGame
from src.core.config import UP, DOWN, LEFT, RIGHT


class CombinedAgent:
    """Hybrid controller that combines A* targeting with Minimax safety."""

    def __init__(self):
        self.astar_agent = AStarPacmanAgent()
        self.minimax_agent = MinimaxPacmanAgent(depth=2)
        self.ghost_minimax_agent = MinimaxGhostAgent(depth=1)
        self.scared_ghost_attack_agent = MinimaxScaredGhostAttackAgent(
            safety_distance=5,
            timer_margin=2,
        )
        self.scared_ghost_defense_agent = MinimaxScaredGhostDefenseAgent(
            activation_distance=5,
            escape_horizon=4,
        )

    def _should_prioritize_food(self, game_state) -> bool:
        """Use food-first behavior when few pellets remain or Pac-Man is close to them."""
        remaining_food = len(game_state.board.pellets) + len(game_state.board.power_pellets)
        if remaining_food <= 3:
            return True

        if remaining_food <= 8:
            pacman_pos = game_state.pacman.position.to_tuple() if game_state.pacman else None
            if not pacman_pos:
                return False
            nearest_food = None
            targets = list(game_state.board.get_pellets()) + list(game_state.board.get_power_pellets())
            if targets:
                nearest_food = min(
                    targets,
                    key=lambda target: self.astar_agent.maze_distance(game_state, pacman_pos, target) or float("inf"),
                )
                distance = self.astar_agent.maze_distance(game_state, pacman_pos, nearest_food)
                return distance is not None and distance <= 4

        return False

    def get_action(self, game_state) -> Tuple[int, int]:
        target_name = self.scared_ghost_attack_agent.select_target(game_state)
        if target_name is not None:
            scared_action = self.scared_ghost_attack_agent.get_action(
                game_state,
                target_name,
            )
            if self.minimax_agent.is_action_safe(
                game_state,
                scared_action,
                minimum_distance=2,
            ):
                return scared_action

        point_action = self.astar_agent.get_action(game_state)
        threat_nearby = self.minimax_agent.is_threat_nearby(game_state)
        prioritize_food = self._should_prioritize_food(game_state)
        if (
            (not threat_nearby or prioritize_food)
            and self.minimax_agent.is_action_safe(
                game_state,
                point_action,
                minimum_distance=0 if prioritize_food else 2,
            )
        ):
            return point_action

        defense_action = self.minimax_agent.get_action(game_state)
        if self.minimax_agent.is_action_safe(
            game_state,
            defense_action,
            minimum_distance=0,
        ):
            return defense_action

        return self.minimax_agent.get_safest_action(game_state)


def move_ghosts(game: PacmanGame, ghost_minimax_agent: MinimaxGhostAgent, scared_defense_agent: MinimaxScaredGhostDefenseAgent):
    """Move ghosts using the same role-based logic as play_game.py."""
    ghost_names = [ghost.name for ghost in game.game_state.ghosts]
    ghost_minimax_agent.begin_turn(game.game_state)
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
            action = scared_defense_agent.get_action(game.game_state, ghost_index)
        else:
            action = ghost_minimax_agent.get_action(game.game_state, ghost_index)

        if action != (0, 0):
            game.move_ghost(ghost_index, action)


def train_q_agent(episodes: int = 40) -> ApproximateQAgent:
    """Train an Approximate Q-learning agent using the same reward shaping as train_q_agent.py."""
    agent = ApproximateQAgent(epsilon=0.2, learning_rate=0.01, discount_factor=0.9)
    for episode in range(1, episodes + 1):
        game = PacmanGame(display=False)
        game.reset()

        prev_action = None
        reverse_actions = {
            (0, -1): (0, 1),
            (0, 1): (0, -1),
            (-1, 0): (1, 0),
            (1, 0): (-1, 0),
        }
        steps = 0
        max_steps = 3000

        while not game.is_game_over() and not game.is_game_won() and steps < max_steps:
            steps += 1
            current_state = game.get_game_state().clone()
            action = agent.get_action(current_state)

            prev_pellets_eaten = current_state.pellets_eaten
            prev_score = current_state.pacman.score
            prev_lives = current_state.pacman.lives

            if action != (0, 0):
                game.move_pacman(action)
            move_ghosts(
                game,
                MinimaxGhostAgent(depth=1),
                MinimaxScaredGhostDefenseAgent(activation_distance=5, escape_horizon=4),
            )
            game.check_collisions()
            game.update_scared_timers()
            game.update_respawn_timers()
            game.check_win_condition()

            next_state = game.get_game_state().clone()
            reward = next_state.pacman.score - prev_score
            reward -= 1
            if prev_action is not None and action == reverse_actions.get(prev_action):
                reward -= 5
            if next_state.pellets_eaten > prev_pellets_eaten:
                reward += 50
            if next_state.pacman.lives < prev_lives:
                reward -= 250
            elif game.is_game_won():
                reward += 500
            reward = reward / 100.0

            agent.update(current_state, action, next_state, reward)
            prev_action = action

        agent.epsilon = max(0.01, agent.epsilon * 0.995)

    return agent


def _state_signature(game: PacmanGame) -> Tuple[Tuple[int, int], Tuple[Tuple[int, int], ...], Tuple[Tuple[bool, int], ...], Tuple[Tuple[int, int], ...], Tuple[Tuple[int, int], ...], int, int]:
    """Create a compact signature for detecting repeated game states during evaluation."""
    state = game.get_game_state()
    pacman_pos = state.pacman.position.to_tuple() if state.pacman else (-1, -1)
    ghost_positions = tuple(ghost.position.to_tuple() for ghost in state.ghosts)
    ghost_modes = tuple((ghost.scared, ghost.scared_timer) for ghost in state.ghosts)
    pellets = tuple(sorted(state.board.pellets))
    power_pellets = tuple(sorted(state.board.power_pellets))
    return (
        pacman_pos,
        ghost_positions,
        ghost_modes,
        pellets,
        power_pellets,
        state.pellets_eaten,
        state.power_pellets_eaten,
    )


def evaluate_agent(agent_name: str, agent, episodes: int = 8, max_steps: int = 3000) -> List[Dict[str, float]]:
    """Run a fixed number of evaluation games and collect summary metrics."""
    results: List[Dict[str, float]] = []
    ghost_minimax_agent = MinimaxGhostAgent(depth=1)
    scared_defense_agent = MinimaxScaredGhostDefenseAgent(activation_distance=5, escape_horizon=4)

    for episode in range(1, episodes + 1):
        game = PacmanGame(display=False)
        game.reset()
        initial_food = len(game.board.pellets) + len(game.board.power_pellets)
        steps = 0
        seen_states = set()
        stall_count = 0

        while not game.is_game_over() and not game.is_game_won() and steps < max_steps:
            state = game.get_game_state()
            action = agent.get_action(state)

            if action != (0, 0):
                game.move_pacman(action)
            move_ghosts(game, ghost_minimax_agent, scared_defense_agent)
            game.check_collisions()
            game.update_scared_timers()
            game.update_respawn_timers()
            game.check_win_condition()
            steps += 1

            signature = _state_signature(game)
            if signature in seen_states:
                stall_count += 1
            else:
                seen_states.add(signature)
                stall_count = 0

            if stall_count >= 40:
                break

        pacman_state = game.get_game_state().pacman
        state = game.get_game_state()
        completion_ratio = (
            state.pellets_eaten / initial_food if initial_food > 0 else 1.0
        )
        success = 1 if game.is_game_won() or completion_ratio >= 0.95 else 0
        results.append(
            {
                "agent": agent_name,
                "episode": episode,
                "score": pacman_state.score if pacman_state else 0,
                "pellets_eaten": state.pellets_eaten,
                "steps": steps,
                "won": success,
                "completion_ratio": completion_ratio,
                "lives": pacman_state.lives if pacman_state else 0,
            }
        )

    return results


def save_results(results: List[Dict[str, float]], output_path: Path):
    fieldnames = ["agent", "episode", "score", "pellets_eaten", "steps", "won", "completion_ratio", "lives"]
    with output_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(results)


def create_plots(results: List[Dict[str, float]], output_path: Path):
    agents = sorted({row["agent"] for row in results})
    summary = []
    for agent_name in agents:
        rows = [row for row in results if row["agent"] == agent_name]
        summary.append(
            {
                "agent": agent_name,
                "avg_score": sum(row["score"] for row in rows) / len(rows),
                "avg_completion": sum(row["completion_ratio"] for row in rows) / len(rows) * 100,
                "win_rate": sum(row["won"] for row in rows) / len(rows) * 100,
                "avg_steps": sum(row["steps"] for row in rows) / len(rows),
            }
        )

    fig, axes = plt.subplots(1, 3, figsize=(15, 4))

    names = [item["agent"] for item in summary]
    scores = [item["avg_score"] for item in summary]
    completion_rates = [item["avg_completion"] for item in summary]
    win_rates = [item["win_rate"] for item in summary]

    def draw_bar(ax, values, title, ylabel, color):
        bars = ax.bar(names, values, color=color, edgecolor="black", width=0.8)
        ax.set_title(title)
        ax.set_ylabel(ylabel)
        ax.tick_params(axis="x", rotation=20)
        ax.set_ylim(0, max(max(values) * 1.2, 1) if values else 1)
        for bar, value in zip(bars, values):
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height() + max(1, max(values) * 0.02),
                f"{value:.1f}",
                ha="center",
                va="bottom",
                fontsize=8,
            )
        return bars

    draw_bar(axes[0], scores, "Average Score", "Score", "tab:blue")
    draw_bar(axes[1], completion_rates, "Average Completion (%)", "Completion %", "tab:green")
    win_ax = axes[2]
    win_bars = win_ax.bar(names, win_rates, color="tab:red", edgecolor="black", width=0.8)
    win_ax.set_title("Win Rate (%)")
    win_ax.set_ylabel("Win Rate")
    win_ax.tick_params(axis="x", rotation=20)
    win_ax.set_ylim(0, 100)
    for bar, value in zip(win_bars, win_rates):
        win_ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 2,
            f"{value:.1f}%",
            ha="center",
            va="bottom",
            fontsize=8,
        )

    plt.tight_layout()
    plt.savefig(output_path, dpi=200)
    plt.close(fig)


def main():
    project_root = Path(__file__).resolve().parent
    csv_path = project_root / "agent_comparison_results.csv"
    plot_path = project_root / "agent_comparison_plots.png"

    agents = {
        "Q-Learning": train_q_agent(episodes=40),
        "Minimax": MinimaxPacmanAgent(depth=2),
        "A*": AStarPacmanAgent(),
        "Combined": CombinedAgent(),
    }

    all_results: List[Dict[str, float]] = []
    for name, agent in agents.items():
        print(f"Evaluating {name}...")
        results = evaluate_agent(name, agent, episodes=10)
        all_results.extend(results)

    save_results(all_results, csv_path)
    create_plots(all_results, plot_path)

    print(f"Saved results to {csv_path}")
    print(f"Saved plots to {plot_path}")

    print("\nSummary:")
    for agent_name in ["Q-Learning", "Minimax", "A*", "Combined"]:
        rows = [row for row in all_results if row["agent"] == agent_name]
        avg_score = sum(row["score"] for row in rows) / len(rows)
        avg_completion = sum(row["completion_ratio"] for row in rows) / len(rows) * 100
        win_rate = sum(row["won"] for row in rows) / len(rows) * 100
        print(
            f"- {agent_name}: avg score={avg_score:.1f}, avg completion={avg_completion:.1f}%, win rate={win_rate:.1f}%"
        )


if __name__ == "__main__":
    main()
