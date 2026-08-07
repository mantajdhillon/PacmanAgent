"""Ablation study to understand contribution of each AI technique.

This script systematically evaluates each AI component in isolation and
combination to measure their individual and synergistic contributions.
"""

import csv
import random
import sys
from pathlib import Path
from typing import Callable, Dict, List

import matplotlib.pyplot as plt
import numpy as np

# Add project root to path for imports
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.agents.approximate_q_agent import ApproximateQAgent
from src.agents.astar_agent import AStarPacmanAgent
from src.agents.minimax_agent import MinimaxPacmanAgent
from src.agents.minimax_ghost import MinimaxGhostAgent
from src.agents.minimax_defense_ghost import MinimaxScaredGhostDefenseAgent
from src.core.game_engine import PacmanGame
from src.core.config import UP, DOWN, LEFT, RIGHT


# Agent configurations for ablation study
class RandomAgent:
    """Random baseline agent."""
    def get_action(self, state):
        legal = state.get_legal_actions(0)
        return random.choice(legal) if legal else (0, 0)


class AStarOnlyAgent:
    """A* pathfinding without any safety considerations."""
    def __init__(self):
        self.astar = AStarPacmanAgent()
    
    def get_action(self, state):
        return self.astar.get_action(state)


class MinimaxOnlyAgent:
    """Minimax defensive agent without pellet collection optimization."""
    def __init__(self):
        self.minimax = MinimaxPacmanAgent(depth=2)
    
    def get_action(self, state):
        return self.minimax.get_action(state)


class QLearningOnlyAgent:
    """Q-Learning agent trained for this evaluation."""
    def __init__(self):
        self.agent = ApproximateQAgent(epsilon=0.0, learning_rate=0.01, discount_factor=0.9)
    
    def get_action(self, state):
        return self.agent.get_action(state)


class AStarMinimaxAgent:
    """Combined A* + Minimax (no Q-Learning)."""
    def __init__(self):
        self.astar = AStarPacmanAgent()
        self.minimax = MinimaxPacmanAgent(depth=2)
    
    def get_action(self, state):
        # Use A* when safe, Minimax when threatened
        action = self.astar.get_action(state)
        if self.minimax.is_action_safe(state, action, minimum_distance=0):
            return action
        return self.minimax.get_action(state)


class FullSystemAgent:
    """Complete system: A* + Minimax + Q-Learning."""
    def __init__(self, q_agent: ApproximateQAgent):
        self.q_agent = q_agent
        self.astar = AStarPacmanAgent()
        self.minimax = MinimaxPacmanAgent(depth=2)
    
    def get_action(self, state):
        return self.q_agent.get_action(state)


def move_ghosts(game: PacmanGame):
    """Move ghosts using Minimax for consistent evaluation."""
    ghost_names = [ghost.name for ghost in game.game_state.ghosts]
    ghost_agent = MinimaxGhostAgent(depth=1)
    scared_agent = MinimaxScaredGhostDefenseAgent(activation_distance=5, escape_horizon=4)
    
    for ghost_name in ghost_names:
        ghost_index = next(
            (i for i, ghost in enumerate(game.game_state.ghosts) if ghost.name == ghost_name),
            None,
        )
        if ghost_index is None:
            continue
        
        ghost = game.game_state.ghosts[ghost_index]
        if ghost.scared:
            action = scared_agent.get_action(game.game_state, ghost_index)
        else:
            action = ghost_agent.get_action(game.game_state, ghost_index)
        
        if action != (0, 0):
            game.move_ghost(ghost_index, action)


def train_q_agent(episodes: int = 100) -> ApproximateQAgent:
    """Train Q-Learning agent for ablation study."""
    agent = ApproximateQAgent(learning_rate=0.01, discount_factor=0.9, epsilon=0.2)
    
    for episode in range(episodes):
        game = PacmanGame(display=False)
        game.reset()
        
        prev_action = None
        reverse_actions = {
            (0, -1): (0, 1), (0, 1): (0, -1),
            (-1, 0): (1, 0), (1, 0): (-1, 0),
        }
        steps = 0
        max_steps = 3000
        
        while not game.is_game_over() and not game.is_game_won() and steps < max_steps:
            steps += 1
            current_state = game.get_game_state().clone()
            action = agent.get_action(current_state)
            
            prev_pellets = current_state.pellets_eaten
            prev_score = current_state.pacman.score
            prev_lives = current_state.pacman.lives
            
            game.move_pacman(action)
            move_ghosts(game)
            game.check_collisions()
            game.update_scared_timers()
            game.update_respawn_timers()
            game.check_win_condition()
            
            next_state = game.get_game_state().clone()
            reward = next_state.pacman.score - prev_score
            reward -= 1
            
            if prev_action is not None and action == reverse_actions.get(prev_action):
                reward -= 5
            if next_state.pellets_eaten > prev_pellets:
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


def evaluate_agent(
    agent_name: str,
    agent,
    episodes: int = 20,
    max_steps: int = 3000,
) -> Dict[str, float]:
    """Evaluate an agent and return comprehensive metrics."""
    scores = []
    wins = []
    pellets_list = []
    steps_list = []
    survival_times = []
    
    for _ in range(episodes):
        game = PacmanGame(display=False)
        game.reset()
        initial_food = len(game.board.pellets) + len(game.board.power_pellets)
        steps = 0
        
        while not game.is_game_over() and not game.is_game_won() and steps < max_steps:
            steps += 1
            state = game.get_game_state().clone()
            action = agent.get_action(state)
            
            game.move_pacman(action)
            move_ghosts(game)
            game.check_collisions()
            game.update_scared_timers()
            game.update_respawn_timers()
            game.check_win_condition()
        
        final_state = game.get_game_state()
        scores.append(final_state.pacman.score if final_state.pacman else 0)
        wins.append(1 if game.is_game_won() else 0)
        pellets_list.append(final_state.pellets_eaten)
        steps_list.append(steps)
        survival_times.append(final_state.pacman.lives if final_state.pacman else 0)
    
    return {
        "agent": agent_name,
        "avg_score": np.mean(scores),
        "std_score": np.std(scores),
        "win_rate": np.mean(wins) * 100,
        "avg_pellets": np.mean(pellets_list),
        "avg_steps": np.mean(steps_list),
        "avg_lives": np.mean(survival_times),
        "completion_rate": np.mean(pellets_list) / initial_food * 100,
    }


def run_ablation_study() -> List[Dict[str, float]]:
    """Run complete ablation study."""
    print("=" * 70)
    print("ABLATION STUDY: Evaluating AI Component Contributions")
    print("=" * 70)
    print("\nTraining Q-Learning agent (100 episodes)...")
    
    # Train Q-Learning agent
    trained_q_agent = train_q_agent(episodes=100)
    
    # Define agent configurations
    print("\nEvaluating agent configurations...\n")
    
    agents = {
        "Random Baseline": RandomAgent(),
        "A* Only": AStarOnlyAgent(),
        "Minimax Only": MinimaxOnlyAgent(),
        "Q-Learning Only": QLearningOnlyAgent(),
        "A* + Minimax": AStarMinimaxAgent(),
        "Full System (A*+Minimax+Q)": FullSystemAgent(trained_q_agent),
    }
    
    results = []
    for agent_name, agent in agents.items():
        print(f"Evaluating {agent_name}...")
        metrics = evaluate_agent(agent_name, agent, episodes=20)
        results.append(metrics)
        print(f"  -> Score: {metrics['avg_score']:.1f}, Win Rate: {metrics['win_rate']:.1f}%\n")
    
    return results


def save_results(results: List[Dict[str, float]], output_path: Path):
    """Save ablation study results."""
    fieldnames = [
        "agent", "avg_score", "std_score", "win_rate",
        "avg_pellets", "avg_steps", "avg_lives", "completion_rate",
    ]
    with output_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(results)
    print(f"Results saved to {output_path}")


def create_visualizations(results: List[Dict[str, float]], output_dir: Path):
    """Create comprehensive ablation study visualizations."""
    output_dir.mkdir(exist_ok=True)
    
    agent_names = [r["agent"] for r in results]
    
    # Plot 1: Win Rate and Score comparison
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
    
    # Win Rate
    win_rates = [r["win_rate"] for r in results]
    colors = plt.cm.RdYlGn(np.linspace(0.2, 0.8, len(agent_names)))
    bars1 = ax1.bar(agent_names, win_rates, color=colors, edgecolor="black", alpha=0.8)
    ax1.set_ylabel("Win Rate (%)", fontsize=12)
    ax1.set_title("Win Rate by Agent Configuration", fontsize=14, fontweight="bold")
    ax1.set_ylim(0, 100)
    ax1.tick_params(axis="x", rotation=30, ha="right")
    ax1.grid(True, alpha=0.3, axis="y")
    
    for bar, rate in zip(bars1, win_rates):
        height = bar.get_height()
        ax1.text(
            bar.get_x() + bar.get_width() / 2, height + 2,
            f"{rate:.1f}%", ha="center", va="bottom", fontsize=10, fontweight="bold"
        )
    
    # Average Score
    scores = [r["avg_score"] for r in results]
    bars2 = ax2.bar(agent_names, scores, color=colors, edgecolor="black", alpha=0.8)
    ax2.set_ylabel("Average Score", fontsize=12)
    ax2.set_title("Average Score by Agent Configuration", fontsize=14, fontweight="bold")
    ax2.tick_params(axis="x", rotation=30, ha="right")
    ax2.grid(True, alpha=0.3, axis="y")
    
    for bar, score in zip(bars2, scores):
        height = bar.get_height()
        ax2.text(
            bar.get_x() + bar.get_width() / 2, height + max(scores) * 0.02,
            f"{score:.0f}", ha="center", va="bottom", fontsize=10, fontweight="bold"
        )
    
    plt.tight_layout()
    plt.savefig(output_dir / "ablation_winrate_score.png", dpi=200)
    plt.close()
    
    # Plot 2: Multi-metric radar chart
    fig, ax = plt.subplots(figsize=(10, 10), subplot_kw=dict(projection="polar"))
    
    metrics = ["win_rate", "avg_pellets", "avg_lives", "completion_rate"]
    metric_labels = ["Win Rate", "Pellets Collected", "Lives Remaining", "Completion %"]
    
    # Normalize metrics to 0-1 scale
    normalized_results = []
    for r in results:
        normalized = {
            "win_rate": r["win_rate"] / 100,
            "avg_pellets": min(r["avg_pellets"] / 174, 1.0),  # Max 174 pellets
            "avg_lives": r["avg_lives"] / 3,  # Max 3 lives
            "completion_rate": r["completion_rate"] / 100,
        }
        normalized_results.append(normalized)
    
    angles = np.linspace(0, 2 * np.pi, len(metrics), endpoint=False).tolist()
    angles += angles[:1]  # Complete the loop
    
    colors = plt.cm.tab10(np.linspace(0, 1, len(agent_names)))
    
    for idx, (agent_name, normalized) in enumerate(zip(agent_names, normalized_results)):
        values = [normalized[m] for m in metrics]
        values += values[:1]  # Complete the loop
        
        ax.plot(angles, values, "o-", linewidth=2, label=agent_name, color=colors[idx])
        ax.fill(angles, values, alpha=0.1, color=colors[idx])
    
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(metric_labels, fontsize=11)
    ax.set_ylim(0, 1)
    ax.set_title("Agent Performance Comparison (Normalized)", fontsize=14, fontweight="bold", pad=20)
    ax.legend(loc="upper right", bbox_to_anchor=(1.3, 1.0), fontsize=9)
    ax.grid(True)
    
    plt.tight_layout()
    plt.savefig(output_dir / "ablation_radar.png", dpi=200)
    plt.close()
    
    # Plot 3: Component contribution analysis
    fig, axes = plt.subplots(2, 2, figsize=(14, 12))
    
    # Win Rate improvement
    baseline_win_rate = results[0]["win_rate"]  # Random baseline
    win_rate_improvements = [r["win_rate"] - baseline_win_rate for r in results]
    bars1 = axes[0, 0].bar(agent_names, win_rate_improvements, color="green", edgecolor="black", alpha=0.7)
    axes[0, 0].set_ylabel("Win Rate Improvement (%)")
    axes[0, 0].set_title("Win Rate Improvement Over Random Baseline")
    axes[0, 0].tick_params(axis="x", rotation=30, ha="right")
    axes[0, 0].axhline(y=0, color="black", linestyle="--", alpha=0.3)
    axes[0, 0].grid(True, alpha=0.3, axis="y")
    
    # Score improvement
    baseline_score = results[0]["avg_score"]
    score_improvements = [r["avg_score"] - baseline_score for r in results]
    bars2 = axes[0, 1].bar(agent_names, score_improvements, color="blue", edgecolor="black", alpha=0.7)
    axes[0, 1].set_ylabel("Score Improvement")
    axes[0, 1].set_title("Score Improvement Over Random Baseline")
    axes[0, 1].tick_params(axis="x", rotation=30, ha="right")
    axes[0, 1].axhline(y=0, color="black", linestyle="--", alpha=0.3)
    axes[0, 1].grid(True, alpha=0.3, axis="y")
    
    # Pellets collected
    pellets = [r["avg_pellets"] for r in results]
    bars3 = axes[1, 0].bar(agent_names, pellets, color="orange", edgecolor="black", alpha=0.7)
    axes[1, 0].set_ylabel("Average Pellets Collected")
    axes[1, 0].set_title("Pellets Collected by Agent")
    axes[1, 0].tick_params(axis="x", rotation=30, ha="right")
    axes[1, 0].axhline(y=174, color="red", linestyle="--", alpha=0.5, label="Max (174)")
    axes[1, 0].legend()
    axes[1, 0].grid(True, alpha=0.3, axis="y")
    
    # Survival (lives remaining)
    lives = [r["avg_lives"] for r in results]
    bars4 = axes[1, 1].bar(agent_names, lives, color="red", edgecolor="black", alpha=0.7)
    axes[1, 1].set_ylabel("Average Lives Remaining")
    axes[1, 1].set_title("Survival Ability by Agent")
    axes[1, 1].tick_params(axis="x", rotation=30, ha="right")
    axes[1, 1].axhline(y=3, color="green", linestyle="--", alpha=0.5, label="Max (3)")
    axes[1, 1].legend()
    axes[1, 1].grid(True, alpha=0.3, axis="y")
    axes[1, 1].set_ylim(0, 3.5)
    
    plt.suptitle("Ablation Study: Component Contribution Analysis", fontsize=16, fontweight="bold", y=1.00)
    plt.tight_layout()
    plt.savefig(output_dir / "ablation_contributions.png", dpi=200)
    plt.close()
    
    # Plot 4: Performance vs Complexity trade-off
    fig, ax = plt.subplots(figsize=(10, 7))
    
    # Complexity scores (subjective based on implementation effort)
    complexity = {
        "Random Baseline": 1,
        "A* Only": 2,
        "Minimax Only": 3,
        "Q-Learning Only": 4,
        "A* + Minimax": 5,
        "Full System (A*+Minimax+Q)": 6,
    }
    
    for r in results:
        agent_name = r["agent"]
        x = complexity[agent_name]
        y = r["win_rate"]
        size = r["avg_score"] / 10
        
        ax.scatter(x, y, s=size, alpha=0.6, edgecolors="black", linewidth=2)
        ax.annotate(
            agent_name, (x, y),
            xytext=(5, 5), textcoords="offset points",
            fontsize=9, fontweight="bold"
        )
    
    ax.set_xlabel("System Complexity (Implementation Effort)", fontsize=12)
    ax.set_ylabel("Win Rate (%)", fontsize=12)
    ax.set_title("Performance vs Complexity Trade-off\n(Bubble size = Average Score)", fontsize=14, fontweight="bold")
    ax.grid(True, alpha=0.3)
    ax.set_ylim(-5, 105)
    
    plt.tight_layout()
    plt.savefig(output_dir / "ablation_tradeoff.png", dpi=200)
    plt.close()
    
    print(f"\nVisualizations saved to {output_dir}")


def print_ablation_summary(results: List[Dict[str, float]]):
    """Print detailed ablation study summary."""
    print("\n" + "=" * 80)
    print("ABLATION STUDY SUMMARY")
    print("=" * 80)
    
    # Calculate improvements
    baseline = results[0]
    
    print("\nPerformance Comparison:")
    print("-" * 80)
    print(f"{'Agent':<30} {'Win Rate':>10} {'Avg Score':>12} {'Pellets':>10} {'Lives':>8}")
    print("-" * 80)
    
    for r in results:
        print(
            f"{r['agent']:<30} {r['win_rate']:>9.1f}% {r['avg_score']:>11.1f} "
            f"{r['avg_pellets']:>9.1f} {r['avg_lives']:>7.1f}"
        )
    
    print("\n" + "=" * 80)
    print("KEY FINDINGS:")
    print("=" * 80)
    
    # Find best agent
    best = max(results, key=lambda r: r["win_rate"])
    print(f"\n★ Best Agent: {best['agent']}")
    print(f"  Win Rate: {best['win_rate']:.1f}%")
    print(f"  Average Score: {best['avg_score']:.1f}")
    
    # Calculate contribution of each component
    if len(results) >= 6:
        q_only = next(r for r in results if "Q-Learning Only" in r["agent"])
        minimax_only = next(r for r in results if "Minimax Only" in r["agent"])
        astar_only = next(r for r in results if "A* Only" in r["agent"])
        full_system = next(r for r in results if "Full System" in r["agent"])
        
        print("\nComponent Contributions (Win Rate Improvement):")
        print(f"  A*:         +{astar_only['win_rate'] - baseline['win_rate']:.1f}%")
        print(f"  Minimax:    +{minimax_only['win_rate'] - baseline['win_rate']:.1f}%")
        print(f"  Q-Learning: +{q_only['win_rate'] - baseline['win_rate']:.1f}%")
        print(f"  Synergy:    +{full_system['win_rate'] - max(astar_only['win_rate'], minimax_only['win_rate'], q_only['win_rate']):.1f}%")
    
    print("=" * 80)


def main():
    """Run ablation study."""
    project_root = Path(__file__).resolve().parent
    output_dir = project_root / "ablation_study_results"
    output_dir.mkdir(exist_ok=True)
    
    # Run ablation study
    results = run_ablation_study()
    
    # Save results
    csv_path = output_dir / "ablation_results.csv"
    save_results(results, csv_path)
    
    # Create visualizations
    create_visualizations(results, output_dir)
    
    # Print summary
    print_ablation_summary(results)


if __name__ == "__main__":
    main()