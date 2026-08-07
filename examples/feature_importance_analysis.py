"""Feature importance analysis for Q-Learning agent.

This script analyzes which features the Q-Learning agent learns to value most
during training, providing insights into the agent's decision-making process.
"""

import csv
from pathlib import Path
from typing import Dict, List

import matplotlib.pyplot as plt
import numpy as np

from src.agents.approximate_q_agent import ApproximateQAgent
from src.core.game_engine import PacmanGame
from src.agents.minimax_ghost import MinimaxGhostAgent
from src.agents.minimax_defense_ghost import MinimaxScaredGhostDefenseAgent


FEATURE_NAMES = [
    "Bias",
    "Eats Food",
    "Food Proximity",
    "Danger (Adjacent Ghost)",
    "Ghost Distance",
    "Scared Ghost Distance",
]


def move_ghosts(game: PacmanGame):
    """Move ghosts using Minimax for realistic training."""
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


def train_agent(episodes: int = 100) -> ApproximateQAgent:
    """Train Q-Learning agent and track weight evolution."""
    agent = ApproximateQAgent(learning_rate=0.01, discount_factor=0.9, epsilon=0.2)
    
    weight_history = []
    episode_rewards = []
    episode_wins = []
    
    for episode in range(1, episodes + 1):
        game = PacmanGame(display=False)
        game.reset()
        
        prev_action = None
        reverse_actions = {
            (0, -1): (0, 1), (0, 1): (0, -1),
            (-1, 0): (1, 0), (1, 0): (-1, 0),
        }
        steps = 0
        max_steps = 3000
        episode_reward = 0
        
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
            episode_reward += reward
            
            agent.update(current_state, action, next_state, reward)
            prev_action = action
        
        # Record weight snapshot
        if agent.weights is not None:
            weight_history.append(agent.weights.copy())
        
        episode_rewards.append(episode_reward)
        episode_wins.append(1 if game.is_game_won() else 0)
        
        agent.epsilon = max(0.01, agent.epsilon * 0.995)
        
        if episode % 10 == 0:
            recent_wins = sum(episode_wins[-10:]) / 10 * 100
            print(f"Episode {episode}/{episodes} | Recent Win Rate: {recent_wins:.1f}%")
    
    return agent, weight_history, episode_rewards, episode_wins


def analyze_feature_importance(weight_history: List[np.ndarray]) -> Dict[str, List[float]]:
    """Analyze how feature weights evolve during training."""
    if not weight_history:
        return {}
    
    # Stack weights into matrix (episodes x features)
    weight_matrix = np.stack(weight_history)
    
    # Calculate statistics for each feature
    feature_stats = {}
    for i, name in enumerate(FEATURE_NAMES):
        weights = weight_matrix[:, i]
        feature_stats[name] = {
            "values": weights.tolist(),
            "final": weights[-1],
            "mean": np.mean(weights),
            "std": np.std(weights),
            "min": np.min(weights),
            "max": np.max(weights),
            "abs_mean": np.mean(np.abs(weights)),
        }
    
    return feature_stats


def create_visualizations(
    feature_stats: Dict[str, List[float]],
    weight_history: List[np.ndarray],
    episode_rewards: List[float],
    episode_wins: List[int],
    output_dir: Path,
):
    """Create comprehensive visualizations of feature importance."""
    output_dir.mkdir(exist_ok=True)
    
    # Plot 1: Weight evolution over training
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    axes = axes.flatten()
    
    weight_matrix = np.stack(weight_history)
    episodes = list(range(1, len(weight_history) + 1))
    
    for idx, (feature_name, stats) in enumerate(feature_stats.items()):
        ax = axes[idx]
        values = stats["values"]
        ax.plot(episodes, values, linewidth=2, color=f"C{idx}")
        ax.axhline(y=0, color="black", linestyle="--", alpha=0.3)
        ax.fill_between(episodes, values, 0, alpha=0.2, color=f"C{idx}")
        ax.set_xlabel("Episode")
        ax.set_ylabel("Weight Value")
        ax.set_title(f"{feature_name}\nFinal: {stats['final']:.3f}")
        ax.grid(True, alpha=0.3)
    
    plt.suptitle("Feature Weight Evolution During Training", fontsize=14, fontweight="bold")
    plt.tight_layout()
    plt.savefig(output_dir / "weight_evolution.png", dpi=200)
    plt.close()
    
    # Plot 2: Final feature importance (absolute values)
    fig, ax = plt.subplots(figsize=(10, 6))
    final_weights = [feature_stats[name]["final"] for name in FEATURE_NAMES]
    abs_weights = [abs(w) for w in final_weights]
    colors = ["green" if w > 0 else "red" for w in final_weights]
    
    bars = ax.barh(FEATURE_NAMES, final_weights, color=colors, edgecolor="black", alpha=0.7)
    ax.axvline(x=0, color="black", linestyle="--", linewidth=1)
    ax.set_xlabel("Weight Value")
    ax.set_title("Final Feature Importance (Positive = Good, Negative = Bad)")
    ax.grid(True, alpha=0.3, axis="x")
    
    # Add value labels
    for bar, weight in zip(bars, final_weights):
        width = bar.get_width()
        ax.text(
            width + (0.01 if width > 0 else -0.01),
            bar.get_y() + bar.get_height() / 2,
            f"{weight:.3f}",
            ha="left" if width > 0 else "right",
            va="center",
            fontsize=10,
            fontweight="bold",
        )
    
    plt.tight_layout()
    plt.savefig(output_dir / "feature_importance.png", dpi=200)
    plt.close()
    
    # Plot 3: Absolute importance ranking
    fig, ax = plt.subplots(figsize=(10, 6))
    sorted_features = sorted(FEATURE_NAMES, key=lambda name: abs(feature_stats[name]["final"]), reverse=True)
    sorted_abs_weights = [abs(feature_stats[name]["final"]) for name in sorted_features]
    
    bars = ax.barh(sorted_features, sorted_abs_weights, color="steelblue", edgecolor="black", alpha=0.7)
    ax.set_xlabel("Absolute Weight Value")
    ax.set_title("Feature Importance Ranking (Absolute Values)")
    ax.grid(True, alpha=0.3, axis="x")
    
    for bar, weight in zip(bars, sorted_abs_weights):
        width = bar.get_width()
        ax.text(
            width + 0.01,
            bar.get_y() + bar.get_height() / 2,
            f"{weight:.3f}",
            ha="left",
            va="center",
            fontsize=10,
            fontweight="bold",
        )
    
    plt.tight_layout()
    plt.savefig(output_dir / "feature_ranking.png", dpi=200)
    plt.close()
    
    # Plot 4: Training progress (rewards and win rate)
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8))
    
    # Rewards with rolling average
    window = 10
    rolling_rewards = [
        np.mean(episode_rewards[max(0, i - window + 1) : i + 1])
        for i in range(len(episode_rewards))
    ]
    ax1.plot(episodes, episode_rewards, alpha=0.3, color="blue", label="Raw Reward")
    ax1.plot(episodes, rolling_rewards, linewidth=2, color="darkblue", label=f"Rolling Avg (window={window})")
    ax1.set_xlabel("Episode")
    ax1.set_ylabel("Total Reward")
    ax1.set_title("Training Progress: Episode Rewards")
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Win rate with rolling average
    rolling_wins = [
        sum(episode_wins[max(0, i - window + 1) : i + 1]) / min(window, i + 1) * 100
        for i in range(len(episode_wins))
    ]
    ax2.plot(episodes, [w * 100 for w in episode_wins], alpha=0.3, color="green", label="Win/Loss")
    ax2.plot(episodes, rolling_wins, linewidth=2, color="darkgreen", label=f"Rolling Win Rate (window={window})")
    ax2.set_xlabel("Episode")
    ax2.set_ylabel("Win Rate (%)")
    ax2.set_title("Training Progress: Win Rate")
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_dir / "training_progress.png", dpi=200)
    plt.close()
    
    # Plot 5: Weight correlation heatmap
    if len(weight_history) > 10:
        fig, ax = plt.subplots(figsize=(10, 8))
        weight_matrix = np.stack(weight_history)
        correlation_matrix = np.corrcoef(weight_matrix.T)
        
        im = ax.imshow(correlation_matrix, cmap="RdBu_r", vmin=-1, vmax=1)
        ax.set_xticks(np.arange(len(FEATURE_NAMES)))
        ax.set_yticks(np.arange(len(FEATURE_NAMES)))
        ax.set_xticklabels(FEATURE_NAMES, rotation=45, ha="right")
        ax.set_yticklabels(FEATURE_NAMES)
        
        # Add correlation values
        for i in range(len(FEATURE_NAMES)):
            for j in range(len(FEATURE_NAMES)):
                text = ax.text(
                    j, i, f"{correlation_matrix[i, j]:.2f}",
                    ha="center", va="center", color="black", fontsize=9
                )
        
        ax.set_title("Feature Weight Correlation Matrix")
        plt.colorbar(im, ax=ax, label="Correlation")
        plt.tight_layout()
        plt.savefig(output_dir / "weight_correlations.png", dpi=200)
        plt.close()
    
    print(f"\nVisualizations saved to {output_dir}")


def save_analysis_report(
    feature_stats: Dict[str, Dict[str, float]],
    output_path: Path,
):
    """Save detailed analysis report to CSV."""
    fieldnames = ["feature", "final_weight", "mean", "std", "min", "max", "abs_mean"]
    
    with output_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        
        for feature_name, stats in feature_stats.items():
            writer.writerow({
                "feature": feature_name,
                "final_weight": stats["final"],
                "mean": stats["mean"],
                "std": stats["std"],
                "min": stats["min"],
                "max": stats["max"],
                "abs_mean": stats["abs_mean"],
            })
    
    print(f"Analysis report saved to {output_path}")


def print_feature_analysis(feature_stats: Dict[str, Dict[str, float]]):
    """Print detailed feature importance analysis."""
    print("\n" + "=" * 70)
    print("FEATURE IMPORTANCE ANALYSIS")
    print("=" * 70)
    
    # Sort by absolute final weight
    sorted_features = sorted(
        feature_stats.items(),
        key=lambda x: abs(x[1]["final"]),
        reverse=True,
    )
    
    print("\nFeature Ranking by Importance (Absolute Weight):")
    print("-" * 70)
    for rank, (name, stats) in enumerate(sorted_features, 1):
        direction = "↑ GOOD" if stats["final"] > 0 else "↓ BAD"
        print(f"{rank}. {name:25s} | Weight: {stats['final']:+.4f} {direction:10s} | Abs Mean: {stats['abs_mean']:.4f}")
    
    print("\n" + "=" * 70)
    print("INTERPRETATION:")
    print("=" * 70)
    print("Positive weights: Agent learns to seek these states")
    print("Negative weights: Agent learns to avoid these states")
    print("Large absolute values: Strong influence on decisions")
    print("Small absolute values: Minor influence on decisions")
    print("=" * 70)


def main():
    """Run feature importance analysis."""
    project_root = Path(__file__).resolve().parent
    output_dir = project_root / "feature_importance_analysis"
    output_dir.mkdir(exist_ok=True)
    
    print("Training Q-Learning agent with weight tracking...")
    print("This will train for 100 episodes and analyze feature importance.\n")
    
    # Train agent
    agent, weight_history, episode_rewards, episode_wins = train_agent(episodes=100)
    
    # Analyze feature importance
    feature_stats = analyze_feature_importance(weight_history)
    
    if not feature_stats:
        print("ERROR: No weight history recorded. Training may have failed.")
        return
    
    # Create visualizations
    create_visualizations(feature_stats, weight_history, episode_rewards, episode_wins, output_dir)
    
    # Save analysis report
    csv_path = output_dir / "feature_importance.csv"
    save_analysis_report(feature_stats, csv_path)
    
    # Print analysis
    print_feature_analysis(feature_stats)
    
    # Final evaluation
    print("\n" + "=" * 70)
    print("FINAL AGENT EVALUATION")
    print("=" * 70)
    agent.epsilon = 0.0
    
    test_scores = []
    test_wins = []
    for _ in range(10):
        game = PacmanGame(display=False)
        game.reset()
        steps = 0
        max_steps = 3000
        
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
        
        test_scores.append(game.get_game_state().pacman.score)
        test_wins.append(1 if game.is_game_won() else 0)
    
    print(f"Test Win Rate: {np.mean(test_wins) * 100:.1f}%")
    print(f"Test Avg Score: {np.mean(test_scores):.1f} ± {np.std(test_scores):.1f}")
    print("=" * 70)


if __name__ == "__main__":
    main()