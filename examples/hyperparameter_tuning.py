"""Hyperparameter tuning for Q-Learning agent.

This script systematically explores different hyperparameter combinations
to find optimal settings for the Approximate Q-Learning agent.
"""

import csv
from pathlib import Path
from typing import Dict, List, Tuple

import matplotlib.pyplot as plt
import numpy as np

from src.agents.approximate_q_agent import ApproximateQAgent
from src.core.game_engine import PacmanGame
from src.agents.minimax_ghost import MinimaxGhostAgent
from src.agents.minimax_defense_ghost import MinimaxScaredGhostDefenseAgent


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


def train_and_evaluate(
    learning_rate: float,
    discount_factor: float,
    epsilon: float,
    train_episodes: int = 50,
    eval_episodes: int = 10,
) -> Dict[str, float]:
    """Train agent with given hyperparameters and evaluate performance."""
    # Train
    agent = ApproximateQAgent(
        learning_rate=learning_rate,
        discount_factor=discount_factor,
        epsilon=epsilon,
    )
    
    for episode in range(train_episodes):
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
    
    # Evaluate
    agent.epsilon = 0.0  # Pure exploitation
    scores = []
    wins = []
    pellets_list = []
    
    for _ in range(eval_episodes):
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
        
        scores.append(game.get_game_state().pacman.score)
        wins.append(1 if game.is_game_won() else 0)
        pellets_list.append(game.get_game_state().pellets_eaten)
    
    return {
        "learning_rate": learning_rate,
        "discount_factor": discount_factor,
        "epsilon": epsilon,
        "avg_score": np.mean(scores),
        "std_score": np.std(scores),
        "win_rate": np.mean(wins) * 100,
        "avg_pellets": np.mean(pellets_list),
    }


def grid_search() -> List[Dict[str, float]]:
    """Perform grid search over hyperparameters."""
    learning_rates = [0.005, 0.01, 0.02, 0.05]
    discount_factors = [0.9, 0.95, 0.99]
    epsilons = [0.1, 0.2, 0.3]
    
    results = []
    total = len(learning_rates) * len(discount_factors) * len(epsilons)
    current = 0
    
    print(f"Starting grid search over {total} combinations...")
    print("This will take several minutes.\n")
    
    for lr in learning_rates:
        for df in discount_factors:
            for eps in epsilons:
                current += 1
                print(f"[{current}/{total}] Testing lr={lr}, df={df}, eps={eps}")
                
                result = train_and_evaluate(lr, df, eps, train_episodes=50, eval_episodes=10)
                results.append(result)
                print(f"  → Score: {result['avg_score']:.1f}, Win Rate: {result['win_rate']:.1f}%\n")
    
    return results


def save_results(results: List[Dict[str, float]], output_path: Path):
    """Save hyperparameter tuning results to CSV."""
    fieldnames = [
        "learning_rate", "discount_factor", "epsilon",
        "avg_score", "std_score", "win_rate", "avg_pellets",
    ]
    with output_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(results)
    print(f"Results saved to {output_path}")


def create_visualizations(results: List[Dict[str, float]], output_dir: Path):
    """Create visualizations of hyperparameter tuning results."""
    output_dir.mkdir(exist_ok=True)
    
    # Sort by win rate
    results_sorted = sorted(results, key=lambda x: x["win_rate"], reverse=True)
    top_10 = results_sorted[:10]
    
    # Plot 1: Top 10 configurations by win rate
    fig, ax = plt.subplots(figsize=(12, 6))
    labels = [
        f"lr={r['learning_rate']}\ndf={r['discount_factor']}\neps={r['epsilon']}"
        for r in top_10
    ]
    win_rates = [r["win_rate"] for r in top_10]
    scores = [r["avg_score"] for r in top_10]
    
    x = np.arange(len(labels))
    width = 0.35
    
    bars1 = ax.bar(x - width/2, win_rates, width, label="Win Rate (%)", color="tab:green")
    ax2 = ax.twinx()
    bars2 = ax2.bar(x + width/2, scores, width, label="Avg Score", color="tab:blue")
    
    ax.set_xlabel("Hyperparameter Configuration")
    ax.set_ylabel("Win Rate (%)", color="tab:green")
    ax2.set_ylabel("Average Score", color="tab:blue")
    ax.set_title("Top 10 Hyperparameter Configurations")
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=45, ha="right", fontsize=8)
    ax.legend(loc="upper left")
    ax2.legend(loc="upper right")
    
    plt.tight_layout()
    plt.savefig(output_dir / "top_10_configs.png", dpi=200)
    plt.close()
    
    # Plot 2: Learning rate vs Win Rate (for best discount factor and epsilon)
    best_df = max(results, key=lambda r: r["win_rate"])["discount_factor"]
    best_eps = max(results, key=lambda r: r["win_rate"])["epsilon"]
    
    lr_results = [r for r in results if r["discount_factor"] == best_df and r["epsilon"] == best_eps]
    lr_results.sort(key=lambda r: r["learning_rate"])
    
    fig, ax = plt.subplots(figsize=(10, 5))
    lrs = [r["learning_rate"] for r in lr_results]
    win_rates = [r["win_rate"] for r in lr_results]
    
    ax.plot(lrs, win_rates, marker="o", linewidth=2, markersize=8, color="tab:red")
    ax.set_xlabel("Learning Rate")
    ax.set_ylabel("Win Rate (%)")
    ax.set_title(f"Learning Rate Impact (df={best_df}, eps={best_eps})")
    ax.grid(True, alpha=0.3)
    ax.set_xscale("log")
    
    plt.tight_layout()
    plt.savefig(output_dir / "learning_rate_impact.png", dpi=200)
    plt.close()
    
    # Plot 3: Epsilon decay impact
    fig, ax = plt.subplots(figsize=(10, 5))
    for eps in [0.1, 0.2, 0.3]:
        eps_results = [r for r in results if r["epsilon"] == eps]
        eps_results.sort(key=lambda r: r["learning_rate"])
        win_rates = [r["win_rate"] for r in eps_results]
        lrs = [r["learning_rate"] for r in eps_results]
        ax.plot(lrs, win_rates, marker="o", label=f"ε={eps}", linewidth=2, markersize=6)
    
    ax.set_xlabel("Learning Rate")
    ax.set_ylabel("Win Rate (%)")
    ax.set_title("Epsilon Impact on Performance")
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_xscale("log")
    
    plt.tight_layout()
    plt.savefig(output_dir / "epsilon_impact.png", dpi=200)
    plt.close()
    
    print(f"\nVisualizations saved to {output_dir}")


def print_best_configuration(results: List[Dict[str, float]]):
    """Print the best hyperparameter configuration."""
    best = max(results, key=lambda r: r["win_rate"])
    
    print("\n" + "=" * 60)
    print("BEST HYPERPARAMETER CONFIGURATION")
    print("=" * 60)
    print(f"Learning Rate:  {best['learning_rate']}")
    print(f"Discount Factor: {best['discount_factor']}")
    print(f"Epsilon:         {best['epsilon']}")
    print(f"\nPerformance Metrics:")
    print(f"  Win Rate:      {best['win_rate']:.1f}%")
    print(f"  Average Score: {best['avg_score']:.1f} ± {best['std_score']:.1f}")
    print(f"  Avg Pellets:   {best['avg_pellets']:.1f}")
    print("=" * 60)


def main():
    """Run hyperparameter tuning."""
    project_root = Path(__file__).resolve().parent
    output_dir = project_root / "hyperparameter_tuning_results"
    output_dir.mkdir(exist_ok=True)
    
    # Run grid search
    results = grid_search()
    
    # Save results
    csv_path = output_dir / "tuning_results.csv"
    save_results(results, csv_path)
    
    # Create visualizations
    create_visualizations(results, output_dir)
    
    # Print best configuration
    print_best_configuration(results)


if __name__ == "__main__":
    main()