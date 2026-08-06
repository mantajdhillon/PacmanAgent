import csv
import random
import matplotlib.pyplot as plt

from src.core.game_engine import PacmanGame
from src.agents.approximate_q_agent import ApproximateQAgent
from src.core.config import UP, DOWN, LEFT, RIGHT


def move_ghosts_randomly(game):
    directions = [UP, DOWN, LEFT, RIGHT]
    for i in range(len(game.game_state.ghosts)):
        game.move_ghost(i, random.choice(directions))


def run_training(episodes=100, csv_path="training_metrics.csv"):
    metrics = []

    for episode in range(1, episodes + 1):
        game = PacmanGame(display=False)
        agent = ApproximateQAgent(epsilon=0.2, learning_rate=0.01, discount_factor=0.9)

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

            game.move_pacman(action)
            move_ghosts_randomly(game)
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

        win = 1 if game.is_game_won() else 0
        metrics.append({
            "episode": episode,
            "score": game.get_game_state().pacman.score,
            "pellets_eaten": game.get_game_state().pellets_eaten,
            "steps": steps,
            "win": win,
            "epsilon": agent.epsilon,
        })

        agent.epsilon = max(0.01, agent.epsilon * 0.995)

    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["episode", "score", "pellets_eaten", "steps", "win", "epsilon"])
        writer.writeheader()
        writer.writerows(metrics)

    print(f"Saved metrics to {csv_path}")

    # Plot 1: score over episodes
    plt.figure(figsize=(12, 4))
    plt.subplot(1, 3, 1)
    plt.plot([m["episode"] for m in metrics], [m["score"] for m in metrics], color="tab:blue")
    plt.title("Training Score")
    plt.xlabel("Episode")
    plt.ylabel("Score")

    # Plot 2: pellets eaten over episodes
    plt.subplot(1, 3, 2)
    plt.plot([m["episode"] for m in metrics], [m["pellets_eaten"] for m in metrics], color="tab:green")
    plt.title("Pellets Eaten")
    plt.xlabel("Episode")
    plt.ylabel("Pellets")

    # Plot 3: win rate over episodes
    win_rates = []
    for i in range(len(metrics)):
        window = metrics[max(0, i - 9):i + 1]
        win_rates.append(sum(m["win"] for m in window) / len(window) * 100)

    plt.subplot(1, 3, 3)
    plt.plot([m["episode"] for m in metrics], win_rates, color="tab:red")
    plt.title("Win Rate (rolling 10-episode)")
    plt.xlabel("Episode")
    plt.ylabel("Win Rate (%)")

    plt.tight_layout()
    plt.savefig("training_plots.png")
    plt.show()


if __name__ == "__main__":
    run_training(episodes=100)