import pygame
import random
from src.core.game_engine import PacmanGame
from src.agents.approximate_q_agent import ApproximateQAgent
from src.core.config import UP, DOWN, LEFT, RIGHT

def move_ghosts_randomly(game: PacmanGame):
    directions = [UP, DOWN, LEFT, RIGHT]
    for i in range(len(game.game_state.ghosts)):
        game.move_ghost(i, random.choice(directions))

def train_agent(episodes: int) -> ApproximateQAgent:
    print(f"--- Starting Training for {episodes} Episodes ---")
    
    game = PacmanGame(display=False)
    agent = ApproximateQAgent(epsilon=0.2, learning_rate=0.01, discount_factor=0.9)
    batch_wins = 0
    for episode in range(1, episodes + 1):
        game.reset()

        prev_action = None

        # Map actions to their exact opposites to easily detect when the agent is backtracking
        reverse_actions = {(0, -1): (0, 1), (0, 1): (0, -1), (-1, 0): (1, 0), (1, 0): (-1, 0)}
        steps = 0

        # Hard cap to prevent infinite loops if the agent learns "cowardly" stalling behavior
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
            
            # Base logic: Reward the agent based on changes to the official game score
            reward = next_state.pacman.score - prev_score

            # Base step penalty to encourage the agent to finish the game quickly
            reward -= 1 

            # Penalize thrashing (vibrating back and forth) to smooth out pathfinding
            if prev_action is not None and action == reverse_actions.get(prev_action):
                reward -= 5

            # Custom reward shaping for critical events to guide the TD update
            if next_state.pellets_eaten > prev_pellets_eaten:
                reward += 50
            if next_state.pacman.lives < prev_lives:
                reward -= 250
            elif game.is_game_won():
                reward += 500

            # Normalize the raw reward so large numbers don't cause massive weight oscillations
            reward = reward / 100.0

            agent.update(current_state, action, next_state, reward)
            prev_action = action

        if game.is_game_won():
                batch_wins += 1    
        if episode % 50 == 0:
            current_win_rate = (batch_wins / 50.0) * 100
            print(f"Episode {episode}/{episodes} | Win Rate (of current batch): {current_win_rate:.1f}% | Epsilon: {agent.epsilon:.3f}")
            batch_wins = 0

        # Decay epsilon slowly to transition the agent from exploration to pure exploitation
        agent.epsilon = max(0.01, agent.epsilon * 0.995)
            
    return agent


def test_agent(agent: ApproximateQAgent, episodes: int):
    game = PacmanGame(display=False)

    # Force pure exploitation (no random moves) for accurate performance evaluation
    agent.epsilon = 0.0
        
    total_test_score = 0
    total_test_wins = 0
    total_test_steps = 0
    total_test_pellets = 0

    for test_ep in range(1, episodes + 1):
        game.reset()
        steps = 0
            
        while not game.is_game_over() and not game.is_game_won():
            current_state = game.get_game_state().clone()
            action = agent.get_action(current_state)
                
            game.move_pacman(action)
            move_ghosts_randomly(game)
            game.check_collisions()
            game.update_scared_timers()
            game.update_respawn_timers()
            game.check_win_condition()
                
            steps += 1
            
        total_test_score += game.get_game_state().pacman.score
        total_test_pellets += game.get_game_state().pellets_eaten
        total_test_steps += steps
            
        if game.is_game_won():
            total_test_wins += 1
                
    test_win_rate = (total_test_wins / episodes) * 100
    avg_test_score = total_test_score / episodes
    avg_test_pellets = total_test_pellets / episodes
    avg_test_steps = total_test_steps / episodes
    
    print(f"       TESTING SUMMARY ({episodes} Games)")
    print(f" Win Rate:           {test_win_rate:.1f}%")
    print(f" Average Score:      {avg_test_score:.1f}")
    print(f" Avg Pellets Eaten:  {avg_test_pellets:.1f}")
    print(f" Average Steps:      {avg_test_steps:.1f}")


if __name__ == "__main__":
    trained_agent = train_agent(episodes=100)
    
    test_agent(trained_agent, episodes=5)