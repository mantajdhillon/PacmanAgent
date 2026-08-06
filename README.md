# Pac-Man Agent: AI Learning Through Game Playing

Autonomous Pac-Man agent implementing A*, Minimax, and Q-Learning for CMPT 310.

## Quick Start

### Installation
```bash
pip install -r requirements.txt
```

### Run the Game with UI
```bash
python src/main/play_game.py
```

### Run Tests
```bash
python tests/run_tests.py
```

## Project Structure

```
PacmanAgent/
├── src/
│   ├── core/                    # Core game engine and state management
│   │   ├── config.py            # Game constants (board size, colors, directions)
│   │   ├── game_state.py        # Position, PacmanState, GhostState, GameBoard, GameState
│   │   ├── game_engine.py       # PacmanGame - main game engine with Pygame
│   │   └── feature_extractor.py # Feature extraction for AI agents
│   ├── agents/                  # AI agent implementations
│   │   ├── astar_agent.py       # A* pathfinding agent
│   │   ├── minimax_agent.py     # Defensive Minimax Pac-Man agent
│   │   ├── minimax_ghost.py     # Team Minimax ghost attack agent
│   │   ├── minimax_attack_agent.py # Scared ghost attack agent
│   │   ├── minimax_defense_ghost.py # Scared ghost defense agent
│   │   └── approximate_q_agent.py # Q-learning agent
│   └── main/
│       └── play_game.py         # Interactive Pygame UI
├── examples/                    # Demo and example scripts
│   ├── run_astar_demo.py        # Headless A* demo
│   ├── run_minimax_demo.py      # Headless Minimax demo
│   ├── train_q_agent.py         # Q-learning training script
│   ├── compare_agents.py        # Agent comparison tool
│   ├── training_plots.py        # Training visualization
│   ├── example_usage.py         # 8 working examples
│   └── demo_testing_options.py  # Testing overview
├── tests/
│   └── run_tests.py             # 19 automated tests
├── requirements.txt
└── README.md
```

## Project Overview

This project develops an intelligent Pac-Man agent that combines multiple AI techniques:
- **A*** for pathfinding to pellets
- **Minimax** with Alpha-Beta pruning for ghost behavior modeling
- **Q-Learning** for long-term strategy optimization

The agent must balance short-term rewards (collecting pellets) with long-term survival (avoiding ghosts).

---

## Core Components

### Game Engine (`src/core/game_engine.py`)
The complete Pac-Man game implementation with Pygame rendering.
- **Key class**: `PacmanGame`
- **Key methods**: `move_pacman()`, `move_ghost()`, `render()`, `check_collisions()`
- **Use for**: Creating game instances and controlling gameplay

### Game State (`src/core/game_state.py`)
Data structures that represent the game state.
- **Key classes**: `Position`, `PacmanState`, `GhostState`, `GameBoard`, `GameState`
- **Use for**: Understanding game data structure, accessing board and entity information

### Feature Extractor (`src/core/feature_extractor.py`)
Converts game state into multiple formats for AI algorithms.
- **Key class**: `FeatureExtractor`
- **Formats**: Dictionaries, NumPy arrays, normalized vectors
- **Use for**: Getting game state data in format your AI needs

### Configuration (`src/core/config.py`)
Centralized constants and bounded-cache support.
- **Board size**: 21 x 21 cells
- **Directions**: UP, DOWN, LEFT, RIGHT
- **Colors**: BLACK, WHITE, YELLOW, RED, PINK, CYAN, ORANGE, BLUE

---

## AI Agents

### A* Agent (`src/agents/astar_agent.py`)
Pathfinding agent for pellet collection.
- Uses A* search with Manhattan distance heuristic
- Finds shortest path to nearest pellet or power pellet
- Includes MST cost estimation for route planning

### Minimax Pac-Man Agent (`src/agents/minimax_agent.py`)
Defensive controller for Pac-Man.
- Models Pac-Man as maximizing agent, ghosts as minimizing agents
- Threat radius: 5 maze steps
- Prioritizes survival, escape routes, and ghost separation
- Uses Alpha-Beta pruning with memoization

### Minimax Ghost Agent (`src/agents/minimax_ghost.py`)
Coordinated team attack for normal ghosts.
- All non-scared ghosts attack at every distance
- Stable named roles (Blinky, Pinky, Inky, Clyde) for approach directions
- Team utility: capture Pac-Man, restrict exits, coordinate coverage

### Minimax Scared Ghost Attack (`src/agents/minimax_attack_agent.py`)
Safe Pac-Man pursuit of scared ghosts.
- Activates when scared ghost is reachable before timer expires
- Requires safety margin from normal ghosts
- Locks target to prevent oscillation

### Minimax Scared Ghost Defense (`src/agents/minimax_defense_ghost.py`)
Defensive behavior for scared ghosts.
- Minimax within distance 5, greedy flee beyond that
- Maximizes survival and escape space
- Avoids dead ends and interception

### Approximate Q-Learning Agent (`src/agents/approximate_q_agent.py`)
Reinforcement learning agent.
- Uses weight vector instead of Q-table
- 6 core features: bias, food proximity, danger, scared ghost distance
- Epsilon-greedy exploration with decay

---

## How to Use

### Play the Game Interactively
```bash
python src/main/play_game.py
```
- Opens Pygame window
- Arrow keys: Move Pac-Man
- A: Toggle manual/autoplay control
- SPACE: Pause/Resume
- R: Restart
- ESC: Quit

### Run Automated Tests
```bash
python tests/run_tests.py
```
- 19 tests covering all components
- Validates game mechanics, AI agents, and feature extraction
- ~30 seconds to complete

### Run A* Demo (Headless)
```bash
python examples/run_astar_demo.py
```
- Pac-Man collects all pellets using A* pathfinding
- No ghosts for clean demonstration
- Shows steps, pellets eaten, and final score

### Run Minimax Demo (Headless)
```bash
python examples/run_minimax_demo.py
```
- Full Minimax system with Pac-Man defense and ghost attack
- 200 frames of simulation
- Reports FPS and mode usage statistics

### Train Q-Learning Agent
```bash
python examples/train_q_agent.py
```
- Trains agent for 100 episodes
- Shows win rate every 50 episodes
- Tests final performance

### Compare All Agents
```bash
python examples/compare_agents.py
```
- Trains and evaluates Q-Learning, Minimax, A*, and Combined agents
- Generates CSV results and comparison plots
- Shows average score, completion rate, and win rate

### View Examples
```bash
python examples/example_usage.py
```
- 8 examples demonstrating:
  1. Basic game setup
  2. Feature extraction
  3. State arrays
  4. State vectors
  5. Board navigation
  6. Pellet tracking
  7. Game state components
  8. Game simulation

---

## Using the Library in Your Code

```python
import sys
sys.path.insert(0, 'c:/Users/manta/OneDrive/Desktop/CMPT310/PacmanAgent')

from src.core.game_engine import PacmanGame
from src.core.config import UP, DOWN, LEFT, RIGHT
from src.agents.astar_agent import AStarPacmanAgent

# Create game (headless - no UI)
game = PacmanGame(display=False)
game.reset()

# Get game state in 3 formats
features = game.get_features()           # Dictionary
state_arrays = game.get_state_arrays()   # NumPy arrays
state_vector = game.get_state_vector()   # Neural network vector

# Use A* agent
agent = AStarPacmanAgent()
action = agent.get_action(game.game_state)

# Move Pac-Man
game.move_pacman(action)

# Get info
print(f"Score: {game.game_state.pacman.score}")
print(f"Pellets: {len(game.board.pellets)}")
```

---

## Game Information

**Board Size**: 21 x 21 cells  
**Pac-Man**: Starts at (10, 15) with 3 lives  
**Ghosts**: 4 independent ghosts (Blinky, Pinky, Inky, Clyde)  
**Pellets**: 174 regular pellets + 4 power pellets at corners

**Scoring**:
- Regular pellet: +10 points
- Power pellet: +50 points
- Eat scared ghost: +200 points

**Win Condition**: Collect all pellets  
**Lose Condition**: Lose all 3 lives

---

## Epic 2: A* Pathfinding

A* pathfinding agent that automatically guides Pac-Man to the nearest pellet using the A* search algorithm with Manhattan distance heuristic.

### Features
- A* search for shortest path planning
- Manhattan distance heuristic
- Exact maze-distance helper
- Minimum spanning tree (MST) cost estimate
- Autoplay mode integrated into play_game.py

### Test Epic 2
```bash
python tests/run_tests.py
python examples/run_astar_demo.py
```

Expected: All tests pass, A* demo collects all pellets and wins.

---

## Epic 3: Adversarial Search (Minimax)

Four-mode adversarial system where Pac-Man and ghosts change objectives based on ghost state, maze distance, scared-timer feasibility, and survival risk.

### Decision Priority
1. **Safe scared-ghost attack** - Pursue edible ghost when safe and timer allows
2. **Safe point collection** - Use A* when no nearby threats
3. **Minimax defense** - Activate when normal ghost within distance 5
4. **Emergency survival** - Choose safest action when objective move is unsafe

### Mode Summary

| Controller | Activation | Primary Objective |
| --- | --- | --- |
| Pac-Man Minimax defense | Normal ghost within distance 5, or unsafe offensive move | Preserve lives, maintain escape space |
| Normal ghost Minimax attack | Ghost not scared; active at every distance | Catch Pac-Man, restrict exits |
| Pac-Man scared-ghost attack | Scared target reachable before timer expiry | Capture scared ghost safely |
| Scared ghost Minimax defense | Ghost scared and Pac-Man within distance 5 | Avoid capture, survive timer |
| Scared ghost greedy defense | Ghost scared and Pac-Man beyond distance 5 | Flee without normal attack |

### Test Epic 3
```bash
python tests/run_tests.py
python examples/run_minimax_demo.py
```

Expected: All tests pass, Minimax demo survives 200 frames with high FPS.

---

## Epic 4: Approximate Q-Learning

Reinforcement Learning agent that learns to play Pac-Man through trial and error using weighted features and Temporal Difference (TD) learning.

### Features
- **Approximate Q-Learning**: Weight vector instead of Q-table for generalization
- **Feature-Based Evaluation**: 6 core metrics (bias, food proximity, danger, ghost distances)
- **Reward Shaping**: Scaled rewards to prevent weight oscillation
- **Epsilon-Greedy**: Gradual decay from exploration to exploitation

### Test Epic 4
```bash
python examples/train_q_agent.py
python examples/compare_agents.py
```

Expected: Agent shows improvement over training episodes, comparison plots generated.

---

## Development

### Adding New Agents
1. Create agent class in `src/agents/`
2. Implement `get_action(game_state)` method
3. Import and use in your scripts

### Running Tests During Development
```bash
python tests/run_tests.py
```

### Debugging
- Use `display=False` for headless testing
- Check feature extraction with `game.get_features()`
- Validate state with `game.get_state_arrays()`

---

## Requirements

- Python 3.8+
- pygame
- numpy
- matplotlib

Install with:
```bash
pip install -r requirements.txt
```

---

## Notes

- All imports use the `src.` prefix (e.g., `from src.core.config import ...`)
- Set `PYTHONPATH` if running scripts from outside the project directory
- The game maintains an invariant of 4 ghost identities (Blinky, Pinky, Inky, Clyde)
- Ghosts cannot occupy the same cell or immediately reverse direction unless blocked