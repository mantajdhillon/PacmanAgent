# Pac-Man Agent: AI Learning Through Game Playing

Autonomous Pac-Man agent implementing A*, Minimax, and Q-Learning for CMPT 310.

## Quick Start

### Installation
```bash
pip install -r requirements.txt
```

### Run the Game with UI
```bash
python play_game.py
```

### Run Tests
```bash
python run_tests.py
```

## Project Overview

This project develops an intelligent Pac-Man agent that combines multiple AI techniques:
- **A*** for pathfinding to pellets
- **Minimax** with Alpha-Beta pruning for ghost behavior modeling
- **Q-Learning** for long-term strategy optimization

The agent must balance short-term rewards (collecting pellets) with long-term survival (avoiding ghosts).

---

## Python Files - What Each Does

### Core Game Files

#### `game_engine.py` - Main Game Engine (450+ lines)
The complete Pac-Man game implementation with Pygame rendering.
- **What it does**: Manages the game loop, board rendering, movement, collisions
- **Key classes**: `PacmanGame`
- **Key methods**: `move_pacman()`, `move_ghost()`, `render()`, `check_collisions()`
- **Use it for**: Creating game instances and controlling gameplay

#### `game_state.py` - Game State Management (250+ lines)
Data structures that represent the game state (positions, scores, board layout).
- **What it does**: Stores all game data - Pac-Man position, ghost positions, pellets, scoring
- **Key classes**: `Position`, `PacmanState`, `GhostState`, `GameBoard`, `GameState`
- **Use it for**: Understanding game data structure, accessing board and entity information

#### `feature_extractor.py` - Feature Extraction System (300+ lines)
Converts game state into multiple formats for AI algorithms.
- **What it does**: Extracts features from game state in 3 formats:
  - Dictionaries (structured data)
  - NumPy arrays (for computation)
  - Normalized vectors (for neural networks)
- **Key classes**: `FeatureExtractor`
- **Key methods**: `extract_all_features()`, `get_state_vector()`, `extract_distance_features()`
- **Use it for**: Getting game state data in format your AI needs

#### `config.py` - Configuration Constants (50 lines)
Centralized constants for board size, colors, game settings.
- **What it does**: Defines board dimensions (19×21), colors, directions, FPS
- **Use it for**: Changing game settings (board size, speed, colors)

### Testing & Demo Files

#### `play_game.py` - Interactive Game with Keyboard Control
Play the game manually with Pygame UI.
- **What it does**: Opens a game window where you can play Pac-Man
- **How to run**: `python play_game.py`
- **Controls**:
  - Arrow keys: Move Pac-Man
  - SPACE: Pause/Resume
  - R: Restart
  - ESC: Quit
- **Output**: Live game board with score, lives, pellet count

#### `run_tests.py` - Automated Test Suite (10 Tests)
Validates that all infrastructure works correctly.
- **What it does**: Runs 19 automated tests covering:
  - Game initialization
  - Feature extraction (all 5 types)
  - State arrays and vectors
  - Movement mechanics
  - Pellet collection
  - Collision detection
  - Win condition
  - Four-ghost lifecycle and long-run cache behavior
  - Approximate Q-Learning weight initialization and TD updates
  - Epsilon-greedy action selection and terminal state handling
- **How to run**: `python run_tests.py`
- **Output**: Test results showing "[PASS] ALL TESTS PASSED!" or failures

#### `example_usage.py` - 8 Working Examples
Demonstrates how to use each component of the system.
- **What it does**: Shows 8 examples:
  1. Basic game setup
  2. Feature extraction
  3. State arrays
  4. State vectors
  5. Board navigation
  6. Pellet tracking
  7. Game state components
  8. Game simulation
- **How to run**: `python example_usage.py`
- **Output**: Example code execution and results

#### `demo_testing_options.py` - Testing Overview Demo
Quick demo showing all testing capabilities.
- **What it does**: Demonstrates game with UI, headless simulation, and all features
- **How to run**: `python demo_testing_options.py`
- **Output**: Overview of game initialization and feature extraction

---

## How to Use

### Option 1: Play the Game
```bash
python play_game.py
```
- Opens Pygame window with interactive game
- Use arrow keys to move Pac-Man
- See real-time score and pellet counter

### Option 2: Run Tests
```bash
python run_tests.py
```
- Validates all features work correctly
- ~30 seconds to complete
- Shows "[PASS] ALL TESTS PASSED!" if everything works

### Option 3: See Examples
```bash
python example_usage.py
```
- Demonstrates all features
- Shows different ways to extract game state
- ~10 seconds to complete

### Option 4: In Your Own Code
```python
from game_engine import PacmanGame
from config import UP, DOWN, LEFT, RIGHT

# Create game (headless - no UI)
game = PacmanGame(display=False)
game.reset()

# Get game state in 3 formats
features = game.get_features()           # Dictionary
state_arrays = game.get_state_arrays()   # NumPy arrays
state_vector = game.get_state_vector()   # Neural network vector

# Move Pac-Man
game.move_pacman(UP)

# Get info
print(f"Score: {game.game_state.pacman.score}")
print(f"Pellets: {len(game.board.pellets)}")
```

---

## Game Information

**Board Size**: 19 × 21 cells  
**Pac-Man**: Starts at center (9, 10) with 3 lives  
**Ghosts**: 4 independent ghosts (Blinky, Pinky, Inky, Clyde)  
**Pellets**: 280 regular pellets + 4 power pellets at corners  

**Scoring**:
- Regular pellet: +10 points
- Power pellet: +50 points
- Eat scared ghost: +200 points

**Win Condition**: Collect all pellets  
**Lose Condition**: Lose all 3 lives

---

## 


## Epic 2: A* Pathfinding

Epic 2 introduces an A* pathfinding agent (`astar_agent.py`) that automatically guides Pac-Man to the nearest pellet or power pellet using the A* search algorithm with a Manhattan distance heuristic.

### Features

- A* search for shortest path planning
- Manhattan distance heuristic
- Exact maze-distance helper
- Minimum spanning tree (MST) cost estimate for future optimization
- Autoplay mode integrated into `play_game.py`
- Automated A* tests added to `run_tests.py`

### Test Epic 2

```bash
python run_tests.py
python run_astar_demo.py
```

Expected results:

- All automated tests pass.
- The A* demo successfully collects every pellet and wins the game without ghosts.

---

##


## Epic 3: Adversarial Search (Minimax)

Epic 3 introduces a defensive adversarial agent (`minimax_agent.py`) designed to calculate optimal evasion routes when under threat. It utilizes a multi-agent Minimax algorithm heavily optimized for real-time Pygame execution, allowing Pac-Man to dynamically prioritize survival over point collection.

### Features

* Multi-agent Minimax search tree handling up to 3 simultaneous actors (Max: Pac-Man, Min: Ghosts).
* Alpha-Beta Pruning to mathematically sever suboptimal branches and reduce time complexity.
* Transposition Tables (Memoization) with custom state-hashing to eliminate redundant state evaluations.
* Active Threat Pruning, dynamically reducing the branching factor by ignoring ghosts outside a 6-tile radius.
* Hybrid AI Orchestrator integrated into `play_game.py` (toggles between A* Offense and Minimax Defense automatically).
* Custom terminal evaluation function penalizing death and maximizing Manhattan distance from threats.
* Automated defensive survival test added to `run_tests.py`.
* Headless adversarial simulation and FPS benchmarking via `run_minimax_demo.py`.

### Test Epic 3

```bash
python run_tests.py
python run_minimax_demo.py
```

Expected results:
* All automated tests pass, specifically `test_minimax_defense` confirming Pac-Man successfully avoids a manufactured mortal threat.
* The Minimax headless demo successfully evades ghosts for 200 frames while outputting a high FPS simulation speed, proving the pruning optimizations prevent computational freezing.
* When running `python play_game.py` with the Hybrid Autoplay toggle ('A'), the UI smoothly transitions between A* Offense (Green) and Minimax Defense (Red) without interpreter lag.
---

## Epic 4: Approximate Q-Learning

Epic 4 introduces a Reinforcement Learning agent in the form of Q-Learning that learns to play Pac-Man through trial and error. Instead of relying on hardcoded decision trees or deep search algorithms, this agent evaluates the board using weighted features and dynamically updates its strategy over time using Temporal Difference (TD) learning.

### Features

- **Approximate Q-Learning:** Uses a weight vector instead of a massive, memory-heavy Q-table to instantly generalize learning across the entire 21x21 board.
- **Feature-Based Evaluation:** Distills complex board states into 6 core metrics (Bias, Eats Food, Nearest Pellet, Imminent Danger, Scared Ghost Distance, and Continuous Ghost Distance).
- **Reward Shaping & Normalization:** Employs scaled rewards and penalties to prevent mathematical weight oscillation and encourage aggressive, decisive pathfinding.
- **Epsilon-Greedy Exploration:** Gradually decays random exploration during training to smoothly transition the agent into pure, optimized exploitation.

### Q-Learning Files

| File | Responsibility |
| --- | --- |
| `approximate_q_agent.py` | Core Q-learning math, TD weight updates, and epsilon-greedy action selection. |
| `train_q_agent.py` | The training environment, reward shaping logic, and headless performance testing. |

### Training and Testing

To train the Q-Learning agent and view its final performance summary, run:

```bash
python train_q_agent.py
