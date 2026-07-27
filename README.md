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

#### `game_engine.py` - Main Game Engine (400+ lines)
The complete Pac-Man game implementation with Pygame rendering.
- **What it does**: Manages the game loop, board rendering, movement, collisions
- **Key classes**: `PacmanGame`
- **Key methods**: `move_pacman()`, `move_ghost()`, `render()`, `check_collisions()`
- **Use it for**: Creating game instances and controlling gameplay

#### `game_state.py` - Game State Management (350+ lines)
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

#### `config.py` - Shared Configuration
Centralized constants and bounded-cache support.
- **What it does**: Defines the 21 x 21 board, colors, directions, FPS, and the bounded LRU cache used by Minimax agents
- **Use it for**: Changing game settings and understanding shared search-cache behavior

### Testing & Demo Files

#### `play_game.py` - Interactive Game with Keyboard Control
Play the game manually or use autonomous control with the Pygame UI.
- **What it does**: Opens a game window for manual or hybrid AI play
- **How to run**: `python play_game.py`
- **Controls**:
  - Arrow keys: Move Pac-Man
  - A: Toggle manual/autoplay control
  - SPACE: Pause/Resume
  - R: Restart
  - ESC: Quit
- **Output**: Live game board with score, lives, pellet count

#### `run_tests.py` - Automated Test Suite (15 Tests)
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
  - A* pathfinding
  - All four Epic 3 Minimax requirements
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

Epic 3 extends the project from a single defensive controller into a
four-mode adversarial system. Pac-Man and the four named ghosts change
objectives according to ghost state, maze distance, scared-timer feasibility,
and immediate survival risk.

The implementation uses maze distance rather than straight-line distance, so
walls and corridors affect every threat, chase, and escape decision.

### Decision Priority

The autoplay controller in `play_game.py` uses the following strict priority:

1. **Safe scared-ghost attack** - pursue an edible ghost only when it is
   reachable before its timer expires and no nearby normal ghost makes the
   chase unsafe.
2. **Safe point collection** - use A* to collect the nearest pellet or power
   pellet when no normal ghost is within the defensive threshold.
3. **Minimax defense** - activate when a normal ghost is within maze distance
   5 or when the proposed offensive action is unsafe.
4. **Emergency survival** - if the preferred Minimax action is unsafe, choose
   the legal action with the strongest immediate survival and escape-space
   score.

Every offensive candidate passes a shared safety gate. The gate rejects
illegal moves, immediate collisions, life loss, insufficient ghost separation,
and positions that a normal ghost can capture on its next move.

### Mode Summary

| Controller | Activation | Primary objective |
| --- | --- | --- |
| Pac-Man Minimax defense | A normal ghost is within maze distance 5, or an offensive move is unsafe | Preserve lives, increase separation, maintain escape space, and avoid dead ends |
| Normal ghost Minimax attack | The ghost is not scared; active at every distance | Catch Pac-Man, reduce maze distance, restrict exits, and coordinate approach directions |
| Pac-Man scared-ghost attack | A scared target is reachable before timer expiry and the chase is safe from normal ghosts | Capture the selected scared ghost without sacrificing a life |
| Scared ghost Minimax defense | The ghost is scared and Pac-Man is within maze distance 5 | Avoid capture, increase separation, avoid traps, and survive until timer expiry |
| Scared ghost greedy defense | The ghost is scared and Pac-Man is farther than distance 5 | Continue fleeing cheaply without reverting to normal attack |

### 1. Updated Pac-Man Minimax Defense

Implemented in `minimax_agent.py`.

- Models Pac-Man as the maximizing agent and nearby normal ghosts as
  minimizing agents.
- Uses an inclusive threat radius of **5 maze steps**.
- Treats terminal loss and lost lives as more important than score or food.
- Rewards maze separation, legal escape actions, reachable local space, and
  secondary food progress.
- Penalizes collision, constrained corridors, dead ends, immediate reversal,
  and recently visited positions.
- Filters unsafe root actions whenever at least one survivable action exists.
- Includes `get_safest_action()` for emergency fallback when an objective move
  cannot be accepted.
- Uses Alpha-Beta pruning, state memoization, action ordering, and a bounded
  least-recently-used distance cache for responsive long-running play.

### 2. Updated Normal-Ghost Minimax Attack

Implemented in `minimax_ghost.py`.

- Every non-scared ghost attacks at every distance.
- Pac-Man is modeled as the minimizing opponent; the controlled ghost
  maximizes the shared ghost-team utility.
- Capture and reduction of Pac-Man's remaining lives dominate positional
  rewards.
- The evaluation rewards reduced maze distance, team pressure, and fewer safe
  Pac-Man escape routes.
- Stable named roles create multiple approach directions:
  - **Blinky** approaches from Pac-Man's left.
  - **Pinky** approaches from above.
  - **Inky** approaches from Pac-Man's right.
  - **Clyde** approaches from below.
- Each ghost phase freezes distinct walkable interception targets so earlier
  ghost movement does not change later assignments.
- Teammate clustering is penalized to prevent all ghosts from following the
  same route.
- Teammate cells are excluded during legal-action planning. If a teammate
  blocks the only forward corridor, a safe reversal is permitted instead of
  allowing both ghosts to freeze.

### 3. Pac-Man Scared-Ghost Attack Mode

Implemented in `minimax_attack_agent.py`.

- Activates only when at least one ghost is scared.
- Requires maze distance plus a timer safety margin to fit inside the
  remaining scared time.
- Cancels the chase when a normal ghost is within the safety radius.
- Prefers targets with greater timer slack, then shorter capture distance.
- Locks a viable target to prevent Pac-Man from switching targets every frame
  and running in circles.
- Models the target ghost's evasive response and nearby normal-ghost threats.
- Cancels the attack instead of restoring unsafe chase actions when no safe
  chase move exists.
- Penalizes reversal and recently visited positions.

### 4. Scared-Ghost Minimax Defense Mode

Implemented in `minimax_defense_ghost.py`.

- Activates Minimax when the controlled ghost is scared and Pac-Man is within
  maze distance 5.
- Maximizes survival, distance from Pac-Man, open escape space, and the number
  of exits.
- Penalizes capture, dead ends, corridors vulnerable to interception, and
  recently visited positions.
- Treats surviving until the scared timer expires as the terminal objective.
- Uses a cheaper greedy flee policy outside distance 5.
- The distant fallback still moves away from Pac-Man and never changes to
  normal attack while the ghost remains scared.

### Epic 3 Files

| File | Responsibility |
| --- | --- |
| `minimax_agent.py` | Normal Pac-Man defense, shared safety gate, and emergency survival |
| `minimax_ghost.py` | Coordinated normal-ghost attack |
| `minimax_attack_agent.py` | Safe Pac-Man pursuit of scared ghosts |
| `minimax_defense_ghost.py` | Scared-ghost Minimax and distant flee fallback |
| `play_game.py` | Live mode routing and UI integration |
| `run_minimax_demo.py` | Headless integrated simulation and performance reporting |
| `run_tests.py` | Behavioral and lifecycle regression tests |

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
