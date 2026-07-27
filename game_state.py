"""Game state data structures and utilities."""

import numpy as np
from dataclasses import dataclass
from typing import List, Tuple, Optional
from config import WALL, EMPTY, PELLET, POWER_PELLET


@dataclass
class Position:
    """Represents a position on the board."""
    x: int
    y: int

    def __eq__(self, other):
        if isinstance(other, tuple):
            return self.x == other[0] and self.y == other[1]
        return self.x == other.x and self.y == other.y

    def __hash__(self):
        return hash((self.x, self.y))

    def to_tuple(self) -> Tuple[int, int]:
        return (self.x, self.y)


@dataclass
class PacmanState:
    """Represents Pac-Man's state."""
    position: Position
    direction: Tuple[int, int] = (0, 0)
    score: int = 0
    lives: int = 3


@dataclass
class GhostState:
    """Represents a ghost's state."""
    position: Position
    color: str
    direction: Tuple[int, int] = (0, 0)
    scared: bool = False
    scared_timer: int = 0
    respawn_timer: int = 0
    name: str = "ghost"
    start_position: Optional[Tuple[int, int]] = None


class GameBoard:
    """Represents the game board with walls, pellets, and power pellets."""

    def __init__(self, width: int, height: int):
        self.width = width
        self.height = height
        self.board = np.zeros((height, width), dtype=np.int32)
        self.pellets = set()
        self.power_pellets = set()

    def add_wall(self, x: int, y: int):
        """Add a wall at position."""
        if 0 <= x < self.width and 0 <= y < self.height:
            self.board[y, x] = WALL

    def add_pellet(self, x: int, y: int):
        """Add a regular pellet at position."""
        if 0 <= x < self.width and 0 <= y < self.height:
            self.pellets.add((x, y))
            self.board[y, x] = PELLET

    def add_power_pellet(self, x: int, y: int):
        """Add a power pellet at position."""
        if 0 <= x < self.width and 0 <= y < self.height:
            self.power_pellets.add((x, y))
            self.board[y, x] = POWER_PELLET

    def remove_pellet(self, x: int, y: int):
        """Remove a pellet at position."""
        self.pellets.discard((x, y))
        if (x, y) not in self.power_pellets:
            self.board[y, x] = EMPTY

    def remove_power_pellet(self, x: int, y: int):
        """Remove a power pellet at position."""
        self.power_pellets.discard((x, y))
        if (x, y) not in self.pellets:
            self.board[y, x] = EMPTY

    def is_wall(self, x: int, y: int) -> bool:
        """Check if position is a wall."""
        if not (0 <= x < self.width and 0 <= y < self.height):
            return True
        return self.board[y, x] == WALL

    def get_board_array(self) -> np.ndarray:
        """Get a copy of the board array."""
        return self.board.copy()

    def get_pellets(self) -> List[Tuple[int, int]]:
        """Get list of pellet positions."""
        return list(self.pellets)

    def get_power_pellets(self) -> List[Tuple[int, int]]:
        """Get list of power pellet positions."""
        return list(self.power_pellets)


class GameState:
    """Complete game state including board, Pac-Man, and ghosts."""

    def __init__(self, board: GameBoard):
        self.board = board
        self.pacman = None
        self.ghosts: List[GhostState] = []
        self.eaten_ghosts: List[Tuple[GhostState, int]] = []  # List of (ghost, respawn_timer)
        self.pellets_eaten = 0
        self.power_pellets_eaten = 0
        self.game_over = False
        self.game_won = False

    def set_pacman(self, pacman: PacmanState):
        """Set Pac-Man state."""
        self.pacman = pacman

    def add_ghost(self, ghost: GhostState):
        """Add a ghost to the game."""
        if any(existing.name == ghost.name for existing in self.ghosts):
            raise ValueError(f"Duplicate active ghost identity: {ghost.name}")
        self.ghosts.append(ghost)

    def update_pacman_position(self, new_x: int, new_y: int):
        """Update Pac-Man's position."""
        if self.pacman:
            self.pacman.position = Position(new_x, new_y)

    def update_ghost_position(self, ghost_index: int, new_x: int, new_y: int):
        """Update a ghost's position."""
        if 0 <= ghost_index < len(self.ghosts):
            self.ghosts[ghost_index].position = Position(new_x, new_y)

    def update_ghost_scared_state(self, ghost_index: int, scared: bool, timer: int = 0):
        """Update scared state for a ghost."""
        if 0 <= ghost_index < len(self.ghosts):
            self.ghosts[ghost_index].scared = scared
            self.ghosts[ghost_index].scared_timer = timer

    def extract_features(self) -> dict:
        """
        Extract features from current game state.
        Returns a dictionary with all relevant game state information.
        """
        features = {
            "pacman_position": self.pacman.position.to_tuple() if self.pacman else None,
            "pacman_direction": self.pacman.direction if self.pacman else None,
            "pacman_score": self.pacman.score if self.pacman else 0,
            "pacman_lives": self.pacman.lives if self.pacman else 0,
            "ghosts": [],
            "pellets": self.board.get_pellets(),
            "power_pellets": self.board.get_power_pellets(),
            "board": self.board.get_board_array(),
            "pellets_eaten": self.pellets_eaten,
            "power_pellets_eaten": self.power_pellets_eaten,
            "game_over": self.game_over,
            "game_won": self.game_won,
        }

        for ghost in self.ghosts:
            ghost_info = {
                "position": ghost.position.to_tuple(),
                "color": ghost.color,
                "direction": ghost.direction,
                "scared": ghost.scared,
                "scared_timer": ghost.scared_timer,
                "name": ghost.name,
            }
            features["ghosts"].append(ghost_info)

        return features

    def get_state_arrays(self) -> dict:
        """
        Get state as numpy arrays for efficient computation.
        Returns:
            dict with arrays for board, pacman_pos, ghost_positions, pellet_map
        """
        board_array = self.board.get_board_array()

        pacman_pos = np.array(self.pacman.position.to_tuple()) if self.pacman else np.array([0, 0])

        ghost_positions = np.array([ghost.position.to_tuple() for ghost in self.ghosts])
        ghost_scared_states = np.array([ghost.scared for ghost in self.ghosts])

        pellet_map = np.zeros_like(board_array, dtype=np.int32)
        for px, py in self.board.pellets:
            pellet_map[py, px] = 1

        power_pellet_map = np.zeros_like(board_array, dtype=np.int32)
        for ppx, ppy in self.board.power_pellets:
            power_pellet_map[ppy, ppx] = 1

        return {
            "board": board_array,
            "pacman_position": pacman_pos,
            "ghost_positions": ghost_positions,
            "ghost_scared": ghost_scared_states,
            "pellet_map": pellet_map,
            "power_pellet_map": power_pellet_map,
            "board_width": self.board.width,
            "board_height": self.board.height,
        }

    def clone(self):
        """
        Creates a fast, lightweight copy of the game state for search trees.
        """
        # Clone the board lightly.
        new_board = GameBoard(self.board.width, self.board.height)
        new_board.board = self.board.board.copy()

        # Sets are highly optimized in Python. Copying them is fast.
        new_board.pellets = set(self.board.pellets)
        new_board.power_pellets = set(self.board.power_pellets)

        # Initialize new state and primitives
        new_state = GameState(new_board)
        new_state.pellets_eaten = self.pellets_eaten
        new_state.power_pellets_eaten = self.power_pellets_eaten
        new_state.game_over = self.game_over
        new_state.game_won = self.game_won

        # Deep-copy respawning ghosts so search states cannot mutate live state.
        new_state.eaten_ghosts = [
            (
                GhostState(
                    position=Position(
                        ghost.position.x,
                        ghost.position.y,
                    ),
                    color=ghost.color,
                    direction=ghost.direction,
                    scared=ghost.scared,
                    scared_timer=ghost.scared_timer,
                    respawn_timer=ghost.respawn_timer,
                    name=ghost.name,
                    start_position=ghost.start_position,
                ),
                timer,
            )
            for ghost, timer in self.eaten_ghosts
        ]

        # Clone Pac-Man
        if self.pacman:
            new_state.pacman = PacmanState(
                position=Position(self.pacman.position.x, self.pacman.position.y),
                direction=self.pacman.direction,
                score=self.pacman.score,
                lives=self.pacman.lives
            )

        # Clone Ghosts
        for ghost in self.ghosts:
            new_ghost = GhostState(
                position=Position(ghost.position.x, ghost.position.y),
                color=ghost.color,
                direction=ghost.direction,
                scared=ghost.scared,
                scared_timer=ghost.scared_timer,
                respawn_timer=ghost.respawn_timer,
                name=ghost.name,
                start_position=ghost.start_position
            )
            new_state.ghosts.append(new_ghost)

        return new_state

    def get_legal_actions(self, agent_index: int) -> List[Tuple[int, int]]:
        """
        Returns a list of legal moves (dx, dy) that do not result in hitting a wall.
        agent_index: 0 for Pac-Man, 1+ for Ghosts.
        """
        if self.game_over or self.game_won:
            return []

        # Standard directions: UP, DOWN, LEFT, RIGHT
        directions = [(0, -1), (0, 1), (-1, 0), (1, 0)]
        legal_actions = []

        # Get the starting coordinates based on the agent
        if agent_index == 0:
            if not self.pacman: return []
            x, y = self.pacman.position.x, self.pacman.position.y
        else:
            ghost_idx = agent_index - 1
            if ghost_idx >= len(self.ghosts): return []
            x, y = self.ghosts[ghost_idx].position.x, self.ghosts[ghost_idx].position.y

        ghost_house_area = [(9, 10), (10, 10), (11, 10), (9, 9), (10, 9), (11, 9)]

        # Validate which moves are open
        for dx, dy in directions:
            new_x, new_y = x + dx, y + dy
            if agent_index == 0 and (new_x, new_y) in ghost_house_area:
                continue

            if not self.board.is_wall(new_x, new_y):
                legal_actions.append((dx, dy))

        if agent_index > 0:
            ghost_index = agent_index - 1
            ghost = self.ghosts[ghost_index]

            # A teammate's cell is a blocked destination during planning, not
            # merely during execution. Otherwise Minimax repeatedly selects a
            # move that PacmanGame.move_ghost() rejects every frame.
            occupied_positions = {
                other.position.to_tuple()
                for index, other in enumerate(self.ghosts)
                if index != ghost_index
            }
            legal_actions = [
                action
                for action in legal_actions
                if (
                    ghost.position.x + action[0],
                    ghost.position.y + action[1],
                )
                not in occupied_positions
            ]

            # Match the classic no-immediate-reversal rule when another route
            # exists. If a teammate blocks the only forward corridor, reversal
            # remains legal so the ghost turns around instead of freezing.
            if len(legal_actions) > 1 and ghost.direction != (0, 0):
                reverse = (-ghost.direction[0], -ghost.direction[1])
                non_reverse = [
                    action
                    for action in legal_actions
                    if action != reverse
                ]
                if non_reverse:
                    legal_actions = non_reverse

        return legal_actions

    def generate_successor(self, agent_index: int, action: Tuple[int, int]):
        """
        Generates a new GameState simulating the given action for the specific agent.
        """
        # Spawn the isolated clone
        successor = self.clone()

        if successor.game_over or successor.game_won:
            return successor

        # Execute Pac-Man's Move (Index 0)
        if agent_index == 0:
            pacman = successor.pacman
            new_x = pacman.position.x + action[0]
            new_y = pacman.position.y + action[1]

            pacman.position = Position(new_x, new_y)
            pacman.direction = action
            pos_tuple = (new_x, new_y)

            # Resolve Pellet Logic
            if pos_tuple in successor.board.pellets:
                successor.board.remove_pellet(new_x, new_y)
                pacman.score += 10
                successor.pellets_eaten += 1
                if not successor.board.pellets and not successor.board.power_pellets:
                    successor.game_won = True

            elif pos_tuple in successor.board.power_pellets:
                successor.board.remove_power_pellet(new_x, new_y)
                pacman.score += 50
                successor.power_pellets_eaten += 1
                for ghost in successor.ghosts:
                    ghost.scared = True
                    ghost.scared_timer = 150
                if not successor.board.pellets and not successor.board.power_pellets:
                    successor.game_won = True

            # Resolve Ghost Collision (Pac-Man moved into a ghost)
            for ghost in successor.ghosts:
                if ghost.position.to_tuple() == pos_tuple:
                    if ghost.scared:
                        successor.ghosts.remove(ghost)
                        pacman.score += 200
                        break
                    else:
                        pacman.lives -= 1
                        if pacman.lives <= 0:
                            successor.game_over = True
                        break

        # Execute Ghost's Move (Index > 0)
        else:
            ghost_idx = agent_index - 1
            if ghost_idx < len(successor.ghosts):
                ghost = successor.ghosts[ghost_idx]
                new_x = ghost.position.x + action[0]
                new_y = ghost.position.y + action[1]

                ghost.position = Position(new_x, new_y)
                ghost.direction = action
                pos_tuple = (new_x, new_y)

                # Resolve Ghost Collision (Ghost moved into Pac-Man)
                if successor.pacman and pos_tuple == successor.pacman.position.to_tuple():
                    if ghost.scared:
                        successor.ghosts.remove(ghost)
                        successor.pacman.score += 200
                    else:
                        successor.pacman.lives -= 1
                        if successor.pacman.lives <= 0:
                            successor.game_over = True

        return successor
