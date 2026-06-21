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
