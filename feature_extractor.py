"""Feature extraction module for converting game state to feature vectors."""

import numpy as np
from typing import List, Tuple, Dict, Optional
from game_state import GameState, Position
from config import BOARD_WIDTH, BOARD_HEIGHT


class FeatureExtractor:
    """Extracts and processes features from game state."""

    def __init__(self, board_width: int = BOARD_WIDTH, board_height: int = BOARD_HEIGHT):
        self.board_width = board_width
        self.board_height = board_height

    def extract_pacman_features(self, game_state: GameState) -> Dict:
        """Extract Pac-Man specific features."""
        pacman = game_state.pacman
        if not pacman:
            return {}

        return {
            "position": pacman.position.to_tuple(),
            "x": pacman.position.x,
            "y": pacman.position.y,
            "direction": pacman.direction,
            "score": pacman.score,
            "lives": pacman.lives,
        }

    def extract_ghost_features(self, game_state: GameState) -> List[Dict]:
        """Extract features for all ghosts."""
        ghost_features = []
        for i, ghost in enumerate(game_state.ghosts):
            features = {
                "id": i,
                "name": ghost.name,
                "position": ghost.position.to_tuple(),
                "x": ghost.position.x,
                "y": ghost.position.y,
                "color": ghost.color,
                "direction": ghost.direction,
                "scared": ghost.scared,
                "scared_timer": ghost.scared_timer,
            }
            ghost_features.append(features)
        return ghost_features

    def extract_pellet_features(self, game_state: GameState) -> Dict:
        """Extract pellet and power pellet features."""
        pellets = game_state.board.get_pellets()
        power_pellets = game_state.board.get_power_pellets()

        return {
            "pellet_count": len(pellets),
            "power_pellet_count": len(power_pellets),
            "pellets": pellets,
            "power_pellets": power_pellets,
            "total_pellets_eaten": game_state.pellets_eaten,
            "total_power_pellets_eaten": game_state.power_pellets_eaten,
        }

    def extract_board_features(self, game_state: GameState) -> Dict:
        """Extract board layout features."""
        board_array = game_state.board.get_board_array()

        return {
            "width": game_state.board.width,
            "height": game_state.board.height,
            "board": board_array,
            "wall_count": np.sum(board_array == 1),
        }

    def extract_distance_features(self, game_state: GameState) -> Dict:
        """Extract distance-based features (nearest pellet, nearest ghost, etc.)."""
        pacman = game_state.pacman
        if not pacman:
            return {}

        pacman_pos = pacman.position.to_tuple()
        pellets = game_state.board.get_pellets()
        ghosts = game_state.ghosts

        # Nearest pellet distance
        nearest_pellet_dist = float('inf')
        nearest_pellet = None
        if pellets:
            distances = [self._manhattan_distance(pacman_pos, p) for p in pellets]
            nearest_pellet_dist = min(distances)
            nearest_pellet = pellets[distances.index(nearest_pellet_dist)]

        # Nearest ghost distance
        nearest_ghost_dist = float('inf')
        nearest_ghost_idx = -1
        for i, ghost in enumerate(ghosts):
            dist = self._manhattan_distance(pacman_pos, ghost.position.to_tuple())
            if dist < nearest_ghost_dist:
                nearest_ghost_dist = dist
                nearest_ghost_idx = i

        # Nearest power pellet
        nearest_power_pellet_dist = float('inf')
        nearest_power_pellet = None
        power_pellets = game_state.board.get_power_pellets()
        if power_pellets:
            distances = [self._manhattan_distance(pacman_pos, p) for p in power_pellets]
            nearest_power_pellet_dist = min(distances)
            nearest_power_pellet = power_pellets[distances.index(nearest_power_pellet_dist)]

        return {
            "nearest_pellet_distance": nearest_pellet_dist if nearest_pellet_dist != float('inf') else -1,
            "nearest_pellet": nearest_pellet,
            "nearest_ghost_distance": nearest_ghost_dist if nearest_ghost_dist != float('inf') else -1,
            "nearest_ghost_index": nearest_ghost_idx,
            "nearest_power_pellet_distance": nearest_power_pellet_dist if nearest_power_pellet_dist != float('inf') else -1,
            "nearest_power_pellet": nearest_power_pellet,
        }

    def extract_all_features(self, game_state: GameState) -> Dict:
        """Extract all available features from game state."""
        features = {
            "pacman": self.extract_pacman_features(game_state),
            "ghosts": self.extract_ghost_features(game_state),
            "pellets": self.extract_pellet_features(game_state),
            "board": self.extract_board_features(game_state),
            "distances": self.extract_distance_features(game_state),
            "game_over": game_state.game_over,
            "game_won": game_state.game_won,
        }
        return features

    def get_state_vector(self, game_state: GameState) -> np.ndarray:
        """
        Convert game state to a single feature vector for neural networks.
        Returns a 1D numpy array of normalized features.
        """
        features = self.extract_all_features(game_state)
        pacman = game_state.pacman
        if not pacman:
            return np.array([])

        vector = []

        # Pac-Man position (normalized)
        vector.append(pacman.position.x / self.board_width)
        vector.append(pacman.position.y / self.board_height)

        # Direction (one-hot encoding)
        direction_map = {(0, -1): [1, 0, 0, 0], (0, 1): [0, 1, 0, 0], (-1, 0): [0, 0, 1, 0], (1, 0): [0, 0, 0, 1], (0, 0): [0, 0, 0, 0]}
        vector.extend(direction_map.get(pacman.direction, [0, 0, 0, 0]))

        # Ghost positions and scared states
        for ghost in game_state.ghosts:
            vector.append(ghost.position.x / self.board_width)
            vector.append(ghost.position.y / self.board_height)
            vector.append(1 if ghost.scared else 0)

        # Pellet information
        vector.append(features["pellets"]["pellet_count"] / max(1, features["pellets"]["pellet_count"] + features["pellets"]["power_pellet_count"]))
        vector.append(features["pellets"]["power_pellet_count"] / max(1, features["pellets"]["pellet_count"] + features["pellets"]["power_pellet_count"]))

        # Distance features
        vector.append(min(1.0, features["distances"]["nearest_pellet_distance"] / self.board_width) if features["distances"]["nearest_pellet_distance"] > 0 else 0)
        vector.append(min(1.0, features["distances"]["nearest_ghost_distance"] / self.board_width) if features["distances"]["nearest_ghost_distance"] > 0 else 0)

        # Score and lives
        vector.append(pacman.score / 10000)
        vector.append(pacman.lives / 3)

        return np.array(vector, dtype=np.float32)

    @staticmethod
    def _manhattan_distance(pos1: Tuple[int, int], pos2: Tuple[int, int]) -> int:
        """Calculate Manhattan distance between two positions."""
        return abs(pos1[0] - pos2[0]) + abs(pos1[1] - pos2[1])

    @staticmethod
    def _euclidean_distance(pos1: Tuple[int, int], pos2: Tuple[int, int]) -> float:
        """Calculate Euclidean distance between two positions."""
        return ((pos1[0] - pos2[0]) ** 2 + (pos1[1] - pos2[1]) ** 2) ** 0.5

    def get_reachable_positions(self, game_state: GameState, pos: Tuple[int, int], max_distance: int) -> List[Tuple[int, int]]:
        """Get all reachable positions within max_distance (BFS)."""
        from collections import deque

        visited = set()
        queue = deque([(pos, 0)])
        reachable = []

        while queue:
            (x, y), dist = queue.popleft()

            if (x, y) in visited or dist > max_distance:
                continue

            if game_state.board.is_wall(x, y):
                continue

            visited.add((x, y))
            reachable.append((x, y))

            for dx, dy in [(0, 1), (0, -1), (1, 0), (-1, 0)]:
                next_pos = (x + dx, y + dy)
                if next_pos not in visited:
                    queue.append((next_pos, dist + 1))

        return reachable
