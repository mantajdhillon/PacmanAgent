"""A* pathfinding agent for Epic 2.

The agent uses the GameState/GameBoard objects from Epic 1 and returns legal
Pac-Man moves that follow the shortest maze path to a pellet.
"""

import heapq
from collections import deque
from typing import Dict, Iterable, List, Optional, Sequence, Set, Tuple

from src.core.config import DIRECTIONS
from src.core.game_state import GameState


Position = Tuple[int, int]
Direction = Tuple[int, int]


class AStarPacmanAgent:
    """Pac-Man controller that uses A* to move toward pellets."""

    def __init__(self, include_power_pellets: bool = True):
        self.include_power_pellets = include_power_pellets

    def get_action(self, game_state: GameState) -> Direction:
        """Return the next direction Pac-Man should move.

        If there is no reachable pellet, the agent returns ``(0, 0)`` so the
        caller can safely skip movement.
        """
        path = self.path_to_nearest_pellet(game_state)
        if len(path) < 2:
            return (0, 0)

        current_x, current_y = path[0]
        next_x, next_y = path[1]
        return (next_x - current_x, next_y - current_y)

    def path_to_nearest_pellet(self, game_state: GameState) -> List[Position]:
        """Find the shortest maze path from Pac-Man to the nearest pellet."""
        if not game_state.pacman:
            return []

        start = game_state.pacman.position.to_tuple()
        targets = self._pellet_targets(game_state)
        targets.discard(start)
        if not targets:
            return [start]

        return self.find_path_to_any_target(game_state, start, targets)

    def find_path_to_any_target(
        self,
        game_state: GameState,
        start: Position,
        targets: Iterable[Position],
    ) -> List[Position]:
        """Run A* from ``start`` until the closest reachable target is found."""
        target_set = set(targets)
        if start in target_set:
            return [start]

        open_heap = []
        heapq.heappush(open_heap, (0, 0, start))

        came_from: Dict[Position, Position] = {}
        g_score: Dict[Position, int] = {start: 0}
        closed: Set[Position] = set()
        tie_breaker = 0

        while open_heap:
            _, _, current = heapq.heappop(open_heap)
            if current in closed:
                continue

            if current in target_set:
                return self._reconstruct_path(came_from, current)

            closed.add(current)

            for neighbor in self._neighbors(game_state, current):
                tentative_g = g_score[current] + 1
                if tentative_g >= g_score.get(neighbor, float("inf")):
                    continue

                came_from[neighbor] = current
                g_score[neighbor] = tentative_g
                tie_breaker += 1
                f_score = tentative_g + self._nearest_manhattan(neighbor, target_set)
                heapq.heappush(open_heap, (f_score, tie_breaker, neighbor))

        return [start]

    def find_path(self, game_state: GameState, start: Position, goal: Position) -> List[Position]:
        """Run A* from ``start`` to one specific goal."""
        return self.find_path_to_any_target(game_state, start, [goal])

    def maze_distance(self, game_state: GameState, start: Position, goal: Position) -> Optional[int]:
        """Return exact shortest-path distance through the maze using BFS."""
        if start == goal:
            return 0

        visited = {start}
        queue = deque([(start, 0)])

        while queue:
            current, distance = queue.popleft()
            for neighbor in self._neighbors(game_state, current):
                if neighbor in visited:
                    continue
                if neighbor == goal:
                    return distance + 1
                visited.add(neighbor)
                queue.append((neighbor, distance + 1))

        return None

    def estimate_mst_cost(self, game_state: GameState, points: Sequence[Position]) -> int:
        """Estimate pellet-clearing cost with a Manhattan minimum spanning tree.

        This is useful for Epic 2 experiments/reporting. It is not needed for
        the simple nearest-pellet controller, but it gives a stronger heuristic
        for reasoning about a set of remaining pellets.
        """
        if len(points) <= 1:
            return 0

        remaining = set(points[1:])
        connected = {points[0]}
        total_cost = 0

        while remaining:
            best_edge = None
            best_cost = float("inf")
            for source in connected:
                for target in remaining:
                    cost = self._manhattan(source, target)
                    if cost < best_cost:
                        best_cost = cost
                        best_edge = target

            total_cost += int(best_cost)
            connected.add(best_edge)
            remaining.remove(best_edge)

        return total_cost

    def _pellet_targets(self, game_state: GameState) -> Set[Position]:
        targets = set(game_state.board.get_pellets())
        if self.include_power_pellets:
            targets.update(game_state.board.get_power_pellets())
        return targets

    def _neighbors(self, game_state: GameState, position: Position) -> List[Position]:
        x, y = position
        neighbors = []
        for dx, dy in DIRECTIONS:
            next_pos = (x + dx, y + dy)
            if not game_state.board.is_wall(*next_pos):
                neighbors.append(next_pos)
        return neighbors

    def _reconstruct_path(self, came_from: Dict[Position, Position], current: Position) -> List[Position]:
        path = [current]
        while current in came_from:
            current = came_from[current]
            path.append(current)
        path.reverse()
        return path

    def _nearest_manhattan(self, position: Position, targets: Set[Position]) -> int:
        return min(self._manhattan(position, target) for target in targets)

    @staticmethod
    def _manhattan(a: Position, b: Position) -> int:
        return abs(a[0] - b[0]) + abs(a[1] - b[1])
