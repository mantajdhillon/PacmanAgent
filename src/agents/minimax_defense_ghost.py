import math
from collections import deque
from src.core.config import BoundedCache


class MinimaxScaredGhostDefenseAgent:
    """
    Defensive controller for one scared ghost.

    Within the activation distance, the scared ghost maximizes survival utility
    while Pac-Man minimizes it by choosing its best capture response. Outside
    that distance, a cheaper greedy flee policy continues moving away.
    """

    def __init__(self, activation_distance: int = 5, escape_horizon: int = 4):
        self.activation_distance = activation_distance
        self.escape_horizon = escape_horizon
        self.distance_cache = BoundedCache(max_size=8192)
        self.board_signature_cache = {}
        self.recent_positions = {}

    def _history_for(self, ghost_name):
        """Return a short per-ghost history used to break local oscillations."""
        if ghost_name not in self.recent_positions:
            self.recent_positions[ghost_name] = deque(maxlen=10)
        return self.recent_positions[ghost_name]

    def _board_signature(self, board):
        """Return a wall-only signature shared by cloned search boards."""
        board_id = id(board)
        if board_id not in self.board_signature_cache:
            self.board_signature_cache[board_id] = (
                board.width,
                board.height,
                (board.board == 1).tobytes(),
            )
        return self.board_signature_cache[board_id]

    def _maze_distance(self, board, start, target):
        """Return shortest walkable distance, or math.inf when unreachable."""
        if start == target:
            return 0

        endpoints = tuple(sorted((start, target)))
        cache_key = (self._board_signature(board), endpoints)
        if cache_key in self.distance_cache:
            return self.distance_cache[cache_key]

        queue = deque([(start, 0)])
        visited = {start}

        while queue:
            (x, y), distance = queue.popleft()
            for dx, dy in ((0, -1), (0, 1), (-1, 0), (1, 0)):
                neighbor = (x + dx, y + dy)
                if neighbor in visited or board.is_wall(*neighbor):
                    continue
                if neighbor == target:
                    result = distance + 1
                    self.distance_cache[cache_key] = result
                    return result
                visited.add(neighbor)
                queue.append((neighbor, distance + 1))

        self.distance_cache[cache_key] = math.inf
        return math.inf

    @staticmethod
    def _find_ghost(state, ghost_name):
        """Find a ghost by stable name after successor cloning."""
        return next(
            (ghost for ghost in state.ghosts if ghost.name == ghost_name),
            None,
        )

    def _pacman_distance(self, state, ghost):
        """Return maze distance from one ghost to Pac-Man."""
        if not state.pacman or ghost is None:
            return math.inf
        return self._maze_distance(
            state.board,
            ghost.position.to_tuple(),
            state.pacman.position.to_tuple(),
        )

    def is_minimax_active(self, state, ghost_index: int) -> bool:
        """Whether this scared ghost should use full defensive Minimax."""
        if ghost_index < 0 or ghost_index >= len(state.ghosts):
            return False

        ghost = state.ghosts[ghost_index]
        return (
            ghost.scared
            and self._pacman_distance(state, ghost)
            <= self.activation_distance
        )

    def _safe_ghost_actions(self, state, ghost_index, legal_actions):
        """Remove actions that make the scared ghost collide with Pac-Man."""
        if not state.pacman:
            return legal_actions

        ghost = state.ghosts[ghost_index]
        pacman_position = state.pacman.position.to_tuple()
        safe_actions = [
            action
            for action in legal_actions
            if (
                ghost.position.x + action[0],
                ghost.position.y + action[1],
            )
            != pacman_position
        ]
        return safe_actions if safe_actions else legal_actions

    def _local_escape_space(self, state, ghost) -> int:
        """
        Count locally reachable cells while treating Pac-Man as an obstacle.

        Small values identify dead ends and narrow pockets where Pac-Man can
        intercept the scared ghost.
        """
        if ghost is None:
            return 0

        start = ghost.position.to_tuple()
        blocked = (
            {state.pacman.position.to_tuple()}
            if state.pacman
            else set()
        )
        queue = deque([(start, 0)])
        visited = {start}

        while queue:
            (x, y), distance = queue.popleft()
            if distance >= self.escape_horizon:
                continue

            for dx, dy in ((0, -1), (0, 1), (-1, 0), (1, 0)):
                neighbor = (x + dx, y + dy)
                if (
                    neighbor in visited
                    or neighbor in blocked
                    or state.board.is_wall(*neighbor)
                ):
                    continue
                visited.add(neighbor)
                queue.append((neighbor, distance + 1))

        return len(visited)

    def _escape_action_count(self, state, ghost_index) -> int:
        """Count legal moves that do not enter Pac-Man's current cell."""
        if ghost_index < 0 or ghost_index >= len(state.ghosts):
            return 0
        legal_actions = state.get_legal_actions(ghost_index + 1)
        return len(
            self._safe_ghost_actions(state, ghost_index, legal_actions)
        )

    def _order_ghost_actions(
        self,
        state,
        ghost_index,
        legal_actions,
        ghost_name,
    ):
        """Search greater separation and escape space first."""
        def escape_score(action):
            successor = state.generate_successor(ghost_index + 1, action)
            successor_ghost = self._find_ghost(successor, ghost_name)
            if successor_ghost is None:
                return (-math.inf, -math.inf)
            return (
                self._pacman_distance(successor, successor_ghost),
                self._local_escape_space(successor, successor_ghost),
            )

        return sorted(legal_actions, key=escape_score, reverse=True)

    def _greedy_flee_action(self, state, ghost_index, ghost_name):
        """
        Cheap fallback used outside the Minimax activation distance.

        The score prioritizes maze separation, then local escape space and the
        number of exits. It never rewards approaching Pac-Man.
        """
        legal_actions = state.get_legal_actions(ghost_index + 1)
        if not legal_actions:
            return (0, 0)

        legal_actions = self._safe_ghost_actions(
            state,
            ghost_index,
            legal_actions,
        )

        best_action = legal_actions[0]
        history = self._history_for(ghost_name)
        best_score = (-math.inf, -math.inf, -math.inf, -math.inf)

        for action in legal_actions:
            successor = state.generate_successor(ghost_index + 1, action)
            successor_ghost = self._find_ghost(successor, ghost_name)
            if successor_ghost is None:
                continue

            successor_index = successor.ghosts.index(successor_ghost)
            successor_position = successor_ghost.position.to_tuple()
            score = (
                self._pacman_distance(successor, successor_ghost),
                0 if successor_position in history else 1,
                self._local_escape_space(successor, successor_ghost),
                self._escape_action_count(successor, successor_index),
            )
            if score > best_score:
                best_score = score
                best_action = action

        return best_action

    def get_action(self, game_state, ghost_index: int):
        """
        Choose Minimax defense nearby or greedy fleeing outside distance 5.
        """
        self.board_signature_cache.clear()

        if ghost_index < 0 or ghost_index >= len(game_state.ghosts):
            return (0, 0)

        ghost = game_state.ghosts[ghost_index]
        if not ghost.scared:
            return (0, 0)

        ghost_name = ghost.name
        history = self._history_for(ghost_name)
        history.append(ghost.position.to_tuple())
        if not self.is_minimax_active(game_state, ghost_index):
            return self._greedy_flee_action(
                game_state,
                ghost_index,
                ghost_name,
            )

        legal_actions = game_state.get_legal_actions(ghost_index + 1)
        if not legal_actions:
            return (0, 0)

        legal_actions = self._safe_ghost_actions(
            game_state,
            ghost_index,
            legal_actions,
        )
        legal_actions = self._order_ghost_actions(
            game_state,
            ghost_index,
            legal_actions,
            ghost_name,
        )

        best_action = legal_actions[0]
        best_value = -math.inf
        alpha = -math.inf
        beta = math.inf

        for action in legal_actions:
            successor = game_state.generate_successor(
                ghost_index + 1,
                action,
            )
            successor_ghost = self._find_ghost(successor, ghost_name)
            if successor_ghost is None:
                value = -1_000_000_000.0
            elif successor_ghost.scared_timer <= 1:
                # The real engine decrements the timer after this ghost round.
                value = self.evaluate_state(
                    successor,
                    ghost_name,
                    elapsed_rounds=1,
                )
            else:
                value = self._minimize_pacman_response(
                    successor,
                    ghost_name,
                    alpha,
                    beta,
                )

            if successor_ghost is not None:
                successor_position = successor_ghost.position.to_tuple()
                if successor_position in history:
                    value -= 2_000.0

            if value > best_value:
                best_value = value
                best_action = action
            alpha = max(alpha, best_value)

        return best_action

    def _minimize_pacman_response(
        self,
        state,
        ghost_name,
        alpha,
        beta,
    ):
        """Let Pac-Man choose its strongest immediate capture response."""
        ghost = self._find_ghost(state, ghost_name)
        if ghost is None:
            return -1_000_000_000.0
        if not state.pacman:
            return 1_000_000_000.0

        legal_actions = state.get_legal_actions(0)
        if not legal_actions:
            return self.evaluate_state(
                state,
                ghost_name,
                elapsed_rounds=1,
            )

        def chase_score(action):
            successor = state.generate_successor(0, action)
            successor_ghost = self._find_ghost(successor, ghost_name)
            if successor_ghost is None:
                return -math.inf
            return self._pacman_distance(successor, successor_ghost)

        value = math.inf
        for action in sorted(legal_actions, key=chase_score):
            successor = state.generate_successor(0, action)
            child_value = self.evaluate_state(
                successor,
                ghost_name,
                elapsed_rounds=1,
            )
            value = min(value, child_value)
            if value <= alpha:
                return value
            beta = min(beta, value)

        return value

    def evaluate_state(self, state, ghost_name, elapsed_rounds=0):
        """Evaluate survival, separation, timer progress, and escape quality."""
        ghost = self._find_ghost(state, ghost_name)
        if ghost is None:
            return -1_000_000_000.0
        if not state.pacman:
            return 1_000_000_000.0

        effective_timer = ghost.scared_timer - elapsed_rounds
        if effective_timer <= 0 or not ghost.scared:
            return 900_000_000.0

        distance = self._pacman_distance(state, ghost)
        if distance == 0:
            return -1_000_000_000.0

        ghost_index = state.ghosts.index(ghost)
        escape_actions = self._escape_action_count(state, ghost_index)
        escape_space = self._local_escape_space(state, ghost)

        if escape_actions == 0:
            dead_end_penalty = 100_000.0
        elif escape_actions == 1:
            dead_end_penalty = 20_000.0
        else:
            dead_end_penalty = 0.0

        immediate_danger_penalty = (
            80_000.0 / (distance ** 2)
            if distance <= self.activation_distance
            else 0.0
        )

        return float(
            distance * 3_000.0
            + escape_space * 500.0
            + escape_actions * 1_000.0
            - dead_end_penalty
            - immediate_danger_penalty
            - effective_timer * 100.0
        )
