import math
from collections import deque

class MinimaxPacmanAgent:
    """
    Defensive adversarial-search agent using Minimax with alpha-beta pruning.

    Pac-Man is the maximizing agent. Non-scared ghosts within the configured
    maze-distance threshold are minimizing agents. The evaluation function
    deliberately ranks survival above mobility, ghost separation, and food.
    """

    def __init__(self, depth: int = 2, threat_distance: int = 5):
        self.depth = depth
        self.memo = {}
        self.active_threat_radius = threat_distance


    @staticmethod
    def _maze_distance(board, start, target, max_distance=None):
        """Return shortest walkable distance, or math.inf when unreachable."""
        if start == target:
            return 0

        queue = deque([(start, 0)])
        visited = {start}

        while queue:
            (x, y), distance = queue.popleft()
            if max_distance is not None and distance >= max_distance:
                continue

            for dx, dy in ((0, -1), (0, 1), (-1, 0), (1, 0)):
                neighbor = (x + dx, y + dy)
                if neighbor in visited or board.is_wall(*neighbor):
                    continue
                if neighbor == target:
                    return distance + 1
                visited.add(neighbor)
                queue.append((neighbor, distance + 1))

        return math.inf


    def _dangerous_ghost_distances(self, state):
        """Return maze distances from Pac-Man to every non-scared ghost."""
        if not state.pacman:
            return []

        pacman_position = state.pacman.position.to_tuple()
        return [
            self._maze_distance(
                state.board,
                pacman_position,
                ghost.position.to_tuple(),
            )
            for ghost in state.ghosts
            if not ghost.scared
        ]


    def is_threat_nearby(self, state) -> bool:
        """Whether a non-scared ghost is within the defensive threshold."""
        return any(
            distance <= self.active_threat_radius
            for distance in self._dangerous_ghost_distances(state)
        )


    @staticmethod
    def _safe_legal_actions(state):
        """Pac-Man moves that do not enter a currently occupied danger cell."""
        if not state.pacman:
            return []

        dangerous_positions = {
            ghost.position.to_tuple()
            for ghost in state.ghosts
            if not ghost.scared
        }
        pacman_position = state.pacman.position

        return [
            action
            for action in state.get_legal_actions(0)
            if (
                pacman_position.x + action[0],
                pacman_position.y + action[1],
            )
            not in dangerous_positions
        ]


    @staticmethod
    def _reachable_area(state, max_distance=4) -> int:
        """
        Count safe cells Pac-Man can reach locally.

        A small reachable area indicates a cul-de-sac or constrained corridor.
        Dangerous ghost cells are treated as blocked for this mobility signal.
        """
        if not state.pacman:
            return 0

        blocked = {
            ghost.position.to_tuple()
            for ghost in state.ghosts
            if not ghost.scared
        }
        start = state.pacman.position.to_tuple()
        queue = deque([(start, 0)])
        visited = {start}

        while queue:
            (x, y), distance = queue.popleft()
            if distance >= max_distance:
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


    def _nearest_food_distance(self, state):
        """Shortest maze distance to any remaining food item."""
        if not state.pacman:
            return math.inf

        food = list(state.board.pellets | state.board.power_pellets)
        if not food:
            return math.inf

        start = state.pacman.position.to_tuple()
        return min(
            self._maze_distance(state.board, start, target)
            for target in food
        )


    def _get_state_hash(self, state, depth, agent_index) -> int:
        """
        Creates a unique integer hash for the current board arrangement.
        """
        pacman_state = None
        if state.pacman:
            pacman_state = (
                state.pacman.position.to_tuple(),
                state.pacman.direction,
                state.pacman.score,
                state.pacman.lives,
            )

        ghost_state = tuple(
            (
                ghost.name,
                ghost.position.to_tuple(),
                ghost.direction,
                ghost.scared,
                ghost.scared_timer,
            )
            for ghost in state.ghosts
        )

        return hash((
            pacman_state,
            ghost_state,
            frozenset(state.board.pellets),
            frozenset(state.board.power_pellets),
            state.game_over,
            state.game_won,
            depth,
            agent_index,
        ))


    def _order_actions(self, state, agent_index, legal_actions):
        """
        Sorts legal actions using a lightweight heuristic to force massive Alpha-Beta cutoffs.
        """
        if not state.pacman or not legal_actions:
            return legal_actions

        if agent_index == 0:
            # Search safer, more mobile Pac-Man actions first for earlier cutoffs.
            def pacman_score(action):
                successor = state.generate_successor(0, action)
                distances = self._dangerous_ghost_distances(successor)
                nearest = min(distances, default=math.inf)
                mobility = len(self._safe_legal_actions(successor))
                return nearest, mobility

            return sorted(legal_actions, key=pacman_score, reverse=True)

        else:
            # Search attacking ghost actions first for earlier cutoffs.
            ghost_idx = agent_index - 1
            if ghost_idx >= len(state.ghosts):
                return legal_actions

            ghost_state = state.ghosts[ghost_idx]
            target_x = state.pacman.position.x
            target_y = state.pacman.position.y

            def ghost_score(action):
                new_x = ghost_state.position.x + action[0]
                new_y = ghost_state.position.y + action[1]
                return self._maze_distance(
                    state.board,
                    (new_x, new_y),
                    (target_x, target_y),
                )

            return sorted(legal_actions, key=ghost_score)


    def get_action(self, game_state) -> tuple:
        """
        Entry point. Evaluates the top-level Max node.
        """
        # Clear the cache
        self.memo.clear()

        legal_actions = game_state.get_legal_actions(0)

        if not legal_actions:
            return (0, 0)

        best_action = None
        max_value = -math.inf
        alpha = -math.inf
        beta = math.inf

        legal_actions = self._order_actions(game_state, 0, legal_actions)

        for action in legal_actions:
            successor = game_state.generate_successor(0, action)
            value = self._minimax(successor, self.depth, 1, len(successor.ghosts), alpha, beta)

            if value > max_value:
                max_value = value
                best_action = action

            alpha = max(alpha, max_value)

        return best_action


    def _minimax(self, state, depth: int, agent_index: int, total_ghosts: int, alpha: float, beta: float) -> float:
        """
        The recursive core with Alpha-Beta Pruning and Memoization.
        """
        # Check the Transposition Table
        state_hash = self._get_state_hash(state, depth, agent_index)
        if state_hash in self.memo:
            return self.memo[state_hash]

        # Base Cases
        if depth == 0 or state.game_over or state.game_won:
            eval_score = self.evaluate_state(state)
            self.memo[state_hash] = eval_score
            return eval_score

        # Max Node: Pac-Man
        if agent_index == 0:
            value = -math.inf
            legal_actions = state.get_legal_actions(0)

            if not legal_actions:
                eval_score = self.evaluate_state(state)
                self.memo[state_hash] = eval_score
                return eval_score

            legal_actions = self._order_actions(state, 0, legal_actions)

            for action in legal_actions:
                successor = state.generate_successor(0, action)
                value = max(value, self._minimax(successor, depth, 1, total_ghosts, alpha, beta))

                if value >= beta:
                    return value

                alpha = max(alpha, value)

            self.memo[state_hash] = value
            return value

        # Min Node: Ghosts
        else:
            value = math.inf
            ghost_idx = agent_index - 1

            if ghost_idx >= len(state.ghosts):
                return self._minimax(state, depth - 1, 0, total_ghosts, alpha, beta)

            ghost = state.ghosts[ghost_idx]
            pacman = state.pacman

            if pacman:
                dist = self._maze_distance(
                    state.board,
                    pacman.position.to_tuple(),
                    ghost.position.to_tuple(),
                    max_distance=self.active_threat_radius,
                )

                # Requirement 1 models only nearby, non-scared threats.
                if dist > self.active_threat_radius or ghost.scared:
                    # Determine next agent in the ply
                    if agent_index == total_ghosts:
                        next_agent = 0
                        next_depth = depth - 1
                    else:
                        next_agent = agent_index + 1
                        next_depth = depth

                    # Skip generating successors
                    # Pass the exact same state down the tree
                    return self._minimax(state, next_depth, next_agent, total_ghosts, alpha, beta)

            legal_actions = state.get_legal_actions(agent_index)

            if not legal_actions:
                eval_score = self.evaluate_state(state)
                self.memo[state_hash] = eval_score
                return eval_score

            legal_actions = self._order_actions(state, agent_index, legal_actions)

            for action in legal_actions:
                successor = state.generate_successor(agent_index, action)

                if agent_index == total_ghosts:
                    next_agent = 0
                    next_depth = depth - 1
                else:
                    next_agent = agent_index + 1
                    next_depth = depth

                value = min(value, self._minimax(successor, next_depth, next_agent, total_ghosts, alpha, beta))

                if value <= alpha:
                    return value

                beta = min(beta, value)

            self.memo[state_hash] = value
            return value


    def evaluate_state(self, state) -> float:
        """
        Evaluate a state for normal defensive Pac-Man behavior.

        Priority order:
        1. Avoid terminal loss and preserve lives.
        2. Avoid immediate collision and dangerous proximity.
        3. Preserve escape routes and avoid locally confined areas.
        4. Increase maze distance from dangerous ghosts.
        5. Prefer score and nearby food only as secondary objectives.
        """
        if state.game_over:
            return -1_000_000_000.0

        if state.game_won:
            return 1_000_000_000.0

        if not state.pacman:
            return -1_000_000_000.0

        distances = self._dangerous_ghost_distances(state)
        nearest_threat = min(distances, default=math.inf)

        lives_score = state.pacman.lives * 25_000.0

        if nearest_threat == 0:
            danger_penalty = 100_000.0
            separation_score = 0.0
        elif nearest_threat <= self.active_threat_radius:
            danger_penalty = 12_000.0 / (nearest_threat ** 2)
            separation_score = nearest_threat * 250.0
        else:
            danger_penalty = 0.0
            separation_score = (
                self.active_threat_radius * 250.0
                if nearest_threat < math.inf
                else 0.0
            )

        safe_actions = self._safe_legal_actions(state)
        if not safe_actions:
            dead_end_penalty = 8_000.0
        elif len(safe_actions) == 1:
            dead_end_penalty = 2_500.0
        else:
            dead_end_penalty = 0.0

        mobility_score = len(safe_actions) * 200.0
        reachable_space_score = self._reachable_area(state) * 20.0

        food_distance = self._nearest_food_distance(state)
        food_score = (
            100.0 / (food_distance + 1.0)
            if food_distance < math.inf
            else 0.0
        )

        return float(
            lives_score
            - danger_penalty
            - dead_end_penalty
            + separation_score
            + mobility_score
            + reachable_space_score
            + state.pacman.score
            + food_score
        )
