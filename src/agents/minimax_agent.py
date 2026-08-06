import math
from collections import deque
from src.core.config import BoundedCache

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
        self.distance_cache = BoundedCache(max_size=8192)
        self.board_signature_cache = {}
        self.recent_positions = deque(maxlen=12)


    def _board_signature(self, board):
        """Return the wall-only signature shared by cloned search boards."""
        board_id = id(board)
        if board_id not in self.board_signature_cache:
            self.board_signature_cache[board_id] = (
                board.width,
                board.height,
                (board.board == 1).tobytes(),
            )
        return self.board_signature_cache[board_id]


    def _maze_distance(self, board, start, target, max_distance=None):
        """Return shortest walkable distance, or math.inf when unreachable."""
        if start == target:
            return 0

        endpoints = tuple(sorted((start, target)))
        cache_key = (
            self._board_signature(board),
            endpoints,
            max_distance,
        )
        if cache_key in self.distance_cache:
            return self.distance_cache[cache_key]

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
                    result = distance + 1
                    self.distance_cache[cache_key] = result
                    return result
                visited.add(neighbor)
                queue.append((neighbor, distance + 1))

        self.distance_cache[cache_key] = math.inf
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


    def is_action_safe(self, state, action, minimum_distance=2) -> bool:
        """
        Whether an action survives both Pac-Man's move and the next ghost move.

        This gate is shared by scared-ghost pursuit, point collection, and the
        defensive fallback, so a scoring objective can never bypass survival.
        """
        if action not in state.get_legal_actions(0):
            return False
        successor = state.generate_successor(0, action)
        if (
            not successor.pacman
            or successor.pacman.lives < state.pacman.lives
            or successor.game_over
        ):
            return False
        distances = self._dangerous_ghost_distances(successor)
        if min(distances, default=math.inf) <= minimum_distance:
            return False

        pacman_position = successor.pacman.position.to_tuple()
        for ghost_index, ghost in enumerate(successor.ghosts):
            if ghost.scared:
                continue
            for ghost_action in successor.get_legal_actions(ghost_index + 1):
                destination = (
                    ghost.position.x + ghost_action[0],
                    ghost.position.y + ghost_action[1],
                )
                if destination == pacman_position:
                    return False

        return True


    def get_safest_action(self, state):
        """
        Return the strongest survival move when an objective move is unsafe.

        Immediate survival dominates distance, escape space, food progress,
        and loop avoidance in that exact order.
        """
        if not state.pacman:
            return (0, 0)

        best_action = (0, 0)
        best_score = None
        for action in state.get_legal_actions(0):
            successor = state.generate_successor(0, action)
            survived_move = (
                successor.pacman is not None
                and successor.pacman.lives == state.pacman.lives
                and not successor.game_over
            )
            safe_next_round = self.is_action_safe(
                state,
                action,
                minimum_distance=0,
            )
            nearest_threat = min(
                self._dangerous_ghost_distances(successor),
                default=math.inf,
            )
            successor_position = successor.pacman.position.to_tuple()
            collected_food = (
                successor.pellets_eaten > state.pellets_eaten
                or successor.power_pellets_eaten
                > state.power_pellets_eaten
            )
            score = (
                1 if survived_move else 0,
                1 if safe_next_round else 0,
                nearest_threat,
                self._reachable_area(successor),
                1 if collected_food else 0,
                0 if successor_position in self.recent_positions else 1,
            )
            if best_score is None or score > best_score:
                best_score = score
                best_action = action

        return best_action


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
        """
        Find the nearest food with one multi-target BFS.

        The previous implementation ran one complete BFS per pellet, per leaf.
        This search stops as soon as it reaches any pellet or power pellet.
        """
        if not state.pacman:
            return math.inf

        food = state.board.pellets | state.board.power_pellets
        if not food:
            return math.inf

        start = state.pacman.position.to_tuple()
        if start in food:
            return 0

        queue = deque([(start, 0)])
        visited = {start}

        while queue:
            (x, y), distance = queue.popleft()
            for dx, dy in ((0, -1), (0, 1), (-1, 0), (1, 0)):
                neighbor = (x + dx, y + dy)
                if neighbor in visited or state.board.is_wall(*neighbor):
                    continue
                if neighbor in food:
                    return distance + 1
                visited.add(neighbor)
                queue.append((neighbor, distance + 1))

        return math.inf


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
        # Distance entries remain valid because their keys include wall layout.
        self.board_signature_cache.clear()

        current_position = game_state.pacman.position.to_tuple()
        if (
            game_state.pacman.score == 0
            and game_state.pellets_eaten == 0
            and current_position == (10, 15)
        ):
            self.recent_positions.clear()
        self.recent_positions.append(current_position)

        legal_actions = game_state.get_legal_actions(0)

        if not legal_actions:
            return (0, 0)

        survival_actions = [
            action
            for action in legal_actions
            if self.is_action_safe(
                game_state,
                action,
                minimum_distance=0,
            )
        ]
        if survival_actions:
            legal_actions = survival_actions

        best_action = None
        max_value = -math.inf
        alpha = -math.inf
        beta = math.inf

        legal_actions = self._order_actions(game_state, 0, legal_actions)

        for action in legal_actions:
            successor = game_state.generate_successor(0, action)
            value = self._minimax(successor, self.depth, 1, len(successor.ghosts), alpha, beta)

            successor_position = successor.pacman.position.to_tuple()
            if successor_position in self.recent_positions:
                value -= 2_500.0

            reverse = (
                -game_state.pacman.direction[0],
                -game_state.pacman.direction[1],
            )
            if game_state.pacman.direction != (0, 0) and action == reverse:
                value -= 750.0

            if successor.pellets_eaten > game_state.pellets_eaten:
                value += 3_500.0
            if (
                successor.power_pellets_eaten
                > game_state.power_pellets_eaten
            ):
                value += 8_000.0

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
            1_000.0 / (food_distance + 1.0)
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
