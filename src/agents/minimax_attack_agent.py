import math
from collections import deque
from src.core.config import BoundedCache


class MinimaxScaredGhostAttackAgent:
    """
    Minimax policy for safely chasing one edible scared ghost.

    Pac-Man maximizes capture utility. The selected scared ghost minimizes it
    by evading, while nearby normal ghosts minimize it by threatening Pac-Man.
    The policy searches one complete ghost-response round and replans every
    real frame.
    """

    PACMAN_FORBIDDEN_CELLS = {
        (9, 10),
        (10, 10),
        (11, 10),
        (9, 9),
        (10, 9),
        (11, 9),
    }

    def __init__(self, safety_distance: int = 5, timer_margin: int = 2):
        self.safety_distance = safety_distance
        self.timer_margin = timer_margin
        self.distance_cache = BoundedCache(max_size=8192)
        self.board_signature_cache = {}
        self.recent_positions = deque(maxlen=12)
        self.locked_target_name = None


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


    def _maze_distance(self, board, start, target, pacman_path=False):
        """Return shortest walkable distance, or math.inf when unreachable."""
        if start == target:
            return 0

        endpoints = tuple(sorted((start, target)))
        cache_key = (
            self._board_signature(board),
            endpoints,
            pacman_path,
        )
        if cache_key in self.distance_cache:
            return self.distance_cache[cache_key]

        queue = deque([(start, 0)])
        visited = {start}

        while queue:
            (x, y), distance = queue.popleft()
            for dx, dy in ((0, -1), (0, 1), (-1, 0), (1, 0)):
                neighbor = (x + dx, y + dy)
                if (
                    neighbor in visited
                    or board.is_wall(*neighbor)
                    or (
                        pacman_path
                        and neighbor in self.PACMAN_FORBIDDEN_CELLS
                    )
                ):
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
    def _find_target(state, target_name):
        """Find a target by stable ghost name after successor cloning."""
        return next(
            (ghost for ghost in state.ghosts if ghost.name == target_name),
            None,
        )


    def _nearest_normal_ghost_distance(self, state):
        """Return maze distance to the closest non-scared ghost."""
        if not state.pacman:
            return 0

        pacman_position = state.pacman.position.to_tuple()
        distances = [
            self._maze_distance(
                state.board,
                pacman_position,
                ghost.position.to_tuple(),
            )
            for ghost in state.ghosts
            if not ghost.scared
        ]
        return min(distances, default=math.inf)


    def select_target(self, state):
        """
        Return the name of the safest viable scared target, or None.

        A chase is forbidden while a normal ghost is within the safety radius.
        A target is viable only when maze distance plus a safety margin is
        strictly below its remaining scared timer.
        """
        if not state.pacman:
            self.locked_target_name = None
            return None

        if self._nearest_normal_ghost_distance(state) <= self.safety_distance:
            self.locked_target_name = None
            return None

        pacman_position = state.pacman.position.to_tuple()
        candidates = []

        locked_target = self._find_target(
            state,
            self.locked_target_name,
        )
        if (
            locked_target
            and locked_target.scared
            and locked_target.scared_timer > 0
        ):
            locked_distance = self._maze_distance(
                state.board,
                pacman_position,
                locked_target.position.to_tuple(),
                pacman_path=True,
            )
            if (
                locked_distance < math.inf
                and locked_target.scared_timer - locked_distance
                > self.timer_margin
            ):
                return self.locked_target_name

        for ghost in state.ghosts:
            if not ghost.scared or ghost.scared_timer <= 0:
                continue

            distance = self._maze_distance(
                state.board,
                pacman_position,
                ghost.position.to_tuple(),
                pacman_path=True,
            )
            if distance == math.inf:
                continue

            slack = ghost.scared_timer - distance
            if slack <= self.timer_margin:
                continue

            # Prefer a generous timer buffer, then the closer target.
            candidates.append((slack, -distance, ghost.name))

        if not candidates:
            self.locked_target_name = None
            return None

        self.locked_target_name = max(candidates)[2]
        return self.locked_target_name


    def has_viable_target(self, state) -> bool:
        """Whether Requirement 3 should replace normal A* offense."""
        return self.select_target(state) is not None


    def _safe_chase_actions(self, state, legal_actions):
        """
        Reject moves entering the normal-ghost danger radius when possible.

        If every legal move is dangerous, retain all moves so Minimax can
        choose the least harmful fallback rather than returning no action.
        """
        safe_actions = []
        for action in legal_actions:
            successor = state.generate_successor(0, action)
            if (
                self._nearest_normal_ghost_distance(successor)
                > self.safety_distance
            ):
                safe_actions.append(action)

        return safe_actions


    def _order_pacman_actions(self, state, legal_actions, target_name):
        """Search safe moves toward the edible target first."""
        target = self._find_target(state, target_name)
        if not state.pacman or not target:
            return legal_actions

        def attack_score(action):
            successor = state.generate_successor(0, action)
            successor_target = self._find_target(successor, target_name)
            if successor_target is None:
                return (-math.inf, -math.inf)

            target_distance = self._maze_distance(
                successor.board,
                successor.pacman.position.to_tuple(),
                successor_target.position.to_tuple(),
                pacman_path=True,
            )
            normal_distance = self._nearest_normal_ghost_distance(successor)
            return (target_distance, -normal_distance)

        return sorted(legal_actions, key=attack_score)


    def _should_model_ghost(self, state, ghost, target_name):
        """Model the target and any normal ghost close enough to threaten."""
        if ghost.name == target_name:
            return True
        if ghost.scared or not state.pacman:
            return False

        distance = self._maze_distance(
            state.board,
            state.pacman.position.to_tuple(),
            ghost.position.to_tuple(),
        )
        return distance <= self.safety_distance + 1


    def get_action(self, game_state, target_name=None):
        """Choose Pac-Man's safe max–min chase action."""
        self.board_signature_cache.clear()

        if target_name is None:
            target_name = self.select_target(game_state)
        if target_name is None:
            return (0, 0)

        target = self._find_target(game_state, target_name)
        if not target or not target.scared:
            self.locked_target_name = None
            return (0, 0)

        current_position = game_state.pacman.position.to_tuple()
        self.recent_positions.append(current_position)

        legal_actions = game_state.get_legal_actions(0)
        if not legal_actions:
            return (0, 0)

        legal_actions = self._safe_chase_actions(game_state, legal_actions)
        if not legal_actions:
            self.locked_target_name = None
            return (0, 0)
        legal_actions = self._order_pacman_actions(
            game_state,
            legal_actions,
            target_name,
        )

        best_action = legal_actions[0]
        best_value = -math.inf
        alpha = -math.inf
        beta = math.inf

        for action in legal_actions:
            successor = game_state.generate_successor(0, action)
            if self._find_target(successor, target_name) is None:
                value = 1_000_000_000.0
            else:
                value = self._minimize_ghost_responses(
                    successor,
                    target_name,
                    1,
                    alpha,
                    beta,
                )

            successor_position = successor.pacman.position.to_tuple()
            if successor_position in self.recent_positions:
                value -= 2_000.0

            reverse = (
                -game_state.pacman.direction[0],
                -game_state.pacman.direction[1],
            )
            if game_state.pacman.direction != (0, 0) and action == reverse:
                value -= 750.0

            if value > best_value:
                best_value = value
                best_action = action
            alpha = max(alpha, best_value)

        return best_action


    def _minimize_ghost_responses(
        self,
        state,
        target_name,
        agent_index,
        alpha,
        beta,
    ):
        """
        Let the target evade and nearby normal ghosts threaten Pac-Man.

        All modeled ghosts minimize Pac-Man's attacking utility. Other ghosts
        are skipped to keep the per-frame search bounded.
        """
        if self._find_target(state, target_name) is None:
            return 1_000_000_000.0
        if state.game_over:
            return -1_000_000_000.0
        if agent_index > len(state.ghosts):
            return self.evaluate_state(state, target_name, elapsed_rounds=1)

        ghost = state.ghosts[agent_index - 1]
        if not self._should_model_ghost(state, ghost, target_name):
            return self._minimize_ghost_responses(
                state,
                target_name,
                agent_index + 1,
                alpha,
                beta,
            )

        legal_actions = state.get_legal_actions(agent_index)
        if not legal_actions:
            return self._minimize_ghost_responses(
                state,
                target_name,
                agent_index + 1,
                alpha,
                beta,
            )

        value = math.inf
        for action in legal_actions:
            successor = state.generate_successor(agent_index, action)
            child_value = self._minimize_ghost_responses(
                successor,
                target_name,
                agent_index + 1,
                alpha,
                beta,
            )
            value = min(value, child_value)
            if value <= alpha:
                return value
            beta = min(beta, value)

        return value


    def evaluate_state(self, state, target_name, elapsed_rounds=0):
        """
        Evaluate capture progress while making normal-ghost safety dominant.
        """
        target = self._find_target(state, target_name)
        if target is None:
            return 1_000_000_000.0
        if state.game_over or not state.pacman:
            return -1_000_000_000.0
        if not target.scared:
            return -500_000.0

        pacman_position = state.pacman.position.to_tuple()
        target_distance = self._maze_distance(
            state.board,
            pacman_position,
            target.position.to_tuple(),
            pacman_path=True,
        )
        effective_timer = target.scared_timer - elapsed_rounds
        timer_slack = effective_timer - target_distance

        if target_distance == math.inf or timer_slack <= 0:
            target_score = -500_000.0
        else:
            target_score = (
                -target_distance * 2_500.0
                + timer_slack * 300.0
            )

        normal_distance = self._nearest_normal_ghost_distance(state)
        if normal_distance == 0:
            normal_danger_penalty = 1_000_000.0
        elif normal_distance <= self.safety_distance:
            normal_danger_penalty = 100_000.0 / (normal_distance ** 2)
        else:
            normal_danger_penalty = 0.0

        safe_mobility = len([
            action
            for action in state.get_legal_actions(0)
            if self._nearest_normal_ghost_distance(
                state.generate_successor(0, action)
            )
            > self.safety_distance
        ])

        return float(
            state.pacman.lives * 100_000.0
            - normal_danger_penalty
            + target_score
            + safe_mobility * 250.0
            + state.pacman.score
        )
