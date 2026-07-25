import math
from collections import deque


class MinimaxGhostAgent:
    """
    Team-oriented Minimax controller for normal (non-scared) ghosts.

    Each ghost maximizes a shared team utility while Pac-Man minimizes it by
    choosing evasive responses. Teammate positions contribute to route
    coverage and pressure without expanding the Cartesian product of every
    teammate's moves in every live ghost decision. Normal ghosts attack at
    every distance; scared ghosts are outside Requirement 2 and are skipped.
    """

    def __init__(self, depth: int = 1):
        self.depth = depth
        self.memo = {}
        self.distance_cache = {}
        self.board_signature_cache = {}


    def _board_signature(self, board):
        """Cache the wall layout shared by cloned search states."""
        board_id = id(board)
        if board_id not in self.board_signature_cache:
            wall_bytes = (board.board == 1).tobytes()
            self.board_signature_cache[board_id] = (
                board.width,
                board.height,
                wall_bytes,
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
    def _normal_ghosts(state):
        """Return the ghosts participating in the attacking team."""
        return [ghost for ghost in state.ghosts if not ghost.scared]

    def _pacman_escape_routes(self, state) -> int:
        """
        Count Pac-Man moves that are not occupied or immediately coverable.

        A destination adjacent to a normal ghost is considered controlled
        because that ghost can capture Pac-Man on the following ghost turn.
        """
        if not state.pacman:
            return 0

        normal_ghosts = self._normal_ghosts(state)
        pacman = state.pacman.position
        escape_routes = 0

        for action in state.get_legal_actions(0):
            destination = (pacman.x + action[0], pacman.y + action[1])
            nearest_ghost = min(
                (
                    self._maze_distance(
                        state.board,
                        destination,
                        ghost.position.to_tuple(),
                    )
                    for ghost in normal_ghosts
                ),
                default=math.inf,
            )
            if nearest_ghost > 1:
                escape_routes += 1

        return escape_routes


    def _get_state_hash(
        self,
        state,
        depth,
        agent_index,
        controlled_ghost_index,
    ):
        """Hash all state fields that affect the attacking utility."""
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
            controlled_ghost_index,
        ))


    def _order_actions(self, state, agent_index, legal_actions):
        """Order promising attacks and evasions first for alpha-beta pruning."""
        if not state.pacman:
            return legal_actions

        if agent_index == 0:
            # Pac-Man is the minimizing opponent. Order likely escapes first
            # using cached distances only; full evaluation happens in Minimax.
            pacman = state.pacman.position
            normal_ghosts = self._normal_ghosts(state)

            def escape_score(action):
                destination = (
                    pacman.x + action[0],
                    pacman.y + action[1],
                )
                nearest_ghost = min(
                    (
                        self._maze_distance(
                            state.board,
                            destination,
                            ghost.position.to_tuple(),
                        )
                        for ghost in normal_ghosts
                    ),
                    default=math.inf,
                )
                # sorted() is ascending, so negate distance to put the
                # strongest Pac-Man escape first.
                return -nearest_ghost

            return sorted(legal_actions, key=escape_score)

        ghost_index = agent_index - 1
        if ghost_index >= len(state.ghosts):
            return legal_actions

        ghost = state.ghosts[ghost_index]
        pacman_position = state.pacman.position.to_tuple()

        def attack_score(action):
            destination = (
                ghost.position.x + action[0],
                ghost.position.y + action[1],
            )
            return self._maze_distance(
                state.board,
                destination,
                pacman_position,
            )


        return sorted(legal_actions, key=attack_score)


    def get_action(self, game_state, ghost_index: int):
        """
        Select the real action for one normal ghost.

        ghost_index is zero-based, matching PacmanGame.move_ghost().
        """
        if ghost_index < 0 or ghost_index >= len(game_state.ghosts):
            return (0, 0)

        ghost = game_state.ghosts[ghost_index]
        if ghost.scared:
            return (0, 0)

        self.memo.clear()
        # Distances remain valid across turns because the cache key includes
        # the complete wall layout. Keeping them avoids repeated BFS work.
        self.board_signature_cache.clear()
        agent_index = ghost_index + 1
        legal_actions = game_state.get_legal_actions(agent_index)
        if not legal_actions:
            return (0, 0)

        legal_actions = self._order_actions(
            game_state,
            agent_index,
            legal_actions,
        )

        best_action = legal_actions[0]
        best_value = -math.inf
        alpha = -math.inf
        beta = math.inf

        for action in legal_actions:
            successor = game_state.generate_successor(agent_index, action)
            value = self._minimax(
                successor,
                self.depth,
                0,
                ghost_index,
                alpha,
                beta,
            )
            if value > best_value:
                best_value = value
                best_action = action
            alpha = max(alpha, best_value)

        return best_action


    def _minimax(
        self,
        state,
        depth,
        agent_index,
        controlled_ghost_index,
        alpha,
        beta,
    ):
        """
        Alternate one controlled ghost with Pac-Man's best evasive response.

        Pac-Man is a minimizing node and the controlled normal ghost is a
        maximizing node. Other normal ghosts remain part of the shared utility,
        so the controlled ghost prefers complementary route coverage.
        """
        state_hash = self._get_state_hash(
            state,
            depth,
            agent_index,
            controlled_ghost_index,
        )
        if state_hash in self.memo:
            return self.memo[state_hash]

        if depth == 0 or state.game_over or state.game_won:
            value = self.evaluate_state(state)
            self.memo[state_hash] = value
            return value

        if agent_index == 0:
            legal_actions = state.get_legal_actions(0)
            if not legal_actions:
                value = self.evaluate_state(state)
                self.memo[state_hash] = value
                return value

            value = math.inf
            for action in self._order_actions(state, 0, legal_actions):
                successor = state.generate_successor(0, action)
                value = min(
                    value,
                    self._minimax(
                        successor,
                        depth - 1,
                        controlled_ghost_index + 1,
                        controlled_ghost_index,
                        alpha,
                        beta,
                    ),
                )
                if value <= alpha:
                    return value
                beta = min(beta, value)

            self.memo[state_hash] = value
            return value

        ghost_index = controlled_ghost_index
        if ghost_index >= len(state.ghosts):
            return self.evaluate_state(state)

        ghost = state.ghosts[ghost_index]

        # Requirement 2 activates only for normal ghosts.
        if ghost.scared:
            return self.evaluate_state(state)

        legal_actions = state.get_legal_actions(ghost_index + 1)
        if not legal_actions:
            return self.evaluate_state(state)

        value = -math.inf
        for action in self._order_actions(
            state,
            ghost_index + 1,
            legal_actions,
        ):
            successor = state.generate_successor(ghost_index + 1, action)
            value = max(
                value,
                self._minimax(
                    successor,
                    depth,
                    0,
                    controlled_ghost_index,
                    alpha,
                    beta,
                ),
            )
            if value >= beta:
                return value
            alpha = max(alpha, value)

        self.memo[state_hash] = value
        return value


    def evaluate_state(self, state) -> float:
        """
        Evaluate a state for the normal ghost team's attacking objective.

        Catching Pac-Man dominates the score. Otherwise the team is rewarded
        for preserving fewer Pac-Man lives, closing maze distance, covering
        nearby cells, and reducing safe escape routes.
        """
        if state.game_over:
            return 1_000_000_000.0
        if state.game_won:
            return -1_000_000_000.0
        if not state.pacman:
            return -1_000_000_000.0

        normal_ghosts = self._normal_ghosts(state)
        if not normal_ghosts:
            return -1_000_000_000.0

        pacman_position = state.pacman.position.to_tuple()
        distances = [
            self._maze_distance(
                state.board,
                ghost.position.to_tuple(),
                pacman_position,
            )
            for ghost in normal_ghosts
        ]

        if any(distance == 0 for distance in distances):
            collision_score = 100_000.0
        else:
            collision_score = 0.0

        finite_distances = [
            distance for distance in distances if distance < math.inf
        ]
        nearest_distance = min(finite_distances, default=100)

        # Shared pressure rewards every teammate for approaching Pac-Man.
        team_pressure = sum(
            1_500.0 / (distance + 1.0)
            for distance in finite_distances
        )
        distance_score = -nearest_distance * 1_000.0

        escape_routes = self._pacman_escape_routes(state)
        restriction_score = -escape_routes * 2_500.0
        if escape_routes == 0:
            restriction_score += 8_000.0

        # Fewer remaining Pac-Man lives is always better for the ghost team.
        lives_score = -state.pacman.lives * 100_000.0

        return float(
            lives_score
            + collision_score
            + distance_score
            + team_pressure
            + restriction_score
        )
