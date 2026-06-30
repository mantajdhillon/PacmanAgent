import math
from feature_extractor import FeatureExtractor

class MinimaxPacmanAgent:
    """
    Adversarial search agent using Minimax with Alpha-Beta Pruning.
    """

    def __init__(self, depth: int = 2):
        self.depth = depth
        self.extractor = FeatureExtractor()
        self.memo = {}
        self.active_threat_radius = 4


    def _get_state_hash(self, state, depth, agent_index) -> int:
        """
        Creates a unique integer hash for the current board arrangement.
        """
        pacman_pos = state.pacman.position.to_tuple() if state.pacman else None
        ghost_pos = tuple([g.position.to_tuple() for g in state.ghosts])

        # Hash the positions, remaining depth, and whose turn it is.
        return hash((pacman_pos, ghost_pos, depth, agent_index))


    def _order_actions(self, state, agent_index, legal_actions):
        """
        Sorts legal actions using a lightweight heuristic to force massive Alpha-Beta cutoffs.
        """
        if not state.pacman or not legal_actions:
            return legal_actions

        if agent_index == 0:
            # PACMAN: Evaluate moves that maximize distance from ghosts first
            def pacman_score(action):
                new_x = state.pacman.position.x + action[0]
                new_y = state.pacman.position.y + action[1]

                min_dist = math.inf
                for ghost in state.ghosts:
                    dist = abs(new_x - ghost.position.x) + abs(new_y - ghost.position.y)
                    if dist < min_dist:
                        min_dist = dist
                return min_dist

            # Puts the largest distance at the front of the list
            return sorted(legal_actions, key=pacman_score, reverse=True)

        else:
            # GHOST: Evaluate moves that minimize distance to Pac-Man first
            ghost_idx = agent_index - 1
            if ghost_idx >= len(state.ghosts):
                return legal_actions

            ghost_state = state.ghosts[ghost_idx]
            target_x = state.pacman.position.x
            target_y = state.pacman.position.y

            def ghost_score(action):
                new_x = ghost_state.position.x + action[0]
                new_y = ghost_state.position.y + action[1]
                return abs(new_x - target_x) + abs(new_y - target_y)

            # Puts the smallest distance at the front
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

                if value > beta:
                    self.memo[state_hash] = value
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
                dist = abs(pacman.position.x - ghost.position.x) + abs(pacman.position.y - ghost.position.y)

                # If ghost is too far away or scared, no simulate its moves
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

                if value < alpha:
                    self.memo[state_hash] = value
                    return value

                beta = min(beta, value)

            self.memo[state_hash] = value
            return value


    def evaluate_state(self, state) -> float:
        """
        Custom terminal utility function to heavily prioritize survival.
        """
        if state.game_over:
            return -99999.0

        if state.game_won:
            return 99999.0

        # Calculate nearest threat distance directly
        nearest_ghost_dist = math.inf
        pacman_pos = state.pacman.position

        for ghost in state.ghosts:
            # Ignore scared ghosts
            if not ghost.scared:
                dist = abs(pacman_pos.x - ghost.position.x) + abs(pacman_pos.y - ghost.position.y)
                if dist < nearest_ghost_dist:
                    nearest_ghost_dist = dist

        if nearest_ghost_dist == math.inf:
            nearest_ghost_dist = -1

        score = state.pacman.score

        # Make every single life incredibly valuable
        lives_reward = state.pacman.lives * 10000

        # Danger Penalty Logic
        danger_penalty = 0
        if nearest_ghost_dist != -1 and nearest_ghost_dist <= 2:
            danger_penalty = 5000 / (nearest_ghost_dist + 0.1)  # +0.1 prevents division by zero

        # Distance Reward Logic
        distance_reward = nearest_ghost_dist * 10 if nearest_ghost_dist != -1 else 0

        # Return the comprehensive utility formula
        return float(score + lives_reward + distance_reward - danger_penalty)
