"""Pac-Man game environment using Pygame."""

import pygame
import numpy as np
from typing import Tuple, List, Optional
from config import (
    BOARD_WIDTH, BOARD_HEIGHT, CELL_SIZE, WINDOW_WIDTH, WINDOW_HEIGHT, FPS,
    UP, DOWN, LEFT, RIGHT, DIRECTIONS,
    WALL, EMPTY, PELLET, POWER_PELLET,
    BLACK, WHITE, YELLOW, RED, PINK, CYAN, ORANGE, BLUE
)
from game_state import GameState, GameBoard, PacmanState, GhostState, Position
from feature_extractor import FeatureExtractor


class PacmanGame:
    """Main Pac-Man game engine."""

    def __init__(self, display: bool = True):
        """
        Initialize the game.
        Args:
            display: Whether to display the game using Pygame
        """
        self.display = display
        self.running = False

        if self.display:
            pygame.init()
            self.screen = pygame.display.set_mode((WINDOW_WIDTH, WINDOW_HEIGHT))
            pygame.display.set_caption("Pac-Man Agent")
            self.clock = pygame.time.Clock()
            self.font = pygame.font.Font(None, 36)
        else:
            self.screen = None
            self.clock = None
            self.font = None

        self.board = GameBoard(BOARD_WIDTH, BOARD_HEIGHT)
        self.game_state = GameState(self.board)
        self.feature_extractor = FeatureExtractor(BOARD_WIDTH, BOARD_HEIGHT)

        self._setup_maze()

    def _setup_maze(self):
        # Create borders
        for x in range(BOARD_WIDTH):
            self.board.add_wall(x, 0)
            self.board.add_wall(x, BOARD_HEIGHT - 1)

        for y in range(BOARD_HEIGHT):
            self.board.add_wall(0, y)
            self.board.add_wall(BOARD_WIDTH - 1, y)

        self._add_interior_walls()

        for x in range(1, BOARD_WIDTH - 1):
            for y in range(1, BOARD_HEIGHT - 1):
                if not self.board.is_wall(x, y):
                    in_ghost_house = (7 <= x <= 13) and (8 <= y <= 11)
                    if not in_ghost_house:
                        self.board.add_pellet(x, y)

        power_pellet_positions = [(1, 3), (19, 3), (1, 15), (19, 15)]
        for x, y in power_pellet_positions:
            if not self.board.is_wall(x, y):
                self.board.remove_pellet(x, y)
                self.board.add_power_pellet(x, y)

        pacman = PacmanState(Position(10, 15))
        self.game_state.set_pacman(pacman)

        ghost_colors = ["red", "pink", "cyan", "orange"]
        ghost_names = ["Blinky", "Pinky", "Inky", "Clyde"]
        ghost_start_positions = [(9, 10), (10, 10), (11, 10), (10, 9)]

        for i, (color, name, (x, y)) in enumerate(zip(ghost_colors, ghost_names, ghost_start_positions)):
            if not self.board.is_wall(x, y):
                ghost = GhostState(Position(x, y), color, name=name)
                self.game_state.add_ghost(ghost)

    def _add_interior_walls(self):
        wall_patterns = [
            (10, [1, 2, 3, 6, 7, 14, 15, 17, 18]),

            (range(2, 5), [2, 3, 5, 14]),
            (range(16, 19), [2, 3, 5, 14]),

            (range(6, 9), [2, 3, 7, 14]),
            (range(12, 15), [2, 3, 7, 14]),

            (range(8, 13), [5, 11, 16]),

            (6, [5, 6, 8, 9, 10, 11, 16, 17]),
            (14, [5, 6, 8, 9, 10, 11, 16, 17]),

            (range(1, 5), [7, 8, 9, 10, 11, 12, 16]),
            (range(16, 20), [7, 8, 9, 10, 11, 12, 16]),

            (8, [9, 10]),
            (12, [9, 10]),

            (4, [15]),
            (16, [15]),

            (range(2, 9), [18]),
            (range(12, 19), [18]),
        ]

        for x_or_range, y_or_range in wall_patterns:
            if isinstance(x_or_range, range):
                for x in x_or_range:
                    for y in y_or_range:
                        if 0 < x < BOARD_WIDTH - 1 and 0 < y < BOARD_HEIGHT - 1:
                            self.board.add_wall(x, y)
            else:
                for y in y_or_range:
                    if 0 < x_or_range < BOARD_WIDTH - 1 and 0 < y < BOARD_HEIGHT - 1:
                        self.board.add_wall(x_or_range, y)

    def reset(self):
        """Reset the game to initial state."""
        self.board = GameBoard(BOARD_WIDTH, BOARD_HEIGHT)
        self.game_state = GameState(self.board)
        self._setup_maze()
        self.running = True

    def move_pacman(self, direction: Tuple[int, int]) -> bool:
        """
        Move Pac-Man in the given direction.
        Args:
            direction: Tuple (dx, dy) representing direction
        Returns:
            True if move was successful, False otherwise
        """
        if not self.game_state.pacman:
            return False

        pacman = self.game_state.pacman
        new_x = pacman.position.x + direction[0]
        new_y = pacman.position.y + direction[1]

        if self.board.is_wall(new_x, new_y):
            return False

        # Check pellet collection
        if (new_x, new_y) in self.board.pellets:
            self.board.remove_pellet(new_x, new_y)
            pacman.score += 10
            self.game_state.pellets_eaten += 1

        # Check power pellet collection
        if (new_x, new_y) in self.board.power_pellets:
            self.board.remove_power_pellet(new_x, new_y)
            pacman.score += 50
            self.game_state.power_pellets_eaten += 1
            # Activate scared mode for ghosts (150 frames = 15 seconds at 10 FPS)
            for i, ghost in enumerate(self.game_state.ghosts):
                self.game_state.update_ghost_scared_state(i, True, 150)

        pacman.position.x = new_x
        pacman.position.y = new_y
        pacman.direction = direction

        return True

    def move_ghost(self, ghost_index: int, direction: Tuple[int, int]) -> bool:
        """
        Move a ghost in the given direction.
        Args:
            ghost_index: Index of the ghost
            direction: Tuple (dx, dy) representing direction
        Returns:
            True if move was successful, False otherwise
        """
        if ghost_index >= len(self.game_state.ghosts):
            return False

        ghost = self.game_state.ghosts[ghost_index]
        new_x = ghost.position.x + direction[0]
        new_y = ghost.position.y + direction[1]

        if self.board.is_wall(new_x, new_y):
            return False

        ghost.position.x = new_x
        ghost.position.y = new_y
        ghost.direction = direction

        return True

    def check_collisions(self) -> bool:
        """
        Check for collisions between Pac-Man and ghosts.
        Returns:
            True if Pac-Man collided with a ghost (game over), False otherwise
        """
        if not self.game_state.pacman:
            return False

        pacman_pos = self.game_state.pacman.position.to_tuple()

        for ghost in list(self.game_state.ghosts):  # Use list() to avoid modifying while iterating
            if ghost.position.to_tuple() == pacman_pos:
                if ghost.scared:
                    # Store ghost for respawn (respawn after 100 frames = 10 seconds at 10 FPS)
                    ghost.start_position = (BOARD_WIDTH // 2, BOARD_HEIGHT // 2)
                    self.game_state.eaten_ghosts.append((ghost, 100))
                    self.game_state.ghosts.remove(ghost)
                    self.game_state.pacman.score += 200
                else:
                    self.game_state.pacman.lives -= 1
                    if self.game_state.pacman.lives <= 0:
                        self.game_state.game_over = True
                        return True
                    else:
                        self._reset_positions()
                return False

        return False

    def _reset_positions(self):
        """Reset Pac-Man and ghost positions after collision."""
        self.game_state.pacman.position = Position(10, 15)
        ghost_start_positions = [(9, 10),(10, 10),(11, 10),(10, 9)]

        for i, ghost in enumerate(self.game_state.ghosts):
            if i < len(ghost_start_positions):
                x, y = ghost_start_positions[i]
                ghost.position = Position(x, y)

    def check_win_condition(self) -> bool:
        """
        Check if all pellets have been collected.
        Returns:
            True if game is won, False otherwise
        """
        # Don't set win condition if game is already over (lives at 0)
        if self.game_state.game_over:
            return False
        
        if not self.board.pellets and not self.board.power_pellets:
            self.game_state.game_won = True
            return True
        return False

    def update_scared_timers(self):
        """Update scared timers for ghosts."""
        for i, ghost in enumerate(self.game_state.ghosts):
            if ghost.scared and ghost.scared_timer > 0:
                ghost.scared_timer -= 1
                if ghost.scared_timer == 0:
                    self.game_state.update_ghost_scared_state(i, False, 0)

    def update_respawn_timers(self):
        """Update respawn timers for eaten ghosts and bring them back when ready."""
        still_eaten = []
        for ghost, timer in self.game_state.eaten_ghosts:
            timer -= 1
            if timer <= 0:
                # Respawn ghost at center in unscared state
                ghost.position = Position(BOARD_WIDTH // 2, BOARD_HEIGHT // 2)
                ghost.scared = False
                ghost.scared_timer = 0
                self.game_state.ghosts.append(ghost)
            else:
                still_eaten.append((ghost, timer))
        
        # Update the eaten ghosts list
        self.game_state.eaten_ghosts = still_eaten

    def render(self):
        """Render the game board and entities."""
        if not self.display or not self.screen:
            return

        self.screen.fill(BLACK)

        # Draw board
        board_array = self.board.get_board_array()
        for y in range(BOARD_HEIGHT):
            for x in range(BOARD_WIDTH):
                cell = board_array[y, x]
                rect = pygame.Rect(x * CELL_SIZE, y * CELL_SIZE, CELL_SIZE, CELL_SIZE)

                if cell == WALL:
                    pygame.draw.rect(self.screen, BLUE, rect)
                elif cell == PELLET:
                    pygame.draw.circle(
                        self.screen, WHITE,
                        (x * CELL_SIZE + CELL_SIZE // 2, y * CELL_SIZE + CELL_SIZE // 2), 2
                    )
                elif cell == POWER_PELLET:
                    pygame.draw.circle(
                        self.screen, WHITE,
                        (x * CELL_SIZE + CELL_SIZE // 2, y * CELL_SIZE + CELL_SIZE // 2), 5
                    )

        # Draw Pac-Man
        if self.game_state.pacman:
            pacman = self.game_state.pacman
            x, y = pacman.position.x * CELL_SIZE + CELL_SIZE // 2, pacman.position.y * CELL_SIZE + CELL_SIZE // 2
            pygame.draw.circle(self.screen, YELLOW, (x, y), CELL_SIZE // 2 - 2)

        # Draw ghosts
        for ghost in self.game_state.ghosts:
            x, y = ghost.position.x * CELL_SIZE + CELL_SIZE // 2, ghost.position.y * CELL_SIZE + CELL_SIZE // 2
            color = {
                "red": RED,
                "pink": PINK,
                "cyan": CYAN,
                "orange": ORANGE,
            }.get(ghost.color, RED)

            if ghost.scared:
                color = BLUE

            pygame.draw.rect(self.screen, color, (x - CELL_SIZE // 3, y - CELL_SIZE // 3, CELL_SIZE // 3 * 2, CELL_SIZE // 3 * 2))

        # Draw HUD
        if self.font:
            score_text = self.font.render(f"Score: {self.game_state.pacman.score if self.game_state.pacman else 0}", True, WHITE)
            lives_text = self.font.render(f"Lives: {self.game_state.pacman.lives if self.game_state.pacman else 0}", True, WHITE)
            self.screen.blit(score_text, (10, WINDOW_HEIGHT - 25))
            self.screen.blit(lives_text, (WINDOW_WIDTH - 100, WINDOW_HEIGHT - 25))

        pygame.display.flip()
        if self.clock:
            self.clock.tick(FPS)

    def get_game_state(self):
        """Get the current game state."""
        return self.game_state

    def get_features(self):
        """Get extracted features from current game state."""
        return self.feature_extractor.extract_all_features(self.game_state)

    def get_state_arrays(self):
        """Get state as numpy arrays."""
        return self.game_state.get_state_arrays()

    def get_state_vector(self):
        """Get state as a feature vector."""
        return self.feature_extractor.get_state_vector(self.game_state)

    def is_game_over(self) -> bool:
        """Check if game is over."""
        return self.game_state.game_over

    def is_game_won(self) -> bool:
        """Check if game is won."""
        return self.game_state.game_won

    def quit(self):
        """Quit the game."""
        if self.display:
            pygame.quit()
        self.running = False
