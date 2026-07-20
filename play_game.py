"""Interactive Pac-Man game with Pygame UI for manual testing."""

import pygame
from astar_agent import AStarPacmanAgent
from minimax_agent import MinimaxPacmanAgent
from game_engine import PacmanGame
from config import UP, DOWN, LEFT, RIGHT, WINDOW_WIDTH, WINDOW_HEIGHT
import sys


class GamePlayer:
    """Interactive game controller for manual testing."""

    def __init__(self):
        pygame.init()
        self.game = PacmanGame(display=True)
        self.game.reset()
        self.running = True
        self.paused = False
        self.autoplay = True
        self.current_ai_mode = "A* AUTOPLAY"
        self.astar_agent = AStarPacmanAgent()
        self.minimax_agent = MinimaxPacmanAgent(depth=2)
        self.move_count = 0
        self.frame_count = 0

    def handle_input(self):
        """Handle keyboard input for Pac-Man movement."""
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                self.running = False

            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE:
                    self.running = False
                elif event.key == pygame.K_SPACE:
                    self.paused = not self.paused
                elif event.key == pygame.K_a:
                    self.autoplay = not self.autoplay
                    mode = "MANUAL" if self.autoplay else "A* AUTOPLAY"
                    print(f"\n[MODE] Switched to {mode}")
                elif event.key == pygame.K_r:
                    self.game.reset()
                    self.move_count = 0
                    print("\n[RESET] Game reset!")
                    
        if not self.paused:
            if self.autoplay:
                game_state = self.game.game_state
                pacman_pos = game_state.pacman.position

                # Find the nearest dangerous ghost
                nearest_threat_dist = float('inf')

                for ghost in game_state.ghosts:
                    if not ghost.scared:
                        dist = abs(pacman_pos.x - ghost.position.x) + abs(pacman_pos.y - ghost.position.y)
                        if dist < nearest_threat_dist:
                            nearest_threat_dist = dist

                # Orchestrator Logic
                THREAT_RADIUS = 5  # Grid squares

                if nearest_threat_dist <= THREAT_RADIUS:
                    # DANGER: Engage Defensive Minimax
                    self.current_ai_mode = "MINIMAX"
                    pygame.event.pump()
                    action = self.minimax_agent.get_action(game_state)
                else:
                    # SAFE: Engage Offensive A*
                    self.current_ai_mode = "A*"
                    action = self.astar_agent.get_action(game_state)

                # Execute the routed action
                if action != (0, 0) and self.game.move_pacman(action):
                    self.move_count += 1
                return

            # Continuous key checking for smooth movement
            self.current_ai_mode = "MANUAL"
            keys = pygame.key.get_pressed()
            if keys[pygame.K_UP]:
                if self.game.move_pacman(UP):
                    self.move_count += 1
            elif keys[pygame.K_DOWN]:
                if self.game.move_pacman(DOWN):
                    self.move_count += 1
            elif keys[pygame.K_LEFT]:
                if self.game.move_pacman(LEFT):
                    self.move_count += 1
            elif keys[pygame.K_RIGHT]:
                if self.game.move_pacman(RIGHT):
                    self.move_count += 1

    def update_ghosts(self):
        """Simple ghost AI - random movement."""
        import random
        directions = [UP, DOWN, LEFT, RIGHT]

        for i in range(len(self.game.game_state.ghosts)):
            # Try random direction, keep current direction if it hits a wall
            direction = random.choice(directions)
            self.game.move_ghost(i, direction)

    def update(self):
        """Update game state."""
        # Stop updating if game is already over or won
        if self.game.is_game_over() or self.game.is_game_won():
            return
        
        if not self.paused:
            # Move ghosts
            self.update_ghosts()

            # Check collisions
            self.game.check_collisions()

            # Update scared timers
            self.game.update_scared_timers()

            # Update respawn timers
            self.game.update_respawn_timers()

            # Check win condition (only if not game over)
            if not self.game.is_game_over():
                self.game.check_win_condition()

    def render(self):
        """Render game and UI."""
        self.game.render()

        # Draw additional UI info
        if self.game.screen and self.game.font:
            info_text = f"Moves: {self.move_count} | Pellets: {len(self.game.board.pellets)} | Frame: {self.frame_count}"
            info_surface = self.game.font.render(info_text, True, (255, 255, 255))
            self.game.screen.blit(info_surface, (5, 5))

            if self.current_ai_mode == "A*":
                mode_text = "A* AUTOPLAY"
            elif self.current_ai_mode == "MINIMAX":
                mode_text = "MINIMAX AUTOPLAY"
            else:
                mode_text = "MANUAL"

            mode_surface = self.game.font.render(mode_text, True, (255, 255, 0))
            self.game.screen.blit(mode_surface, (WINDOW_WIDTH - 340, WINDOW_HEIGHT - 25))

            if self.paused:
                pause_text = self.game.font.render("PAUSED (SPACE to resume)", True, (255, 0, 0))
                self.game.screen.blit(pause_text, (WINDOW_WIDTH // 2 - 150, WINDOW_HEIGHT // 2))

            if self.game.is_game_over():
                game_over_text = self.game.font.render(f"GAME OVER! Final Score: {self.game.game_state.pacman.score}", True, (255, 0, 0))
                self.game.screen.blit(game_over_text, (WINDOW_WIDTH // 2 - 162, WINDOW_HEIGHT // 2 - 50))
                restart_text = self.game.font.render("Press R to restart", True, (255, 255, 255))
                self.game.screen.blit(restart_text, (WINDOW_WIDTH // 2 - 100, WINDOW_HEIGHT // 2 + 50))

            if self.game.is_game_won():
                won_text = self.game.font.render(f"YOU WIN! Final Score: {self.game.game_state.pacman.score}", True, (0, 255, 0))
                self.game.screen.blit(won_text, (WINDOW_WIDTH // 2 - 162, WINDOW_HEIGHT // 2 - 50))
                restart_text = self.game.font.render("Press R to restart", True, (255, 255, 255))
                self.game.screen.blit(restart_text, (WINDOW_WIDTH // 2 - 100, WINDOW_HEIGHT // 2 + 50))

    def run(self):
        """Main game loop."""
        print("\n" + "=" * 60)
        print("PAC-MAN GAME - MANUAL TESTING")
        print("=" * 60)
        print("\nControls:")
        print("  Arrow Keys - Move Pac-Man (UP, DOWN, LEFT, RIGHT)")
        print("  A - Toggle A* autopilot")
        print("  SPACE - Pause/Resume")
        print("  R - Restart game")
        print("  ESC - Quit")
        print("\nGame State Information:")
        print(f"  Board: {self.game.board.width}x{self.game.board.height}")
        print(f"  Pacman: {self.game.game_state.pacman.position.to_tuple()}")
        print(f"  Ghosts: {len(self.game.game_state.ghosts)}")
        print(f"  Pellets: {len(self.game.board.pellets)}")
        print(f"  Power Pellets: {len(self.game.board.power_pellets)}")
        print("=" * 60 + "\n")

        clock = pygame.time.Clock()

        while self.running:
            self.handle_input()

            if not self.paused:
                self.update()

            self.render()
            self.frame_count += 1

            if self.game.is_game_over() or self.game.is_game_won():
                # Final render the "GAME OVER" or "YOU WIN" UI displays
                self.render()
                pygame.display.flip()

                # Pause the thread for 2 seconds
                pygame.time.delay(2000)

                # Break the loop
                self.running = False

            # Cap at 10 FPS for reasonable gameplay
            clock.tick(10)

        self.quit()

    def quit(self):
        """Clean up and quit."""
        print("\n" + "=" * 60)
        print("GAME STATISTICS")
        print("=" * 60)
        print(f"Final Score: {self.game.game_state.pacman.score}")
        print(f"Moves Made: {self.move_count}")
        print(f"Frames: {self.frame_count}")
        print(f"Pellets Eaten: {self.game.game_state.pellets_eaten}")
        print(f"Power Pellets Eaten: {self.game.game_state.power_pellets_eaten}")
        print(f"Remaining Lives: {self.game.game_state.pacman.lives}")
        print(f"Game Won: {self.game.game_state.game_won}")
        print(f"Game Over: {self.game.game_state.game_over}")
        print("=" * 60 + "\n")

        self.game.quit()
        pygame.quit()
        sys.exit(0)


if __name__ == "__main__":
    player = GamePlayer()
    player.run()
