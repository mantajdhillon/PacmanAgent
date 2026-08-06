"""Game configuration constants."""

from collections import OrderedDict


class BoundedCache(OrderedDict):
    """Small LRU cache that cannot grow without limit during long games."""

    def __init__(self, max_size=8192):
        super().__init__()
        self.max_size = max_size

    def __getitem__(self, key):
        value = super().__getitem__(key)
        self.move_to_end(key)
        return value

    def __setitem__(self, key, value):
        if key in self:
            self.move_to_end(key)
        super().__setitem__(key, value)
        while len(self) > self.max_size:
            self.popitem(last=False)


# Board dimensions
BOARD_WIDTH = 21
BOARD_HEIGHT = 21

# Display settings
CELL_SIZE = 30
WINDOW_WIDTH = BOARD_WIDTH * CELL_SIZE
WINDOW_HEIGHT = BOARD_HEIGHT * CELL_SIZE
FPS = 10

# Game element types
WALL = 1
EMPTY = 0
PELLET = 2
POWER_PELLET = 3

# Directions
UP = (0, -1)
DOWN = (0, 1)
LEFT = (-1, 0)
RIGHT = (1, 0)
DIRECTIONS = [UP, DOWN, LEFT, RIGHT]

# Colors
BLACK = (0, 0, 0)
WHITE = (255, 255, 255)
YELLOW = (255, 255, 0)
RED = (255, 0, 0)
PINK = (255, 184, 255)
CYAN = (0, 255, 255)
ORANGE = (255, 184, 82)
BLUE = (100, 100, 255)

# Timeouts and game settings
GHOST_SPEED = 1
PACMAN_SPEED = 1
