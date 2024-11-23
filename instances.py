"""
A collection of instances.
"""

import dataclasses
from board import LabelInvariantBoard


@dataclasses.dataclass
class Instance:
    grid: None
    best: int

    def __post_init__(self):
        self.board = LabelInvariantBoard(self.grid, check=True)


# These boards were retrieved from NRK. Each board lives for 24 hours, and the
# keys are the dates in november 2024 that the boards were live. The "best"
# key is the record solution within the 24 hours the board was live.
{
    16: Instance(
        grid=[
            [1, 2, 2, 3, 1, 4, 4],
            [1, 2, 1, 3, 4, 4, 4],
            [2, 4, 4, 4, 1, 3, 1],
            [1, 2, 1, 1, 3, 4, 3],
            [2, 3, 3, 3, 4, 1, 3],
            [3, 2, 4, 2, 4, 4, 4],
            [1, 3, 4, 2, 1, 4, 4],
            [3, 4, 2, 3, 4, 3, 4],
            [1, 3, 4, 1, 4, 1, 4],
        ],
        best=12,
    ),
    19: Instance(
        grid=[
            [1, 1, 2, 3, 1, 4, 2],
            [3, 4, 4, 3, 1, 1, 2],
            [4, 1, 4, 1, 4, 4, 3],
            [1, 2, 4, 3, 3, 1, 4],
            [3, 2, 1, 2, 3, 2, 4],
            [4, 1, 4, 1, 3, 3, 1],
            [1, 1, 1, 1, 1, 4, 3],
            [3, 4, 1, 4, 2, 2, 3],
            [2, 3, 1, 4, 3, 2, 4],
        ],
        best=12,
    ),
    20: Instance(
        grid=[
            [1, 1, 2, 1, 1, 3, 3],
            [2, 2, 2, 4, 4, 1, 2],
            [4, 2, 3, 2, 2, 3, 4],
            [3, 1, 1, 4, 4, 1, 2],
            [1, 4, 4, 2, 3, 1, 2],
            [1, 4, 1, 2, 4, 1, 2],
            [2, 2, 4, 1, 2, 3, 2],
            [2, 3, 1, 1, 4, 4, 4],
            [2, 1, 1, 4, 3, 2, 4],
        ],
        best=11,
    ),
    21: Instance(
        grid=[
            [1, 1, 1, 1, 2, 3, 1],
            [3, 4, 4, 4, 2, 4, 1],
            [4, 1, 1, 3, 2, 1, 2],
            [3, 2, 3, 2, 2, 2, 3],
            [4, 1, 4, 1, 3, 1, 1],
            [2, 3, 3, 2, 4, 1, 3],
            [2, 1, 3, 4, 4, 3, 2],
            [4, 4, 2, 1, 4, 3, 3],
            [1, 3, 4, 2, 1, 3, 1],
        ],
        best=15,
    ),
    22: Instance(
        grid=[
            [1, 2, 1, 3, 4, 3, 4],
            [4, 4, 2, 4, 2, 3, 1],
            [2, 4, 2, 4, 4, 3, 4],
            [2, 1, 2, 2, 3, 2, 4],
            [2, 1, 4, 1, 4, 2, 1],
            [1, 1, 4, 2, 4, 1, 1],
            [1, 4, 1, 2, 2, 4, 1],
            [2, 4, 2, 2, 2, 1, 3],
            [4, 4, 3, 2, 3, 4, 3],
        ],
        best=12,
    ),
    23: Instance(
        grid=[
            [1, 2, 2, 2, 2, 1, 1],
            [1, 3, 4, 2, 4, 1, 1],
            [1, 4, 2, 1, 2, 3, 1],
            [2, 2, 2, 4, 2, 2, 3],
            [4, 1, 1, 4, 1, 4, 4],
            [3, 4, 2, 1, 1, 2, 1],
            [2, 1, 3, 4, 3, 2, 4],
            [2, 2, 1, 1, 3, 2, 4],
            [3, 4, 4, 2, 4, 3, 3],
        ],
        best=13,
    ),
}


# These are example boards that are "hard" to solve. The best solution has
# been verified by running A* with admissible heuristic.
example_boards = [
    Instance(grid=[[1, 1, 2], [2, 3, 1], [4, 2, 2]], best=4),
    Instance(grid=[[1, 2, 2, 1], [3, 2, 4, 4], [2, 1, 1, 2], [3, 4, 3, 1]], best=6),
    Instance(
        grid=[
            [1, 2, 3, 4, 4],
            [4, 1, 2, 3, 1],
            [2, 4, 3, 3, 2],
            [3, 1, 4, 1, 2],
            [3, 1, 2, 2, 1],
        ],
        best=9,
    ),
]