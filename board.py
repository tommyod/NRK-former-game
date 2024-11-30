"""
Game board definition
---------------------

An implementation of the NRK.no game "Former" (Shapes), found at:
https://www.nrk.no/former-1.17105310

A Board is an immutable class that represents a game.
Colors are represented as integers 1, 2, 3, ... on a grid.

    >>> grid = [[1, 2],
    ...         [2, 1]]
    >>> board = Board(grid)
    >>> board
    Board([[1, 2], [2, 1]])
    >>> print(board)
    12
    21

Clicking on a board returns a new instance, that can be printed:

    >>> board = board.click(0, 1)
    >>> print(board)
    10
    21

All valid clicks (non-zero cells) are given by the `yield_clicks` method:

    >>> list(board.yield_clicks())
    [(0, 0), (1, 0), (1, 1)]

Statistics can be retrieved using these methods:

    >>> len(board)  # The board has size 2 x 2 = 4
    4
    >>> board.remaining  # There are three cells remaining (non-zero)
    3
    >>> board.cleared  # One cell has been cleared
    1
    >>> board.unique_remaining  # There are two unique remaining numbers
    2

The Board instances are immutable - a new instance is always returned.

    >>> original_board = Board(grid)
    >>> new_board = original_board.click(0, 0)
    >>> print(new_board)
    02
    21
    >>> print(original_board)
    12
    21
    >>> new_board == original_board
    False

A canonical representation re-labels the numbers in increasing order and
flips the board if needed.

    >>> grid = [[3, 3, 3],
    ...         [2, 2, 3],
    ...         [2, 1, 2]]
    >>> board = Board(grid).canonicalize()
    >>> print(board)
    111
    122
    232
    >>> print(board.click(1, 1).canonicalize())  # Here the board is flipped
    001
    011
    123

A simpler approach is to simply relabel the board:

    >>> grid = [[3, 3, 3],
    ...         [2, 2, 3],
    ...         [2, 1, 2]]
    >>> board = Board(grid).relabel()
    >>> print(board)
    111
    221
    232

Boards are considered equal iff all cells are equal, not if their canonical
forms are equal. Canonicalization is an indempotent function, so applying it
twice gives the same result as applying it twice.

    >>> Board(grid) == Board(grid).canonicalize()
    False
    >>> Board(grid).canonicalize() == Board(grid).canonicalize().canonicalize()
    True

Finally, all children (results of all valid clicks) can be retrieved:

    >>> for move, child in board.children():
    ...     assert board.click(*move) == child
    ...     print(child)
    ...     print()
    000
    220
    232
    <BLANKLINE>
    001
    011
    132
    <BLANKLINE>
    101
    211
    222
    <BLANKLINE>
    110
    221
    231
    <BLANKLINE>
"""

from typing import List, Tuple
from typing import Set
from copy import deepcopy
import itertools
import random
import functools


class Board:
    """Board for game instances.

    Examples
    --------
    >>> grid = [[1, 2, 3],
    ...         [2, 3, 3],
    ...         [4, 4, 4]]
    >>> board = Board(grid).click(0, 0).display()
    023
    233
    444
    >>> board = Board(grid).click(2, 0).display()
    000
    123
    233
    >>> board = Board(grid).click(0, 2).display()
    100
    220
    444
    """

    def __init__(self, grid: List[List[int]]):
        self.grid = [row[:] for row in grid]
        self.rows = len(grid)
        self.cols = len(grid[0]) if grid else 0

    @classmethod
    def generate_random(cls, shape=(4, 4), maximum=4, seed=None):
        """Generate a random board of a given shape.

        Examples
        --------
        >>> print(Board.generate_random(shape=(2, 4), seed=42))
        1132
        2211
        """
        rows, cols = shape
        rng = random.Random(seed)
        grid = [[rng.randint(1, maximum) for j in range(cols)] for i in range(rows)]
        return cls(grid)

    def __len__(self) -> int:
        """Returns rows times columns."""
        return self.rows * self.cols

    def flip(self):
        """Reverse the order of the columns.

        Examples
        --------
        >>> grid = [[1, 2, 3],
        ...         [1, 2, 3],
        ...         [1, 2, 3]]
        >>> Board(grid).flip().display()
        321
        321
        321
        """
        return type(self)([row[::-1] for row in self.grid])

    def relabel(self):
        """Relabel the board, going from row by row from top left,
        relabeing unique numbers are they are seen to 1, 2, 3, ...

        Examples
        --------
        >>> grid = [[3, 1, 2],
        ...         [3, 3, 4],
        ...         [1, 3, 1]]
        >>> print(Board(grid).relabel())
        123
        114
        212
        """
        board = self.copy()
        next_label = 1
        mapping = {0: 0}  # Keep 0 mapped to 0

        for i, j in itertools.product(range(board.rows), range(board.cols)):
            val = board.grid[i][j]
            if val not in mapping:
                mapping[val] = next_label
                next_label += 1
            board.grid[i][j] = mapping[val]

        return board

    def canonicalize(self, was_flipped=False):
        """Rename the grid starting with 1, 2, ... and flip it.

        If `would_flip` is True, then a tuple (board, was_flipped) is returned.

        Examples
        --------
        >>> grid = [[3, 1],
        ...         [1, 2]]
        >>> Board(grid).canonicalize().display()
        12
        23
        >>> grid = [[3, 0],
        ...         [2, 1]]
        >>> Board(grid).canonicalize().display()
        01
        23
        >>> Board(grid).canonicalize(was_flipped=True)
        (Board([[0, 1], [2, 3]]), True)
        """

        # Once a board is relabeled, there are only four options.
        # They are generated by the sequence:
        #   relabel() -> flip() -> relabel() -> flip()
        # Our of these four, we choose the lowest number (lexicographic order)

        b0 = Board(self.grid).relabel()
        b1 = b0.flip()  # Flipped board
        b2 = b1.relabel()  # Flipped board
        b3 = b2.flip()

        boards = ((b, i) for (i, b) in enumerate([b0, b1, b2, b3]))

        board, board_idx = min(boards, key=lambda t: t[0]._to_number())

        if was_flipped:
            return board, board_idx in (1, 2)  # Second are third are flipped
        else:
            return board

    def _to_number(self):
        """Convert the board to an integer."""
        return int("".join(str(num) for row in self.grid for num in row))

    @functools.cached_property
    def unique_remaining(self):
        """Count number of unique non-zeros remaining in the board.

        Examples
        --------
        >>> grid = [[1, 2, 3],
        ...         [2, 3, 3],
        ...         [4, 4, 4]]
        >>> Board(grid).click(0, 0).unique_remaining
        3
        >>> Board(grid).click(1, 0).unique_remaining
        4
        """
        return len(set(self.grid[i][j] for (i, j) in self.yield_clicks()))

    def yield_clicks(self):
        """Yield all combinations of (i, j) that are non-zero in the board."""
        for i, j in itertools.product(range(self.rows), range(self.cols)):
            if self.grid[i][j] > 0:
                yield (i, j)

    def children(self, return_removed=False):
        """Yields (move, board) for all children boards.

        Examples
        --------
        >>> grid = [[1, 2, 3],
        ...         [2, 3, 3],
        ...         [4, 4, 4]]
        >>> for move, board in Board(grid).children():
        ...     print(move)
        ...     print(board)
        (0, 0)
        023
        233
        444
        (0, 1)
        103
        233
        444
        (0, 2)
        100
        220
        444
        (1, 0)
        023
        133
        444
        (2, 0)
        000
        123
        233
        """
        seen = set()  # Do not yield the same board twice

        if self.is_solved:
            return

        for i, j in self.yield_clicks():
            board, num_removed = self.click(i, j, return_removed=True)

            if board in seen:
                continue

            if return_removed:
                yield (i, j), board, num_removed
            else:
                yield (i, j), board

            seen.add(board)

    def click(self, i: int, j: int, return_removed=False) -> bool:
        """Handle a click at position (i,j).

        Examples
        --------
        >>> board = Board([[1, 2], [3, 2]])
        >>> board.click(0, 1, return_removed=True)
        (Board([[1, 0], [3, 0]]), 2)
        >>> board.click(1, 0, return_removed=True)
        (Board([[0, 2], [1, 2]]), 1)
        """
        outside = not (0 <= i < self.rows and 0 <= j < self.cols)
        if outside or self.grid[i][j] == 0:
            raise ValueError("Invalid click.")

        new_board, num_removed = self.copy()._apply_move(i, j)
        return (new_board, num_removed) if return_removed else new_board

    def _apply_move(self, i: int, j: int):
        """Apply move and gravity, modifies the board in place."""
        connected = self._find_connected(i, j, value=self.grid[i][j], visited=set())

        # Remove connected cells
        for row, col in connected:
            self.grid[row][col] = 0

        # Apply gravity
        self._apply_gravity(self.grid)
        return self, len(connected)

    def _find_connected(
        self, i: int, j: int, value: int, visited: Set[Tuple[int, int]]
    ) -> Set[Tuple[int, int]]:
        """Find all cells connected to (i,j) with the same value."""
        outside = not (0 <= i < self.rows and 0 <= j < self.cols)

        if outside or ((i, j) in visited) or (self.grid[i][j] != value):
            return visited

        visited.add((i, j))
        for di, dj in [(0, 1), (1, 0), (0, -1), (-1, 0)]:
            # Recursive call, which updates the `visisted` argument
            self._find_connected(i + di, j + dj, value, visited)
        return visited

    def _apply_gravity(self, grid: List[List[int]]):
        """Make cells fall down to fill cleared spaces,
        modifying the board in place."""

        for j in range(self.cols):
            # Get non-zero values in column
            values = [grid[i][j] for i in range(self.rows) if grid[i][j] != 0]

            # Pad with zeros at top
            values = [0] * (self.rows - len(values)) + values

            # Update column
            for i in range(self.rows):
                grid[i][j] = values[i]

        return self

    @functools.cached_property
    def is_solved(self) -> bool:
        """Check if the board is cleared (all cells are zero)."""
        # Only need to check bottom row because of gravity
        return all(cell == 0 for cell in self.grid[-1])

    @functools.cached_property
    def remaining(self) -> int:
        """Count number of remaining non-cleared (non-zero) cells.

        Examples
        --------
        >>> grid = [[1, 2, 3],
        ...         [2, 3, 3],
        ...         [4, 4, 4]]
        >>> Board(grid).remaining
        9
        >>> Board(grid).click(2, 2).remaining
        6
        """
        return sum(cell > 0 for row in self.grid for cell in row)

    @functools.cached_property
    def cleared(self) -> int:
        """Returns the number of cleared cells (zero cells)."""
        return len(self) - self.remaining

    def display(self, click=None):
        """Print current board state."""
        if click:
            grid = deepcopy(self.grid)
            i, j = click
            grid[i][j] = "X"
            for row in grid:
                print("".join(str(x) for x in row))
        else:
            for row in self.grid:
                print("".join(str(x) for x in row))

    def __str__(self):
        """String representation of the board."""
        return "\n".join("".join(str(cell) for cell in row) for row in self.grid)

    def __repr__(self):
        return f"{type(self).__name__}({repr(self.grid)})"

    def copy(self):
        return type(self)(self.grid)  # The __init__ method takes a copy

    def __hash__(self):
        return hash(tuple(tuple(row) for row in self.grid))  # 1.15 μs
        return self._to_number()  # 5.59 μs

    def __eq__(self, other):
        if not isinstance(other, Board):
            return False
        if not (self.rows == other.rows and self.cols == other.cols):
            return False
        return self.grid == other.grid

    def verify_solution(self, moves):
        """Returns True if a sequence of moves solves the board."""
        board = self.copy()
        for move in moves:
            board = board.click(*move)
        return board.is_solved

    def plot(self, ax=None, click=None, n_colors=None, show_values=False):
        """Plot the current board state using matplotlib.

        Returns
        -------
        matplotlib.axes.Axes
            The axis containing the plot.

        Examples
        --------
        >>> import matplotlib.pyplot as plt
        >>> grid = [[1, 2, 3], [2, 3, 3], [4, 4, 4]]
        >>> board = Board(grid)
        >>> ax = board.plot()
        >>> plt.close()
        >>> ax = board.plot(click=(1, 1))
        >>> plt.close()
        """
        try:
            import matplotlib.pyplot as plt
        except ModuleNotFoundError:
            return None

        if ax is None:
            _, ax = plt.subplots(figsize=(self.cols * 0.33, self.rows * 0.33))

        # Create distinct colors for numbers 1-9 using a colormap
        # Skip pure white (reserved for 0) and very light colors
        unique_numbers = sorted(set(num for row in self.grid for num in row if num > 0))
        n_colors = len(unique_numbers) if n_colors is None else n_colors
        color_map = {i: plt.cm.Set3.colors[i % 12] for i in range(1, n_colors + 1)}
        color_map[0] = "white"  # Cleared cells are white

        # Create the grid
        for i in range(self.rows):
            for j in range(self.cols):
                value = self.grid[i][j]
                color = color_map[value]

                # Plot cell
                circle = plt.Circle(
                    (j + 0.5, self.rows - i - 0.5),
                    0.4,
                    color=color,
                    ec="black",
                    zorder=1,
                )
                ax.add_patch(circle)

                # Add number text
                if value > 0 and show_values:
                    ax.text(
                        j + 0.5,
                        self.rows - i - 0.5,
                        str(value),
                        ha="center",
                        va="center",
                        fontsize=12,
                        fontweight="bold",
                        zorder=2,
                    )

        # Add X marker if click position is given
        if click is not None:
            i, j = click
            ax.plot(
                [j + 0.5 - 0.2, j + 0.5 + 0.2],
                [self.rows - i - 0.5 - 0.2, self.rows - i - 0.5 + 0.2],
                "k-",
                linewidth=2,
                zorder=3,
            )
            ax.plot(
                [j + 0.5 - 0.2, j + 0.5 + 0.2],
                [self.rows - i - 0.5 + 0.2, self.rows - i - 0.5 - 0.2],
                "k-",
                linewidth=2,
                zorder=3,
            )

        # Set up the axes
        ax.set_xlim(-0.1, self.cols + 0.1)
        ax.set_ylim(-0.1, self.rows + 0.1)
        ax.set_xticks(range(self.cols))
        ax.set_yticks(range(self.rows))
        ax.grid(True)
        ax.set_aspect("equal")

        # Hide axis labels since we're showing a grid
        ax.set_xticklabels([])
        ax.set_yticklabels([])

        return ax


class LabelInvariantBoard(Board):
    """A board that is canonical and always yields canonical children."""

    def __init__(self, grid, check=True):
        super().__init__(grid)
        if check:
            assert self == Board(grid).relabel(), "Input must be correctly labeled"

    def click(self, i: int, j: int, return_removed=False) -> bool:
        """Handle a click at position (i,j).

        Examples
        --------
        >>> board = LabelInvariantBoard([[1, 2], [1, 1]])
        >>> print(board.click(0, 0))
        00
        01
        """
        board, num_removed = Board(self.grid).click(i, j, return_removed=True)
        board = board.relabel()
        if return_removed:
            return type(self)(board.grid, check=False), num_removed
        else:
            return type(self)(board.grid, check=False)

    def children(self, return_removed=False):
        """Yields (move, flip, board) for all children boards.
        To get the new board, apply the move, then flip it.

        Examples
        --------
        >>> board = LabelInvariantBoard([[1, 2, 3], [1, 3, 2]])
        >>> for move, child in board.children():
        ...     print(move)
        ...     print(board)
        (0, 0)
        123
        132
        (0, 1)
        123
        132
        (0, 2)
        123
        132
        (1, 1)
        123
        132
        (1, 2)
        123
        132
        """
        seen = set()

        for i, j in self.yield_clicks():
            board, num_removed = self.click(i, j, return_removed=True)
            board = type(self)(board.grid, check=False)

            if board in seen:
                continue
            else:
                if return_removed:
                    yield (i, j), board, num_removed
                else:
                    yield (i, j), board
                seen.add(board)


class CanonicalBoard(Board):
    """A board that is canonical and always yields canonical children."""

    def __init__(self, grid):
        super().__init__(grid)
        if not self == Board(grid).canonicalize():
            msg = f"Input must be canonical. Got: \n{Board(grid)}"
            raise TypeError(msg)

    def click(self, i: int, j: int, return_removed=False) -> bool:
        """Handle a click at position (i,j).

        Examples
        --------
        >>> board = CanonicalBoard([[1, 2], [1, 1]])
        >>> print(board.click(0, 0))
        00
        01
        >>> board.click(0, 0, return_removed=True)
        (CanonicalBoard([[0, 0], [0, 1]]), 3)
        """
        board, num_removed = Board(self.grid).click(i, j, return_removed=True)
        board = type(self)(board.canonicalize().grid)
        return (board, num_removed) if return_removed else board

    def canonicalize(self):
        return Board(self.grid).canonicalize()

    def children(self, return_removed=False, return_flip=False):
        """Yields (move, flip, board) for all children boards.
        To get the new board, apply the move, then flip it.

        Examples
        --------
        >>> board = CanonicalBoard([[1, 2, 3], [1, 3, 2]])
        >>> for move, child, flip in board.children(return_flip=True):
        ...     print(move, flip)
        ...     print(board)
        (0, 0) False
        123
        132
        (0, 1) False
        123
        132
        (0, 2) True
        123
        132
        (1, 1) False
        123
        132
        (1, 2) True
        123
        132
        """
        seen = set()
        non_canonical_board = Board(self.grid)

        for i, j in non_canonical_board.yield_clicks():
            board, num_removed = non_canonical_board.click(i, j, return_removed=True)

            # TODO: To recover solution path, flips must be taken into account.
            # This is not implemented
            board, flip = board.canonicalize(was_flipped=True)
            # j = self.cols - j - 1 if flip else j

            if board in seen:
                continue
            else:
                if return_removed and return_flip:
                    yield (i, j), type(self)(board.grid), num_removed, flip
                elif return_removed and not return_flip:
                    yield (i, j), type(self)(board.grid), num_removed
                elif not return_removed and return_flip:
                    yield (i, j), type(self)(board.grid), flip
                else:
                    yield (i, j), type(self)(board.grid)

                seen.add(board)


if __name__ == "__main__":
    import pytest

    pytest.main(
        [
            __file__,
            "-v",
            "-v",
            "--doctest-modules",
            "-l",
        ]
    )
