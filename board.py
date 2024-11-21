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
    >>> board.remaining()  # There are three cells remaining (non-zero)
    3
    >>> board.cleared()  # One cell has been cleared
    1
    >>> board.unique_remaining()  # There are two unique remaining numbers
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
import pytest
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

        b0 = self.copy().relabel()
        b1 = b0.flip()  # Flipped board
        b2 = b1.relabel()  # Flipped board
        b3 = b2.flip()

        boards = ((b, i) for (i, b) in enumerate([b0, b1, b2, b3]))

        board, board_idx = min(boards, key=lambda t: hash(t[0]))

        if was_flipped:
            return board, board_idx in (1, 2)  # Second are third are flipped
        else:
            return board

    def _to_number(self):
        """Convert the board to an integer."""
        return int("".join(str(num) for row in self.grid for num in row))

    def unique_remaining(self):
        """Count number of unique non-zeros remaining in the board.

        Examples
        --------
        >>> grid = [[1, 2, 3],
        ...         [2, 3, 3],
        ...         [4, 4, 4]]
        >>> Board(grid).click(0, 0).unique_remaining()
        3
        >>> Board(grid).click(1, 0).unique_remaining()
        4

        """
        return len(set(self.grid[i][j] for (i, j) in self.yield_clicks()))

    def yield_clicks(self):
        """Yield all combinations of (i, j) that are non-zero in the board."""
        for i, j in itertools.product(range(self.rows), range(self.cols)):
            if self.grid[i][j] > 0:
                yield (i, j)

    def children(self):
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

        if self.is_solved():
            return

        for i, j in self.yield_clicks():
            board = self.copy().click(i, j)

            if board in seen:
                continue

            yield (i, j), board
            seen.add(board)

    def click(self, i: int, j: int) -> bool:
        """Handle a click at position (i,j)."""
        outside = not (0 <= i < self.rows and 0 <= j < self.cols)
        if outside or self.grid[i][j] == 0:
            raise ValueError("Invalid click.")

        return self.copy()._apply_move(i, j)

    def _apply_move(self, i: int, j: int):
        """Apply move and gravity, modifies the board in place."""
        connected = self._find_connected(i, j, color=self.grid[i][j], visited=set())

        # Remove connected cells
        for row, col in connected:
            self.grid[row][col] = 0

        # Apply gravity
        self._apply_gravity(self.grid)
        return self

    def _find_connected(
        self, i: int, j: int, color: int, visited: Set[Tuple[int, int]]
    ) -> Set[Tuple[int, int]]:
        """Find all cells connected to (i,j) with the same color."""
        outside = not (0 <= i < self.rows and 0 <= j < self.cols)

        if outside or ((i, j) in visited) or (self.grid[i][j] != color):
            return visited

        visited.add((i, j))
        for di, dj in [(0, 1), (1, 0), (0, -1), (-1, 0)]:
            # Recursive call, which updates the `visisted` argument
            self._find_connected(i + di, j + dj, color, visited)
        return visited

    def _apply_gravity(self, grid: List[List[int]]):
        """Make cells fall down to fill empty spaces,
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

    def is_solved(self) -> bool:
        """Check if the board is cleared (all cells are zero)."""
        # Only need to check bottom row because of gravity
        return all(cell == 0 for cell in self.grid[-1])

    def remaining(self) -> int:
        """Count number of remaining non-zero cells.

        Examples
        --------
        >>> grid = [[1, 2, 3],
        ...         [2, 3, 3],
        ...         [4, 4, 4]]
        >>> Board(grid).remaining()
        9
        >>> Board(grid).click(2, 2).remaining()
        6
        """
        return sum(cell > 0 for row in self.grid for cell in row)

    @functools.cache
    def cleared(self) -> int:
        """Returns the number of cleared cells (zero cells)."""
        return len(self) - self.remaining()

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
        # return hash(tuple(tuple(row) for row in self.grid))
        return self._to_number()

    def __eq__(self, other):
        if not isinstance(other, Board):
            return False
        if not (self.rows == other.rows and self.cols == other.cols):
            return False

        coords = itertools.product(range(self.rows), range(self.cols))
        return all(self.grid[i][j] == other.grid[i][j] for (i, j) in coords)

    def plot(self, ax=None, click=None, n_colors=None, show_values=False):
        """Plot the current board state using matplotlib.

        Parameters
        ----------
        ax : matplotlib.axes.Axes, optional
            The axis to plot on. If None, creates a new figure and axis.
        click : tuple(int, int), optional
            If provided, marks the position (i,j) with an X.

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
        color_map[0] = "white"  # Empty cells are white

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
            assert self == Board(grid).relabel(), "Input must be canonical"

    def click(self, i: int, j: int) -> bool:
        """Handle a click at position (i,j).

        Examples
        --------
        >>> board = LabelInvariantBoard([[1, 2], [1, 1]])
        >>> print(board.click(0, 0))
        00
        01
        """
        board = Board(self.grid).click(i, j).relabel()
        return type(self)(board.grid, check=False)

    def children(self):
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
            board = self.click(i, j)

            if board in seen:
                continue
            else:
                yield (i, j), type(self)(board.grid, check=False)
                seen.add(board)


class CanonicalBoard(Board):
    """A board that is canonical and always yields canonical children."""

    def __init__(self, grid):
        super().__init__(grid)
        if not self == Board(grid).canonicalize():
            msg = f"Input must be canonical. Got: \n{Board(grid)}"
            raise TypeError(msg)

    def click(self, i: int, j: int) -> bool:
        """Handle a click at position (i,j).

        Examples
        --------
        >>> board = CanonicalBoard([[1, 2], [1, 1]])
        >>> print(board.click(0, 0))
        00
        01
        """
        board = Board(self.grid).click(i, j).canonicalize()
        return type(self)(board.grid)

    def canonicalize(self):
        return Board(self.grid).canonicalize()

    def children(self):
        """Yields (move, flip, board) for all children boards.
        To get the new board, apply the move, then flip it.

        Examples
        --------
        >>> board = CanonicalBoard([[1, 2, 3], [1, 3, 2]])
        >>> for move, flip, child in board.children():
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
            board = non_canonical_board.click(i, j)

            board, flip = board.canonicalize(was_flipped=True)
            # j = self.cols - j - 1 if flip else j

            if board in seen:
                continue
            else:
                yield (i, j), flip, type(self)(board.grid)
                seen.add(board)


class TestBoard:
    @pytest.fixture
    def simple_board(self):
        """Create a simple 2x2 board for basic tests"""
        return Board([[1, 2], [2, 1]])

    @pytest.fixture
    def complex_board(self):
        """Create a more complex board for testing game mechanics"""
        return Board([[1, 2, 3], [2, 3, 3], [4, 4, 4]])

    def test_board_initialization(self):
        """Test that boards are initialized correctly"""
        grid = [[1, 2], [3, 4]]
        board = Board(grid)
        assert board.rows == 2
        assert board.cols == 2
        assert board.grid == grid
        assert board.grid is not grid  # Should be a deep copy

    def test_board_equality(self):
        """Test board equality comparisons"""
        board1 = Board([[1, 2], [3, 4]])
        board2 = Board([[1, 2], [3, 4]])
        board3 = Board([[2, 1], [3, 4]])

        assert board1 == board2
        assert board1 != board3
        assert board1 != "not a board"

    def test_basic_click(self, simple_board):
        """Test basic clicking behavior"""
        clicked = simple_board.click(0, 0)
        assert clicked.grid == [[0, 2], [2, 1]]
        assert clicked != simple_board  # Should create new instance

    def test_invalid_clicks(self, simple_board):
        """Test that invalid clicks raise appropriate errors"""
        with pytest.raises(ValueError):
            simple_board.click(-1, 0)  # Out of bounds
        with pytest.raises(ValueError):
            simple_board.click(0, 2)  # Out of bounds

        # Click empty cell after clearing it
        clicked = simple_board.click(0, 0)
        with pytest.raises(ValueError):
            clicked.click(0, 0)  # Clicking on empty space

    def test_gravity(self, complex_board):
        """Test that pieces fall correctly after clearing"""
        # Click bottom row to test gravity
        board = complex_board.click(2, 0)
        expected = [[0, 0, 0], [1, 2, 3], [2, 3, 3]]
        assert board.grid == expected

    def test_connected_removal(self, complex_board):
        """Test that connected pieces are removed properly"""
        # Click on a '4' in the bottom row
        board = complex_board.click(2, 1)
        # All connected 4's should be removed
        assert all(4 not in row for row in board.grid)

    def test_board_statistics(self, complex_board):
        """Test statistical methods of the board"""
        assert len(complex_board) == 9  # 3x3 board
        assert complex_board.remaining() == 9
        assert complex_board.unique_remaining() == 4

        clicked = complex_board.click(2, 0)
        assert clicked.remaining() == 6
        assert clicked.unique_remaining() == 3

    def test_canonicalization(self):
        """Test board canonicalization"""
        board = Board([[3, 1], [1, 2]])
        canonical = board.canonicalize()
        assert canonical.grid == [[1, 2], [2, 3]]  # Numbers should be remapped

    def test_children_generation(self, simple_board):
        """Test generation of child boards"""
        children = list(simple_board.children())
        assert len(children) == 4  # Should have 4 possible moves

        # Verify all children are unique
        child_boards = [board for _, board in children]
        assert len(set(hash(board) for board in child_boards)) == len(child_boards)

    def test_is_solved(self, simple_board):
        """Test win condition detection"""
        assert not simple_board.is_solved()

        # Create a solved board
        solved = Board([[0, 0], [0, 0]])
        assert solved.is_solved()

    def test_board_copy(self, complex_board):
        """Test that board copying works correctly"""
        copy = complex_board.copy()
        assert copy == complex_board
        assert copy is not complex_board
        assert copy.grid is not complex_board.grid

        # Modify copy shouldn't affect original
        copy.grid[0][0] = 9
        assert complex_board.grid[0][0] != 9

    def test_display_output(self, simple_board, capsys):
        """Test display output formatting"""
        simple_board.display()
        captured = capsys.readouterr()
        assert captured.out == "12\n21\n"

        # Test display with click
        simple_board.display(click=(0, 0))
        captured = capsys.readouterr()
        assert captured.out == "X2\n21\n"

    def test_flipping(self):
        """Test board flipping functionality"""
        board = Board([[1, 2, 3], [4, 5, 6]])
        flipped = board.flip()
        assert flipped.grid == [[3, 2, 1], [6, 5, 4]]

    @pytest.mark.parametrize("seed", range(10))
    def test_that_relabeling_and_flipping(self, seed):
        # Create a random board with a random shape
        rng = random.Random(seed)
        rows, cols = rng.randint(2, 4), rng.randint(2, 4)
        board = Board.generate_random(shape=(rows, cols), seed=seed).canonicalize()
        assert board.relabel().flip().relabel().flip().relabel() == board.relabel()

    @pytest.mark.parametrize("seed", range(10))
    def test_that_canonicalization_is_idempotent(self, seed):
        # Create a random board with a random shape
        rng = random.Random(seed)
        rows, cols = rng.randint(2, 4), rng.randint(2, 4)
        board = Board.generate_random(shape=(rows, cols), seed=seed).canonicalize()
        assert board == board.canonicalize()

    @pytest.mark.parametrize("seed", range(1))
    def test_canonical_clicks(self, seed):
        # Create a random board with a random shape
        rng = random.Random(seed)
        rows, cols = rng.randint(2, 2), rng.randint(2, 3)
        board = Board.generate_random(shape=(rows, cols), seed=seed).canonicalize()

        # Create a canonical board
        canonical_board = CanonicalBoard(board.grid)

        for (i, j), flip, child in canonical_board.children():
            assert child.canonicalize() == child

            if flip:
                assert child == board.click(i, j).flip().canonicalize()
            else:
                assert child == board.click(i, j).canonicalize()


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
