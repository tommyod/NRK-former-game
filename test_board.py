"""
Tests for the board.
"""

import pytest
import random


from board import Board, CanonicalBoard
from solvers import a_star_search


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
        assert complex_board.remaining == 9
        assert complex_board.unique_remaining == 4

        clicked = complex_board.click(2, 0)
        assert clicked.remaining == 6
        assert clicked.unique_remaining == 3

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
        assert not simple_board.is_solved

        # Create a solved board
        solved = Board([[0, 0], [0, 0]])
        assert solved.is_solved

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

    @pytest.mark.parametrize("seed", range(100))
    def test_lower_and_upper_bounds(self, seed):
        # Create a random board with a random shape
        rng = random.Random(seed)
        rows, cols = rng.randint(2, 4), rng.randint(2, 4)
        board = Board.generate_random(shape=(rows, cols), seed=seed)

        # A* returns an optimal solution. Check that it's within bounds
        opt_moves = len(a_star_search(board))
        assert board.lower_bound <= opt_moves <= board.upper_bound

    @pytest.mark.parametrize("seed", range(100))
    def test_that_canonicalization_is_idempotent(self, seed):
        # Create a random board with a random shape
        rng = random.Random(seed)
        rows, cols = rng.randint(2, 4), rng.randint(2, 4)
        board = Board.generate_random(shape=(rows, cols), seed=seed).canonicalize()

        # Already canonical, so canonicalization does nothing
        assert board == board.canonicalize()

        # This sequence always leads to the same result
        board = Board.generate_random(shape=(rows, cols), seed=seed)
        assert board.relabel().flip().relabel().flip().relabel() == board.relabel()

    @pytest.mark.parametrize("seed", range(1))
    def test_canonical_clicks(self, seed):
        # Create a random board with a random shape
        rng = random.Random(seed)
        rows, cols = rng.randint(2, 2), rng.randint(2, 3)
        board = Board.generate_random(shape=(rows, cols), seed=seed).canonicalize()

        # Create a canonical board
        canonical_board = CanonicalBoard(board.grid)

        for (i, j), child, flip in canonical_board.children(return_flip=True):
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
