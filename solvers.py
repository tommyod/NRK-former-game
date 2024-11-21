#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""

Best-first search
-----------------

The board below can be solved in 3 moves, but best-first uses 4 moves.

>>> grid = [[3, 3, 3],
...         [2, 2, 3],
...         [2, 1, 2]]
>>> board = Board(grid)
>>> moves = list(best_first_search(board))
>>> moves
[(0, 0), (1, 0), (2, 2), (2, 1)]
>>> for move in moves:
...     board = board.click(*move)
>>> board.is_solved()
True


Breadth-first search (BFS)
--------------------------

>>> grid = [[3, 3, 3],
...         [2, 2, 3],
...         [2, 1, 2]]
>>> board = Board(grid)
>>> moves = breath_first_search(board)
>>> moves
[(0, 0), (2, 1), (1, 0)]


A* search
---------

A* search with an admissible heuristic. This algorithm is guaranteed to
always return a minimum path, solving the board in the fewest moves possible.

>>> grid = [[3, 3, 3],
...         [2, 2, 3],
...         [2, 1, 2]]
>>> board = Board(grid)
>>> moves = a_star_search(board)
>>> moves
[(0, 0), (2, 1), (1, 0)]

>>> grid = [[0, 0, 3, 0, 0],
...         [0, 3, 2, 3, 3],
...         [3, 2, 1, 2, 3]]
>>> board = Board(grid)
>>> moves = a_star_search(board)
>>> moves
[(2, 2), (2, 1), (1, 4)]


Heuristic search
----------------

>>> grid = [[3, 3, 3],
...         [2, 2, 3],
...         [2, 1, 2]]
>>> board = Board(grid)
>>> solutions = heuristic_search(board)
>>> for moves in solutions:
...     print(moves)
[(0, 0), (2, 1), (1, 0)]

>>> board = Board([[1, 4, 4, 3],
...                [3, 4, 2, 2],
...                [4, 4, 2, 4],
...                [3, 4, 2, 2],
...                [3, 1, 1, 3]])
>>> for moves in heuristic_search(board):
...     print(moves)
[(1, 2), (0, 1), (2, 0), (4, 0), (3, 3)]
>>> len(moves) == len(a_star_search(board))
True



"""

from dataclasses import dataclass
from heapq import heappush, heappop
from collections import deque
import pytest
import random
import itertools
import functools

from board import Board, LabelInvariantBoard


def best_first_search(board: Board, power=None, seed=None):
    """Greedy search. Choose the move that clears the most cells.

    If power is a number, then the algorithm is no longer deterministic.
    Instead, it records the number of cleared cells per child and chooses
    a random move with probability weights: cleared**power
    """

    rng = random.Random(seed)
    stack = [board.copy()]

    while stack:
        board = stack.pop()

        if board.is_solved():
            return

        # Go through all children and record how many are removed
        possibilities = []

        for move, next_board in board.children():
            cleared = board.remaining() - next_board.remaining()
            possibilities.append((cleared, move))

        # Deterministic selection of the next move
        if power is None:
            _, move = max(possibilities)

        # Randomized selection of the next move
        else:
            assert power >= 0
            weights = [cleared**power for (cleared, _) in possibilities]
            ((_, move),) = rng.choices(possibilities, weights=weights, k=1)

        # Apply the best move and add board to the stack
        stack.append(board.click(*move))
        yield move


def breath_first_search(board: Board) -> list:
    """Breadth-first search to find shortest solution path.

    This approach is not very efficient, but it is guaranteed to return
    a minimum path, solving the board in the fewest moves possible.
    """

    # Queue of (board, moves) tuples using a deque for efficient popleft
    queue = deque([(board.copy(), [])])

    # Track visited states to avoid cycles
    visited = {board.copy()}

    while queue:
        current_board, moves = queue.popleft()

        # Check if we've found a solution
        if current_board.is_solved():
            return moves

        # Try all possible moves from current state
        for (i, j), next_board in current_board.children():
            # Skip if we've seen this state before
            if next_board in visited:
                continue

            # Add new state to queue and visited set
            visited.add(next_board)
            queue.append((next_board, moves + [(i, j)]))

    # No solution found
    return None


# =============================================================================


@dataclass(frozen=True)
class AStarNode:
    """Make board states comparable for search."""

    board: Board
    moves: tuple  # Using tuple instead of list since lists aren't hashable

    def g(self):
        return len(self.moves)

    def h(self):
        return estimate_remaining(self.board)

    def f(self):
        num_moves = len(self.moves)
        # Return (admissible_heuristic(), non_admissible(), non_admissible())
        # The overall result is still admissible, but the second and third
        # component of the tuple act as tie-breakers
        cleared_per_move = self.board.cleared() / num_moves
        return (self.g() + self.h(), -cleared_per_move, -num_moves)

    def __lt__(self, other):
        return self.f() < other.f()


def a_star_search(board: Board) -> list:
    """A star search with a consistent heuristic."""

    # f(n) = num_moves + heuristic(n), board, moves in a SearchNode class
    heap = [AStarNode(board.copy(), ())]
    g_scores = {board.copy(): 0}  # Keep track of nodes seen and number of moves

    while heap:
        # Pop the smallest item from the heap
        current = heappop(heap)
        current_g = len(current.moves)

        # The path to current node is larger than what we've seen, so skip it
        if current_g > g_scores[current.board]:
            continue

        # The board is solved, return the list of moves
        if current.board.is_solved():
            return list(current.moves)

        # Go through all children, created by applying a single move
        for (i, j), next_board in current.board.children():
            # Increment by one, since we need to make one more move to get here
            g = current_g + 1

            # If not seen before, or the path is lower than recorded
            if (next_board not in g_scores) or (g < g_scores[next_board]):
                g_scores[next_board] = g
                next_node = AStarNode(next_board, current.moves + ((i, j),))
                heappush(heap, next_node)


@functools.cache
def consecutive_groups(tuple_):
    """Count how many consecutive groups of True there are.

    Examples
    --------
    >>> consecutive_groups((True, True, True))
    1
    >>> consecutive_groups((True, False, True))
    2
    >>> consecutive_groups((True, True, False))
    1
    >>> consecutive_groups((False, True, True, False, False, True, True))
    2
    """
    return sum(1 for key, group in itertools.groupby(tuple_) if key)


@functools.cache
def estimate_remaining(board: Board) -> int:
    """A lower bound on how many moves are needed to solve.

    A simple heuristic (not this one) is to use unique number of non-zero
    integers. A better heuristic (this one) looks at each color in turn.
    For each color, count whether it appears in each column. For each group
    of columns, separated by columns where the color does not appear, check
    if the color appears. Each color separated by non-colors needs at most
    one move.

    Examples
    --------
    >>> board = Board([[1, 2, 1],
    ...                [2, 2, 2],
    ...                [1, 2, 1]])
    >>> estimate_remaining(board)
    3
    >>> board = Board([[1, 3, 1],
    ...                [2, 3, 1],
    ...                [1, 3, 1]])
    >>> estimate_remaining(board)
    4
    """
    # Transpose the grid
    matrix = list(map(list, zip(*board.grid)))

    # Get the unique integers larger than zero in the matrix
    unique_integers = set(c for col in matrix for c in col if c > 0)

    # For every integer, see if it exists in each column
    integer_in_col = (tuple((c in col) for col in matrix) for c in unique_integers)

    # Count groups of integers separated by other integers
    return sum(consecutive_groups(int_in_col) for int_in_col in integer_in_col)


# =============================================================================


@dataclass(frozen=True)
class HeuristicNode:
    """Make board states comparable for search."""

    board: Board
    moves: tuple

    @functools.cache
    def heuristic(self):
        # No moves have been made, special case
        if not self.moves:
            return 0

        moves = len(self.moves)

        # Clearing 10 nodes in 2 moves is better than 5 in 1 move
        bias = 1  # Bias that can be used to search deep first
        cleared_per_move = self.board.cleared() / moves + bias * moves
        total_estimate = moves + estimate_remaining(self.board)

        # Negate signs, because lower is better in heapq
        return (-cleared_per_move, -moves, total_estimate)

    def __lt__(self, other):
        return self.heuristic() < other.heuristic()


def heuristic_search(board: Board, verbose=False, max_nodes=0):
    """A heuristic search that yields solutions as they are found.

    If run long enough, then this function will eventually find the optimal
    path. The optimal path will be the last path it yields, but as it
    searches the graph it will yield the best paths found so far.
    """
    board = board.copy()

    # Yield a greedy solution, which also gives a lower bound on the solution
    yield (greedy_solution := list(best_first_search(board)))
    shortest_path = len(greedy_solution)

    # Add the board to the heap
    heap = [HeuristicNode(board, moves=())]
    g_scores = {board: 0}  # Keep track of nodes seen and number of moves

    popped_counter = 0
    while heap:
        # Terminate the search
        if max_nodes and popped_counter > max_nodes:
            return

        # Pop the the highest-priority node from the heap
        current = heappop(heap)
        current_g = len(current.moves)  # g(node) = number of moves
        popped_counter += 1

        if popped_counter % 10_000 == 0 and verbose:
            print(f"Heuristic function value: {current.heuristic()}")
            print(f"Number of moves (depth): {len(current.moves)}")
            print(f"Nodes popped: {popped_counter}")
            print(f"Nodes in queue: {len(heap)}")
            print(f"Nodes seen: {len(g_scores)}")

        # The lower bound f(n) = g(n) + h(n) >= best we've seen, so skip it
        if current_g + estimate_remaining(current.board) >= shortest_path:
            continue

        # The path to current node is longer than what we've seen, so skip it
        if current_g > g_scores[current.board]:
            continue

        # The board is solved. If the path is shorter than what we have,
        # then yield it and update the lower bound.
        if current.board.is_solved() and len(current.moves) < shortest_path:
            yield list(current.moves)
            shortest_path = len(current.moves)

        # Go through all children, created by applying a single move
        for (i, j), next_board in current.board.children():
            g = current_g + 1  # One more move is needed to reach the child

            # If not seen before, or the path is lower than recorded
            if (next_board not in g_scores) or (g < g_scores[next_board]):
                g_scores[next_board] = g
                next_node = HeuristicNode(next_board, current.moves + ((i, j),))
                heappush(heap, next_node)


class TestSolvers:
    @pytest.mark.parametrize("seed", range(25))
    @pytest.mark.parametrize("relabel", [True, False])
    def test_that_breadth_first_solution_length_equals_a_star(self, seed, relabel):
        # Create a random board with a random shape
        rng = random.Random(seed)
        shape = rng.randint(1, 4), rng.randint(1, 4)
        board = Board.generate_random(shape=shape, seed=seed)

        if relabel:
            board = LabelInvariantBoard(board.relabel().grid)

        # Solve it using both algorithms
        moves_bfs = breath_first_search(board)
        moves_astar = a_star_search(board)
        assert len(moves_bfs) == len(moves_astar)

        # Verify that solutions yield the solved board
        board_bfs = board.copy()
        for move in moves_bfs:
            board_bfs = board_bfs.click(*move)
        assert board_bfs.is_solved()

        # Verify that solutions yield the solved board
        board_astar = board.copy()
        for move in moves_astar:
            board_astar = board_astar.click(*move)
        assert board_astar.is_solved()

    @pytest.mark.parametrize("seed", range(100))
    def test_that_heuristic_solver_yields_optimal_solution(self, seed):
        # Create a random board with a random shape
        rng = random.Random(seed)
        shape = rng.randint(2, 5), rng.randint(2, 5)
        board = Board.generate_random(shape=shape, seed=seed)

        # Solve it using both algorithms
        moves_astar = a_star_search(board)
        *_, moves_heuristic = heuristic_search(board)
        assert len(moves_astar) == len(moves_heuristic)


def plot_solution(board, moves):
    """Plot a solution sequence."""

    board = board.copy()

    import matplotlib.pyplot as plt

    sqrt_moves = int(len(moves) ** 0.5)

    nrows, ncols = sqrt_moves, sqrt_moves + 1
    fig, axes = plt.subplots(nrows=nrows, ncols=ncols)

    n_colors = max(c for row in board.grid for c in row)
    for move, ax in zip(moves, iter(axes.ravel())):
        board.plot(ax=ax, click=move, n_colors=n_colors)
        board = board.click(*move)


if __name__ == "__main__":
    pytest.main(
        [
            __file__,
            "-v",
            "--doctest-modules",
            "-l",
        ]
    )
