"""
Investigate solvers and create figures for reporting.
"""

import time
from typing import Callable, List
from board import Board, LabelInvariantBoard, CanonicalBoard
from solvers import (
    breadth_first_search,
    a_star_search,
    heuristic_search,
    monte_carlo_search,
    best_first_search,
)
import statistics
import matplotlib.pyplot as plt
import itertools
from collections import deque
from instances import NRK_boards
import random


def time_solver(solver: Callable, num_boards: int = 10, relabel=False) -> List[float]:
    """
    Time the performance of a board solver across different board sizes.

    Args:
        solver: Function that takes a Board and returns a solution
        max_size: Maximum board dimension to test (e.g. 5 for 5x5)
        num_boards: Number of random boards to test for each size

    Returns:
        List of average solving times for each board size (1x1 to max_size x max_size)
    """

    def yield_board_shapes():
        for size in itertools.count(1):
            yield (size, size)
            yield (size, size + 1)
            yield (size + 1, size)

    for shape in yield_board_shapes():
        times = []
        for i in range(num_boards):
            board = Board.generate_random(shape=shape, seed=i)

            if relabel:
                board = LabelInvariantBoard(board.relabel().grid)

            start_time = time.perf_counter()
            moves = solver(board)
            elapsed_time = time.perf_counter() - start_time
            print(
                f"Solved board of shape {shape} in {elapsed_time:.6f} ({solver.__name__})"
            )

            # Verify that solutions yield the solved board
            for move in moves:
                board = board.click(*move)
            assert board.is_solved()

            times.append(elapsed_time)

        yield shape, statistics.geometric_mean(times)


def bfs_counter(board: Board, max_depth=1) -> list:
    """Breadth-first search to find shortest solution path.

    This approach is not very efficient, but it is guaranteed to return
    a minimum path, solving the board in the fewest moves possible.
    """
    board = board.copy()

    # board = board.copy()
    board.depth = 0

    # Queue of (board, moves) tuples using a deque for efficient popleft
    queue = deque([board])
    current_depth = 0
    current_counter = 0

    while queue:
        current_board = queue.popleft()
        if current_board.depth > current_depth:
            print(f"Found {current_counter} boards at depth {current_depth}")
            current_depth += 1
            current_counter = 1
        else:
            current_counter += 1

        # Check if we've found a solution
        if current_board.is_solved():
            return

        if current_board.depth > max_depth:
            return

        # Try all possible moves from current state
        for _, next_board in current_board.children():
            # Add new state to queue and visited set
            next_board.depth = current_board.depth + 1
            queue.append(next_board)


def dfs_counter(board: Board, max_depth=1) -> int:
    """Depth-first search counter that explores the game tree up to max_depth.
    Returns the total number of nodes at and above max_depth.

    Args:
        board (Board): The starting board position
        max_depth (int): Maximum depth to explore

    Returns:
        int: Total number of nodes found at and above max_depth
    """
    # Initialize counters for each depth
    depth_counts = [0] * (max_depth + 1)

    def dfs_helper(current_board: Board, depth: int):
        # Count the current node at its depth
        depth_counts[depth] += 1

        # Base cases
        if current_board.is_solved() or depth >= max_depth:
            return

        # Recursive case: explore all children
        for _, next_board in current_board.children():
            next_board.depth = depth + 1
            dfs_helper(next_board, depth + 1)

    # Start the recursion with initial board
    board = board.copy()
    board.depth = 0
    dfs_helper(board, 0)

    # Print counts at each depth
    for depth, count in enumerate(depth_counts):
        print(f"Found {count} boards at depth {depth}")

    return depth_counts


def best_first_performance(board, simulations=100):
    """Try best-first search with various powers on a board many times."""

    # The lower the value of power, the more randomess
    # The higher the value of power, the more we focus on large groups of colors
    for power in [0, 0.5, 1, 2, 5, 10, 25, None]:
        if power is None:
            # Pure best first search
            moves = list(best_first_search(board, power=None, seed=None))
            yield power, [len(moves)]
            continue

        results = []
        for simulation in range(simulations):
            moves = list(best_first_search(board, power=power, seed=simulation))
            results.append(len(moves))

        yield power, results


def search_timer(algorithm, *args, **kwargs):
    start_time = time.perf_counter()
    for moves in algorithm(*args, **kwargs):
        elapsed_time = time.perf_counter() - start_time
        print(f"Found solution of length {len(moves)} after {elapsed_time:.3f} s")
        yield elapsed_time, len(moves)


def plot_solution(board, moves):
    """Plot a solution sequence."""
    import matplotlib.pyplot as plt

    board = board.copy()
    # Determine shape of the board
    sqrt_moves = int(len(moves) ** 0.5)
    if sqrt_moves * sqrt_moves == len(moves):
        nrows = sqrt_moves
        ncols = sqrt_moves
    elif sqrt_moves * (sqrt_moves + 1) >= len(moves):
        nrows = sqrt_moves
        ncols = sqrt_moves + 1
    else:
        nrows = sqrt_moves + 1
        ncols = sqrt_moves + 1

    assert nrows * ncols >= len(moves)

    # Create figure with minimal spacing
    fig = plt.figure(figsize=(ncols * 2, nrows * 2))
    gs = plt.GridSpec(nrows, ncols, figure=fig)
    gs.update(wspace=0.1, hspace=0.1)  # Reduce spacing between subplots

    # Create axes using GridSpec
    axes = []
    for i in range(nrows):
        for j in range(ncols):
            ax = fig.add_subplot(gs[i, j])
            axes.append(ax)

    axes = iter(axes)

    for move, ax in zip(moves, axes):
        board.plot(ax=ax, click=move, n_colors=4)
        board = board.click(*move)
        # Remove ticks and grid for cleaner look
        ax.set_xticks([])
        ax.set_yticks([])
        ax.grid(False)

    # Turn off remaining axes
    for ax in axes:
        ax.axis("off")

    # Adjust the layout to be more compact
    plt.subplots_adjust(left=0.02, right=0.98, bottom=0.02, top=0.98)

    return fig, axes


if __name__ == "__main__":
    if False:
        plt.figure(figsize=(6, 3))
        plt.title("Solving random boards to optimality")
        N = 3 * 5
        n = N
        solutions = list(itertools.islice(time_solver(breadth_first_search), n))
        times = [time for (_, time) in solutions]
        plt.semilogy(times, label="BFS")

        n = N + 1
        solutions = list(itertools.islice(time_solver(a_star_search), n))
        times = [time for (_, time) in solutions]
        plt.semilogy(times, label="A*")

        n = N + 1
        solutions = list(itertools.islice(time_solver(a_star_search, relabel=True), n))
        times = [time for (_, time) in solutions]
        shapes = [shape for (shape, _) in solutions]
        plt.semilogy(times, label="A* (relabel)")

        plt.xticks(list(range(n)), shapes)
        plt.legend()

        plt.xlabel("Board size")
        plt.ylabel("Solution time")
        plt.grid(True, ls="--", zorder=0, alpha=0.33)
        plt.tight_layout()
        plt.savefig("random_boards_to_optimality.png", dpi=200)
        plt.show()

    if False:
        board = NRK_boards[16].board

        board = CanonicalBoard(board.canonicalize().grid)
        # board = Board([[1, 2], [2, 1]])
        depth_counts = dfs_counter(board, max_depth=2)

        # Found 1 boards at depth 0
        # Found 39 boards at depth 1
        # Found 1446 boards at depth 2
        # Found 51162 boards at depth 3
        # Found 1730312 boards at depth 4
        # Found 55950299 boards at depth 5
        # branching factors = [39.0, 37.077, 35.382, 33.82, 32.335]
        # geometric mean of branching factors => 35.445449

    if False:
        for BOARD_NUMBER in [16, 19, 20, 21, 22, 23, 24]:
            plt.figure(figsize=(6, 3))
            plt.title("Comparing heuristic search and Monte Carlo search")
            # BOARD_NUMBER = 16
            board = NRK_boards[BOARD_NUMBER].board

            max_nodes = 500_000
            results = list(search_timer(heuristic_search, board, max_nodes=max_nodes))
            times = [seconds for (seconds, num_moves) in results]
            num_moves = [num_moves for (seconds, num_moves) in results]
            plt.semilogx(
                times, num_moves, "-o", label=f"heuristic_search({max_nodes=})"
            )

            iterations = 20_000
            results = list(
                search_timer(monte_carlo_search, board, iterations=iterations, seed=42)
            )
            times = [seconds for (seconds, num_moves) in results]
            num_moves = [num_moves for (seconds, num_moves) in results]
            plt.semilogx(
                times, num_moves, "-o", label=f"monte_carlo_search({iterations=})"
            )

            plt.axhline(
                y=NRK_boards[BOARD_NUMBER].best,
                label="Best known solution",
                ls="--",
                color="black",
                alpha=0.33,
            )

            plt.legend()

            plt.xlabel("Solution time")
            plt.ylabel("Number of moves in solution")
            plt.grid(True, ls="--", zorder=0, alpha=0.33)
            plt.tight_layout()
            plt.savefig(f"heuristic_searches_board_no_{BOARD_NUMBER}.png", dpi=200)
            plt.show()
            
    

    if False:
        BOARD_NUMBER = 25
        board = NRK_boards[BOARD_NUMBER].board
        board = Board(board.grid)
        
        # board = Board.generate_random((4,4), seed=3)

        iterations = 100_000
        results = list(
            monte_carlo_search(board, iterations=iterations, seed=2, verbosity=1)
        )
        moves = results[-1]

        fig, axes = plot_solution(board, moves)
        plt.savefig(f"best_solution_found_no_{BOARD_NUMBER}.png", dpi=200)
        plt.show()

    if False:
        plt.figure(figsize=(6, 3))
        rng = random.Random(42)

        board = NRK_boards[16].board
        plt.title(f"Best-first search on a board with shape {(board.rows, board.cols)}")

        labels = []
        sim_results = best_first_performance(board, simulations=100)
        for i, (power, results) in enumerate(sim_results):
            x = [i + (rng.random() - 0.5) * 0.33 for _ in results]
            labels.append(str(power))
            plt.scatter([i], [statistics.mean(results)], color="black")

            if power is not None:
                plt.scatter(x, results, s=10, alpha=0.33)
                plt.errorbar(
                    i,
                    statistics.mean(results),
                    yerr=statistics.stdev(results),
                    capsize=3,
                    fmt="k--o",
                    ecolor="black",
                )

        plt.xlabel("Power")
        plt.xticks(list(range(len(labels))), labels)
        plt.ylabel("Number of moves in solution")
        plt.grid(True, ls="--", zorder=0, alpha=0.33)
        plt.tight_layout()
        plt.savefig("randomized_best_first_search.png", dpi=200)
        plt.show()
        
        
    if True:
        # Test various levels of exploration
        for exploration in [0.25, 0.5, 1, 2, 4]:
            results = []
            for seed in range(10):
                
                board = Board.generate_random((9, 7), seed=seed)
                

                iterations = 10_000
                *_, moves = list(
                    monte_carlo_search(board, iterations=iterations, seed=2, verbosity=0,
                                       exploration=exploration)
                )
                results.append(len(moves))
                
            import statistics
            print(exploration, statistics.mean(results))
                
