"""
Investigate solvers and create figures for reporting.
"""

import time
from typing import Callable, List
from board import Board, LabelInvariantBoard
from solvers import (
    breadth_first_search,
    a_star_search,
    heuristic_search,
    monte_carlo_search,
    greedy_search,
    iterative_deepening_search,
    beam_search,
    anytime_beam_search,
    middle_bound,
)
import statistics
import matplotlib.pyplot as plt
import itertools
from collections import deque
from instances import NRK_boards
import random
import functools
import math


def yield_board_shapes():
    """Yield board shapes of increasing size."""
    for size in itertools.count(2):
        yield (size, size)
        yield (size + 1, size)
        yield (size, size + 1)


def time_solver(
    solver: Callable, num_boards: int = 10, relabel=False, max_time=10
) -> List[float]:
    """Time the performance of a board solver across different board sizes."""

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
            assert board.is_solved

            times.append(elapsed_time)

        # Use geometric mean since numbers are on vastly different
        # orders of magnitude
        yield shape, statistics.geometric_mean(times)

        # Stop here
        if statistics.geometric_mean(times) > max_time:
            return


def bfs_counter(board: Board, max_depth=1) -> list:
    """Breadth-first search to find shortest solution path.

    This approach is not very efficient, but it is guaranteed to return
    a minimum path, solving the board in the fewest moves possible.
    """
    board = board.copy()
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
        if current_board.is_solved:
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
    """
    # Initialize counters for each depth
    depth_counts = [0] * (max_depth + 1)

    def dfs_helper(current_board: Board, depth: int):
        # Count the current node at its depth
        depth_counts[depth] += 1

        # Base cases
        if current_board.is_solved or depth >= max_depth:
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


def randomized_greedy_performance(board, simulations=25):
    """Try randomized greedy search on a board many times."""

    def random_key(node, exponent=None):
        """Random weighted sampling chooses element k with probability equal
        to the weight w_k / sum_i^n w_i. This is equivalent to computing
        Uniform(0, 1)**(1/w_i) for each i and choosing the maximal element.
        See 'Weighted random sampling with a reservoir', Efraimidis et al."""
        # Compute score
        score = middle_bound(node)
        if exponent is None:
            return score

        # Lower is better, so set w = exp(-score). Since we're maximizing and
        # taking maximum is unaffected by monotonic transformations, we take
        # logarithms: log[U(0, 1)**(1/w)] = 1/w * log(U) = exp(w) * log(U)
        # Finally, we take minimum since lower is better on the output.
        return -pow(exponent, score) * math.log(rng.random())

    # The lower the value of power, the more randomess
    # The higher the value of power, the more we focus on large groups of colors
    for exponent in [1, 2, 5, 10, 25, 100, None]:
        results = []
        for simulation in range(simulations):
            key = functools.partial(random_key, exponent=exponent)
            moves = greedy_search(board, key=key)
            results.append(len(moves))
        yield exponent, results


def search_timer(algorithm, *args, **kwargs):
    """Time an algorithm and yield (time_ran, moves) as it progresses."""
    start_time = time.perf_counter()
    for moves in algorithm(*args, **kwargs):
        elapsed_time = time.perf_counter() - start_time
        print(f"Found solution of length {len(moves)} after {elapsed_time:.3f} s")
        yield elapsed_time, moves


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


def benchmark(algorithm, *, simulations, **kwargs):
    """Benchmark and algorithm on `simulations` random instances."""
    LABEL_INVARIANT = False

    print(f"Benchmarking: {algorithm.__name__}(board, {kwargs})\n ", end="")
    times, solution_lengths = [], []
    for seed in range(simulations):
        print(".", end="")
        board = Board.generate_random(shape=(9, 7), seed=seed)
        if LABEL_INVARIANT:
            board = LabelInvariantBoard(board.relabel().grid)
        start_time = time.perf_counter()

        # If "moves" is a keyword argument, provide an initial solution
        if "moves" in algorithm.__kwdefaults__:
            moves = greedy_search(board)
            generator_or_list = algorithm(board, moves=moves, **kwargs)
        else:
            generator_or_list = algorithm(board, **kwargs)

        # Extract the results
        if isinstance(generator_or_list, list):
            moves = generator_or_list
        else:
            try:
                *_, moves = generator_or_list
            except ValueError:  # Nothing came out from it :(
                moves = moves

        times.append(time.perf_counter() - start_time)
        board.verify_solution(moves)
        solution_lengths.append(len(moves))

    avg_time = statistics.mean(times)
    avg_solution = statistics.mean(solution_lengths)
    print(f"\n Solutions of avg. length {avg_solution:.3f} in {avg_time:.3f} sec.")
    print(f" Avg. length times sec (lower is better) => {avg_solution*avg_time:.3f}\n")


if __name__ == "__main__":
    # Benchmark the solvers.
    if False:
        benchmark = functools.partial(benchmark, simulations=10)

        # Parameters such that each solver takes ~15s to solve an instance
        benchmark(greedy_search)
        benchmark(anytime_beam_search, power=5)
        benchmark(heuristic_search, iterations=500, verbose=False)
        benchmark(monte_carlo_search, iterations=50, seed=1)

    # Solve random boards to optimality and create a plot
    if False:
        plt.figure(figsize=(7, 3))
        plt.title("Solving random boards to optimality")

        # Set time limit and number of boards here
        time_solver = functools.partial(time_solver, max_time=0.1, num_boards=5)

        solutions = list(time_solver(iterative_deepening_search))
        n = len(solutions)
        times = [time for (_, time) in solutions]
        plt.semilogy(times, label="IDS")

        solutions = list(time_solver(breadth_first_search))
        n = max(n, len(solutions))
        times = [time for (_, time) in solutions]
        plt.semilogy(times, label="BFS")

        solutions = list(time_solver(a_star_search))
        n = max(n, len(solutions))
        times = [time for (_, time) in solutions]
        plt.semilogy(times, label="A*")

        solutions = list(time_solver(a_star_search, relabel=True))
        n = max(n, len(solutions))
        times = [time for (_, time) in solutions]
        plt.semilogy(times, label="A* (relabel)")

        shapes = list(itertools.islice(yield_board_shapes(), n))
        plt.xticks(list(range(n)), shapes, rotation=30)
        plt.legend()

        plt.xlabel("Board size")
        plt.ylabel("Solution time")
        plt.grid(True, ls="--", zorder=0, alpha=0.33)
        plt.tight_layout()
        plt.savefig("random_boards_to_optimality.png", dpi=200)
        plt.show()

    # Investigate the branching factor of a board instance
    if False:
        board = NRK_boards[26].board
        for max_depth in range(1, 4):
            print()
            depth_counts = dfs_counter(board, max_depth=max_depth)

        # On board from November 16th
        # ---------------------------
        # Found 1 boards at depth 0
        # Found 39 boards at depth 1
        # Found 1446 boards at depth 2
        # Found 51162 boards at depth 3
        # Found 1730312 boards at depth 4
        # Found 55950299 boards at depth 5
        # branching factors = [39.0, 37.077, 35.382, 33.82, 32.335]
        # geometric mean of branching factors => 35.445449

    # Plot solution times on NRK boards
    if False:
        for board_no, instance in sorted(NRK_boards.items()):
            plt.figure(figsize=(6, 3))
            plt.title(f"Comparing search algorithms (NRK board {board_no})")
            board = Board(instance.board.grid)
            print(f"Board number: {board_no} (best known: {instance.best}) \n{board}")

            # Start with a greedy search
            moves = greedy_search(board)
            print(f"Initial bound by greedy: {len(moves)}")

            print("Running beam search")
            power = 12
            st = time.perf_counter()
            results = list(search_timer(anytime_beam_search, board, power=power))
            print(f"Ran in: {time.perf_counter() - st:.2f}")
            times = [seconds for (seconds, num_moves) in results]
            num_moves = [len(moves) for (seconds, moves) in results]
            plt.semilogx(times, num_moves, "-o", label=f"anytime_beam_search({power=})")
            best_moves = results[-1][1]

            print("Running heuristic search")
            iterations = 100_000
            st = time.perf_counter()
            results = list(
                search_timer(
                    heuristic_search,
                    board,
                    iterations=iterations,
                    verbose=False,
                    moves=moves,
                )
            )
            print(f"Ran in: {time.perf_counter() - st:.2f}")
            times = [seconds for (seconds, num_moves) in results]
            num_moves = [len(moves) for (seconds, moves) in results]
            plt.semilogx(
                times, num_moves, "-o", label=f"heuristic_search({iterations=:.1e})"
            )
            best_moves = min([best_moves, results[-1][1]], key=len)

            print("Running Monte Carlo search")
            iterations = 100_000
            st = time.perf_counter()
            results = list(
                search_timer(
                    monte_carlo_search,
                    board,
                    iterations=iterations,
                    verbosity=0,
                    moves=moves,
                )
            )
            print(f"Ran in: {time.perf_counter() - st:.2f}")
            times = [seconds for (seconds, num_moves) in results]
            num_moves = [len(moves) for (seconds, moves) in results]
            plt.semilogx(
                times, num_moves, "-o", label=f"monte_carlo_search({iterations=:.1e})"
            )
            best_moves = min([best_moves, results[-1][1]], key=len)

            plt.axhline(
                y=NRK_boards[board_no].best,
                label="Best known solution",
                ls="--",
                color="black",
                alpha=0.33,
            )

            plt.legend(fontsize=8)

            plt.xlabel("Solution time")
            plt.ylabel("Number of moves in solution")
            plt.grid(True, ls="--", zorder=0, alpha=0.33)
            plt.tight_layout()
            plt.savefig(f"heuristic_searches_board_no_{board_no}.png", dpi=200)
            plt.show()

            # Plot the best move sequence
            fig, axes = plot_solution(board, best_moves)
            plt.savefig(f"best_solution_found_board_no_{board_no}.png", dpi=200)
            plt.show()

    # Beam search on NRK instances
    if False:
        plt.figure(figsize=(6, 3))
        plt.title("Beam search on NRK instances")
        MAX_POWER = 5

        for board_no, instance in sorted(NRK_boards.items()):
            print(f"Beam search on board number {board_no}")

            board = Board(instance.board.grid)

            shortest_path = float("inf")
            powers, solutions = [], []
            for power in range(MAX_POWER + 1):
                print(f"p={power}  ", end="")
                solution = beam_search(
                    board, beam_width=2**power, shortest_path=shortest_path
                )
                if solution and len(solution) < shortest_path:
                    shortest_path = len(solution)
                    powers.append(power)
                    solutions.append(len(solution))

            print()
            plt.plot(powers, solutions, "-o", alpha=0.8, markersize=4)

        plt.xticks(
            list(range(MAX_POWER + 1)),
            [r"$2^{" + str(i) + r"}$" for i in range(MAX_POWER + 1)],
        )
        plt.xlabel("Beam width")
        plt.ylabel("Number of moves in solution")
        plt.grid(True, ls="--", zorder=0, alpha=0.33)
        plt.tight_layout()
        plt.savefig("beam_search_NRK_instances.png", dpi=200)
        plt.show()

    # Try best-first seach with various values of `power` on a board
    if False:
        plt.figure(figsize=(6, 3))
        rng = random.Random(42)

        board = NRK_boards[16].board
        plt.title(f"Greedy search on a board with shape {(board.rows, board.cols)}")

        rng = random.Random(0)

        labels = []
        sim_results = randomized_greedy_performance(board, simulations=100)
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

        plt.xlabel("Exponent")
        plt.xticks(list(range(len(labels))), labels)
        plt.ylabel("Number of moves in solution")
        plt.grid(True, ls="--", zorder=0, alpha=0.33)
        plt.tight_layout()
        plt.savefig("randomized_best_first_search.png", dpi=200)
        plt.show()
