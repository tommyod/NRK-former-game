"""
Tests for solvers.
"""

import pytest
import random
import statistics
import itertools


from board import Board, LabelInvariantBoard
from solvers import (
    breadth_first_search,
    a_star_search,
    heuristic_search,
    monte_carlo_search,
    iterative_deepening_search,
    anytime_beam_search,
    greedy_search,
)


class TestSolvers:
    @pytest.mark.parametrize("seed", range(50))
    def test_that_heuristic_search_initial_solution_never_hurts(self, seed):
        # Create a random board with a random shape
        board = Board.generate_random(shape=(5, 5), seed=seed)

        # Solve without any initial solution
        *_, moves1 = heuristic_search(board, moves=None, iterations=99)

        # Solve using an initial solution
        init_moves = greedy_search(board)
        try:
            # Try to get an improvement
            *_, moves2 = heuristic_search(board, moves=init_moves, iterations=99)
            assert len(moves2) <= len(moves1)
            moves2 = min(init_moves, moves2, key=len)
        except ValueError:
            # No improvement was made
            moves2 = init_moves

        assert len(moves2) <= len(moves1)

    @pytest.mark.parametrize("seed", range(50))
    def test_that_monte_carlo_search_initial_solution_never_hurts(self, seed):
        # Create a random board with a random shape
        board = Board.generate_random(shape=(5, 5), seed=seed)

        # Solve without any initial solution
        *_, moves1 = monte_carlo_search(board, moves=None, seed=0, iterations=66)

        # Solve using an initial solution
        init_moves = greedy_search(board)
        try:
            # Try to get an improvement
            *_, moves2 = monte_carlo_search(
                board, moves=init_moves, seed=0, iterations=66
            )
            assert len(moves2) <= len(moves1)
            moves2 = min(init_moves, moves2, key=len)
        except ValueError:
            # No improvement was made
            moves2 = init_moves

        assert len(moves2) <= len(moves1)

    @pytest.mark.parametrize("seed", range(25))
    @pytest.mark.parametrize("relabel", [True, False])
    def test_that_BFS_solution_equals_IDS(self, seed, relabel):
        # Create a random board with a random shape
        rng = random.Random(seed)
        shape = rng.randint(1, 3), rng.randint(1, 3)
        board = Board.generate_random(shape=shape, seed=seed)

        if relabel:
            board = LabelInvariantBoard(board.relabel().grid)

        # Solve it using both algorithms
        moves_bfs = breadth_first_search(board)
        moves_ids = iterative_deepening_search(board)

        # Verify that solutions yield the solved board
        assert board.verify_solution(moves_bfs)
        assert board.verify_solution(moves_ids)

        # Verify that IDS is as good as BFS, which is more naive
        assert len(moves_ids) == len(moves_bfs)

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
        moves_bfs = breadth_first_search(board)
        moves_astar = a_star_search(board)

        # Verify that solutions yield the solved board
        assert board.verify_solution(moves_bfs)
        assert board.verify_solution(moves_astar)

        # Verify that A* is as good as BFS, which is more naive
        assert len(moves_bfs) == len(moves_astar)

    @pytest.mark.parametrize("seed", range(100))
    def test_that_heuristic_solver_yields_optimal_solution(self, seed):
        # Create a random board with a random shape
        rng = random.Random(seed)
        shape = rng.randint(2, 5), rng.randint(2, 5)
        board = Board.generate_random(shape=shape, seed=seed)

        # Solve it using both algorithms
        moves_astar = a_star_search(board)
        assert board.verify_solution(moves_astar)

        moves_heuristic = list(heuristic_search(board, iterations=99))
        # Check that each solution yielded is lower than the previous
        for m1, m2 in itertools.pairwise(moves_heuristic):
            assert len(m1) > len(m2)

        for moves in moves_heuristic:
            assert board.verify_solution(moves)
            if len(moves_astar) == len(moves):
                break
        else:
            assert False

    @pytest.mark.parametrize("seed", range(100))
    def test_that_beam_search_yields_optimal_solution(self, seed):
        # Create a random board with a random shape
        rng = random.Random(seed)
        shape = rng.randint(2, 4), rng.randint(2, 4)
        board = Board.generate_random(shape=shape, seed=seed)

        # Solve it using both algorithms
        moves_astar = a_star_search(board)
        assert board.verify_solution(moves_astar)

        for moves in anytime_beam_search(board, power=5):
            assert board.verify_solution(moves)
            if len(moves_astar) == len(moves):
                break
        else:
            assert False

    @pytest.mark.parametrize("seed", range(100))
    def test_that_mcts_solver_yields_optimal_solution(self, seed):
        # Create a random board with a random shape
        rng = random.Random(seed)
        shape = rng.randint(2, 4), rng.randint(2, 4)
        board = Board.generate_random(shape=shape, seed=seed)

        # Solve the board with A* first, which guarantees an optimal solution
        moves_astar = a_star_search(board)
        assert board.verify_solution(moves_astar)

        moves_MCTS = list(monte_carlo_search(board, seed=42, iterations=25))
        # Check that each solution yielded is lower than the previous
        for m1, m2 in itertools.pairwise(moves_MCTS):
            assert len(m1) > len(m2)

        # Solving the board with MCTS
        for moves in moves_MCTS:
            assert board.verify_solution(moves)

            if len(moves_astar) == len(moves):
                break
        else:
            assert False

    def test_key_functions(self):
        # Create a random board
        board = Board.generate_random(shape=(9, 7), seed=0)

        def node_to_scores(node):
            # Priority 1: Compute a biased average of total moves => lower is better
            alpha = 0.5
            avg = (1 - alpha) * node.board.lower_bound + alpha * node.board.num_moves
            expected = len(node.moves) + avg

            # Priority 2: Compute the range between low and high => lower is better
            assert node.board.num_moves >= node.board.lower_bound
            range_ = abs(node.board.num_moves - node.board.lower_bound) ** 0.5

            # Priority 3: Cleared per move => higher is better
            cleared_per_move = node.cleared / len(node.moves) if node.moves else 0
            return (expected, range_, cleared_per_move)

        def scalar_key(node):
            (expected, range_, cleared_per_move) = node_to_scores(node)
            return 1.0 * expected + 0.1 * range_ - 0.01 * cleared_per_move

        # Call solvers with key function
        moves = greedy_search(board, key=scalar_key)
        *_, moves = anytime_beam_search(board, power=4, key=scalar_key)
        assert board.verify_solution(moves)
        *_, moves = monte_carlo_search(board, iterations=5, seed=0, key=scalar_key)
        assert board.verify_solution(moves)
        *_, moves = heuristic_search(board, iterations=25, key=scalar_key)
        assert board.verify_solution(moves)

        def tuple_key(node):
            (expected, range_, cleared_per_move) = node_to_scores(node)
            return (expected, range_, -cleared_per_move)

        # Call solvers with key function
        moves = greedy_search(board, key=tuple_key)
        *_, moves = anytime_beam_search(board, power=4, key=tuple_key)
        assert board.verify_solution(moves)
        *_, moves = monte_carlo_search(board, iterations=5, seed=0, key=tuple_key)
        assert board.verify_solution(moves)
        *_, moves = heuristic_search(board, iterations=25, key=tuple_key)
        assert board.verify_solution(moves)


@pytest.mark.parametrize("seed", range(99))
def test_that_astar_solution_is_within_bounds(seed):
    # Create a random board with a random shape
    rng = random.Random(seed)
    shape = rng.randint(2, 5), rng.randint(2, 5)
    board = Board.generate_random(shape=shape, seed=seed)
    assert board.lower_bound <= len(a_star_search(board))


class TestNonRegressionOnPerformance:
    def test_performance_anytime_beam_search(self):
        # Takes around 1.5s per board
        solution_lengths = []
        for seed in range(15):
            board = Board.generate_random(shape=(9, 7), seed=seed)
            *_, moves = anytime_beam_search(board, power=5)
            assert board.verify_solution(moves)
            solution_lengths.append(len(moves))

        assert statistics.mean(solution_lengths) <= 15.8

    def test_performance_heuristic_search(self):
        # Takes around 1.5s per board
        solution_lengths = []
        for seed in range(15):
            board = Board.generate_random(shape=(9, 7), seed=seed)
            *_, moves = heuristic_search(board, iterations=600)
            assert board.verify_solution(moves)
            solution_lengths.append(len(moves))

        assert statistics.mean(solution_lengths) <= 15.8

    def test_performance_monte_carlo_search(self):
        # Takes around 1.5s per board
        solution_lengths = []
        for seed in range(15):
            board = Board.generate_random(shape=(9, 7), seed=seed)
            *_, moves = monte_carlo_search(board, iterations=50, seed=seed)
            assert board.verify_solution(moves)
            solution_lengths.append(len(moves))

        assert statistics.mean(solution_lengths) <= 16.5


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--doctest-modules", "-l", "-x", "--durations=10"])
