"""
Tests for solvers.
"""

import pytest
import random


from board import Board, LabelInvariantBoard
from solvers import (
    breadth_first_search,
    a_star_search,
    heuristic_search,
    monte_carlo_search,
    iterative_deepening_search,
    anytime_beam_search,
)


class TestSolvers:
    @pytest.mark.parametrize("seed", range(100))
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

        # Verify that A* is as good as BFS, which is more naive
        assert len(moves_bfs) == len(moves_ids)

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

        for moves in heuristic_search(board):
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

        for moves in anytime_beam_search(board, power=12):
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

        # Solving the board with MCTS
        for moves in monte_carlo_search(board, iterations=999999, seed=42):
            assert board.verify_solution(moves)

            if len(moves_astar) == len(moves):
                break
        else:
            assert False


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--doctest-modules", "-l", "-x"])
