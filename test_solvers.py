"""
Tests for solvers.
"""

import pytest
import random


from board import Board, LabelInvariantBoard
from solvers import (
    breath_first_search,
    a_star_search,
    heuristic_search,
    monte_carlo_search,
)


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
        for moves in heuristic_search(board):
            if len(moves_astar) == len(moves):
                break
        else:
            assert False

    @pytest.mark.parametrize("seed", range(100))
    def test_that_mcts_finds_valid_solutions(self, seed):
        rng = random.Random(seed)
        shape = rng.randint(2, 4), rng.randint(2, 4)
        board = Board.generate_random(shape=shape, seed=seed)

        for moves in monte_carlo_search(board, iterations=1000, seed=42):
            test_board = board.copy()

            for move in moves:
                test_board = test_board.click(*move)
            assert test_board.is_solved()

    @pytest.mark.parametrize("seed", range(50))
    def test_that_mcts_solver_yields_optimal_solution(self, seed):
        # Create a random board with a random shape
        rng = random.Random(seed)
        shape = rng.randint(2, 5), rng.randint(2, 5)
        board = Board.generate_random(shape=shape, seed=seed)

        # Solve it using both algorithms
        moves_astar = a_star_search(board)
        for moves in monte_carlo_search(board, iterations=9999, seed=42):
            if len(moves_astar) == len(moves):
                break
        else:
            assert False


if __name__ == "__main__":
    pytest.main(
        [
            __file__,
            "-v",
            "--doctest-modules",
            "-l",
        ]
    )
