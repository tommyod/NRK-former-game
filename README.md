# NRK-former-game

Code that attempts to solve NRK's game "[Former](https://www.nrk.no/former-1.17105310)" (similar to [SameGame](https://en.wikipedia.org/wiki/SameGame)).

![GamePlay](./gameplay.gif)

- This repo has an accompanying blog post: **[Solving NRK’s game "Former"](https://tommyodland.com/articles/2024/solving-nrks-game-former/)**
- PRs are welcome! Make an issue before you make significant changes in a PR.

## The Board class

A Board is an immutable class that represents a game.
Colors are represented as integers 1, 2, 3, ... on a grid.

```pycon
>>> from board import Board
>>> grid = [[1, 2],
...         [2, 1]]
>>> board = Board(grid)
>>> board
Board([[1, 2], [2, 1]])
>>> print(board)
12
21

```

Clicking on a board returns a new instance, that can be printed:

```pycon
>>> board = board.click(0, 1)
>>> print(board)
10
21

```

Statistics can be retrieved using these methods:

```pycon
>>> len(board)  # The board has size 2 x 2 = 4
4
>>> board.remaining  # There are three cells remaining (non-zero)
3
>>> board.cleared  # One cell has been cleared
1
>>> board.unique_remaining  # There are two unique remaining numbers
2

```

All children (results of all valid clicks) can be retrieved:

```pycon
>>> for move, child in board.children():
...     assert board.click(*move) == child
...     print(child)
...     print()
00
21
<BLANKLINE>
00
11
<BLANKLINE>
10
20
<BLANKLINE>

```

## The solvers

**Best first search** chooses the move that clears the most cells.
There are no guarantees that this results in an optimal solution.

```pycon
>>> from solvers import best_first_search
>>> board = Board([[4, 3, 3, 1, 1], 
...                [1, 4, 4, 3, 2], 
...                [3, 2, 4, 4, 3], 
...                [3, 1, 4, 1, 4], 
...                [4, 1, 2, 4, 3]])
>>> moves = best_first_search(board)
>>> moves
[(3, 3), (1, 4), (1, 3), (1, 1), (3, 1), (4, 1), (1, 0), (2, 0), (3, 4), (3, 0)]
>>> board.verify_solution(moves)
True

```

**A\* search with an admissible heuristic** is guaranteed to solve the problem,
but with large boards we run out of memory and the compute time is too long.
A better heuristic could help---make a PR if you have an idea!

```pycon
>>> from solvers import a_star_search
>>> moves = a_star_search(board)
>>> moves
[(2, 0), (3, 0), (4, 1), (0, 3), (3, 3), (2, 2), (3, 4), (4, 4)]
>>> board.verify_solution(moves)
True

```

**Heuristic search** uses a priority queue and yields solutions as they are discovered.
If you let it run long enough, it will produce an optimal solution.
You might run out of memory before that happens though.

```pycon
>>> from solvers import heuristic_search
>>> for moves in heuristic_search(board):
...    print(f"Found solution of length {len(moves)}: {moves}")
Found solution of length 10: [(3, 3), (1, 4), (1, 3), (1, 1), (3, 1), (4, 1), (1, 0), (2, 0), (3, 4), (3, 0)]
Found solution of length 9: [(3, 3), (1, 4), (1, 3), (1, 1), (2, 0), (3, 0), (4, 1), (3, 4), (3, 0)]
Found solution of length 8: [(3, 3), (1, 4), (1, 3), (2, 0), (3, 0), (1, 2), (4, 1), (3, 4)]

```

**Monte Carlo tree search** also yields solutions as they are discovered.
In the long run it produces a solution, but again you might run out of memory.

```pycon
>>> from solvers import monte_carlo_search
>>> for moves in monte_carlo_search(board, seed=1):
...    print(f"Found solution of length {len(moves)}: {moves}")
Found solution of length 10: [(3, 3), (1, 4), (1, 3), (1, 1), (3, 1), (4, 1), (1, 0), (2, 0), (3, 4), (3, 0)]
Found solution of length 9: [(1, 4), (3, 3), (2, 0), (3, 0), (1, 2), (4, 1), (3, 3), (2, 4), (3, 4)]
Found solution of length 8: [(0, 3), (3, 3), (2, 0), (3, 0), (1, 2), (4, 1), (3, 4), (4, 4)]

```

**Beam search** expands all children, keeps the `beam_width` best nodes, 
expands all children of those nodes, and repeats.

```pycon
>>> from solvers import beam_search
>>> len(beam_search(board, beam_width=2))
9
>>> len(beam_search(board, beam_width=4))
8

```

You can also run it for `beam_width=1, 2, 4, 8, ..., 2^power`:

```pycon
>>> from solvers import anytime_beam_search
>>> for moves in anytime_beam_search(board, power=5):
...    print(f"Found solution of length {len(moves)}: {moves}")
Found solution of length 10: [(3, 3), (1, 4), (1, 3), (1, 1), (3, 1), (4, 1), (1, 0), (2, 0), (3, 4), (3, 0)]
Found solution of length 9: [(2, 0), (3, 0), (4, 1), (2, 2), (3, 4), (3, 3), (3, 3), (4, 1), (4, 4)]
Found solution of length 8: [(2, 0), (3, 0), (3, 3), (1, 2), (4, 1), (3, 4), (4, 4), (4, 3)]

```