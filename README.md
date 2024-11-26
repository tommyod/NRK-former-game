# NRK-former-game

Code that attempts to solve NRK's game "[Former](https://www.nrk.no/former-1.17105310)" (similar to [SameGame](https://en.wikipedia.org/wiki/SameGame)).

![GamePlay](./gameplay.gif)


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
>>> board.remaining()  # There are three cells remaining (non-zero)
3
>>> board.cleared()  # One cell has been cleared
1
>>> board.unique_remaining()  # There are two unique remaining numbers
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
>>> board = Board(grid=[[1, 1, 2], [2, 3, 1], [4, 2, 2]])
>>> moves = list(best_first_search(board))
>>> moves
[(2, 1), (2, 2), (2, 2), (2, 1), (2, 1), (2, 0), (2, 0), (2, 0)]
>>> for move in moves:
...     board = board.click(*move)
>>> board.is_solved()
True

```

**A\* search with an admissible heuristic** is guaranteed to solve the problem,
but with large boards we run out of memory and the compute time is too long.
A better heuristic could help---make a PR if you have an idea!

```pycon
>>> from solvers import a_star_search
>>> board = Board(grid=[[1, 1, 2], [2, 3, 1], [4, 2, 2]])
>>> moves = a_star_search(board)
>>> moves
[(1, 1), (2, 0), (1, 0), (1, 2)]
>>> board.verify_solution(moves)
True

```

**Heuristic search** uses a priority queue and yields solutions as they are discovered.
If you let it run long enough, it will produce an optimal solution.
You might run out of memory before that happens though.

```pycon
>>> from solvers import heuristic_search
>>> board = Board(grid=[[1, 1, 2], [2, 3, 1], [4, 2, 2]])
>>> for moves in heuristic_search(board):
...    print(f"Found solution of length {len(moves)}: {moves}")
Found solution of length 8: [(2, 1), (2, 2), (2, 2), (2, 1), (2, 1), (2, 0), (2, 0), (2, 0)]
Found solution of length 7: [(0, 0), (2, 1), (1, 0), (1, 2), (2, 0), (2, 1), (2, 2)]
Found solution of length 6: [(0, 0), (1, 1), (1, 2), (1, 2), (1, 0), (2, 0)]
Found solution of length 5: [(0, 0), (1, 1), (1, 2), (2, 0), (1, 2)]
Found solution of length 4: [(1, 1), (2, 0), (1, 0), (1, 2)]

```

**Monte Carlo tree search** also yields solutions as they are discovered.
In the long run it produces a solution, but again you might run out of memory.

```pycon
>>> from solvers import monte_carlo_search
>>> board = Board(grid=[[1, 1, 2], [2, 3, 1], [4, 2, 2]])
>>> for moves in monte_carlo_search(board, seed=1):
...    print(f"Found solution of length {len(moves)}: {moves}")
Found solution of length 8: [(2, 1), (2, 2), (2, 2), (2, 1), (2, 1), (2, 0), (2, 0), (2, 0)]
Found solution of length 6: [(1, 1), (1, 1), (1, 2), (1, 0), (2, 0), (2, 0)]
Found solution of length 5: [(1, 1), (2, 0), (2, 0), (2, 0), (2, 2)]
Found solution of length 4: [(2, 0), (1, 1), (1, 0), (1, 2)]

```

**Beam search** expands all children, keeps the `beam_width` best nodes, 
expands all children of those nodes, and repeats.

```pycon
>>> from solvers import beam_search
>>> board = Board(grid=[[1, 1, 2], [2, 3, 1], [4, 2, 2]])
>>> moves = beam_search(board, beam_width=2)
>>> len(moves), moves
(6, [(0, 0), (1, 2), (1, 2), (2, 0), (2, 0), (2, 1)])

```

You can also run it for `beam_width=1, 2, 4, 8, ..., 2^power`:

```pycon
>>> from solvers import anytime_beam_search
>>> board = Board(grid=[[1, 1, 2], [2, 3, 1], [4, 2, 2]])
>>> for moves in anytime_beam_search(board, power=5):
...    print(f"Found solution of length {len(moves)}: {moves}")
Found solution of length 7: [(0, 0), (2, 1), (1, 0), (1, 2), (2, 0), (2, 1), (2, 2)]
Found solution of length 6: [(0, 0), (1, 2), (1, 2), (2, 0), (2, 0), (2, 1)]
Found solution of length 5: [(2, 0), (2, 0), (2, 1), (2, 0), (2, 2)]
Found solution of length 4: [(2, 0), (1, 1), (1, 0), (1, 2)]

```






