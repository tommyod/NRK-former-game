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
>>> for move in moves:
...     board = board.click(*move)
>>> board.is_solved()
True

```


