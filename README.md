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

Let us try to solve the NRK board for November 25th 2024 with a few solvers:

```pycon
>>> board = Board([[1, 2, 1, 2, 1, 3, 3], 
...                [2, 1, 2, 3, 4, 2, 1], 
...                [4, 2, 1, 2, 1, 1, 3], 
...                [2, 4, 2, 2, 1, 2, 1], 
...                [3, 1, 4, 2, 2, 1, 1], 
...                [1, 1, 3, 4, 3, 2, 2], 
...                [3, 3, 3, 2, 3, 3, 1], 
...                [3, 1, 2, 1, 3, 2, 2], 
...                [3, 3, 1, 2, 2, 1, 4]])

```

### Greedy search

**Greedy search** chooses the best move at each level.
The _best move_ is defined by a key function that takes a node as input and returns a number.
The lower the number, the better the move.

Greedy search is quite flexible!
A first heuristic we can try is to maximize cells cleared per move.

```pycon
>>> from solvers import greedy_search
>>> def cleared_per_move(node):
...     # We flip the sign because lower is better
...     return -node.cleared / len(node.moves)
>>> len(greedy_search(board, key=cleared_per_move))
27

```

The heuristic above looks at _what has worked so far_.
We get better results if we look at _what will happen in the future_:

```pycon
>>> def average(node):
...     return (node.board.lower_bound + node.board.upper_bound) / 2
>>> len(greedy_search(board, key=average))
17

```

A range on lower and upper bound like `[8, 12]` is probably better than `[6, 14]`
even though the average is the same.
We can break ties by adding more scores and returning a tuple:

```pycon
>>> def modified_average(node):
...     avg = (node.board.lower_bound + node.board.upper_bound) / 2
...     range_ = node.board.upper_bound - node.board.lower_bound
...     cleared_per_move = node.cleared / len(node.moves)
...     return (avg, range_, -cleared_per_move)
>>> len(greedy_search(board, key=modified_average))
16

```

Randomized search can also be expressed with a key function:

```pycon
>>> import random
>>> rng = random.Random(0)
>>> def random_key(node):
...     return rng.random()
>>> len(greedy_search(board, key=random_key))
30

```

### Heuristic search

**Heuristic search** uses a priority queue and yields solutions as they are discovered.
If we let it run long enough, it will produce an optimal solution.
We might run out of memory before that happens though.

```pycon
>>> from solvers import heuristic_search
>>> for moves in heuristic_search(board, key=cleared_per_move, iterations=999):
...    print(f"Found solution of length {len(moves)}")
...    assert board.verify_solution(moves)
Found solution of length 27
Found solution of length 22
Found solution of length 20
Found solution of length 18

```

Better key functions give better results faster:

```pycon
>>> for moves in heuristic_search(board, key=modified_average, iterations=999):
...    print(f"Found solution of length {len(moves)}")
Found solution of length 16

```

The solver can be given an initial solution via the `moves` argument.
The default `key=None` uses a good key.

```pycon

>>> moves = greedy_search(board)
>>> len(moves)
19
>>> for moves in heuristic_search(board, iterations=9999, moves=moves):
...    print(f"Found solution of length {len(moves)}")
...    assert board.verify_solution(moves)
Found solution of length 15
Found solution of length 14
Found solution of length 13

```

### Beam search

**Beam search** expands all children, keeps the `beam_width` best nodes, 
expands all children of those nodes, and repeats.

```pycon
>>> from solvers import beam_search
>>> len(beam_search(board, beam_width=1, key=modified_average))
16
>>> len(beam_search(board, beam_width=2, key=modified_average))
15
>>> len(beam_search(board, beam_width=4, key=modified_average))
14

```

We can also run it for `beam_width=1, 2, 4, 8, ..., 2^power`:

```pycon
>>> from solvers import anytime_beam_search
>>> for moves in anytime_beam_search(board, power=6, key=None):
...    print(f"Found solution of length {len(moves)}")
Found solution of length 19
Found solution of length 16
Found solution of length 15
Found solution of length 14

```

### Monte Carlo search

**Monte Carlo tree search** also yields solutions as they are discovered.
In the long run it produces a solution, but again we might run out of memory.

```pycon
>>> from solvers import monte_carlo_search
>>> for moves in monte_carlo_search(board, seed=1, iterations=99, key=cleared_per_move):
...    print(f"Found solution of length {len(moves)}")
Found solution of length 24
Found solution of length 23
Found solution of length 20
Found solution of length 19

```

Better keys help here too, because they help tie-beak between UCB scores and
help guide the simulation (rollout) stage. Simulations use a weighted sampling
of key values, where weights are given by `w_i = exp(-k * key(node))`.

```pycon
>>> for moves in monte_carlo_search(board, seed=1, iterations=99, key=modified_average):
...    print(f"Found solution of length {len(moves)}")
Found solution of length 17
Found solution of length 16
Found solution of length 15


```
