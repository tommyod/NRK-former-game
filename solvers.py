"""
This module contains solvers.

Best-first search
-----------------

The board below can be solved in 3 moves, but best-first uses 4 moves.

>>> grid = [[3, 3, 3],
...         [2, 2, 3],
...         [2, 1, 2]]
>>> board = Board(grid)
>>> moves = greedy_search(board)
>>> moves
[(0, 0), (2, 1), (1, 0)]
>>> board.verify_solution(moves)
True


Breadth-first search (BFS)
--------------------------

>>> grid = [[3, 3, 3],
...         [2, 2, 3],
...         [2, 1, 2]]
>>> board = Board(grid)
>>> moves = breadth_first_search(board)
>>> moves
[(0, 0), (2, 1), (1, 0)]


Iterative deepening search
--------------------------

>>> iterative_deepening_search(board)
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


Monte Carlo search
------------------

>>> grid = [[3, 3, 3],
...         [2, 2, 3],
...         [2, 1, 2]]
>>> board = Board(grid)
>>> solutions = monte_carlo_search(board, seed=0)
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

import math
from typing import Optional, List, Set, Callable
import dataclasses
from heapq import heappush, heappop, heapify, nsmallest
import pytest
import random
import itertools
import functools

from board import Board


def middle_bound(node):
    """Default key function. Used to prioritize nodes in various algorithms."""

    # Priority 1: Compute a biased average of total moves => lower is better
    alpha = 0.3  # Experiments show that 0.3 is a good value
    avg = (1 - alpha) * node.board.lower_bound + alpha * node.board.upper_bound
    expected = len(node.moves) + avg

    # Priority 2: Compute the range between low and high => lower is better
    assert node.board.upper_bound >= node.board.lower_bound
    range_ = abs(node.board.upper_bound - node.board.lower_bound) ** 0.5

    # Priority 3: Cleared per move => higher is better
    cleared_per_move = node.cleared / len(node.moves) if node.moves else 0

    return 1.0 * expected + 0.1 * range_ - 0.01 * cleared_per_move


def unique_everseen(iterable, key=None):
    """
    Yield unique elements, preserving order.

    Examples
    --------
    >>> list(unique_everseen('AAAABBBCCDAABBB'))
    ['A', 'B', 'C', 'D']
    >>> list(unique_everseen('ABBCcAD', str.lower))
    ['A', 'B', 'C', 'D']
    """
    # https://github.com/more-itertools/more-itertools/blob/abbcfbce24cb59e62e01daac4d80f9658202708a/more_itertools/recipes.py#L483
    seenset = set()
    seenset_add = seenset.add
    seenlist = []
    seenlist_add = seenlist.append
    use_key = key is not None

    for element in iterable:
        k = key(element) if use_key else element
        try:
            if k not in seenset:
                seenset_add(k)
                yield element
        except TypeError:
            if k not in seenlist:
                seenlist_add(k)
                yield element


@dataclasses.dataclass(frozen=True, eq=False, order=False)
class Node:
    board: Board
    moves: tuple  # Using tuple instead of list since lists aren't hashable
    cleared: int = 0


def greedy_search(board: Board, *, key: Optional[Callable[Board, float]] = None):
    """Expand all children, selects the best one and repeat.

    The `key` argument can be function that evalutes a node. A node has three
    attributes: the board (`board`), a tuple containing moves (`moves`) and
    the number of cleared cells (`cleared`) in the board. The function returns
    a number, and a low number means a high priority.

    Examples
    --------

    The default heuristic used when key=None is quite good:

    >>> board = Board([[0, 0, 0, 3],
    ...                [3, 3, 3, 2],
    ...                [3, 2, 2, 1]])
    >>> greedy_search(board, key=None)
    [(2, 3), (1, 0), (2, 1)]

    To maximize the number of cells cleared per mode:

    >>> def cleared_per_move(node):
    ...     return -node.cleared / len(node.moves)
    >>> greedy_search(board, key=cleared_per_move)
    [(1, 0), (2, 1), (0, 3), (1, 3), (2, 3)]

    A randomized algorithm can be implemented as follows:

    >>> import random
    >>> rng = random.Random(0)
    >>> def random(node):
    ...     return rng.random()
    >>> greedy_search(board, key=random)
    [(2, 1), (1, 3), (2, 3), (1, 0)]
    """
    key = middle_bound if key is None else key
    node = Node(board.copy(), moves=(), cleared=board.cleared)

    while not node.board.is_solved:
        # Generate all children
        child_nodes = (
            Node(child, moves=node.moves + (move,), cleared=node.cleared + num_removed)
            for move, child, num_removed in node.board.children(return_removed=True)
        )

        node = min(child_nodes, key=key)

    # The board is solved, return the moves
    return list(node.moves)


def breadth_first_search(board: Board) -> list:
    """Breadth-first search to find shortest solution path.

    This approach is not very efficient, but it is guaranteed to return
    a minimum path, solving the board in the fewest moves possible.

    Examples
    --------
    >>> board = Board([[0, 0, 0, 3],
    ...                [3, 3, 3, 2],
    ...                [3, 2, 2, 1]])
    >>> list(breadth_first_search(board))  # Finds optimal solution
    [(2, 3), (1, 0), (2, 1)]
    """

    def path_length(node):
        """Breadth-first-search always explores depth i before i+1."""
        return len(node.moves)

    *_, moves = heuristic_search(board, key=path_length)
    return moves


def depth_limited_search(board: Board, *, depth_limit: int) -> Optional[list]:
    """Recursive depth-limited search implementation. Will not find the optimal
    solution unless it's length equals the depth limit.

    Examples
    --------
    >>> board = Board([[1, 2],
    ...                [2, 1]])
    >>> depth_limited_search(board, depth_limit=2)
    >>> moves = depth_limited_search(board, depth_limit=3)
    >>> moves
    [(0, 0), (1, 1), (1, 0)]
    >>> board.verify_solution(moves)
    True

    With a depth limit of three, the optimal solution is found:

    >>> board = Board([[0, 0, 0, 3],
    ...                [3, 3, 3, 2],
    ...                [3, 2, 2, 1]])
    >>> depth_limited_search(board, depth_limit=3)  # Finds optimal solution
    [(2, 3), (1, 0), (2, 1)]
    """

    def dfs(board: Board, depth: int, moves: list) -> Optional[list]:
        if board.is_solved:
            return moves

        if depth >= depth_limit:
            return

        for move, child in board.children():
            if result := dfs(child, depth + 1, moves + [move]):
                return result

    return dfs(board.copy(), 0, [])


def iterative_deepening_search(board: Board) -> list:
    """Iterative deepening depth limited search to find shortest solution path.

    Examples
    --------
    >>> grid = [[3, 3, 3],
    ...         [2, 2, 3],
    ...         [2, 1, 2]]
    >>> board = Board(grid)
    >>> moves = iterative_deepening_search(board)
    >>> moves
    [(0, 0), (2, 1), (1, 0)]
    """
    for depth in itertools.count(0):
        if result := depth_limited_search(board, depth_limit=depth):
            return result


# =============================================================================


def a_star_search(board: Board) -> list:
    """A star search with a consistent heuristic. Guaranteed to find the
    optimal solution.

    Examples
    --------
    >>> grid = [[3, 3, 3],
    ...         [2, 2, 3],
    ...         [2, 1, 2]]
    >>> board = Board(grid)
    >>> moves = a_star_search(board)
    >>> moves
    [(0, 0), (2, 1), (1, 0)]
    """

    def admissible_heuristic(node):
        """Admissible heurisic used for A*"""
        if not node.moves:
            return 0
        # Return (admissible_heuristic(), non_admissible(), non_admissible())
        # The overall result is still admissible, but the second and third
        # component of the tuple act as tie-breakers
        lower_bound = len(node.moves) + node.board.lower_bound
        upper_bound = len(node.moves) + node.board.upper_bound
        cleared_per_move = node.board.cleared / len(node.moves)
        return (lower_bound, upper_bound, -cleared_per_move)

    *_, moves = heuristic_search(board, key=admissible_heuristic)
    return moves


# =============================================================================


class BeamNode(Node):
    """Node for beam search with evaluation function."""

    def __eq__(self, other):
        # Implemented to apply `unique_everseen` to nodes, to remove duplicates
        return self.board == other.board

    def __hash__(self):
        # Implemented to apply `unique_everseen` to nodes, to remove duplicates
        return hash(self.board)


def beam_search(
    board: Board, *, beam_width: int = 3, shortest_path=None, key=None
) -> list:
    """Beam search with specified beam width.

    Maintains only the top beam_width nodes at each depth level.
    Returns the first solution found.

    Examples
    --------
    >>> grid = [[3, 3, 3],
    ...         [2, 2, 3],
    ...         [2, 1, 2]]
    >>> board = Board(grid)
    >>> moves = beam_search(board)
    >>> moves  # May not find optimal solution
    [(0, 0), (2, 1), (1, 0)]
    """
    key = middle_bound if key is None else key

    beam = [BeamNode(board.copy(), moves=(), cleared=0)]
    while beam:
        # Check if any are solved
        for node in beam:
            if node.board.is_solved:
                return list(node.moves)

        # Generate all children of current beam
        next_beam = (
            BeamNode(
                next_board,
                moves=node.moves + (move,),
                cleared=node.cleared + num_removed,
            )
            for node in beam
            for (move, next_board, num_removed) in node.board.children(
                return_removed=True
            )
        )

        # Only keep unique boards. If two boards are unique we know the path
        # length must be unique too, so we can discard the duplicates
        next_beam = unique_everseen(next_beam)

        # Only keep boards where we can do better
        if shortest_path:
            next_beam = (
                node
                for node in next_beam
                if len(node.moves) + node.board.lower_bound < shortest_path
            )

        # Keep only the best beam_width nodes
        beam = nsmallest(n=beam_width, iterable=next_beam, key=key)


def anytime_beam_search(board, *, power: int = 1, key=None, verbose: bool = False):
    """Run beam search with width=1,2,4,8,...,2**power, yielding solutions.
    If power is None, then power will be increased until no improvement occurs
    for 3 iterations.

    Examples
    --------
    >>> board = Board([[0, 0, 0, 3],
    ...                [3, 3, 3, 2],
    ...                [3, 2, 2, 1]])
    >>> for moves in anytime_beam_search(board, power=None):
    ...     assert board.verify_solution(moves)
    ...     print(f'Solution of length {len(moves)}: {moves}')
    Solution of length 3: [(2, 3), (1, 0), (2, 1)]
    >>> board = Board([[1, 2, 2, 1, 3, 3],
    ...                [4, 1, 2, 1, 1, 1],
    ...                [3, 2, 1, 1, 3, 1],
    ...                [3, 1, 3, 3, 4, 4]])
    >>> for moves in anytime_beam_search(board, power=6):
    ...     assert board.verify_solution(moves)
    ...     print(f'Solution of length {len(moves)}')
    Solution of length 7
    """
    power_is_None = power is None
    shortest_path = float("inf")
    no_improvement_count = 0

    for p in itertools.count(0):
        # Break conditions on power
        if not power_is_None and p > power:
            break

        # If we cannot possibly do better, stop searching
        if shortest_path <= board.lower_bound:
            break

        if verbose:
            print(f"Beam search with beam_width=2**{p}={2**p}")

        moves = beam_search(
            board, beam_width=2**p, shortest_path=shortest_path, key=key
        )

        # Only yield if we found a better solution
        if moves and len(moves) < shortest_path:
            if verbose:
                print(f" Found new best solution with length: {len(moves)}")
            yield moves
            shortest_path = len(moves)
            no_improvement_count = 0
        else:
            no_improvement_count += 1

        # If power is None, stop after no improvements for several iterations
        if power_is_None and no_improvement_count >= 3:
            break


# =============================================================================


def heuristic_search(
    board: Board, *, iterations=0, moves=None, key=None, verbose=False
):
    """A heuristic search that yields solutions as they are found.

    If run long enough, then this function will eventually find the optimal
    path. The optimal path will be the last path it yields, but as it
    searches the graph it will yield the best paths found so far.

    Examples
    --------
    >>> board = Board([[0, 0, 0, 3],
    ...                [3, 3, 3, 2],
    ...                [3, 2, 2, 1]])
    >>> for moves in heuristic_search(board):
    ...     assert board.verify_solution(moves)
    ...     print(f'Solution of length {len(moves)}: {moves}')
    Solution of length 3: [(2, 3), (1, 0), (2, 1)]
    """
    key = middle_bound if key is None else key
    shortest_path = len(moves) if moves else float("inf")

    class HeuristicNode(Node):
        def __lt__(self, other):
            return key(self) < key(other)

    def prune(heap, num_moves, shortest_path):
        """Prune nodes that will lead nowhere."""

        # Filter the heap
        def should_be_kept(node):
            return len(node.moves) + node.board.lower_bound < shortest_path

        heap = list(filter(should_be_kept, heap))
        heapify(heap)

        # Filter the scores
        boards = set(n.board for n in heap)
        num_moves = {b: v for (b, v) in num_moves.items() if b in boards}
        return heap, num_moves

    # Add the board to the heap
    heap = [HeuristicNode(board.copy(), moves=(), cleared=board.cleared)]
    num_moves = {board: 0}  # Keep track of nodes seen and number of moves

    popped_counter = 0
    while heap:
        # We have popped `max_nodes` => terminate the search
        if iterations and popped_counter >= iterations:
            return

        # Pop the the highest-priority node from the heap
        current = heappop(heap)
        depth = len(current.moves)  # g(node) = number of moves
        popped_counter += 1

        if popped_counter % max((iterations // 100), 1) == 0 and verbose:
            msg = f"""Nodes popped:{popped_counter:_}  Iterations:{iterations:_}  Shortest path:{shortest_path}
 Heuristic function value:{key(current):.2f}
 Depth:{depth}  Bounds:[{depth + current.board.lower_bound}, {depth + current.board.upper_bound}] 
 In queue:{len(heap):_}  Seen:{len(num_moves):_}"""
            print(msg)

        # The lower bound f(n) = g(n) + h(n) >= best we've seen, so skip it
        if depth + current.board.lower_bound >= shortest_path:
            continue

        # The path to current node is longer than what we've seen, so skip it
        if depth > num_moves[current.board]:
            continue

        # The board is solved. If the path is shorter than what we have,
        # then yield it and update the lower bound.
        if current.board.is_solved and len(current.moves) < shortest_path:
            yield list(current.moves)
            shortest_path = len(current.moves)
            heap, num_moves = prune(heap, num_moves, shortest_path)

        # The upper bound on this node is lower than what we have seen,
        # so we attempt to attain the bound immediately
        if depth + current.board.upper_bound < shortest_path:
            moves = greedy_search(current.board, key=key)
            if len(moves) < current.board.upper_bound:
                yield list(current.moves) + moves
                shortest_path = len(current.moves) + len(moves)
                heap, num_moves = prune(heap, num_moves, shortest_path)

        # Go through all children, created by applying a single move
        children = current.board.children(return_removed=True)
        for (i, j), child_board, num_removed in children:
            g = depth + 1  # One more move is needed to reach the child

            # If not seen before, or the path is lower than recorded
            if (child_board not in num_moves) or (g < num_moves[child_board]):
                num_moves[child_board] = g
                next_node = HeuristicNode(
                    board=child_board,
                    moves=current.moves + ((i, j),),
                    cleared=current.cleared + num_removed,
                )
                heappush(heap, next_node)


# =============================================================================


# We need to asssign to children after creation, so we set frozen=False
@dataclasses.dataclass(frozen=False)
class MCTSNode:
    """Make board states comparable for tree search."""

    board: Board
    move: Optional[tuple] = None
    parent: Optional["MCTSNode"] = dataclasses.field(repr=False, default=None)
    visits: int = 0
    score: float = 0  # Cleared per node in subtree if this node is chosen
    children: Set["MCTSNode"] = dataclasses.field(repr=False, default=None)
    depth: int = 0
    remaining_cells: int = 0  # Total remaining
    cleared_cells_in_move: int = 0  # Cleared by last move

    def __post_init__(self):
        self.children = []  # Cache for children

    def ucb_score(self, exploration=1.41):
        """Calculate UCB score for node selection."""
        if not self.visits:
            # If not visisted, return UCB of +inf and tie-break with h(n)
            return -float("inf")

        exploit = self.score / self.visits
        # Fewer visits => 1/sqrt(visits) is larger => min() selects this node
        explore = exploration * math.sqrt(math.log(self.parent.visits) / self.visits)
        return exploit - explore

    def expand(self) -> List["MCTSNode"]:
        """Create child nodes for all possible moves."""
        # Use cache if we have it
        if self.children:
            return self.children

        # Expand all children and create child nodes, store in cache
        children = self.board.children(return_removed=True)
        for move, next_board, num_removed in children:
            child = type(self)(
                next_board,
                move=move,
                parent=self,
                depth=self.depth + 1,
                remaining_cells=self.remaining_cells - num_removed,
                cleared_cells_in_move=num_removed,
            )
            self.children.append(child)

        return self.children

    @functools.cached_property
    def moves(self):
        """Return a list of moves from the root node to this node."""

        # Search up in the tree until there is no parent
        moves, node = [], self
        while node.parent is not None:
            moves.append(node.move)
            node = node.parent

        # Reverse it and return
        return tuple(reversed(moves))

    @property
    def cleared(self):
        return self.board.cleared

    def prune(self):
        """Prune a node by removing it from the tree."""
        # Remove references and help garbage collector
        self.parent.children.remove(self)  # Unhook reference
        self.parent = None  # Unhook this node from the parent

    def size(self):
        """Return the size of the tree, counting from this node down."""
        if not self.children and self.visits:
            return 1
        return sum(child.size() for child in self.children)

    def search(self, shortest_path, vprint, key=None):
        """Returns a tuple (node, path) by searching down the tree.

        By default, the UCB scores is used.

        """

        # Start at this node
        node, path = self, [self]

        # While the current node has not been visisted
        while node.visits > 0:
            # If the node is solved, break out and return it
            if node.board.is_solved:
                break

            # We cannot possibly improve on what we already have
            if node.depth + node.board.lower_bound >= shortest_path:
                vprint(
                    f" Pruning: depth + h(board) >= shortest ({node.depth} + {node.board.lower_bound} >= {shortest_path})",
                    v=3,
                )

                # Pruning condition on the root node. We cannot possibly
                # do better, so we terminate the algorithm completely.
                if len(path) == 1:
                    return None, None

                node.prune()
                path.pop()  # Remove this node from path
                node = path[-1]  # Reset the node to parent
                continue

            # Expand children (.expand() is cached if we have it before)
            children = node.expand()

            # All children are pruned, go to neighbor
            if not children:
                vprint(
                    f" All children have been pruned. Pruning this node at depth {node.depth}.",
                    v=3,
                )
                node.prune()
                path.pop()  # Remove this node from path

                # All children of root node were pruned, return
                if not path:
                    return None, None
                node = path[-1]  # Reset the node to parent
                continue

            # Any unvisited node will get UCB score -inf and be chosen
            node = min(children, key=key)
            path.append(node)

        return node, path


def monte_carlo_search(
    board: Board,
    *,
    iterations=0,
    seed=None,
    verbosity=0,
    moves=None,
    key=None,
) -> list:
    """Monte Carlo Tree Search to find solution path.

    Examples
    --------
    >>> board = Board([[0, 0, 0, 3],
    ...                [3, 3, 3, 2],
    ...                [3, 2, 2, 1]])
    >>> for moves in monte_carlo_search(board, iterations=100, seed=42):
    ...     assert board.verify_solution(moves)
    ...     print(f'Solution of length {len(moves)}: {moves}')
    Solution of length 3: [(2, 3), (2, 1), (1, 0)]
    """
    key = middle_bound if key is None else key
    shortest_path = len(moves) if moves else float("inf")
    rng = random.Random(seed)

    def vprint(*args, v=0, **kwargs):
        """Verbose printing function with a filter v."""
        if verbosity >= v:
            print(*args, **kwargs)

    def random_key(node, exponent=12.0):
        """Random weighted sampling chooses element k with probability equal
        to the weight w_k / sum_i^n w_i. This is equivalent to computing
        Uniform(0, 1)**(1/w_i) for each i and choosing the maximal element.
        See 'Weighted random sampling with a reservoir', Efraimidis et al."""
        score = key(node)

        # If the score is a tuple, reduce it to a number
        if isinstance(score, tuple):
            score = sum(s_i / 10**i for i, s_i in enumerate(score))

        # No randomness
        if exponent is None:
            return score

        # Lower is better, so set w = pow(exponent, -score). Min/max is
        # unaffected by monotonic transformations, so we take
        # logarithms: log[U(0, 1)**(1/w)] = 1/w * log(U) = exp(w) * log(U)
        # Finally, we take minimum since lower is better on the output.
        return -pow(exponent, score) * math.log(rng.random())
        # Ad-hoc alternative: return score * math.exp(rng.gauss(0, sigma=0.05))

    # Root note for all iterations
    root = MCTSNode(board.copy(), remaining_cells=board.remaining)

    # Main loop
    iteration = 1
    while True:
        if iterations and iteration > iterations:
            break

        if verbosity >= 2:
            vprint(f"Iteration: {iteration} ({shortest_path=})", v=2)

        # Start at the top of the explored tree
        node, path = root.search(
            shortest_path=shortest_path,
            vprint=vprint,
            key=lambda n: (n.ucb_score(), key(n)),
        )

        # Everything got pruned, so terminate the algorithm alltogether
        if node is None:
            return

        # Simulate from leaf node of explored tree down to the end of the game
        # The randomized moves might overestimate, so we can run two simulations.
        # simulation_moves = min(greedy_search(node.board, key=random_key),
        #                       greedy_search(node.board, key=key), key=len)
        simulation_moves = greedy_search(node.board, key=random_key)
        sim_num_moves = len(simulation_moves)
        vprint(
            f" Sim @ depth={node.depth} gave path of length {sim_num_moves} (total={node.depth + sim_num_moves})",
            v=2,
        )

        if verbosity == 1 and iteration % max(((iterations + 1) // 100), 1) == 0:
            vprint(
                f"Iter: {iteration} ({iteration/iterations:.1%}) (sim. @ d={node.depth}, moves @ sim={sim_num_moves}) (treesize: {root.size()}) ({shortest_path=})",
                v=1,
            )

        # The simulation lead to a new shortest path
        if node.depth + sim_num_moves < shortest_path:
            shortest_path = node.depth + sim_num_moves
            vprint(
                f" => Sim @ depth={node.depth} gave new best path of length {shortest_path}",
                v=1,
            )
            yield list(node.moves) + simulation_moves

        # Backpropagation starting at the bottom node in the seen tree and
        # going up to the root. Scores are set to length of the path at each
        # node (simulation path length plus distance from leaf node). Example:
        # node #4 (leaf) : (10)
        # node #3        : (10 + 2)
        # node #2        : (10 + 2 + 1)
        # node #1 (root) : (10 + 2 + 1 + 1)
        vprint(" Backpropagation (starting at bottom node and going up)", v=3)
        for path_num_moves, path_node in enumerate(reversed(path)):
            path_node.visits += 1
            path_node.score += sim_num_moves + path_num_moves
            vprint(
                f"  Updating node at depth {node.depth}: score = {sim_num_moves=} + {path_num_moves=}",
                v=3,
            )

        iteration += 1

    # Make one final attempt to extract a solution. Follow expected
    # return down the tree as far as possible, then use greedy search
    def greedy_key(node):
        # Not visisted => Do not visit these
        if node.visits == 0:
            return (float("inf"), key(node))

        # If visisted, minimize UCB first, then the key
        return (node.ucb_score(exploration=0), key(node))

    node, _ = root.search(shortest_path, vprint, key=greedy_key)
    moves = greedy_search(node.board, key=key)
    if len(node.moves) + len(moves) < shortest_path:
        yield list(node.moves) + moves


# =============================================================================


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--doctest-modules", "-l"])

    if False:
        board = Board.generate_random((9, 7), seed=0)

        *_, moves = anytime_beam_search(board, power=7, verbose=True)

        for moves in monte_carlo_search(board, iterations=9999, verbosity=1, seed=0):
            print(len(moves), moves)
