"""
This module contains solvers.

Best-first search
-----------------

The board below can be solved in 3 moves, but best-first uses 4 moves.

>>> grid = [[3, 3, 3],
...         [2, 2, 3],
...         [2, 1, 2]]
>>> board = Board(grid)
>>> moves = list(best_first_search(board))
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
>>> solutions = monte_carlo_search(board)
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
from typing import Optional, List, Set
import dataclasses
from heapq import heappush, heappop, heapify, nsmallest
from collections import deque
import pytest
import random
import itertools
import functools

from board import Board


def middle_bound(node):
    alpha = 0.4
    avg = (1 - alpha) * node.board.lower_bound + alpha * node.board.upper_bound
    return len(node.moves) + avg - 0.1 * node.cleared


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
class BestFirstNode:
    """Make board states comparable for search."""

    board: Board
    moves: tuple  # Using tuple instead of list since lists aren't hashable
    cleared: int = 0


def best_first_search(board: Board, *, exponent=None, seed=None, key=None):
    """Greedy search. Choose the move that clears the most cells.

    If power is a number, then the algorithm is no longer deterministic.
    Instead, it records the number of cleared cells per child and chooses
    a random move with probability weights: cleared**power

    Examples
    --------
    >>> board = Board([[0, 0, 0, 3],
    ...                [3, 3, 3, 2],
    ...                [3, 2, 2, 1]])
    >>> best_first_search(board)
    [(2, 3), (1, 0), (2, 1)]
    """
    assert exponent is None or exponent >= 1
    key = middle_bound if key is None else key

    rng = random.Random(seed)
    node = BestFirstNode(board.copy(), moves=(), cleared=board.cleared)

    while True:
        # Board is solved, return the moves
        if node.board.is_solved:
            return list(node.moves)

        # Generate all children
        child_nodes = [
            BestFirstNode(
                child, moves=node.moves + (move,), cleared=node.cleared + num_removed
            )
            for move, child, num_removed in node.board.children(return_removed=True)
        ]

        # Deterministic selection of the next move
        if exponent is None:
            node = min(child_nodes, key=key)

        # Randomized selection of the next move
        else:
            # Score all child nodes
            scores = [key(node) for node in child_nodes]

            # Normalize all scores to be positive, then compute 2**-score
            min_scores = min(scores)
            weights = [exponent ** (min_scores - score) for score in scores]
            node = rng.choices(child_nodes, weights=weights, k=1)[0]


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

    # Queue of (board, moves) tuples using a deque for efficient popleft
    queue = deque([(board.copy(), [])])

    # Track visited states to avoid cycles
    visited = {board.copy()}

    while queue:
        current_board, moves = queue.popleft()

        # Check if we've found a solution
        if current_board.is_solved:
            return moves

        # Try all possible moves from current state
        for (i, j), next_board in current_board.children():
            # Skip if we've seen this state before
            if next_board in visited:
                continue

            # Add new state to queue and visited set
            visited.add(next_board)
            queue.append((next_board, moves + [(i, j)]))


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


@dataclasses.dataclass(frozen=True, eq=False, order=False)
class AStarNode:
    """Make board states comparable for search."""

    board: Board
    moves: tuple  # Using tuple instead of list since lists aren't hashable

    def g(self):
        return len(self.moves)

    def h(self):
        return self.board.lower_bound

    @functools.cached_property
    def f(self):
        num_moves = len(self.moves)
        # Return (admissible_heuristic(), non_admissible(), non_admissible())
        # The overall result is still admissible, but the second and third
        # component of the tuple act as tie-breakers
        cleared_per_move = self.board.cleared / num_moves
        return (self.g() + self.h(), -cleared_per_move, -num_moves)

    def __lt__(self, other):
        return self.f < other.f


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

    # f(n) = num_moves + heuristic(n), board, moves in a SearchNode class
    heap = [AStarNode(board.copy(), moves=())]
    g_scores = {board.copy(): 0}  # Keep track of nodes seen and number of moves

    while heap:
        # Pop the smallest item from the heap
        current = heappop(heap)
        current_g = len(current.moves)

        # The path to current node is larger than what we've seen, so skip it
        if current_g > g_scores[current.board]:
            continue

        # The board is solved, return the list of moves
        if current.board.is_solved:
            return list(current.moves)

        # Go through all children, created by applying a single move
        for (i, j), next_board in current.board.children():
            # Increment by one, since we need to make one more move to get here
            g = current_g + 1

            # If not seen before, or the path is lower than recorded
            if (next_board not in g_scores) or (g < g_scores[next_board]):
                g_scores[next_board] = g
                next_node = AStarNode(next_board, moves=current.moves + ((i, j),))
                heappush(heap, next_node)


# =============================================================================


@dataclasses.dataclass(frozen=True, eq=False, order=False)
class BeamNode:
    """Node for beam search with evaluation function."""

    board: Board
    moves: tuple
    cleared: int = 0

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
    Solution of length 8
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

        # If power is None, stop after no improvements for several
        if power_is_None and no_improvement_count >= 3:
            break


# =============================================================================


@dataclasses.dataclass(frozen=True, eq=False, order=False)
class HeuristicNode:
    """Make board states comparable for search.

    The comparison operator < is needed for heapq. Here we switch signs in the
    __lt__ implementation, so that every property is better if it is higher.
    """

    board: Board
    moves: tuple
    cleared: int = None


def heuristic_search(
    board: Board, *, max_nodes=0, shortest_path=None, key=None, verbose=False
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
    shortest_path = shortest_path or float("inf")

    class LocalHeuristicNode(HeuristicNode):
        def __lt__(self, other):
            return key(self) < key(other)

    # Yield a greedy solution, which also gives a lower bound on the solution
    yield (greedy_solution := best_first_search(board, key=key, exponent=None))
    shortest_path = min(len(greedy_solution), shortest_path)

    # Add the board to the heap
    heap = [LocalHeuristicNode(board.copy(), moves=(), cleared=board.cleared)]
    g_scores = {board: 0}  # Keep track of nodes seen and number of moves

    popped_counter = 0
    while heap:
        # Terminate the search
        if max_nodes and popped_counter > max_nodes:
            return

        # Pop the the highest-priority node from the heap
        current = heappop(heap)
        current_g = len(current.moves)  # g(node) = number of moves
        popped_counter += 1

        # if current_g + current.board.upper_bound < shortest_path:
        #    print(f"{current_g=} + {current.board.upper_bound=} < {shortest_path=}")

        if popped_counter % max((max_nodes // 100), 1) == 0 and verbose:
            msg = f"""Nodes popped:{popped_counter}  Progress:{popped_counter/max_nodes:.1%}  Shortest path:{shortest_path}
 Heuristic function value:{key(current):.2f}
 Depth:{len(current.moves)}  In queue:{len(heap)}  Seen:{len(g_scores)}"""
            print(msg)

        # The lower bound f(n) = g(n) + h(n) >= best we've seen, so skip it
        if current_g + current.board.lower_bound >= shortest_path:
            continue

        # The path to current node is longer than what we've seen, so skip it
        if current_g > g_scores[current.board]:
            continue

        # The board is solved. If the path is shorter than what we have,
        # then yield it and update the lower bound.
        if current.board.is_solved and len(current.moves) < shortest_path:
            yield list(current.moves)
            shortest_path = len(current.moves)

            # Filter the heap, removing every node that cannot be better
            heap = [
                n for n in heap if len(n.moves) + n.board.lower_bound < shortest_path
            ]
            heapify(heap)

            # Filter the scores
            boards = set(n.board for n in heap)
            g_scores = {b: v for (b, v) in g_scores.items() if b in boards}
            del boards

        # Go through all children, created by applying a single move
        children = current.board.children(return_removed=True)
        for (i, j), child_board, num_removed in children:
            g = current_g + 1  # One more move is needed to reach the child

            # If not seen before, or the path is lower than recorded
            if (child_board not in g_scores) or (g < g_scores[child_board]):
                g_scores[child_board] = g
                next_node = LocalHeuristicNode(
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
        self.children = []

    @functools.cached_property
    def remaining(self):
        return self.board.lower_bound

    @functools.cached_property
    def remaining_groups(self):
        return self.board.upper_bound

    def ucb_score(self, exploration=1.41):
        """Calculate UCB score for node selection."""
        if not self.visits:
            # If not visisted, return UCB of +inf and tie-break with h(n)
            return float("inf")

        exploit = self.score / self.visits
        explore = exploration * math.sqrt(math.log(self.parent.visits) / self.visits)

        # Return UCB score, and use the admissible heuristic as tie-breaker
        return exploit + explore

    def __lt__(self, other):
        """Implementing < (__lt__) is needed for max() to work."""

        # UCB score is the main comparison metric
        if self.ucb_score() != other.ucb_score():
            return self.ucb_score() < other.ucb_score()

        # If UCB score is the same
        attrs = ["remaining", "remaining_groups"]
        self_values = (getattr(self, attr) for attr in attrs)
        other_values = (getattr(other, attr) for attr in attrs)

        for self_v, other_v in zip(self_values, other_values):
            if self_v != other_v:
                # Switch comparison from < to > means that lower is better
                return self_v > other_v

        return False

    def expand(self) -> List["MCTSNode"]:
        """Create child nodes for all possible moves."""
        if self.children:
            return self.children

        children = self.board.children(return_removed=True)
        for move, next_board, num_removed in children:
            child = MCTSNode(
                next_board,
                move=move,
                parent=self,
                depth=self.depth + 1,
                remaining_cells=self.remaining_cells - num_removed,
                cleared_cells_in_move=num_removed,
            )
            self.children.append(child)

        return self.children

    def construct_moves(self):
        """Return a list of moves from the root to this node."""
        moves, node = [], self
        while node.parent is not None:
            moves.append(node.move)
            node = node.parent

        return list(reversed(moves))

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


def monte_carlo_search(
    board: Board,
    *,
    iterations=1000,
    seed=None,
    verbosity=0,
    shortest_path=None,
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
    Solution of length 3: [(2, 3), (1, 0), (2, 1)]
    """

    def vprint(*args, v=0, **kwargs):
        """Verbose printing function with a filter v."""
        if verbosity >= v:
            print(*args, **kwargs)

    shortest_path = shortest_path or float("inf")

    # Yield a greedy solution, which also gives a lower bound on the solution
    yield (greedy_solution := best_first_search(board))
    shortest_path = min(len(greedy_solution), shortest_path)

    # Root note for all iterations
    root = MCTSNode(board.copy(), remaining_cells=board.remaining)

    for iteration in range(1, iterations + 1):
        if verbosity >= 2:
            vprint(f"Iteration: {iteration} ({shortest_path=})", v=2)

        # Start at the top of the explored tree
        node = root
        path = [node]

        # While the current node has not been visisted, search down by UCB
        while node.visits > 0:
            # If the node is solved, skip ahead to simulating it.
            # Simulation returns no moves, so we'll yield the path to the node
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
                    return

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
                    return
                node = path[-1]  # Reset the node to parent
                continue

            # Any unvisited node will get UCB score +inf and be chosen
            # TODO: The parameter `exploration` for UCB could be tuned
            node = max(children)
            path.append(node)

        # Simulate from leaf node of explored tree down to the end of the game
        # TODO: Here `power` could be considered a hyperparameter to be tuned.
        simulation_seed = None if seed is None else seed + iteration
        simulation_moves = best_first_search(
            node.board, exponent=10.0, seed=simulation_seed
        )

        sim_num_cleared = node.remaining_cells
        sim_num_moves = len(simulation_moves)
        vprint(
            f" Simulation from depth {node.depth} gave path of length {sim_num_moves} (total={node.depth + sim_num_moves})",
            v=2,
        )
        vprint(
            f" Cleared per move in simulation: {sim_num_cleared / sim_num_moves:.3f}",
            v=2,
        )

        if verbosity == 1 and iteration % max((iterations // 100), 1) == 0:
            vprint(
                f"Iter: {iteration} ({iteration/iterations:.1%}) (sim. @ d={node.depth}, cleared/move @ sim={sim_num_cleared / sim_num_moves:.3f}) (treesize: {root.size()}) ({shortest_path=})",
                v=1,
            )

        # The simulation lead to a new shortest path
        if node.depth + sim_num_moves < shortest_path:
            shortest_path = node.depth + sim_num_moves
            vprint(
                f"Simulation from depth {node.depth} gave new best path of length {shortest_path}",
                v=1,
            )
            yield node.construct_moves() + simulation_moves

        # Backpropagation starting at the bottom node in the seen tree and
        # going up to the root. If we have, from root, cleared [3, 1, 2]
        # and simulation clears 10 in 3 moves, then the update will set scores:
        # node #4 (leaf) : (10)             / 3
        # node #3        : (10 + 2)         / 4
        # node #2        : (10 + 2 + 1)     / 5
        # node #1 (root) : (10 + 2 + 1 + 1) / 6
        # The score at each node is average number cleared from that node.
        vprint(" Backpropagation (starting at bottom node and going up)", v=3)
        cleared_in_path = 0
        for path_num_moves, node in enumerate(reversed(path)):
            node.visits += 1
            score_cleared = cleared_in_path + sim_num_cleared
            score_moves = path_num_moves + sim_num_moves
            node.score += score_cleared / score_moves
            vprint(f"  Updated node at depth {node.depth}", v=3)
            vprint(
                f"   Cleared / move = ({cleared_in_path} + {sim_num_cleared}) / ({path_num_moves} + {sim_num_moves})",
                v=3,
            )
            cleared_in_path += node.cleared_cells_in_move

    # Make one final attempt to extract a solution. Follow expected
    # return down the tree as far as possible, then best-first search
    moves = []
    node = root
    while True:
        if node.board.is_solved:
            break

        children = node.expand()
        if any(n.visits == 0 for n in children):
            moves.extend(best_first_search(node.board, exponent=None))
            break

        # Pure exploitation strategy by following expected return
        node = max(children, key=lambda n: n.score / n.visits)
        moves.append(node.move)

    if len(moves) < shortest_path:
        yield moves


# =============================================================================


if __name__ == "__main__":
    pytest.main(
        [
            __file__,
            "-v",
            "--doctest-modules",
            "-l",
        ]
    )
