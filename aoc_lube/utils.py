"""Helpful functions for AoC.

Imports are deferred so that this file loads faster. Requires `networkx` and `numpy`.
"""

from collections.abc import Iterable, Iterator, ValuesView
from typing import Final, Literal, NamedTuple, Self

__all__ = [
    "GRID_NEIGHBORHOODS",
    "UnionFind",
    "Vec2",
    "chinese_remainder_theorem",
    "chunk",
    "distribute",
    "dot_print",
    "grid_steps",
    "extract_ints",
    "extract_maze",
    "ilen",
    "int_grid",
    "maximum_matching",
    "ndigits",
    "nth",
    "oscillate_range",
    "pairwise",
    "partitions",
    "shift_cipher",
    "shiftmod",
    "sliding_window",
    "sliding_window_cycle",
    "spiral_grid",
    "split",
]


class UnionFind[T]:
    """A collection for fast unions of disjoint sets."""

    def __init__(self, iterable=Iterable[T] | None):
        self._parents: dict[T, T] = {}
        self._ranks: dict[T, int] = {}
        self._components: dict[T, set[T]] = {}

        if iterable is not None:
            for item in iterable:
                self.add(item)

    @property
    def components(self) -> ValuesView[set[T]]:
        """Return the disjoint sets."""
        return self._components.values()

    @property
    def size(self) -> int:
        """Number of items in all disjoint sets."""
        return len(self._parents)

    def __contains__(self, item: T) -> bool:
        return item in self._parents

    def __len__(self) -> int:
        return len(self._components)

    def __getitem__(self, item: T) -> T:
        if item not in self:
            raise KeyError(item)

        root = self._find(item)
        return self._components[root]

    def __iter__(self) -> Iterator[T]:
        """Yield each disjoint set."""
        yield from self._components

    def elements(self) -> Iterator[T]:
        """Yield each element of each disjoint set."""
        yield from self._parents.keys()

    def add(self, item: T) -> None:
        """Add a new item to the disjoint set forest."""
        if item in self:
            return
        self._parents[item] = item
        self._ranks[item] = 0
        self._components[item] = {item}

    def _find(self, item: T) -> T:
        """Find the representive of the item's disjoint set."""
        if item not in self:
            raise KeyError(item)

        if self._parents[item] != item:
            self._parents[item] = self._find(self._parents[item])
        return self._parents[item]

    def merge(self, a: T, b: T) -> None:
        """Merge the set containing ``a`` and the set containing ``b``."""
        a_root = self._find(a)
        b_root = self._find(b)
        if a_root == b_root:
            return

        if self._ranks[a_root] < self._ranks[b_root]:
            self._parents[a_root] = b_root
            self._components[b_root] |= self._components.pop(a_root)
        elif self._ranks[a_root] > self._ranks[b_root]:
            self._parents[b_root] = a_root
            self._components[a_root] |= self._components.pop(b_root)
        else:
            self._parents[a_root] = b_root
            self._ranks[b_root] += 1
            self._components[b_root] |= self._components.pop(a_root)


class Vec2(NamedTuple):
    """A 2D point."""

    y: int
    x: int

    def __add__(self, other: tuple[int, int]) -> Self:
        y1, x1 = self
        y2, x2 = other
        return Vec2(y1 + y2, x1 + x2)

    def __sub__(self, other: tuple[int, int]) -> Self:
        y1, x1 = self
        y2, x2 = other
        return Vec2(y1 - y2, x1 - x2)

    def __neg__(self) -> Self:
        y, x = self
        return Vec2(-y, -x)

    def __mul__(self, n: int) -> Self:
        y, x = self
        return Vec2(n * y, n * x)

    def rotate(self, clockwise: bool = True) -> Self:
        """Rotate vector 90 degrees."""
        y, x = self
        if clockwise:
            return Vec2(x, -y)
        return Vec2(-x, y)

    def inbounds(self, size: tuple[int, int], pos: tuple[int, int] = (0, 0)) -> bool:
        """Return whether vec is within some rect."""
        y, x = self
        h, w = size
        oy, ox = pos
        return 0 <= y - oy < h and 0 <= x - ox < w

    @classmethod
    def iter_rect(
        self, size: tuple[int, int], pos: tuple[int, int] = (0, 0)
    ) -> Iterator[Self]:
        """Generate all points in some rect."""
        h, w = size
        oy, ox = pos
        for y in range(h):
            for x in range(w):
                yield Vec2(y + oy, x + ox)

    def adj(self, neighborhood: Literal[4, 5, 8, 9] = 4) -> Iterator[Self]:
        """Yield adjacent points given by neighborhood."""
        return (pos + self for pos in GRID_NEIGHBORHOODS[neighborhood])


# fmt: off
GRID_NEIGHBORHOODS: Final = {
    4: [Vec2(0, 1), Vec2(0, -1), Vec2(1, 0), Vec2(-1, 0)],
    5: [Vec2(0, 0), Vec2(0, 1), Vec2(0, -1), Vec2(1, 0), Vec2(-1, 0)],
    8: [
        Vec2(0, 1), Vec2(0, -1), Vec2(1, 0), Vec2(-1, 0),
        Vec2(1, 1), Vec2(-1, -1), Vec2(1, -1), Vec2(-1, 1),
    ],
    9: [
        Vec2(0, 0), Vec2(0, 1), Vec2(0, -1), Vec2(1, 0), Vec2(-1, 0),
        Vec2(1, 1), Vec2(-1, -1), Vec2(1, -1), Vec2(-1, 1),
    ],
}
# fmt: on


def chinese_remainder_theorem(moduli, residues):
    """Find the solution to a system of modular equations."""
    from math import prod

    N = prod(moduli)

    return (
        sum(
            (div := (N // modulus)) * pow(div, -1, modulus) * residue
            for modulus, residue in zip(moduli, residues)
        )
        % N
    )


def chunk(it, n: int, fillvalue=None):
    """Chunk an iterable into non-overlapping fixed sized pieces."""
    from itertools import zip_longest

    args = [iter(it)] * n
    return zip_longest(*args, fillvalue=fillvalue)


def distribute(iterable, n):
    """Distribute an iterable amoung `n` smaller iterables."""
    from itertools import islice, tee

    children = tee(iterable, n)
    return [islice(it, index, None, n) for index, it in enumerate(children)]


def dot_print(array):
    """Pretty print a binary or boolean array."""
    for row in array:
        print("".join(" #"[i] for i in row))


def grid_steps(neighborhood, height, width):
    """Yield all steps ((y, x), (y + dy, x + dx)) with dy, dx given by `neighborhood`.

    `neighborhood` is one of 4, 5, 8 or 9 (For Von-Neumann or Moore neighborhoods with
    or without center cell. The number is the number of cells in the neighborhood.)
    """
    deltas = GRID_NEIGHBORHOODS[neighborhood]

    for y in range(height):
        for x in range(width):
            for dy, dx in deltas:
                new_y = y + dy
                new_x = x + dx

                if 0 <= new_y < height and 0 <= new_x < width:
                    yield (y, x), (new_y, new_x)


def extract_ints(raw: str):
    """Extract integers from a string."""
    import re

    return map(int, re.findall(r"(-?\d+)", raw))


def extract_maze(raw: str, wall="#", largest_component=False):
    """Parse an ascii maze into a networkx graph. Return a tuple
    `(np.array, nx.Graph)`.
    """
    import networkx as nx
    import numpy as np

    lines = raw.splitlines()
    max_width = max(map(len, lines))
    maze = np.array([list(line + " " * (max_width - len(line))) for line in lines])

    G = nx.grid_graph(maze.shape[::-1])

    walls = np.stack(np.where(maze == wall)).T
    G.remove_nodes_from(map(tuple, walls))

    if largest_component:
        G.remove_nodes_from(
            G.nodes - max(nx.connected_components(G), key=lambda g: len(g))
        )

    return maze, G


def ilen(iterable):
    """Return number of items in `iterable`.

    This will consume the iterable.
    """
    return sum(1 for _ in iterable)


def int_grid(raw, np=True, separator=""):
    """Parse a grid of ints into a 2d list or numpy array (if np==True)."""
    array = [
        [int(i) for i in (line.split(separator) if separator else line) if i]
        for line in raw.splitlines()
    ]

    if np:
        import numpy as np

        return np.array(array)

    return array


def maximum_matching(items: dict[list]):
    """Return a maximum matching from a dict of lists."""
    import networkx as nx

    G = nx.from_dict_of_lists(items)

    for k, v in nx.bipartite.maximum_matching(G, top_nodes=items).items():
        if k in items:  # Filter edges pointing the wrong direction.
            yield k, v


def ndigits(n: int) -> int:
    """Return the number of digits in ``n``."""
    from math import log10

    return int(log10(n)) + 1


def nth(iterable, n):
    """Return nth item of `iterable`."""
    from itertools import islice

    return next(islice(iterable, n, None))


def oscillate_range(start=None, stop=None, step=None, /):
    """Yield values around start."""
    match start, stop, step:
        case (int(), None, None):
            start, stop, step = 0, start, 1 if start > 0 else -1
        case (int(), int(), None):
            step = 1 if start < stop else -1
        case (int(), int(), int()) if step != 0:
            pass
        case _:
            ValueError(f"non-integer values or 0 step ({start=}, {stop=}, {step=})")

    stop_n = (stop - start) // step

    if stop_n <= 0:
        return

    yield start

    n = 1
    while n < stop_n:
        yield start + step * n
        yield start - step * n
        n += 1


def pairwise(iterable, offset=1):
    """Return successive pairs from an iterable separated by `offset`."""
    from itertools import islice, tee

    a, b = tee(iterable)

    return zip(a, islice(b, offset, None))


def partitions(n, r):
    """Generate integer partitions of  `n` into `r` parts."""
    if r == 1:
        yield (n,)
        return

    for i in range(n + 1):
        for j in partitions(n - i, r - 1):
            yield i, *j


def shift_cipher(text, n):
    """Shift all letters `n` characters in text."""

    def _shift_letter(letter):
        if letter.isupper():
            first_ord = ord("A")
        elif letter.islower():
            first_ord = ord("a")
        else:
            return letter

        return chr(shiftmod(ord(letter) + n, 26, shift=first_ord))

    return "".join(map(_shift_letter, text))


def shiftmod(n, m, shift=1):
    """Simlar to n % m except the result lies within [shift, m + shift).

    Examples
    --------
    ```py
    shiftmod(10, 10, shift=1) == 10
    shiftmod(11, 10, shift=1) == 1
    shiftmod(11, 10, shift=2) == 11
    shiftmod(12, 10, shift=2) == 2
    ```

    """
    return (n - shift) % m + shift


def sliding_window(iterable, length=2):
    """Return a sliding window over an iterable."""
    from itertools import islice, tee

    its = (islice(it, i, None) for i, it in enumerate(tee(iterable, length)))

    return zip(*its)


def sliding_window_cycle(iterable, length=2):
    """Return a sliding window over an iterable that wraps around."""
    from itertools import chain, islice

    it = iter(iterable)
    start = tuple(islice(it, length - 1))
    return sliding_window(chain(start, it, start), length)


def spiral_grid():
    """Yield 2D coordinates spiraling around the origin."""
    x = y = 0
    d = m = 1
    while True:
        while 2 * x * d < m:
            yield x, y
            x = x + d

        while 2 * y * d < m:
            yield x, y
            y = y + d

        d *= -1
        m += 1


def split(sequence, n=2):
    """Split a sequence into `n` equal parts. `n` is assumed to divide the
    length of the sequence.
    """
    div = len(sequence) // n
    return (sequence[i * div : (i + 1) * div] for i in range(n))
