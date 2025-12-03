"""Helpful functions for AoC."""

from __future__ import annotations

import re
from collections.abc import Iterable, Iterator, ValuesView
from itertools import chain, islice, tee, zip_longest
from math import log10, prod
from typing import Final, Literal, NamedTuple, Self, overload

import networkx as nx
import numpy as np
from numpy.typing import NDArray

__all__ = [
    "GRID_NEIGHBORHOODS",
    "UnionFind",
    "Vec2",
    "chinese_remainder_theorem",
    "chunk",
    "diff",
    "distribute",
    "dot_print",
    "first_unique",
    "grid_steps",
    "extract_ints",
    "extract_maze",
    "ilen",
    "int_grid",
    "maximum_matching",
    "ndigits",
    "nth",
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

    def __init__(self, iterable: Iterable[T] | None = None):
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

    def __getitem__(self, item: T) -> set[T]:
        if item not in self:
            raise KeyError(item)

        root = self.find(item)
        return self._components[root]

    def __iter__(self) -> Iterator[set[T]]:
        """Yield each disjoint set."""
        yield from self._components.values()

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

    def find(self, item: T) -> T:
        """Find the representive of the item's disjoint set."""
        if item not in self:
            raise KeyError(item)

        if self._parents[item] != item:
            self._parents[item] = self.find(self._parents[item])
        return self._parents[item]

    def merge(self, a: T, b: T) -> None:
        """Merge the set containing ``a`` and the set containing ``b``."""
        a_root = self.find(a)
        b_root = self.find(b)
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

    def __add__(self, other: int | tuple[int, int]) -> Vec2:  # type: ignore
        y1, x1 = self
        if isinstance(other, int):
            return Vec2(y1 + other, x1 + other)
        y2, x2 = other
        return Vec2(y1 + y2, x1 + x2)

    def __sub__(self, other: int | tuple[int, int]) -> Vec2:
        y1, x1 = self
        if isinstance(other, int):
            return Vec2(y1 - other, x1 - other)
        y2, x2 = other
        return Vec2(y1 - y2, x1 - x2)

    def __neg__(self) -> Vec2:
        y, x = self
        return Vec2(-y, -x)

    def __mul__(self, n: int) -> Vec2:  # type: ignore
        y, x = self
        return Vec2(n * y, n * x)

    def __floordiv__(self, n: int) -> Vec2:
        y, x = self
        return Vec2(y // n, x // n)

    def __abs__(self) -> int:
        """Length of vec with manhattan metric."""
        y, x = self
        return abs(y) + abs(x)

    def rotate(self, clockwise: bool = True) -> Vec2:
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
        cls, size: tuple[int, int], pos: tuple[int, int] = (0, 0)
    ) -> Iterator[Self]:
        """Generate all points in some rect."""
        h, w = size
        oy, ox = pos
        for y in range(h):
            for x in range(w):
                yield cls(y + oy, x + ox)

    def adj(self, neighborhood: Literal[4, 5, 8, 9] = 4) -> Iterator[Vec2]:
        """Yield adjacent points given by neighborhood."""
        return (pos + self for pos in GRID_NEIGHBORHOODS[neighborhood])


# fmt: off
GRID_NEIGHBORHOODS: Final = {
    4: (Vec2( 0,  1), Vec2( 0, -1), Vec2( 1,  0), Vec2(-1,  0)),
    5: (Vec2( 0,  1), Vec2( 0, -1), Vec2( 1,  0), Vec2(-1,  0),
        Vec2( 0,  0),
    ),
    8: (
        Vec2( 0,  1), Vec2( 0, -1), Vec2( 1,  0), Vec2(-1,  0),
        Vec2( 1,  1), Vec2(-1, -1), Vec2( 1, -1), Vec2(-1,  1),
    ),
    9: (
        Vec2( 0,  1), Vec2( 0, -1), Vec2( 1,  0), Vec2(-1,  0),
        Vec2( 1,  1), Vec2(-1, -1), Vec2( 1, -1), Vec2(-1,  1),
        Vec2( 0,  0),
    ),
}
# fmt: on


def chinese_remainder_theorem(moduli: list[int], residues: list[int]) -> int:
    """Find the solution to a system of modular equations."""
    N = prod(moduli)

    return (
        sum(
            (div := (N // modulus)) * pow(div, -1, modulus) * residue
            for modulus, residue in zip(moduli, residues)
        )
        % N
    )


@overload
def chunk[T](iterable: Iterable[T], n: Literal[2]) -> Iterator[tuple[T, T]]: ...


@overload
def chunk[T](iterable: Iterable[T], n: Literal[3]) -> Iterator[tuple[T, T, T]]: ...


@overload
def chunk[T](iterable: Iterable[T], n: Literal[4]) -> Iterator[tuple[T, T, T, T]]: ...


def chunk[T](iterable: Iterable[T], n: int) -> Iterator[tuple[T, ...]]:
    """Chunk an iterable into non-overlapping fixed sized pieces."""
    args = [iter(iterable)] * n
    return zip_longest(*args)


def diff(iterable: Iterable[int]) -> Iterator[int]:
    """Return the pairwise differences of ``iterable``.

    For instance, ``a, b, c, d`` is transformed into ``b - a, c - b, d - c``.
    """
    return (b - a for a, b in pairwise(iterable))


def distribute[T](iterable: Iterable[T], n: int) -> list[Iterator[T]]:
    """Distribute an iterable amoung ``n`` smaller iterables."""
    children = tee(iterable, n)
    return [islice(it, index, None, n) for index, it in enumerate(children)]


def dot_print(array) -> None:
    """Pretty print a binary or boolean array."""
    for row in array:
        print("".join(".#"[i] for i in row))


def first_unique[T](iterable: Iterable[T]) -> Iterator[T]:
    """Yield only unique items from ``iterable``.

    For instance, ``a, b, c, b, a, d, e, c`` is transformed into ``a, b, c, d, e``.
    """
    seen = set()
    for item in iterable:
        if item in seen:
            continue
        yield item
        seen.add(item)


def grid_steps(
    neighborhood: Literal[4, 5, 8, 9], height: int, width: int
) -> Iterator[tuple[Vec2, Vec2]]:
    """Yield all steps ((y, x), (y + dy, x + dx)) with dy, dx given by ``neighborhood``.

    ``neighborhood`` is one of 4, 5, 8 or 9 for Von-Neumann or Moore neighborhoods with
    or without center cell. The number is the number of cells in the neighborhood.
    """
    deltas = GRID_NEIGHBORHOODS[neighborhood]

    for y in range(height):
        for x in range(width):
            for dy, dx in deltas:
                new_y = y + dy
                new_x = x + dx

                if 0 <= new_y < height and 0 <= new_x < width:
                    yield Vec2(y, x), Vec2(new_y, new_x)


def extract_ints(raw: str) -> Iterator[int]:
    """Extract integers from a string."""
    return map(int, re.findall(r"(-?\d+)", raw))


def extract_maze(
    raw: str, wall="#", empty=".", largest_component=False
) -> tuple[NDArray[np.str_], nx.Graph, dict[str, Vec2]]:
    """Parse an ascii maze into a networkx graph."""
    lines = raw.splitlines()
    points = {}
    G = nx.Graph()
    for y, line in enumerate(lines):
        for x, char in enumerate(line):
            if char == wall:
                continue
            G.add_node(Vec2(y, x))
            if char != empty:
                points.setdefault(char, []).append(Vec2(y, x))
    G.add_edges_from((u, v) for u in G for v in u.adj() if v in G)

    max_width = max(map(len, lines))
    maze = np.array([list(line + " " * (max_width - len(line))) for line in lines])

    if largest_component:
        G.remove_nodes_from(
            G.nodes - max(nx.connected_components(G), key=lambda g: len(g))
        )

    return maze, G, points


def ilen(iterable: Iterable) -> int:
    """Return number of items in ``iterable``."""
    return sum(1 for _ in iterable)


def int_grid(raw: str, numpy=True, separator="") -> list[list[int]] | NDArray[np.int64]:
    """Parse a grid of ints into a 2d list or numpy array (if np==True)."""
    array = [
        [int(i) for i in (line.split(separator) if separator else line) if i]
        for line in raw.splitlines()
    ]

    if numpy:
        return np.array(array)

    return array


def maximum_matching[T](items: dict[T, list[T]]) -> Iterator[tuple[T, T]]:
    """Return a maximum matching from a dict of lists."""
    G = nx.from_dict_of_lists(items)  # type: ignore

    for k, v in nx.bipartite.hopcroft_karp_matching(G, top_nodes=items).items():
        if k in items:  # Filter edges pointing the wrong direction.
            yield k, v


def ndigits(n: int) -> int:
    """Return the number of digits in ``n``."""
    return int(log10(n)) + 1


def nth[T](iterable: Iterable[T], n: int) -> T:
    """Return nth item of ``iterable``."""
    return next(islice(iterable, n, None))


def pairwise[T](iterable: Iterable[T], offset: int = 1) -> Iterator[tuple[T, T]]:
    """Return successive pairs from an iterable separated by `offset`."""
    a, b = tee(iterable)

    return zip(a, islice(b, offset, None))


def partitions(n: int, r: int) -> Iterator[tuple[int, ...]]:
    """Generate integer partitions of  ``n`` into ``r`` parts."""
    if r == 1:
        yield (n,)
        return

    for i in range(n + 1):
        for j in partitions(n - i, r - 1):
            yield i, *j


def shift_cipher(text: str, n: int) -> str:
    """Shift all letters ``n`` characters in text."""

    def _shift_letter(letter):
        if letter.isupper():
            first_ord = ord("A")
        elif letter.islower():
            first_ord = ord("a")
        else:
            return letter

        return chr(shiftmod(ord(letter) + n, 26, shift=first_ord))

    return "".join(map(_shift_letter, text))


def shiftmod(n: int, m: int, shift: int = 1) -> int:
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


def sliding_window[T](
    iterable: Iterable[T], length: int = 2
) -> Iterator[tuple[T, ...]]:
    """Return a sliding window over an iterable."""
    its = (islice(it, i, None) for i, it in enumerate(tee(iterable, length)))

    return zip(*its)


def sliding_window_cycle[T](
    iterable: Iterable[T], length: int = 2
) -> Iterator[tuple[T, ...]]:
    """Return a sliding window over an iterable that wraps around."""
    it = iter(iterable)
    start = tuple(islice(it, length - 1))
    return sliding_window(chain(start, it, start), length)


def spiral_grid() -> Iterator[Vec2]:
    """Yield 2D coordinates spiraling around the origin."""
    x = y = 0
    d = m = 1
    while True:
        while 2 * x * d < m:
            yield Vec2(x, y)
            x += d

        while 2 * y * d < m:
            yield Vec2(x, y)
            y += d

        d *= -1
        m += 1


def split[T](sequence: list[T], n: int = 2) -> Iterator[list[T]]:
    """Split a sequence into ``n`` equal parts.

    ``n`` is assumed to divide the length of the sequence.
    """
    div = len(sequence) // n
    return (sequence[i * div : (i + 1) * div] for i in range(n))
