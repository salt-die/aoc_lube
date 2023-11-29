"""Helpful functions for AoC.

Imports are deferred so that this file is loads faster. Requires `networkx` and `numpy`.
"""
__all__ = [
    "GRID_NEIGHBORHOODS",
    "grid_steps",
    "extract_ints",
    "chunk",
    "extract_maze",
    "maximum_matching",
    "chinese_remainder_theorem",
    "pairwise",
    "sliding_window",
    "sliding_window_cycle",
    "oscillate_range",
    "int_grid",
    "dot_print",
    "shiftmod",
    "ilen",
    "nth",
    "partitions",
    "shift_cipher",
    "split",
    "distribute",
]

GRID_NEIGHBORHOODS = {
    4: [(0, 1), (0, -1), (1, 0), (-1, 0)],
    5: [(0, 0), (0, 1), (0, -1), (1, 0), (-1, 0)],
    8: [(0, 1), (0, -1), (1, 0), (-1, 0), (1, 1), (-1, -1), (1, -1), (-1, 1)],
    9: [(0, 0), (0, 1), (0, -1), (1, 0), (-1, 0), (1, 1), (-1, -1), (1, -1), (-1, 1)],
}


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


def chunk(it, n: int):
    """Chunk an iterable into non-overlapping fixed sized pieces."""
    args = [iter(it)] * n
    return zip(*args, strict=True)


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


def maximum_matching(items: dict[list]):
    """Return a maximum matching from a dict of lists."""
    import networkx as nx

    G = nx.from_dict_of_lists(items)

    for k, v in nx.bipartite.maximum_matching(G, top_nodes=items).items():
        if k in items:  # Filter edges pointing the wrong direction.
            yield k, v


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


def pairwise(iterable, offset=1):
    """Return successive pairs from an iterable separated by `offset`."""
    from itertools import islice, tee

    a, b = tee(iterable)

    return zip(a, islice(b, offset, None))


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


def dot_print(array):
    """Pretty print a binary or boolean array."""
    for row in array:
        print("".join(" #"[i] for i in row))


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


def ilen(iterable):
    """Return number of items in `iterable`.

    This will consume the iterable.
    """
    return sum(1 for _ in iterable)


def nth(iterable, n):
    """Return nth item of `iterable`."""
    from itertools import islice

    return next(islice(iterable, n, None))


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


def split(sequence, n=2):
    """Split a sequence into `n` equal parts. `n` is assumed to divide the
    length of the sequence.
    """
    div = len(sequence) // n
    return (sequence[i * div : (i + 1) * div] for i in range(n))


def distribute(iterable, n):
    """Distribute an iterable amoung `n` smaller iterables."""
    from itertools import islice, tee

    children = tee(iterable, n)
    return [islice(it, index, None, n) for index, it in enumerate(children)]
