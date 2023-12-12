"""Setup directory from command line with `python -m aoc_lube [YYYY]`."""
import datetime
import sys

from . import setup_dir

try:
    _, year, *rest = sys.argv
except ValueError:
    setup_dir(datetime.date.today().year)
else:
    if not year.isdigit():
        print(f"What is year {year}?")
    elif rest:
        print(f"Extra argument(s) not recognized: {rest}")
    else:
        setup_dir(int(year))
