"""Setup directory from command line with
`python -m aoc_lube [-y YEAR] [-t TEMPLATE]`.
"""
import argparse
from pathlib import Path

from . import setup_dir

parser = argparse.ArgumentParser(
    prog="python -m aoc_lube", description="Setup a directory for Advent of Code."
)
parser.add_argument("-y", "--year", help="Advent of Code year.", default=None)
parser.add_argument("-t", "--template", help="A code template file.", default=None)
args = parser.parse_args()

setup_dir(args.year, args.template if args.template is None else Path(args.template))
