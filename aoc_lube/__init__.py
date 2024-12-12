"""Propert Advent of Code lubricant."""

import re
import subprocess
import time
import webbrowser
from datetime import date, datetime, timedelta, timezone
from pathlib import Path
from typing import Callable, Literal

import bs4
import requests
import tomlkit

from . import utils

__all__ = ["fetch", "setup_dir", "submit", "utils"]

__version__ = "1.3.0"

CONFIG_DIR = Path.home() / ".aoc_lube"
if not CONFIG_DIR.exists():
    CONFIG_DIR.mkdir()

HEADERS = {
    "User-Agent": (
        f"github.com/salt-die/aoc_lube v{__version__} by salt-die@protonmail.com"
    ),
}
SUBMISSIONS_FILE = "submissions.toml"
TEMPLATE_FILE = Path(__file__).parent / "code_template.txt"
TOKEN_FILE = CONFIG_DIR / ".token"
# AoC puzzle inputs unlock at midnight -5 UTC during month of December.
UNLOCK_TIME_INFO = dict(
    month=12,
    hour=0,
    minute=0,
    second=5,
    microsecond=0,
    tzinfo=timezone(timedelta(hours=-5), "Eastern US"),
)
URL = "https://adventofcode.com/{year}/day/{day}"

# Ansi Escapes
RED = "\x1b[31m"
GREEN = "\x1b[32m"
YELLOW = "\x1b[33m"
BLUE = "\x1b[34m"
BOLD = "\x1b[1m"
RESET = "\x1b[0m"
HIDE_CURSOR = "\x1b[?25l"
SHOW_CURSOR = "\x1b[?25h"

try:
    TOKEN = {"session": TOKEN_FILE.read_text().strip()}
except FileNotFoundError:
    print(
        f"\n{BOLD}{RED}::WARNING::{RESET}\n\n"
        f"Token not found at {BLUE}{TOKEN_FILE.absolute()}{RESET}.\n"
        "`fetch` and `submit` functions will fail without a user token.\n"
        f"See {YELLOW}README{RESET} for instructions on how to get your user token.\n"
    )


def setup_dir(year: int | None = None, template: Path | None = None) -> None:
    r"""Run once to setup directory with templates for daily solutions.

    A file can be provided as a template. The file's text should be able to be formatted
    with `year` and `day` keyword arguments. See `code_template.txt` (the default
    template) as an example.
    """
    if year is None:
        year = date.today().year

    if template is None:
        template = TEMPLATE_FILE.read_text()
    else:
        template = template.read_text()

    for day in range(1, 26):
        file = Path(f"day_{day:02}.py")
        if not file.exists():
            file.write_text(template.format(year=year, day=day))


def _fix_old_inputs(year: int) -> bool:
    input_file = CONFIG_DIR / f"{year}" / "inputs.toml"

    if input_file.exists():
        inputs = tomlkit.loads(input_file.read_text())
        for day, input_ in inputs.items():
            (input_file.parent / f"{int(day):02}.txt").write_text(input_)

        input_file.unlink()


def fetch(year: int, day: int, open_with: Literal["vscode"] | None = None) -> str:
    """Fetch puzzle input.

    Inputs are cached so that an input won't be re-fetched. The `open_with` parameter
    can be used to open the fetched input as a separate file in your IDE/text editor.
    """
    _fix_old_inputs(year)

    year_dir = CONFIG_DIR / f"{year}"
    if not year_dir.exists():
        year_dir.mkdir()
    input_file = year_dir / f"{day:02}.txt"

    if input_file.exists():
        input_ = input_file.read_text()
    else:
        _wait_for_unlock(year, day)

        response = requests.get(
            url=URL.format(year=year, day=day) + "/input",
            headers=HEADERS,
            cookies=TOKEN,
        )
        if not response.ok:
            raise ValueError("Request failed.")

        input_ = response.text.rstrip()
        input_file.write_text(input_)

    if open_with is not None:
        # Open an issue to add a command to open the input file in your favorite editor.
        editors = {"vscode": ["code", "-r"]}
        subprocess.run([*editors[open_with], input_file], shell=True, check=False)

    return input_


def submit(
    year: int,
    day: int,
    part: Literal[1, 2],
    solution: Callable,
    sanity_check=True,
) -> None:
    """Submit a solution.

    Submissions are cached so that the same solution won't be resubmitted.
    """
    submissions_file = CONFIG_DIR / f"{year}" / SUBMISSIONS_FILE
    if not submissions_file.parent.exists():
        submissions_file.parent.mkdir()
    if not submissions_file.exists():
        submissions_file.touch()
    submissions = tomlkit.loads(submissions_file.read_text())
    current = submissions.setdefault(str(day), {}).setdefault(str(part), {})

    if "solution" in current:
        print(
            f"Day {day}, part {part} has already been solved. "
            f"The solution was:\n{current['solution']}"
        )
        return

    if (answer := solution()) is None:
        print("Solution produced no answer. Did you forget a return?")
        return

    answer = str(answer)

    if answer in current:
        print(
            f"Solution {answer} to part {part} has already been submitted,"
            " response was:",
        )
        _pretty_print(current[answer])
        return

    if sanity_check and input(f"Submit {answer}? [y]/n\n").startswith(("n", "N")):
        return

    while True:
        print(f"Submitting {answer} as solution to part {part}:")
        response = requests.post(
            url=URL.format(year=year, day=day) + "/answer",
            data={"level": part, "answer": answer},
            headers=HEADERS,
            cookies=TOKEN,
        )

        if not response.ok:
            raise ValueError("Request failed.")

        message = bs4.BeautifulSoup(response.text, "html.parser").article.text
        _pretty_print(message)

        if message[4] == "g":  # "You gave an answer too recently"
            minutes, seconds = re.search(r"(?:(\d+)m )?(\d+)s", message).groups()

            timeout = 60 * int(minutes or 0) + int(seconds)
            try:
                print(HIDE_CURSOR)
                while timeout > 0:
                    print(
                        f"Waiting {BOLD}{YELLOW}{timeout}{RESET} seconds to "
                        "retry...".ljust(50),
                        end="\r",
                    )
                    timeout -= 1
                    time.sleep(1)
            finally:
                print(SHOW_CURSOR)
        else:
            break

    if message[7] == "t":  # "That's the right answer! ..."
        current["solution"] = answer

        if day == 25:  # Automatically submit part 2 on day 25.
            response = requests.post(
                url=URL.format(year=year, day=day) + "/answer",
                data={"level": 2, "answer": ""},
                headers=HEADERS,
                cookies=TOKEN,
            )

        if part == 1:
            webbrowser.open(response.url)  # View part 2 in browser

    current[answer] = message
    submissions_file.write_text(tomlkit.dumps(submissions))


def _wait_for_unlock(year, day):
    """Display a countdown in the terminal to when puzzle input is available."""
    now = datetime.now().astimezone()
    unlock = datetime(year=year, day=day, **UNLOCK_TIME_INFO)

    if now < unlock:
        try:
            print(HIDE_CURSOR)
            while True:
                now = datetime.now().astimezone()
                if (delay := (unlock - now).total_seconds()) <= 0:
                    break
                print(
                    f"Waiting {BOLD}{YELLOW}{delay:.2f}{RESET} seconds for puzzle input"
                    " to unlock...".ljust(50),
                    end="\r",
                )
                time.sleep(0.1)
        finally:
            print(SHOW_CURSOR)


def _pretty_print(message):
    """Color submission response."""
    match message[7]:
        case "t":
            # "That's the right answer! ..."
            color = GREEN
        case "'" | "e":
            # "You don't seem to be solving the right level. ..."
            # "You gave an answer too recently; you have to wait ..."
            color = YELLOW
        case "n":
            # "That's not the right answer. If you're stuck, ..."
            color = RED
        case _:
            raise ValueError("Unexpected message.", message)
    print(f"{BOLD}{color}{message}{RESET}")
