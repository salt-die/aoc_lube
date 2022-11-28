import re
import time
import webbrowser
from datetime import datetime
from pathlib import Path
from typing import Callable, Literal

import bs4
import requests
import tomlkit

from .constants import *
from . import utils

__version__ = "0.2.0"

__all__ = (
    "setup_dir",
    "fetch",
    "submit",
    "utils",
)

try:
    TOKEN = {"session": TOKEN_FILE.read_text().strip()}
except FileNotFoundError:
    print(
        "\n\x1b[1m\x1b[31m::WARNING::\x1b[0m\n\n"
        f"Token not found at \x1b[34m{TOKEN_FILE.absolute()}\x1b[0m.\n"
        "`fetch` and `submit` functions will fail without a user token.\n"
        "See \x1b[33mREADME\x1b[0m for instructions on how to get your user token.\n"
    )

def setup_dir(year: int):
    """
    Run once to setup directory with templates for daily solutions.
    """
    template = TEMPLATE_FILE.read_text()

    for day in range(1, 26):
        file = Path(f"day_{day:02}.py")

        if not file.exists():
            file.write_text(template.format(year=year, day=day))

def _ensure(path):
    if not path.parent.exists():
        path.parent.mkdir()

    if not path.exists():
        path.touch()

def fetch(year: int, day: int) -> str:
    """
    Fetch puzzle input. Inputs are cached.
    """
    input_file = CONFIG_DIR / f"{year}" / INPUTS_FILE
    _ensure(input_file)
    inputs = tomlkit.loads(input_file.read_text())

    if str(day) not in inputs:
        _wait_for_unlock(year, day)

        response = requests.get(url=URL.format(year=year, day=day) + "/input", cookies=TOKEN)
        if not response.ok:
            raise ValueError("Request failed.")

        # Save input data
        inputs[str(day)] = response.text.strip()
        input_file.write_text(tomlkit.dumps(inputs))

    return inputs[str(day)]

def submit(year: int, day: int, part: Literal[1, 2], solution: Callable, sanity_check=True):
    """
    Submit a solution. Submissions are cached.
    """
    submissions_file = CONFIG_DIR / f"{year}" / SUBMISSIONS_FILE
    _ensure(submissions_file)
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
        print(f"Solution {answer} to part {part} has already been submitted, response was:")
        _pretty_print(current[answer])
        return

    if (
        sanity_check
        and input(f"Submit {answer}? [y]/n\n").startswith(("n", "N"))
    ):
        return

    while True:
        print(f"Submitting {answer} as solution to part {part}:")
        response = requests.post(
            url=URL.format(year=year, day=day) + "/answer",
            cookies=TOKEN,
            data={"level": part, "answer": answer}
        )

        if not response.ok:
            raise ValueError("Request failed.")

        message = bs4.BeautifulSoup(response.text, "html.parser").article.text
        _pretty_print(message)

        if message[4] == "g":  # "You gave an answer too recently"
            minutes, seconds = re.search(r"(?:(\d+)m )?(\d+)s", message).groups()

            timeout = 60 * int(minutes or 0) + int(seconds)
            try:
                print("\x1b[?25l")  # Hide cursor.
                while timeout > 0:
                    bold_yellow_delay = f"\x1b[1m\x1b[33m{timeout}\x1b[0m"
                    print(f"Waiting {bold_yellow_delay} seconds to retry...".ljust(50), end="\r")
                    timeout -= 1
                    time.sleep(1)
            finally:
                print("\x1b[?25h")  # Show cursor.
        else:
            break

    if message[7] == "t":  # "That's the right answer! ..."
        current["solution"] = answer

        if part == 1:
            webbrowser.open(response.url)  # View part 2 in browser

    current[answer] = message
    submissions_file.write_text(tomlkit.dumps(submissions))

def _wait_for_unlock(year, day):
    now = datetime.now().astimezone()
    unlock = datetime(year=year, day=day, **UNLOCK_TIME_INFO)

    if now < unlock:
        try:
            print("\x1b[?25l")  # Hide cursor.

            while True:
                now = datetime.now().astimezone()

                if (delay := (unlock - now).total_seconds()) <= 0:
                    break

                bold_yellow_delay = f"\x1b[1m\x1b[33m{delay:.2f}\x1b[0m"

                print(
                    f"Waiting {bold_yellow_delay} seconds for puzzle input to unlock...".ljust(50),
                    end="\r",
                )

                time.sleep(.1)
        finally:
            print("\x1b[?25h")  # Show cursor.

def _pretty_print(message):
    match message[7]:
        case "t":
            # "That's the right answer! ..."
            COLOR = "\x1b[32m"  # Green
        case "'" | "e":
            # "You don't seem to be solving the right level. ..."
            # "You gave an answer too recently; you have to wait ..."
            COLOR = "\x1b[33m"  # Yellow
        case "n":
            # "That's not the right answer. If you're stuck, ..."
            COLOR = "\x1b[31m"  # Red
        case _:
            raise ValueError("Unexpected message.", message)
    print(
        "\x1b[1m",  # Bold
        COLOR,
        message,
        "\x1b[0m",  # Reset
        sep="",
    )
