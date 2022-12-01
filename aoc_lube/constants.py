from datetime import timezone, timedelta
from pathlib import Path

__all__ = (
    "__version__",
    "CONFIG_DIR",
    "HEADERS",
    "INPUTS_FILE",
    "SUBMISSIONS_FILE",
    "TEMPLATE_FILE",
    "TOKEN_FILE",
    "UNLOCK_TIME_INFO",
    "URL",
)

__version__ = "0.2.2"

CONFIG_DIR = Path.home() / ".aoc_lube"
if not CONFIG_DIR.exists():
    CONFIG_DIR.mkdir()

HEADERS = {"User-Agent": f"github.com/salt-die/aoc_lube v{__version__} by salt-die@protonmail.com"}
INPUTS_FILE = "inputs.toml"
SUBMISSIONS_FILE = "submissions.toml"
TEMPLATE_FILE = Path(__file__).parent / "code_template.txt"
TOKEN_FILE =  CONFIG_DIR / ".token"
# AoC puzzle inputs unlock at midnight -5 UTC during month of December.
UNLOCK_TIME_INFO = dict(
    month=12,
    hour=0,
    minute=0,
    second=5,
    microsecond=0,
    tzinfo=timezone(timedelta(hours=-5), 'Eastern US'),
)
URL = "https://adventofcode.com/{year}/day/{day}"
