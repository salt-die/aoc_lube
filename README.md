# aoc_lube - Proper *Advent of Code* lubricant.

Setting up aoc_lube
---------------------
* `pip install aoc-lube` for just `fetch` and `submit` functions or `pip install aoc-lube[utils]` for optional dependencies (networkx, numpy) for utility functions.
* Save your user token to `~/.aoc_lube/.token`.
* [Optional] Create a new directory for your solutions. In this directory, run the function `aoc_lube.setup_dir()` or from the command line, `python -m aoc_lube [-y YEAR] [-t TEMPLATE]`. (If no year is given, the current year will be used.)

Getting your user token
-----------------------
* Navigate to https://adventofcode.com/. Make sure you are logged in.
* Right-click the page and click `Inspect`.
* Click on the `Network` tab.
* Reload the page.
* The token will be the session cookie in the request header to the page. It's a long hex string.

![User Token](token.png)
