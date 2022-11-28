# aoc_helper - Advent of Code fetch/submit with various utilities.

Setting up aoc_helper
---------------------
* Save your user token to `~/.aoc_helper/.token`.
* [Optional] Create a new directory for your solutions. In this directory, run `aoc_helper.setup_dir`.

Getting your user token
-----------------------
* Navigate to https://adventofcode.com/. Make sure you are logged in.
* Right-click the page and click `Inspect`.
* Click on the `Network` tab.
* Reload the page.
* The token will be the session cookie in the request header to the page. It's a long hex string.

![User Token](token.png)
