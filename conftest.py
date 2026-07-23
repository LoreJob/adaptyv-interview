"""Pytest bootstrap: put the repo root on sys.path so `import src...` resolves.

Present at the repo root, pytest imports this before collecting tests, making
the `src` package importable regardless of the working directory or whether the
run is launched from the CLI or the IDE's test runner.
"""

import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
