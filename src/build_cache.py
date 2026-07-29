"""Precompute the heavy read-only endpoints into a JSON cache.

metrics + backtest train sklearn models; doing that per-request spikes memory
and latency on small hosts. Running this once at build time writes the results
to ``data/processed/site_cache.json`` so the API can serve them instantly.

Run: python -m src.build_cache
"""

from __future__ import annotations

import json
from pathlib import Path

if __package__ in (None, ""):  # allow `python src/build_cache.py`
    import sys
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

# Import the endpoint compute functions. The cache file does not exist yet on a
# clean build, so these compute live (rather than short-circuiting on cache).
from src import api

OUT_PATH = Path(__file__).resolve().parent.parent / "data" / "processed" / "site_cache.json"


def main() -> None:
    data = {
        "stats": api._stats(),
        "example_binder": api._example_binder(),
        "metrics": api._metrics(),
        "backtest": api._backtest(),
    }
    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    OUT_PATH.write_text(json.dumps(data, indent=2))
    print(f"wrote {OUT_PATH} ({OUT_PATH.stat().st_size} bytes)")


if __name__ == "__main__":
    main()
