from __future__ import annotations

import os

ASSETS_DIR = os.path.dirname(os.path.abspath(__file__))
"""Directory holding the package's own USD assets."""

ICRA_MAP_FLAT_USD = os.path.join(ASSETS_DIR, "icra_map_flat.usd")
"""Flat ICRA evaluation map, swapped in by the ICRA eval env configs."""
