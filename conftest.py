"""Pytest configuration.

Living at the project root, this file puts the root on ``sys.path`` so tests
can ``import src.scene`` without an installed package.
"""

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
