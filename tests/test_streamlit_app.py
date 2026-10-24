"""Static boundary tests for app/streamlit_app.py.

Streamlit is an app/deploy dependency, not a test dependency, so it is not
installed in the test environment. These tests therefore inspect the module's
source with the ``ast`` module rather than importing it:

- the UI module imports only ``streamlit`` and ``app.logic`` (no ``src/`` math,
  no ``pygame``, no ``render``), honoring the UI -> logic -> core boundary;
- it forces the matplotlib Agg backend before any ``pyplot`` import.
"""

from __future__ import annotations

import ast
from pathlib import Path

APP = Path(__file__).resolve().parent.parent / "app" / "streamlit_app.py"


def _imported_modules(source: str) -> list[str]:
    """Return every top-level module name imported by ``source``."""
    tree = ast.parse(source)
    names: list[str] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            names.extend(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom):
            if node.module is not None and node.level == 0:
                names.append(node.module)
                # Also record `from pkg import name` as `pkg.name`, so
                # `from app import logic` is seen as `app.logic`.
                for alias in node.names:
                    names.append(f"{node.module}.{alias.name}")
    return names


def test_app_source_exists() -> None:
    """The Streamlit UI module is present."""
    assert APP.is_file()


def test_app_imports_only_streamlit_logic_and_allowed_helpers() -> None:
    """No src/ math, no pygame, no render in the UI module's imports.

    Allowed: streamlit, app.logic, matplotlib (for the Agg backend), pandas,
    and standard-library modules. The simulation math must be reached only
    through app.logic.
    """
    modules = _imported_modules(APP.read_text())

    forbidden_prefixes = ("src.", "pygame", "render")
    offenders = [
        m
        for m in modules
        if m.startswith(forbidden_prefixes) or m in {"src", "render", "pygame"}
    ]
    assert offenders == [], f"forbidden imports in streamlit_app.py: {offenders}"

    # app.logic must be the computation entry point.
    assert "app.logic" in modules


def test_app_forces_agg_before_pyplot() -> None:
    """matplotlib.use('Agg') must appear before any pyplot import.

    Stage 2 imports no pyplot, so if pyplot is absent the requirement is
    trivially satisfied; if present, the Agg call must come first.
    """
    source = APP.read_text()
    assert 'matplotlib.use("Agg")' in source or "matplotlib.use('Agg')" in source

    agg_pos = source.find("matplotlib.use")
    pyplot_pos = source.find("matplotlib.pyplot")
    if pyplot_pos != -1:
        assert agg_pos != -1 and agg_pos < pyplot_pos
