"""Controlled cross-module importer for the sensor-fusion stack.

The Lidar/, Vision/, and Radar/ modules each expose a *top-level* ``src`` package
(plus a shared root ``config`` package). They therefore cannot all live on
``sys.path`` simultaneously — their ``src`` packages shadow one another. The
``fusion`` container needs to drive all three modalities from a single process,
so this helper imports each modality with an isolated, one-at-a-time ``sys.path``
state, purging the ``src`` import cache between switches.

This module lives at the repo root (not under any ``src`` package) so it survives
the cache purges it performs.

Example
-------
    from module_loader import use_module, load

    with use_module("Vision"):
        from src.detectors.detector_factory import get_object_detector
        from config.utils.path_manager import path_manager
        yolo = get_object_detector("yolo", model_path=..., device="cuda")

    pp = load("Lidar", "src.detectors.pointpillars")

For hard isolation (separate CUDA contexts, no shared global state) prefer running
each modality as its own subprocess/container and fusing the emitted detections;
this helper is the in-process path for orchestration and visualisation.
"""
from __future__ import annotations

import importlib
import os
import sys
from contextlib import contextmanager
from typing import Iterator

# Repo root = directory containing this file.
REPO_ROOT = os.path.dirname(os.path.abspath(__file__))

# Module roots that each ship a conflicting top-level ``src`` package.
MODULE_ROOTS = {
    "Lidar": os.path.join(REPO_ROOT, "Lidar"),
    "Vision": os.path.join(REPO_ROOT, "Vision"),
    "Radar": os.path.join(REPO_ROOT, "Radar"),
    "Fusion": os.path.join(REPO_ROOT, "Fusion"),
}

# Import-cache prefixes to drop when switching modules. ``config`` is shared and
# resolved as a namespace package (root config.utils + per-module config.*), so we
# also purge it to avoid a stale single-root binding leaking across switches.
_PURGE_PREFIXES = ("src", "config")


def _purge(prefixes: tuple[str, ...] = _PURGE_PREFIXES) -> None:
    for name in list(sys.modules):
        if any(name == p or name.startswith(p + ".") for p in prefixes):
            del sys.modules[name]


def _all_module_paths() -> set[str]:
    return set(MODULE_ROOTS.values())


@contextmanager
def use_module(module: str) -> Iterator[str]:
    """Make exactly one module root importable as ``src`` for the duration.

    Removes sibling module roots from ``sys.path``, purges the ``src``/``config``
    import cache, and prepends the requested root. Restores the previous
    ``sys.path`` and cache state on exit.

    Args:
        module: one of ``"Lidar"``, ``"Vision"``, ``"Radar"``, ``"Fusion"``.
    """
    if module not in MODULE_ROOTS:
        raise KeyError(f"Unknown module {module!r}. Known: {sorted(MODULE_ROOTS)}")

    root = MODULE_ROOTS[module]
    saved_path = list(sys.path)
    siblings = _all_module_paths()

    # Drop every module root, then prepend just the requested one.
    sys.path = [p for p in sys.path if p not in siblings]
    sys.path.insert(0, root)
    _purge()
    try:
        yield root
    finally:
        sys.path[:] = saved_path
        _purge()


def load(module: str, dotted: str):
    """Import ``dotted`` (e.g. ``"src.detectors.pointpillars"``) from ``module``.

    Returns the imported module object. Note the returned module's own imports are
    resolved at import time; do not interleave ``load()`` calls with deferred
    imports inside the returned module without re-entering ``use_module``.
    """
    with use_module(module):
        return importlib.import_module(dotted)
