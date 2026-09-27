"""
fpl_optimiser
=============

A small, self-contained toolkit for building an optimal Fantasy Premier
League (FPL) squad using integer optimisation (via cvxpy).

Public API:
    - Player            : data model for a single player
    - load_from_csv      : load player data from a local CSV
    - fetch_fpl_data     : pull live data from the official FPL API
    - FPLOptimiser       : the optimisation engine
    - OptimiserConfig    : configuration / constraints for a run
    - SquadResult        : structured result of an optimisation run
"""

from .models import Player
from .data_loader import load_from_csv, fetch_fpl_data, transform_kaggle_data

__all__ = [
    "Player",
    "load_from_csv",
    "fetch_fpl_data",
    "FPLOptimiser",
    "OptimiserConfig",
    "SquadResult",
    "render_result",
]


def __getattr__(name):
    # Lazily import cvxpy-dependent pieces so that data loading / the
    # Player model can be used (and tested) in environments where cvxpy
    # or a MIP solver backend isn't installed. The ImportError with a
    # helpful message only surfaces once you actually try to optimise.
    if name in {"FPLOptimiser", "OptimiserConfig", "SquadResult"}:
        from . import optimiser as _optimiser

        return getattr(_optimiser, name)
    if name == "render_result":
        from .reporting import render_result as _render_result

        return _render_result
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")

__version__ = "1.0.0"
