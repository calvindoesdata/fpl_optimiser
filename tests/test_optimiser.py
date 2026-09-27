"""
End-to-end tests of the optimiser itself. These require cvxpy plus at
least one installed MIP-capable backend; they are skipped automatically
if neither is available so the rest of the suite still runs cleanly in
minimal environments.

Run with: pytest tests/test_optimiser.py
"""

import os
import sys

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from fpl_optimiser.data_loader import load_from_csv
from fpl_optimiser.models import (
    SQUAD_SIZE,
    STARTING_XI_SIZE,
    SQUAD_REQUIREMENTS,
    FORMATION_LIMITS,
    MAX_PER_TEAM,
)

SAMPLE_CSV = os.path.join(
    os.path.dirname(__file__), "..", "sample_data", "players_sample.csv"
)

cvxpy = pytest.importorskip("cvxpy", reason="cvxpy not installed")

from fpl_optimiser.optimiser import FPLOptimiser, OptimiserConfig  # noqa: E402

_HAS_MIP_SOLVER = bool(
    {"CBC", "GLPK_MI", "SCIP", "GUROBI", "CPLEX", "XPRESS", "ECOS_BB", "HIGHS"}
    & set(cvxpy.installed_solvers())
)
pytestmark = pytest.mark.skipif(
    not _HAS_MIP_SOLVER, reason="No MIP-capable cvxpy solver installed"
)


@pytest.fixture
def players_df():
    return load_from_csv(SAMPLE_CSV)


def test_squad_satisfies_all_hard_constraints(players_df):
    config = OptimiserConfig(budget=100.0)
    result = FPLOptimiser(players_df, config).solve()

    assert result.status in {"optimal", "optimal_inaccurate"}
    assert len(result.squad) == SQUAD_SIZE
    assert len(result.starting_xi) == STARTING_XI_SIZE
    assert result.total_cost <= config.budget + 1e-6

    for pos, quota in SQUAD_REQUIREMENTS.items():
        assert (result.squad["position"] == pos).sum() == quota

    for pos, (lo, hi) in FORMATION_LIMITS.items():
        count = (result.starting_xi["position"] == pos).sum()
        assert lo <= count <= hi

    team_counts = result.squad["team"].value_counts()
    assert (team_counts <= MAX_PER_TEAM).all()

    assert result.captain["web_name"] in set(result.starting_xi["web_name"])
    assert result.vice_captain["web_name"] != result.captain["web_name"]


def test_must_include_player_is_selected(players_df):
    target_id = int(players_df.iloc[0]["id"])
    config = OptimiserConfig(budget=100.0, must_include=[target_id])
    result = FPLOptimiser(players_df, config).solve()
    assert target_id in set(result.squad["id"])


def test_must_exclude_player_is_absent(players_df):
    target_id = int(players_df.iloc[0]["id"])
    config = OptimiserConfig(budget=100.0, must_exclude=[target_id])
    result = FPLOptimiser(players_df, config).solve()
    assert target_id not in set(result.squad["id"])


def test_lower_budget_lowers_or_maintains_objective(players_df):
    high = FPLOptimiser(players_df, OptimiserConfig(budget=100.0)).solve()
    low = FPLOptimiser(players_df, OptimiserConfig(budget=80.0)).solve()
    assert low.starting_metric_total <= high.starting_metric_total + 1e-6
