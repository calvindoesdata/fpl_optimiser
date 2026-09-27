"""
The optimisation engine.

This models the FPL squad-selection problem as a Mixed-Integer Program
(MIP) and solves it with ``cvxpy``. Three sets of binary decision
variables are used:

    squad[i]    -- 1 if player i is one of the 15 squad members
    starting[i] -- 1 if player i is in the starting XI (implies squad[i]=1)
    captain[i]  -- 1 if player i is the captain (implies starting[i]=1)

Objective
---------
Maximise the total of the chosen metric (default: ``form``) across the
starting XI, with the captain's contribution counted twice (the extra
copy is added via the ``captain`` variable), which mirrors the real
scoring rule that the captain's points are doubled.

Constraints
-----------
    * Squad size == 15, with exact position quotas (2 GK / 5 DEF / 5 MID / 3 FWD)
    * Total spend <= budget
    * At most `max_per_team` players from any one real club
    * Starting XI size == 11, drawn only from the squad, respecting valid
      formation limits (1 GK; 3-5 DEF; 2-5 MID; 1-3 FWD)
    * Exactly one captain, chosen from the starting XI
    * Optional: forced-include / forced-exclude player lists
    * Optional: unavailable players (injured/suspended/ruled out) excluded
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Iterable, Optional

import numpy as np
import pandas as pd

try:
    import cvxpy as cp
except ImportError as exc:  # pragma: no cover
    raise ImportError(
        "cvxpy is required for FPLOptimiser. Install it with `pip install cvxpy` "
        "(and a MIP-capable backend such as `pip install cvxpy[CBC]` or "
        "`pip install pulp`/`scip`)."
    ) from exc

from .models import (
    SQUAD_REQUIREMENTS,
    FORMATION_LIMITS,
    MAX_PER_TEAM,
    DEFAULT_BUDGET,
    SQUAD_SIZE,
    STARTING_XI_SIZE,
)

# Preference order for MIP-capable solvers that cvxpy might have available.
# HiGHS ships bundled with cvxpy >= 1.6 (via the `highspy` dependency), so
# it works out of the box with no extra install and is tried first. If the
# user happens to have a licensed commercial solver installed it is used
# next (faster on very large problems), then other free backends that
# require a separate install (CBC via `cylp`, GLPK_MI via `cvxopt`, SCIP
# via `pyscipopt`), with the older/deprecated ECOS_BB as a last resort.
_SOLVER_PREFERENCE = [
    # "HIGHS",
    # "GUROBI",
    # "CPLEX",
    # "XPRESS",
    "SCIP",
    # "CBC",
    # "GLPK_MI",
    # "ECOS_BB",
]


@dataclass
class OptimiserConfig:
    """All tunable constraints/settings for one optimisation run."""

    budget: float = DEFAULT_BUDGET
    squad_requirements: dict = field(default_factory=lambda: dict(SQUAD_REQUIREMENTS))
    formation_limits: dict = field(default_factory=lambda: dict(FORMATION_LIMITS))
    max_per_team: int = MAX_PER_TEAM
    squad_size: int = SQUAD_SIZE  # derived default (15); override only if you also
    starting_xi_size: int = STARTING_XI_SIZE  # change squad_requirements/formation_limits
    metric: str = "form"  # column in the player data to maximise
    captain_doubles: bool = True  # captain's metric counted twice, as in real FPL
    exclude_unavailable: bool = True  # drop injured/suspended/0%-chance players
    must_include: Iterable[int] = field(default_factory=list)  # player ids
    must_exclude: Iterable[int] = field(default_factory=list)  # player ids
    solver: Optional[str] = None  # force a specific cvxpy solver name
    solver_verbose: bool = False


@dataclass
class SquadResult:
    """Structured output of a solved optimisation run."""

    status: str
    squad: pd.DataFrame
    starting_xi: pd.DataFrame
    bench: pd.DataFrame
    captain: pd.Series
    vice_captain: pd.Series
    formation: str
    total_cost: float
    budget: float
    squad_metric_total: float
    starting_metric_total: float
    metric: str
    solver_used: str

    def summary(self) -> str:
        lines = [
            f"Status: {self.status}  (solver: {self.solver_used})",
            f"Budget used: £{self.total_cost:.1f}m / £{self.budget:.1f}m",
            f"Formation: {self.formation}",
            f"Captain: {self.captain['player_name']} ({self.captain['club_name']})",
            f"Vice-captain: {self.vice_captain['player_name']} ({self.vice_captain['club_name']})",
            f"Projected starting-XI {self.metric} (captain doubled): "
            f"{self.starting_metric_total:.1f}",
        ]
        return "\n".join(lines)


class FPLOptimiser:
    """Builds and solves the FPL squad-selection MIP for a given player pool."""

    def __init__(self, players: pd.DataFrame, config: Optional[OptimiserConfig] = None):
        self.config = config or OptimiserConfig()
        self.players_raw = players.reset_index(drop=True)
        self.players = self._prepare(self.players_raw)

    # ------------------------------------------------------------------
    # Data preparation
    # ------------------------------------------------------------------
    def _prepare(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()

        if self.config.exclude_unavailable and "status" in df.columns:
            available_mask = ~df["status"].isin(["i", "s", "u"])
            if "chance_of_playing_next_round" in df.columns:
                zero_chance = df["chance_of_playing_next_round"] == 0
                available_mask &= ~zero_chance.fillna(False)
            df = df[available_mask]

        if self.config.must_exclude:
            df = df[~df["id"].isin(list(self.config.must_exclude))]

        if self.config.metric not in df.columns:
            raise ValueError(
                f"metric '{self.config.metric}' not found in player data columns: "
                f"{list(df.columns)}"
            )

        df = df.reset_index(drop=True)

        required = self.config.squad_size
        if len(df) < required:
            raise ValueError(
                f"Only {len(df)} eligible players remain after filtering, but a "
                f"squad needs {required}. Loosen exclusions/availability filters."
            )
        return df

    # ------------------------------------------------------------------
    # Solve
    # ------------------------------------------------------------------
    def solve(self) -> SquadResult:
        df = self.players
        n = len(df)
        cfg = self.config

        metric = df[cfg.metric].to_numpy(dtype=float)
        price = df["now_cost"].to_numpy(dtype=float)
        position = df["position_name"].to_numpy()
        team = df["club_name"].to_numpy()

        squad = cp.Variable(n, boolean=True)
        starting = cp.Variable(n, boolean=True)
        captain = cp.Variable(n, boolean=True)

        constraints = []

        # -- Squad-level constraints -------------------------------------------------
        constraints.append(cp.sum(squad) == cfg.squad_size)
        constraints.append(price @ squad <= cfg.budget)

        for pos, quota in cfg.squad_requirements.items():
            mask = (position == pos).astype(float)
            constraints.append(mask @ squad == quota)

        for club in np.unique(team):
            mask = (team == club).astype(float)
            constraints.append(mask @ squad <= cfg.max_per_team)

        # -- Starting XI constraints (subset of the squad) ---------------------------
        constraints.append(starting <= squad)
        constraints.append(cp.sum(starting) == cfg.starting_xi_size)

        for pos, (lo, hi) in cfg.formation_limits.items():
            mask = (position == pos).astype(float)
            constraints.append(mask @ starting >= lo)
            constraints.append(mask @ starting <= hi)

        # -- Captaincy: exactly one captain, and only from the starting XI -----------
        constraints.append(captain <= starting)
        constraints.append(cp.sum(captain) == 1)

        # -- Forced selections ---------------------------------------------------------
        if cfg.must_include:
            id_to_idx = {pid: idx for idx, pid in enumerate(df["id"])}
            missing = [pid for pid in cfg.must_include if pid not in id_to_idx]
            if missing:
                raise ValueError(f"must_include player id(s) not in pool: {missing}")
            for pid in cfg.must_include:
                constraints.append(squad[id_to_idx[pid]] == 1)

        # -- Objective: maximise metric across the XI, captain counted twice ---------
        objective_terms = metric @ starting
        if cfg.captain_doubles:
            objective_terms = objective_terms + metric @ captain
        objective = cp.Maximize(objective_terms)

        problem = cp.Problem(objective, constraints)
        solver_used = self._solve_with_fallback(problem)

        return self._build_result(df, squad, starting, captain, problem.status, solver_used)

    def _solve_with_fallback(self, problem: "cp.Problem") -> str:
        """Try the user's chosen solver, else walk down a preference list."""
        cfg = self.config
        candidates = [cfg.solver] if cfg.solver else _SOLVER_PREFERENCE
        installed = set(cp.installed_solvers())

        last_error = None
        for solver_name in candidates:
            if solver_name not in installed:
                continue
            try:
                problem.solve(solver=getattr(cp, solver_name), verbose=cfg.solver_verbose)
                if problem.status in {"optimal", "optimal_inaccurate"}:
                    return solver_name
            except Exception as exc:  # noqa: BLE001 - we want to try the next solver
                last_error = exc
                continue

        raise RuntimeError(
            "No installed MIP-capable cvxpy solver could solve the problem "
            f"(tried: {candidates}; installed: {sorted(installed)}). "
            "Install one of CBC, GLPK_MI, SCIP or HiGHS, e.g. `pip install pulp` "
            "or `conda install -c conda-forge glpk cvxpy`."
            + (f" Last error: {last_error}" if last_error else "")
        )

    # ------------------------------------------------------------------
    # Result assembly
    # ------------------------------------------------------------------
    def _build_result(
        self,
        df: pd.DataFrame,
        squad_var,
        starting_var,
        captain_var,
        status: str,
        solver_used: str,
    ) -> SquadResult:
        squad_mask = np.round(squad_var.value).astype(bool)
        starting_mask = np.round(starting_var.value).astype(bool)
        captain_mask = np.round(captain_var.value).astype(bool)

        squad_df = df[squad_mask].copy().sort_values(
            ["position_name", self.config.metric], ascending=[True, False]
        )
        starting_df = df[starting_mask].copy()
        bench_df = df[squad_mask & ~starting_mask].copy()

        starting_df = starting_df.assign(
            is_captain=[bool(captain_mask[i]) for i in df.index[starting_mask]]
        )
        starting_df = starting_df.sort_values(
            ["position_name", self.config.metric], ascending=[True, False]
        )

        captain_row = df[captain_mask].iloc[0]
        vice_candidates = starting_df[~starting_df["is_captain"]].sort_values(
            self.config.metric, ascending=False
        )
        vice_row = vice_candidates.iloc[0]

        formation_counts = starting_df["position_name"].value_counts()
        formation = "-".join(
            str(formation_counts.get(pos, 0)) for pos in ["DEF", "MID", "FWD"]
        )

        total_cost = float(squad_df["now_cost"].sum())
        squad_metric_total = float(squad_df[self.config.metric].sum())
        starting_metric_total = float(starting_df[self.config.metric].sum())
        if self.config.captain_doubles:
            starting_metric_total += float(captain_row[self.config.metric])

        return SquadResult(
            status=status,
            squad=squad_df.reset_index(drop=True),
            starting_xi=starting_df.reset_index(drop=True),
            bench=bench_df.sort_values("position_name").reset_index(drop=True),
            captain=captain_row,
            vice_captain=vice_row,
            formation=formation,
            total_cost=total_cost,
            budget=self.config.budget,
            squad_metric_total=squad_metric_total,
            starting_metric_total=starting_metric_total,
            metric=self.config.metric,
            solver_used=solver_used,
        )
