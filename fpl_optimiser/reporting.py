"""Human-readable rendering of a SquadResult."""

from __future__ import annotations

import pandas as pd

from .optimiser import SquadResult

_POSITION_ORDER = ["GK", "DEF", "MID", "FWD"]
_DISPLAY_COLUMNS = ["player_name", "club_name", "position_name", "now_cost", "form", "total_points"]


def _fmt_table(df: pd.DataFrame, metric: str) -> str:
    cols = [c for c in _DISPLAY_COLUMNS if c in df.columns]
    if metric not in cols:
        cols.append(metric)
    view = df[cols].rename(
        columns={
            "player_name": "Player",
            "club_name": "Team",
            "position_name": "Pos",
            "now_cost": "£m",
            "form": "Form",
            "total_points": "Pts",
        }
    )
    return view.to_string(index=False)


def render_result(result: SquadResult) -> str:
    """Return a full, readable report of a solved squad."""
    sections = []
    sections.append(result.summary())

    sections.append("\n--- Starting XI (formation {}) ---".format(result.formation))
    starting = result.starting_xi.copy()
    starting["role"] = starting["is_captain"].map(
        lambda c: "(C)" if c else ""
    )
    starting.loc[
        starting["player_name"] == result.vice_captain["player_name"], "role"
    ] = starting.loc[
        starting["player_name"] == result.vice_captain["player_name"], "role"
    ].apply(lambda x: x + " (VC)" if x else "(VC)")
    starting["position_name"] = pd.Categorical(
        starting["position_name"], categories=_POSITION_ORDER, ordered=True
    )
    starting = starting.sort_values(["position_name", result.metric], ascending=[True, False])
    display = starting[_DISPLAY_COLUMNS + ["role"]].rename(
        columns={
            "player_name": "Player",
            "team": "Team",
            "position": "Pos",
            "price": "£m",
            "form": "Form",
            "total_points": "Pts",
            "role": "Role",
        }
    )
    sections.append(display.to_string(index=False))

    sections.append("\n--- Bench ---")
    bench = result.bench.copy()
    bench["position_name"] = pd.Categorical(
        bench["position_name"], categories=_POSITION_ORDER, ordered=True
    )
    bench = bench.sort_values(["position_name", result.metric], ascending=[True, False])
    sections.append(_fmt_table(bench, result.metric))

    sections.append(
        "\nSquad total: £{:.1f}m spent, {} = {:.1f} (squad-wide)".format(
            result.total_cost, result.metric, result.squad_metric_total
        )
    )
    return "\n".join(sections)
