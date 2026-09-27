"""
Sourcing player data for the optimiser.

Two sources are supported:

1. ``fetch_fpl_data()``  - pulls the live ``bootstrap-static`` payload from
   the official (undocumented but widely used) FPL API. Requires network
   access.
2. ``load_from_csv()``   - loads a local CSV snapshot, useful for offline
   development, testing, or working from a curated data export.

Both return a plain ``pandas.DataFrame`` with a fixed, normalised schema
(see ``REQUIRED_COLUMNS``) that ``FPLOptimiser`` and ``Player`` know how to
consume, so the rest of the codebase does not care which source was used.
"""

from __future__ import annotations

import pandas as pd

import kagglehub
import os

from .models import POSITION_MAP

FPL_BOOTSTRAP_URL = "https://fantasy.premierleague.com/api/bootstrap-static/"

#: Columns every downstream consumer can rely on being present.
REQUIRED_COLUMNS = [
    "id",
    "web_name",
    "team",
    "position",
    "price",
    "form",
    "total_points",
    "selected_by_percent",
    "minutes",
    "chance_of_playing_next_round",
    "status",
]


def fetch_fpl_data(url: str = FPL_BOOTSTRAP_URL, timeout: int = 15) -> pd.DataFrame:
    """
    Fetch the current player universe directly from the official FPL API.

    Parameters
    ----------
    url : str
        The bootstrap-static endpoint (rarely needs changing).
    timeout : int
        Request timeout in seconds.

    Returns
    -------
    pd.DataFrame
        Normalised player data, ready for ``FPLOptimiser``.

    Notes
    -----
    Requires the ``requests`` package and outbound network access. If you
    are working offline, use :func:`load_from_csv` instead (see
    ``sample_data/players_sample.csv`` for the expected shape).
    """
    try:
        import requests
    except ImportError as exc:  # pragma: no cover
        raise ImportError(
            "The 'requests' package is required to fetch live FPL data. "
            "Install it with `pip install requests`, or use load_from_csv() "
            "for an offline data source."
        ) from exc

    response = requests.get(url, timeout=timeout)
    response.raise_for_status()
    payload = response.json()

    elements = pd.DataFrame(payload["elements"])
    teams = pd.DataFrame(payload["teams"])[["id", "name"]].rename(
        columns={"id": "team_id", "name": "team_name"}
    )

    elements = elements.merge(
        teams, left_on="team", right_on="team_id", how="left"
    )

    df = pd.DataFrame(
        {
            "id": elements["id"],
            "web_name": elements["web_name"],
            "team": elements["team_name"],
            "position": elements["element_type"].map(POSITION_MAP),
            "price": elements["now_cost"] / 10.0,
            "form": pd.to_numeric(elements["form"], errors="coerce").fillna(0.0),
            "total_points": elements["total_points"],
            "selected_by_percent": pd.to_numeric(
                elements["selected_by_percent"], errors="coerce"
            ).fillna(0.0),
            "minutes": elements["minutes"],
            "chance_of_playing_next_round": elements[
                "chance_of_playing_next_round"
            ],
            "status": elements["status"],
        }
    )
    return _finalise(df)


def load_from_csv(path: str) -> pd.DataFrame:
    """
    Load player data from a local CSV file.

    The CSV must contain (at least) the columns listed in
    ``REQUIRED_COLUMNS``. Extra columns are preserved and ignored by the
    optimiser. See ``sample_data/players_sample.csv`` for a working example.
    """
    df = pd.read_csv(path)
    missing = set(REQUIRED_COLUMNS) - set(df.columns)
    if missing:
        raise ValueError(
            f"CSV at '{path}' is missing required column(s): {sorted(missing)}. "
            f"Required columns are: {REQUIRED_COLUMNS}"
        )
    return _finalise(df)


def _finalise(df: pd.DataFrame) -> pd.DataFrame:
    """Shared cleanup/validation applied to any data source."""
    df = df.copy()

    invalid_positions = set(df["position"].unique()) - set(POSITION_MAP.values())
    if invalid_positions:
        raise ValueError(f"Unrecognised position code(s) in data: {invalid_positions}")

    df["form"] = pd.to_numeric(df["form"], errors="coerce").fillna(0.0)
    df["price"] = pd.to_numeric(df["price"], errors="coerce")
    df["chance_of_playing_next_round"] = pd.to_numeric(
        df["chance_of_playing_next_round"], errors="coerce"
    )
    df["status"] = df["status"].fillna("a")

    if df["price"].isna().any():
        bad_ids = df.loc[df["price"].isna(), "id"].tolist()
        raise ValueError(f"Missing/invalid price for player id(s): {bad_ids}")

    df = df.reset_index(drop=True)
    return df

def fetch_kaggle_data():
    path = kagglehub.dataset_download("calvinrostanto/fantasy-premier-league-2025-2026")
    df = pd.read_csv(os.path.join(path, os.listdir(path)[0]))

    return df

def transform_kaggle_data():
    df = fetch_kaggle_data()
    avg_points = df.groupby("player_name")["total_points"].mean()
    df_players = df.copy()
    df_players["avg_next_points"] = df["player_name"].map(avg_points)
    df_players = df_players[~df_players['form'].isna()]
    df_players.index = range(df_players.shape[0])

    means = {}
    stds = {}
    for col in ['form', 'total_points']:
        means[col] = df_players[col].mean()
        stds[col] = df_players[col].std()
        df[col + '_norm'] = (df[col] - means[col])/stds[col]

    df['metric'] = df['ict_index'] * 0.4 + df['form_norm'] * 0.45 + df['total_points_norm'] * 0.15

    return df