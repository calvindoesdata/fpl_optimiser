"""
Tests that don't require cvxpy/a MIP solver: data loading, validation,
and the Player model. Run with: pytest tests/test_data_loader.py
"""

import os
import sys

import pandas as pd
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from fpl_optimiser.data_loader import load_from_csv, REQUIRED_COLUMNS
from fpl_optimiser.models import Player, POSITION_MAP

SAMPLE_CSV = os.path.join(
    os.path.dirname(__file__), "..", "sample_data", "players_sample.csv"
)


def test_load_sample_csv_has_required_columns():
    df = load_from_csv(SAMPLE_CSV)
    for col in REQUIRED_COLUMNS:
        assert col in df.columns


def test_load_sample_csv_positions_are_valid():
    df = load_from_csv(SAMPLE_CSV)
    assert set(df["position"].unique()) <= set(POSITION_MAP.values())


def test_load_sample_csv_enough_players_per_position():
    df = load_from_csv(SAMPLE_CSV)
    counts = df["position"].value_counts()
    assert counts.get("GK", 0) >= 2
    assert counts.get("DEF", 0) >= 5
    assert counts.get("MID", 0) >= 5
    assert counts.get("FWD", 0) >= 3


def test_missing_column_raises(tmp_path):
    bad_csv = tmp_path / "bad.csv"
    pd.DataFrame({"id": [1], "web_name": ["X"]}).to_csv(bad_csv, index=False)
    with pytest.raises(ValueError):
        load_from_csv(str(bad_csv))


def test_invalid_position_raises(tmp_path):
    df = load_from_csv(SAMPLE_CSV).copy()
    df.loc[0, "position"] = "ZZZ"
    bad_csv = tmp_path / "bad_pos.csv"
    df.to_csv(bad_csv, index=False)
    with pytest.raises(ValueError):
        load_from_csv(str(bad_csv))


def test_player_model_availability_flags():
    fit = Player(
        id=1, web_name="Fit Player", team="Arsenal", position="MID",
        price=7.0, form=5.0, status="a", chance_of_playing_next_round=100,
    )
    injured = Player(
        id=2, web_name="Injured Player", team="Arsenal", position="MID",
        price=7.0, form=5.0, status="i",
    )
    doubtful_ruled_out = Player(
        id=3, web_name="Zero Chance", team="Arsenal", position="MID",
        price=7.0, form=5.0, status="d", chance_of_playing_next_round=0,
    )
    assert fit.is_available is True
    assert injured.is_available is False
    assert doubtful_ruled_out.is_available is False


def test_player_model_rejects_unknown_position():
    with pytest.raises(ValueError):
        Player(id=1, web_name="X", team="Arsenal", position="STRIKER", price=5.0, form=1.0)


def test_value_score():
    p = Player(id=1, web_name="X", team="Arsenal", position="MID", price=8.0, form=8.0)
    assert p.value_score("form") == pytest.approx(1.0)
