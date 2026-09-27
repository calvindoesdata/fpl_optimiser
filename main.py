#!/usr/bin/env python3
"""
Command-line entry point for the FPL team optimiser.

Examples
--------
Run against the bundled sample data, maximising form, £100m budget:
    python main.py --source csv --csv-path sample_data/players_sample.csv

Run against live FPL data (requires internet + `requests`):
    python main.py --source api

Force two specific players into the squad, and cap spend at £99.5m:
    python main.py --csv-path sample_data/players_sample.csv \\
        --must-include 12 47 --budget 99.5

Optimise on total season points instead of current form:
    python main.py --csv-path sample_data/players_sample.csv --metric total_points
"""

from __future__ import annotations

import argparse
import sys

from fpl_optimiser import (
    FPLOptimiser,
    OptimiserConfig,
    fetch_fpl_data,
    load_from_csv,
    render_result,
    transform_kaggle_data
)
from fpl_optimiser.models import DEFAULT_BUDGET, MAX_PER_TEAM


def parse_args(argv=None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Optimise an FPL squad under the official game constraints.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "--source",
        choices=["csv", "api", "kaggle"],
        default="kaggle",
        help="Where to load player data from (default: kaggle).",
    )
    parser.add_argument(
        "--csv-path",
        default="sample_data/players_sample.csv",
        help="Path to a local CSV of player data (used when --source csv).",
    )
    parser.add_argument(
        "--metric",
        default="form",
        help="Column to maximise, e.g. 'form' or 'total_points' (default: form).",
    )
    parser.add_argument(
        "--budget",
        type=float,
        default=DEFAULT_BUDGET,
        help=f"Total squad budget in £m (default: {DEFAULT_BUDGET}).",
    )
    parser.add_argument(
        "--max-per-team",
        type=int,
        default=MAX_PER_TEAM,
        help=f"Max players from one club (default: {MAX_PER_TEAM}).",
    )
    parser.add_argument(
        "--must-include",
        type=int,
        nargs="*",
        default=[],
        metavar="PLAYER_ID",
        help="Player id(s) that must be included in the squad.",
    )
    parser.add_argument(
        "--must-exclude",
        type=int,
        nargs="*",
        default=[],
        metavar="PLAYER_ID",
        help="Player id(s) to exclude from consideration entirely.",
    )
    parser.add_argument(
        "--include-unavailable",
        action="store_true",
        help="Don't filter out injured/suspended/doubtful players.",
    )
    parser.add_argument(
        "--no-captain-double",
        action="store_true",
        help="Don't double-count the captain's metric in the objective.",
    )
    parser.add_argument(
        "--solver",
        default=None,
        help="Force a specific cvxpy MIP solver (e.g. CBC, GLPK_MI, SCIP).",
    )
    parser.add_argument(
        "--verbose", action="store_true", help="Print solver progress output."
    )
    return parser.parse_args(argv)


def main(argv=None) -> int:
    args = parse_args(argv)

    if args.source == "api":
        print("Fetching live player data from the official FPL API...")
        players = fetch_fpl_data()
    elif args.source =="kaggle":
        print("Fetching live player data from Kaggle resource...")
        players = transform_kaggle_data()
    else:
        print(f"Loading player data from {args.csv_path} ...")
        players = load_from_csv(args.csv_path)

    config = OptimiserConfig(
        budget=args.budget,
        max_per_team=args.max_per_team,
        metric=args.metric,
        captain_doubles=not args.no_captain_double,
        exclude_unavailable=not args.include_unavailable,
        must_include=args.must_include,
        must_exclude=args.must_exclude,
        solver=args.solver,
        solver_verbose=args.verbose,
    )

    print(f"Optimising {len(players)} eligible players (metric: {config.metric})...\n")
    optimiser = FPLOptimiser(players, config)

    try:
        result = optimiser.solve()
    except RuntimeError as exc:
        print(f"Optimisation failed: {exc}", file=sys.stderr)
        return 1

    print(render_result(result))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
