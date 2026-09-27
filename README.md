# FPL Team Optimiser

Selects a mathematically optimal Fantasy Premier League squad by solving a
Mixed-Integer Program (MIP) with [cvxpy](https://www.cvxpy.org/), maximising
player **form** (or any other metric you choose) subject to the real FPL
rules.

## What it optimises

Three sets of binary decisions are solved jointly:

| Variable      | Meaning                                             |
|---------------|------------------------------------------------------|
| `squad[i]`    | Is player *i* one of your 15 squad picks?            |
| `starting[i]` | Is player *i* in your starting XI?                   |
| `captain[i]`  | Is player *i* your captain (2x points)?              |

**Objective:** maximise `sum(form of starting XI) + form of captain`
(the `+ captain` term reproduces the real rule that the captain's score is
doubled).

## Constraints enforced (official FPL rules)

- Squad = exactly 15 players: **2 GK, 5 DEF, 5 MID, 3 FWD**
- Total spend ≤ budget (default **£100.0m**)
- **Max 3 players from any one real club**
- Starting XI = exactly 11 players, drawn from the squad, in a valid
  formation: **1 GK, 3–5 DEF, 2–5 MID, 1–3 FWD**
- Exactly **one captain**, chosen from the starting XI (a vice-captain is
  reported too: the highest-metric non-captain starter)
- Optional: force-include specific players (e.g. players you already own),
  force-exclude specific players, and drop unavailable
  (injured/suspended/ruled-out) players automatically

All of these live in one place, [`fpl_optimiser/models.py`](fpl_optimiser/models.py),
so they're easy to audit or adjust if the rules change.

## Project layout

```
fpl_optimiser/
├── fpl_optimiser/
│   ├── models.py       # Player data model + official FPL rule constants
│   ├── data_loader.py  # Load data from the live FPL API or a local CSV
│   ├── optimiser.py    # The cvxpy MIP formulation and solver
│   └── reporting.py    # Pretty-printed squad/XI/bench report
├── main.py             # CLI entry point
├── sample_data/
│   └── players_sample.csv   # Synthetic offline dataset for demos/tests
├── tests/
│   ├── test_data_loader.py  # No cvxpy required
│   └── test_optimiser.py    # Requires cvxpy + a MIP solver (auto-skips otherwise)
└── requirements.txt
```

## Installation

```bash
pip install -r requirements.txt
```

`cvxpy>=1.6` bundles the open-source **HiGHS** solver, which is MIP-capable,
so this is all you need — there's no separate solver license or install
step required for the core use case.

## Quick start

Run against the bundled synthetic sample data (no network needed):

```bash
python main.py --source csv --csv-path sample_data/players_sample.csv
```

Run against **live** FPL data (requires internet + `requests`):

```bash
python main.py --source api
```

Optimise on total season points instead of current form:

```bash
python main.py --csv-path sample_data/players_sample.csv --metric total_points
```

Lock in players you already own, and tighten the budget:

```bash
python main.py --csv-path sample_data/players_sample.csv \
    --must-include 12 47 --budget 99.5
```

Full option list:

```bash
python main.py --help
```

## Using it as a library

```python
from fpl_optimiser import load_from_csv, FPLOptimiser, OptimiserConfig, render_result

players = load_from_csv("sample_data/players_sample.csv")

config = OptimiserConfig(
    budget=100.0,
    metric="form",          # or "total_points", or any numeric column you add
    must_include=[],        # player ids to force into the squad
    must_exclude=[],        # player ids to ban
)

result = FPLOptimiser(players, config).solve()
print(render_result(result))

# Structured access:
result.squad          # 15-row DataFrame
result.starting_xi    # 11-row DataFrame (has an `is_captain` column)
result.bench          # 4-row DataFrame
result.captain        # Series
result.vice_captain   # Series
result.total_cost     # float, £m
result.formation      # e.g. "4-4-2"
```

## Bringing your own data

`load_from_csv` expects (at least) these columns — see
`sample_data/players_sample.csv` for a full working example:

| Column                         | Meaning                                             |
|--------------------------------|------------------------------------------------------|
| `id`                           | Unique player id                                     |
| `web_name`                     | Display name                                         |
| `team`                         | Club name                                            |
| `position`                     | One of `GK`, `DEF`, `MID`, `FWD`                     |
| `price`                        | Current price, £m (e.g. `7.5`)                       |
| `form`                         | FPL "form" stat (avg. points, recent games)          |
| `total_points`                 | Season total points                                  |
| `selected_by_percent`          | Ownership %                                          |
| `minutes`                      | Minutes played this season                           |
| `chance_of_playing_next_round` | 0–100, or blank if fully fit                         |
| `status`                       | `a`=available, `i`=injured, `s`=suspended, `d`=doubt  |

You can add any extra numeric columns (e.g. `expected_points`,
`xg_last_5`) and point `--metric` / `OptimiserConfig.metric` at them to
optimise on something other than `form`.

`fetch_fpl_data()` fills this schema automatically from the official
`bootstrap-static` FPL API endpoint.

## Testing

```bash
pip install pytest
pytest tests/test_data_loader.py     # no cvxpy required
pytest tests/test_optimiser.py       # requires cvxpy + a MIP solver; auto-skips if absent
```

`tests/test_optimiser.py` asserts, on the bundled sample data, that every
official constraint holds for the solved squad (exact position quotas,
budget respected, ≤3-per-club, valid starting formation, a single captain
distinct from the vice-captain, and that `must_include`/`must_exclude`
are honoured).

## Notes & caveats

- This models a **single gameweek** selection (maximise current form /
  points right now). It does not model transfers, chip strategy,
  price-change speculation, or multi-gameweek fixture planning — those
  would be natural extensions on top of the same `OptimiserConfig` /
  `FPLOptimiser` building blocks.
- `form` is a short-term, backward-looking stat (FPL's own rolling
  average). It's a reasonable proxy for "who's in good shape right now",
  but isn't a fixture-adjusted projection — swap in your own
  `expected_points` column via `--metric` if you have one.
