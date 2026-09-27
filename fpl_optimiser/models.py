"""
Data model for players and the fixed rules of the FPL game.

Keeping the official game constraints in one place (this module) makes it
easy to see exactly what the optimiser is being asked to respect, and to
update them if the game rules change in a future season.
"""

from dataclasses import dataclass, field
from typing import Optional


# ---------------------------------------------------------------------------
# Official FPL rules (2025/26 season). Update here if the rules change.
# ---------------------------------------------------------------------------

#: Element type id -> short position code, as used by the FPL API.
POSITION_MAP = {1: "GK", 2: "DEF", 3: "MID", 4: "FWD"}

#: Exact number of each position required in a 15-man squad.
SQUAD_REQUIREMENTS = {"GK": 2, "DEF": 5, "MID": 5, "FWD": 3}

#: Total squad size.
SQUAD_SIZE = sum(SQUAD_REQUIREMENTS.values())  # 15

#: Starting XI must total 11, with these min/max per position.
#: (GK is always exactly 1 in a valid formation.)
FORMATION_LIMITS = {
    "GK": (1, 1),
    "DEF": (3, 5),
    "MID": (2, 5),
    "FWD": (1, 3),
}
STARTING_XI_SIZE = 11

#: Default season budget, in £m (100.0 = £100.0m).
DEFAULT_BUDGET = 100.0

#: Maximum number of players allowed from any single real-world club.
MAX_PER_TEAM = 3

#: Squad slots not in the starting XI ("the bench").
BENCH_SIZE = SQUAD_SIZE - STARTING_XI_SIZE  # 4


@dataclass
class Player:
    """A single FPL-eligible player and the attributes the optimiser needs."""

    id: int
    web_name: str
    team: str
    position: str  # one of GK, DEF, MID, FWD
    price: float  # £m, e.g. 5.5
    form: float  # FPL "form" stat: average points over last 30 days
    total_points: float = 0.0
    selected_by_percent: float = 0.0
    minutes: int = 0
    chance_of_playing_next_round: Optional[float] = None  # 0-100, None = fit
    status: str = "a"  # 'a' available, 'i' injured, 's' suspended, 'd' doubt
    full_name: Optional[str] = None
    extra: dict = field(default_factory=dict)

    def __post_init__(self):
        if self.position not in POSITION_MAP.values():
            raise ValueError(
                f"Unknown position '{self.position}' for player {self.web_name}; "
                f"expected one of {sorted(set(POSITION_MAP.values()))}"
            )
        if self.full_name is None:
            self.full_name = self.web_name

    @property
    def is_available(self) -> bool:
        """True unless flagged injured/suspended or ruled out (0% chance)."""
        if self.status in {"i", "s", "u"}:  # injured / suspended / unavailable
            return False
        if self.chance_of_playing_next_round is not None and self.chance_of_playing_next_round == 0:
            return False
        return True

    def value_score(self, metric: str = "form") -> float:
        """Points-per-million for the chosen metric; used only for reporting."""
        metric_value = getattr(self, metric)
        return metric_value / self.price if self.price else 0.0
