"""Causal feature engineering for race-winner prediction.

Every feature for a race is computed only from races that happened before it.
`rows_known` uses the actual grid/qualifying data; `rows_prior` hides that
information and estimates it from current form, so the same model can be used
before and after qualifying.
"""

from __future__ import annotations

import json
import math
from collections import defaultdict, deque
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from .data import Entry, Race

WINDOW = 10
FORM_WINDOW = 5
MAX_DRIVERS = 24
PL_K = 5

FEATURE_NAMES = [
    "grid_pos",
    "quali_pos",
    "quali_gap",
    "grid_penalty",
    "quali_known",
    "driver_finish_5",
    "driver_quali_5",
    "driver_points_5",
    "driver_gap_5",
    "driver_dnf_10",
    "driver_exp",
    "driver_last_finish",
    "team_finish_5",
    "team_quali_5",
    "team_points_5",
    "team_dnf_10",
    "circuit_driver_finish",
    "circuit_driver_starts",
    "circuit_team_finish",
    "circuit_team_starts",
    "driver_season_points",
    "team_season_points",
    "teammate_quali_delta",
    "round_frac",
    "sprint_weekend",
]
N_FEATURES = len(FEATURE_NAMES)


def _num(value: float | int | None) -> float:
    return float("nan") if value is None else float(value)


class FeatureState:
    """Rolling driver/team form, updated race by race."""

    def __init__(self) -> None:
        self.driver_hist: dict[str, deque] = defaultdict(lambda: deque(maxlen=WINDOW))
        self.team_hist: dict[str, deque] = defaultdict(lambda: deque(maxlen=WINDOW))
        self.driver_races: dict[str, int] = defaultdict(int)
        self.driver_circuit: dict[tuple[str, str], list[float]] = defaultdict(list)
        self.team_circuit: dict[tuple[str, str], list[float]] = defaultdict(list)
        self.driver_season_points: dict[tuple[int, str], float] = defaultdict(float)
        self.team_season_points: dict[tuple[int, str], float] = defaultdict(float)
        self.teammate_delta: dict[str, deque] = defaultdict(lambda: deque(maxlen=WINDOW))

    @staticmethod
    def _values(hist: deque, key: str, window: int) -> list[float]:
        return [r[key] for r in list(hist)[-window:] if r.get(key) is not None]

    @classmethod
    def _avg(cls, hist: deque, key: str, window: int = FORM_WINDOW) -> float | None:
        values = cls._values(hist, key, window)
        return sum(values) / len(values) if values else None

    @classmethod
    def _rate(cls, hist: deque, key: str, window: int = WINDOW) -> float | None:
        values = cls._values(hist, key, window)
        return sum(values) / len(values) if values else None

    def update(self, race: Race) -> None:
        entries = race.entries
        n = len(entries)
        pole = min(
            (e.quali.time for e in entries if e.quali and e.quali.time),
            default=None,
        )
        by_team: dict[str, list[Entry]] = defaultdict(list)

        for entry in entries:
            finish_value = entry.finish if entry.finish is not None else n + 1
            qpos = entry.quali.position if entry.quali else None
            qtime = entry.quali.time if entry.quali else None
            gap = (qtime / pole - 1.0) if qtime and pole else None
            self.driver_hist[entry.driver_id].append(
                {
                    "finish": float(finish_value),
                    "quali": float(qpos) if qpos else None,
                    "points": entry.points,
                    "gap": gap,
                    "dnf": 0.0 if entry.classified else 1.0,
                }
            )
            self.driver_races[entry.driver_id] += 1
            self.driver_circuit[(entry.driver_id, race.circuit_id)].append(
                float(finish_value)
            )
            self.driver_season_points[(race.season, entry.driver_id)] += entry.points
            by_team[entry.team_id].append(entry)

        for team_id, team_entries in by_team.items():
            finishes = [
                float(e.finish if e.finish is not None else n + 1)
                for e in team_entries
            ]
            qualis = [
                float(e.quali.position)
                for e in team_entries
                if e.quali and e.quali.position
            ]
            gaps = [
                (e.quali.time / pole - 1.0)
                for e in team_entries
                if pole and e.quali and e.quali.time
            ]
            self.team_hist[team_id].append(
                {
                    "finish": sum(finishes) / len(finishes),
                    "quali": sum(qualis) / len(qualis) if qualis else None,
                    "points": sum(e.points for e in team_entries) / len(team_entries),
                    "gap": sum(gaps) / len(gaps) if gaps else None,
                    "dnf": sum(0.0 if e.classified else 1.0 for e in team_entries)
                    / len(team_entries),
                }
            )
            self.team_circuit[(team_id, race.circuit_id)].extend(finishes)
            self.team_season_points[(race.season, team_id)] += sum(
                e.points for e in team_entries
            )

            if len(team_entries) == 2 and all(
                e.quali and e.quali.position for e in team_entries
            ):
                a, b = team_entries
                delta = float(a.quali.position - b.quali.position)
                self.teammate_delta[a.driver_id].append(delta)
                self.teammate_delta[b.driver_id].append(-delta)

    def feature_rows(
        self, race: Race, entries: list[Entry], *, quali_known: bool
    ) -> np.ndarray:
        n = len(entries)
        effective_known = quali_known and any(
            e.quali and e.quali.position is not None for e in entries
        )
        pole = (
            min(
                (e.quali.time for e in entries if e.quali and e.quali.time),
                default=None,
            )
            if effective_known
            else None
        )

        # Estimate the qualifying order from rolling form for the prior model.
        est_scores: list[float] = []
        for entry in entries:
            dq = self._avg(self.driver_hist[entry.driver_id], "quali")
            tq = self._avg(self.team_hist[entry.team_id], "quali")
            values = [v for v in (dq, tq) if v is not None]
            est_scores.append(sum(values) / len(values) if values else n / 2 + 1)
        est_order = sorted(range(n), key=lambda i: (est_scores[i], i))
        est_rank = {idx: rank + 1 for rank, idx in enumerate(est_order)}

        rows: list[list[float]] = []
        for i, entry in enumerate(entries):
            dh = self.driver_hist[entry.driver_id]
            th = self.team_hist[entry.team_id]

            if effective_known:
                qpos = entry.quali.position if entry.quali else None
                qpos_value = float(qpos) if qpos else float(n + 2)
                qtime = entry.quali.time if entry.quali else None
                gap = (
                    qtime / pole - 1.0
                    if qtime and pole
                    else self._avg(dh, "gap")
                )
                if entry.grid is None:
                    grid = qpos_value
                elif entry.grid == 0:  # pit lane
                    grid = float(n + 1)
                else:
                    grid = float(entry.grid)
                penalty = grid - qpos_value
            else:
                qpos_value = float(est_rank[i])
                gap = self._avg(dh, "gap")
                grid = float(est_rank[i])
                penalty = 0.0

            d_finish = self._avg(dh, "finish")
            d_quali = self._avg(dh, "quali")
            team_finish = self._avg(th, "finish")
            team_quali = self._avg(th, "quali")

            driver_circuit = self.driver_circuit.get(
                (entry.driver_id, race.circuit_id), []
            )
            team_circuit = self.team_circuit.get((entry.team_id, race.circuit_id), [])
            circuit_driver_finish = (
                sum(driver_circuit[-WINDOW:]) / len(driver_circuit[-WINDOW:])
                if driver_circuit
                else d_finish
            )
            circuit_team_finish = (
                sum(team_circuit[-WINDOW:]) / len(team_circuit[-WINDOW:])
                if team_circuit
                else team_finish
            )
            teammate_values = self.teammate_delta.get(entry.driver_id)
            teammate_delta = (
                sum(teammate_values) / len(teammate_values)
                if teammate_values
                else 0.0
            )

            rows.append(
                [
                    grid,
                    qpos_value,
                    _num(gap),
                    penalty,
                    1.0 if effective_known else 0.0,
                    _num(d_finish),
                    _num(d_quali),
                    _num(self._avg(dh, "points")),
                    _num(self._avg(dh, "gap")),
                    _num(self._rate(dh, "dnf")),
                    math.log1p(self.driver_races[entry.driver_id]),
                    _num(dh[-1]["finish"] if dh else None),
                    _num(team_finish),
                    _num(team_quali),
                    _num(self._avg(th, "points")),
                    _num(self._rate(th, "dnf")),
                    _num(circuit_driver_finish),
                    math.log1p(len(driver_circuit)),
                    _num(circuit_team_finish),
                    math.log1p(len(team_circuit)),
                    self.driver_season_points.get((race.season, entry.driver_id), 0.0),
                    self.team_season_points.get((race.season, entry.team_id), 0.0),
                    teammate_delta,
                    (race.round - 1) / 23.0,
                    1.0 if race.has_sprint else 0.0,
                ]
            )
        return np.asarray(rows, dtype=np.float64)


@dataclass
class RaceSample:
    season: int
    round: int
    race_name: str
    circuit_id: str
    date: str
    drivers: list[str]
    codes: list[str]
    names: list[str]
    teams: list[str]
    rows_known: np.ndarray
    rows_prior: np.ndarray
    finishes: list[int | None]
    winner: int | None


def build_samples(races: list[Race]) -> list[RaceSample]:
    state = FeatureState()
    samples: list[RaceSample] = []
    for race in sorted(races, key=lambda r: (r.season, r.round)):
        rows_known = state.feature_rows(race, race.entries, quali_known=True)
        rows_prior = state.feature_rows(race, race.entries, quali_known=False)
        finishes = [e.finish for e in race.entries]
        winner = next((i for i, f in enumerate(finishes) if f == 1), None)
        samples.append(
            RaceSample(
                season=race.season,
                round=race.round,
                race_name=race.name,
                circuit_id=race.circuit_id,
                date=race.date,
                drivers=[e.driver_id for e in race.entries],
                codes=[e.code for e in race.entries],
                names=[e.name for e in race.entries],
                teams=[e.team_name for e in race.entries],
                rows_known=rows_known,
                rows_prior=rows_prior,
                finishes=finishes,
                winner=winner,
            )
        )
        state.update(race)
    return samples


def state_from_races(races: list[Race]) -> FeatureState:
    state = FeatureState()
    for race in sorted(races, key=lambda r: (r.season, r.round)):
        state.update(race)
    return state


@dataclass
class Scaler:
    median: np.ndarray
    mean: np.ndarray
    std: np.ndarray

    def transform(self, rows: np.ndarray) -> np.ndarray:
        values = np.asarray(rows, dtype=np.float64)
        values = np.where(np.isnan(values), self.median[None, :], values)
        scaled = (values - self.mean[None, :]) / self.std[None, :]
        return np.clip(scaled, -6.0, 6.0).astype(np.float32)

    def to_dict(self) -> dict:
        return {
            "median": self.median.tolist(),
            "mean": self.mean.tolist(),
            "std": self.std.tolist(),
            "features": FEATURE_NAMES,
        }

    @classmethod
    def from_dict(cls, payload: dict) -> "Scaler":
        return cls(
            median=np.asarray(payload["median"], dtype=np.float64),
            mean=np.asarray(payload["mean"], dtype=np.float64),
            std=np.asarray(payload["std"], dtype=np.float64),
        )

    def save(self, path: Path) -> None:
        path.write_text(json.dumps(self.to_dict(), indent=2))

    @classmethod
    def load(cls, path: Path) -> "Scaler":
        return cls.from_dict(json.loads(path.read_text()))


def fit_scaler(samples: list[RaceSample]) -> Scaler:
    rows = np.concatenate(
        [sample.rows_known for sample in samples]
        + [sample.rows_prior for sample in samples],
        axis=0,
    )
    with np.errstate(invalid="ignore"):
        median = np.nanmedian(rows, axis=0)
    median = np.where(np.isfinite(median), median, 0.0)
    filled = np.where(np.isnan(rows), median[None, :], rows)
    mean = filled.mean(axis=0)
    std = filled.std(axis=0)
    std = np.where(std < 1e-6, 1.0, std)
    return Scaler(median=median, mean=mean, std=std)


def build_circuit_map(samples: list[RaceSample]) -> dict[str, int]:
    circuit_ids = sorted({sample.circuit_id for sample in samples})
    return {circuit_id: index + 1 for index, circuit_id in enumerate(circuit_ids)}


def make_arrays(
    samples: list[RaceSample],
    scaler: Scaler,
    circuit_map: dict[str, int],
    *,
    variant: str,
) -> dict[str, np.ndarray]:
    count = len(samples)
    x = np.zeros((count, MAX_DRIVERS, N_FEATURES), dtype=np.float32)
    mask = np.zeros((count, MAX_DRIVERS), dtype=bool)
    winner = np.zeros(count, dtype=np.int32)
    rank_idx = np.zeros((count, PL_K), dtype=np.int32)
    rank_valid = np.zeros((count, PL_K), dtype=bool)
    circuit = np.zeros(count, dtype=np.int32)

    for i, sample in enumerate(samples):
        rows = sample.rows_known if variant == "known" else sample.rows_prior
        field_size = len(rows)
        if field_size > MAX_DRIVERS:
            raise ValueError(f"{sample.race_name}: {field_size} drivers exceeds padding")
        x[i, :field_size] = scaler.transform(rows)
        mask[i, :field_size] = True
        winner[i] = sample.winner or 0
        circuit[i] = circuit_map.get(sample.circuit_id, 0)
        ranked = sorted(
            (position, index)
            for index, position in enumerate(sample.finishes)
            if position is not None
        )
        for k, (_, index) in enumerate(ranked[:PL_K]):
            rank_idx[i, k] = index
            rank_valid[i, k] = True

    return {
        "x": x,
        "mask": mask,
        "winner": winner,
        "rank_idx": rank_idx,
        "rank_valid": rank_valid,
        "circuit": circuit,
    }


def make_prediction_arrays(
    rows: np.ndarray,
    circuit_id: str,
    scaler: Scaler,
    circuit_map: dict[str, int],
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    field_size = len(rows)
    if field_size > MAX_DRIVERS:
        raise ValueError(f"{field_size} drivers exceeds padding")
    x = np.zeros((1, MAX_DRIVERS, N_FEATURES), dtype=np.float32)
    x[0, :field_size] = scaler.transform(rows)
    mask = np.zeros((1, MAX_DRIVERS), dtype=bool)
    mask[0, :field_size] = True
    circuit = np.asarray([circuit_map.get(circuit_id, 0)], dtype=np.int32)
    return x, mask, circuit
