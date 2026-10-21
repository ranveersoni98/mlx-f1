"""Jolpica/Ergast F1 API client and race-result parsing."""

from __future__ import annotations

import json
import os
import time
import urllib.error
import urllib.request
from dataclasses import dataclass, field
from pathlib import Path

API_BASE = "https://api.jolpi.ca/ergast/f1"
PROJECT_ROOT = Path(os.environ.get("MLX_F1_HOME", Path(__file__).resolve().parents[2]))
DATA_DIR = PROJECT_ROOT / "data"
RAW_DIR = DATA_DIR / "raw"
PROCESSED_DIR = DATA_DIR / "processed"

START_SEASON = 2014
PAGE_SIZE = 100


@dataclass
class QualiInfo:
    position: int | None
    time: float | None  # best lap in seconds


@dataclass
class Entry:
    driver_id: str
    code: str
    name: str
    team_id: str
    team_name: str
    grid: int | None
    finish: int | None
    classified: bool
    points: float
    status: str
    quali: QualiInfo | None = None
    sprint_pos: int | None = None


@dataclass
class Race:
    season: int
    round: int
    name: str
    circuit_id: str
    circuit_name: str
    country: str
    date: str
    has_sprint: bool
    entries: list[Entry] = field(default_factory=list)


def _http_json(url: str, retries: int = 4) -> dict:
    last: Exception | None = None
    for attempt in range(retries):
        try:
            req = urllib.request.Request(
                url, headers={"User-Agent": "mlx-f1/0.1 (race winner predictor)"}
            )
            with urllib.request.urlopen(req, timeout=30) as resp:
                return json.loads(resp.read().decode())
        except (urllib.error.URLError, TimeoutError, json.JSONDecodeError, ValueError) as exc:
            last = exc
            time.sleep(1.5 * (attempt + 1))
    raise RuntimeError(f"Failed to fetch {url}: {last}")


_LIST_KEYS = {
    "results": "Results",
    "qualifying": "QualifyingResults",
    "sprint": "SprintResults",
}


def _fetch_endpoint(season: int, endpoint: str) -> dict:
    """Fetch every page of a season endpoint and cache the merged payload."""
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    cache = RAW_DIR / f"{season}_{endpoint}.json"
    if cache.exists():
        return json.loads(cache.read_text())

    list_key = _LIST_KEYS[endpoint]
    merged: dict[int, dict] = {}
    offset = 0
    total = 1
    while offset < total:
        url = f"{API_BASE}/{season}/{endpoint}.json?limit={PAGE_SIZE}&offset={offset}"
        try:
            payload = _http_json(url)
        except RuntimeError:
            if offset == 0:
                out = {"season": season, "endpoint": endpoint, "races": []}
                cache.write_text(json.dumps(out))
                return out
            raise
        mr = payload["MRData"]
        total = int(mr["total"])
        for race in mr["RaceTable"]["Races"]:
            rnd = int(race["round"])
            dst = merged.setdefault(
                rnd, {k: v for k, v in race.items() if k != list_key}
            )
            dst.setdefault(list_key, []).extend(race.get(list_key, []))
        offset += PAGE_SIZE
        if offset < total:
            time.sleep(0.25)

    out = {
        "season": season,
        "endpoint": endpoint,
        "races": [merged[k] for k in sorted(merged)],
    }
    cache.write_text(json.dumps(out))
    return out


def current_season() -> int:
    cache = RAW_DIR / "_current.json"
    try:
        payload = _http_json(f"{API_BASE}/current.json")
        season = int(payload["MRData"]["RaceTable"]["season"])
        RAW_DIR.mkdir(parents=True, exist_ok=True)
        cache.write_text(json.dumps({"season": season}))
        return season
    except Exception:
        if cache.exists():
            return int(json.loads(cache.read_text())["season"])
        raise


def parse_lap_time(value: str | None) -> float | None:
    if not value:
        return None
    try:
        if ":" in value:
            minutes, seconds = value.split(":", 1)
            return int(minutes) * 60 + float(seconds)
        return float(value)
    except ValueError:
        return None


def _to_int(value: object) -> int | None:
    try:
        return int(str(value))
    except (TypeError, ValueError):
        return None


def _is_classified(status: str) -> bool:
    return status == "Finished" or status.startswith("+")


def _parse_quali_map(race: dict) -> dict[str, QualiInfo]:
    out: dict[str, QualiInfo] = {}
    for q in race.get("QualifyingResults", []):
        times = [parse_lap_time(q.get(k)) for k in ("Q1", "Q2", "Q3")]
        times = [t for t in times if t is not None]
        out[q["Driver"]["driverId"]] = QualiInfo(
            position=_to_int(q.get("position")),
            time=min(times) if times else None,
        )
    return out


def _build_quali_index(payload: dict) -> dict[int, dict[str, QualiInfo]]:
    out: dict[int, dict[str, QualiInfo]] = {}
    for race in payload.get("races", []):
        parsed = _parse_quali_map(race)
        if parsed:
            out[int(race["round"])] = parsed
    return out


def _build_sprint_index(payload: dict) -> dict[int, dict[str, int]]:
    out: dict[int, dict[str, int]] = {}
    for race in payload.get("races", []):
        parsed = {
            r["Driver"]["driverId"]: pos
            for r in race.get("SprintResults", [])
            if (pos := _to_int(r.get("position"))) is not None
        }
        if parsed:
            out[int(race["round"])] = parsed
    return out


def load_races(
    start_season: int = START_SEASON, end_season: int | None = None
) -> list[Race]:
    """Load completed races with results, qualifying and sprint data."""
    if end_season is None:
        end_season = current_season()

    races: list[Race] = []
    for season in range(start_season, end_season + 1):
        results = _fetch_endpoint(season, "results")
        qualifying = _fetch_endpoint(season, "qualifying")
        sprint = _fetch_endpoint(season, "sprint")
        quali_index = _build_quali_index(qualifying)
        sprint_index = _build_sprint_index(sprint)

        for race in results.get("races", []):
            rnd = int(race["round"])
            quali_map = quali_index.get(rnd, {})
            sprint_map = sprint_index.get(rnd, {})
            entries: list[Entry] = []
            for r in race.get("Results", []):
                status = r.get("status", "")
                pos_text = str(r.get("positionText", ""))
                finish = int(r["position"]) if pos_text.isdigit() else None
                driver = r["Driver"]
                constructor = r["Constructor"]
                entries.append(
                    Entry(
                        driver_id=driver["driverId"],
                        code=driver.get("code") or driver["familyName"][:3].upper(),
                        name=f"{driver['givenName']} {driver['familyName']}",
                        team_id=constructor["constructorId"],
                        team_name=constructor["name"],
                        grid=_to_int(r.get("grid")),
                        finish=finish,
                        classified=_is_classified(status),
                        points=float(r.get("points", 0) or 0),
                        status=status,
                        quali=quali_map.get(driver["driverId"]),
                        sprint_pos=sprint_map.get(driver["driverId"]),
                    )
                )
            if not entries:
                continue
            circuit = race["Circuit"]
            races.append(
                Race(
                    season=season,
                    round=rnd,
                    name=race["raceName"],
                    circuit_id=circuit["circuitId"],
                    circuit_name=circuit["circuitName"],
                    country=circuit.get("Location", {}).get("country", ""),
                    date=race.get("date", ""),
                    has_sprint=bool(sprint_map),
                    entries=entries,
                )
            )
    return races


def fetch_next_race() -> dict | None:
    payload = _http_json(f"{API_BASE}/current/next.json")
    races = payload["MRData"]["RaceTable"]["Races"]
    return races[0] if races else None


def fetch_future_qualifying(season: int, round_number: int) -> dict[str, QualiInfo]:
    """Return qualifying results for a round if they have been published."""
    try:
        payload = _http_json(f"{API_BASE}/{season}/{round_number}/qualifying.json")
    except RuntimeError:
        return {}
    races = payload["MRData"]["RaceTable"].get("Races", [])
    if not races:
        return {}
    return _parse_quali_map(races[0])
