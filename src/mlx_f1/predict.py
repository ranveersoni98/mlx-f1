"""Predict the winner of the next Grand Prix."""

from __future__ import annotations

import json
from dataclasses import replace

import mlx.core as mx
import numpy as np

from .data import (
    Entry,
    Race,
    fetch_future_qualifying,
    fetch_next_race,
    load_races,
)
from .features import (
    N_FEATURES,
    Scaler,
    make_prediction_arrays,
    state_from_races,
)
from .model import RaceNet, probabilities
from .train import CIRCUITS_PATH, CONFIG_PATH, METRICS_PATH, MODEL_PATH


def load_artifacts():
    if not MODEL_PATH.exists():
        raise FileNotFoundError(
            "No trained model found. Run `uv run mlx-f1 train` first."
        )
    config = json.loads(CONFIG_PATH.read_text())
    scaler = Scaler.load(METRICS_PATH.parent / "scaler.json")
    circuit_map = json.loads(CIRCUITS_PATH.read_text())
    model = RaceNet(
        n_features=config.get("n_features", N_FEATURES),
        n_circuits=config["n_circuits"],
        hidden=config.get("hidden", 64),
        embed_dim=config.get("embed_dim", 8),
    )
    model.load_weights(str(MODEL_PATH))
    return model, scaler, circuit_map, config


def _next_race_object(payload: dict) -> Race:
    circuit = payload["Circuit"]
    return Race(
        season=int(payload["season"]),
        round=int(payload["round"]),
        name=payload["raceName"],
        circuit_id=circuit["circuitId"],
        circuit_name=circuit["circuitName"],
        country=circuit.get("Location", {}).get("country", ""),
        date=payload.get("date", ""),
        has_sprint="Sprint" in payload,
    )


def run_prediction(as_json: bool = False) -> dict | None:
    model, scaler, circuit_map, config = load_artifacts()
    next_payload = fetch_next_race()
    if next_payload is None:
        print("The current season has no upcoming race.")
        return None

    race = _next_race_object(next_payload)
    races = load_races()
    season_races = [r for r in races if r.season == race.season]
    source = (
        max(season_races, key=lambda r: r.round) if season_races else races[-1]
    )

    # Start from the most recent confirmed entry list. Qualifying/grid slots
    # are filled in only if they have actually been published.
    entries = [
        replace(
            entry,
            grid=None,
            finish=None,
            classified=True,
            points=0.0,
            status="",
            quali=None,
            sprint_pos=None,
        )
        for entry in source.entries
    ]
    quali_map = fetch_future_qualifying(race.season, race.round)
    if quali_map:
        for entry in entries:
            entry.quali = quali_map.get(entry.driver_id)
    race.entries = entries
    quali_known = bool(quali_map)

    state = state_from_races(races)
    rows = state.feature_rows(race, entries, quali_known=quali_known)
    x, mask, circuit = make_prediction_arrays(
        rows, race.circuit_id, scaler, circuit_map
    )
    probs = np.asarray(
        probabilities(model, mx.array(x), mx.array(mask), mx.array(circuit))
    )[0, : len(entries)]

    order = np.argsort(-probs)
    results = [
        {
            "driver": entries[i].name,
            "code": entries[i].code,
            "team": entries[i].team_name,
            "probability": float(probs[i]),
        }
        for i in order
    ]
    metrics = json.loads(METRICS_PATH.read_text()) if METRICS_PATH.exists() else {}
    trained_through = metrics.get("trained_through", {})
    payload = {
        "race": {
            "season": race.season,
            "round": race.round,
            "name": race.name,
            "circuit": race.circuit_name,
            "country": race.country,
            "date": race.date,
        },
        "mode": "post-qualifying" if quali_known else "pre-qualifying",
        "trained_through": trained_through,
        "predictions": results,
    }

    if as_json:
        print(json.dumps(payload, indent=2))
        return payload

    print(f"\nNext race: {race.name}")
    print(
        f"{race.circuit_name}, {race.country}  |  {race.date}  |  "
        f"round {race.round}/{race.season}"
    )
    if trained_through:
        print(
            f"Model trained through {trained_through.get('season')} "
            f"round {trained_through.get('round')}"
        )
    if quali_known:
        print("Mode: POST-QUALIFYING (actual grid and qualifying included)\n")
    else:
        print("Mode: PRE-QUALIFYING (grid estimated from current form)\n")

    print(f"  {'#':>2}  {'Driver':<22} {'Team':<22} {'Win %':>6}")
    print(f"  {'-' * 2}  {'-' * 22} {'-' * 22} {'-' * 6}")
    for rank, row in enumerate(results[:12], start=1):
        bar = "#" * int(round(row["probability"] * 40))
        print(
            f"  {rank:>2}  {row['driver'][:22]:<22} "
            f"{row['team'][:22]:<22} {row['probability']:6.1%}  {bar}"
        )
    top = results[0]
    print(f"\nPrediction: {top['driver']} ({top['team']}) - {top['probability']:.1%}")
    return payload
