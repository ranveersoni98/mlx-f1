"""Train and backtest the MLX race-winner model."""

from __future__ import annotations

import json
import time
from pathlib import Path

import mlx.core as mx
import mlx.nn as nn
import mlx.optimizers as optim
import numpy as np

from .data import PROCESSED_DIR, START_SEASON, Race, load_races
from .features import (
    FEATURE_NAMES,
    N_FEATURES,
    PL_K,
    RaceSample,
    Scaler,
    build_circuit_map,
    build_samples,
    fit_scaler,
    make_arrays,
)
from .model import RaceNet, loss_terms, probabilities, total_loss

TRAIN_END = 2023
VAL_SEASONS = (2024, 2025)
TEST_START = 2026

MODEL_PATH = PROCESSED_DIR / "model.safetensors"
CONFIG_PATH = PROCESSED_DIR / "model_config.json"
SCALER_PATH = PROCESSED_DIR / "scaler.json"
CIRCUITS_PATH = PROCESSED_DIR / "circuits.json"
METRICS_PATH = PROCESSED_DIR / "metrics.json"


def _mx_arrays(arrays: dict[str, np.ndarray]) -> tuple:
    return (
        mx.array(arrays["x"]),
        mx.array(arrays["mask"]),
        mx.array(arrays["winner"]),
        mx.array(arrays["circuit"]),
        mx.array(arrays["rank_idx"]),
        mx.array(arrays["rank_valid"]),
    )


def _evaluation_loss(model: RaceNet, arrays: dict[str, np.ndarray]) -> tuple[float, float]:
    x, mask, winner, circuit, rank_idx, rank_valid = _mx_arrays(arrays)
    total, win_loss, _ = loss_terms(
        model, x, mask, winner, circuit, rank_idx, rank_valid
    )
    return float(total.item()), float(win_loss.item())


def _metrics(
    model: RaceNet, arrays: dict[str, np.ndarray], samples: list[RaceSample]
) -> dict[str, float]:
    x, mask, _, circuit, _, _ = _mx_arrays(arrays)
    probs = np.asarray(probabilities(model, x, mask, circuit))

    top1 = top3 = 0
    reciprocal_rank = 0.0
    log_loss = 0.0
    count = 0
    for i, sample in enumerate(samples):
        if sample.winner is None:
            continue
        row = probs[i, : len(sample.drivers)]
        order = np.argsort(-row)
        rank = int(np.where(order == sample.winner)[0][0]) + 1
        top1 += int(rank == 1)
        top3 += int(rank <= 3)
        reciprocal_rank += 1.0 / rank
        log_loss -= float(np.log(max(row[sample.winner], 1e-12)))
        count += 1

    return {
        "races": count,
        "top1": top1 / count if count else 0.0,
        "top3": top3 / count if count else 0.0,
        "mrr": reciprocal_rank / count if count else 0.0,
        "log_loss": log_loss / count if count else 0.0,
    }


def _baseline(
    samples: list[RaceSample], feature_index: int
) -> dict[str, float]:
    """Pick the lowest raw value of a feature (grid, quali, rolling finish)."""
    top1 = top3 = 0
    count = 0
    for sample in samples:
        if sample.winner is None:
            continue
        column = sample.rows_known[:, feature_index]
        scores = np.where(np.isnan(column), np.inf, column)
        order = np.argsort(scores)
        rank = int(np.where(order == sample.winner)[0][0]) + 1
        top1 += int(rank == 1)
        top3 += int(rank <= 3)
        count += 1
    return {
        "top1": top1 / count if count else 0.0,
        "top3": top3 / count if count else 0.0,
    }


def _backtest_rows(
    model: RaceNet, arrays: dict[str, np.ndarray], samples: list[RaceSample]
) -> list[dict]:
    x, mask, _, circuit, _, _ = _mx_arrays(arrays)
    probs = np.asarray(probabilities(model, x, mask, circuit))
    rows = []
    for i, sample in enumerate(samples):
        if sample.winner is None:
            continue
        field_probs = probs[i, : len(sample.drivers)]
        pick = int(np.argmax(field_probs))
        rows.append(
            {
                "season": sample.season,
                "round": sample.round,
                "race": sample.race_name,
                "winner": sample.names[sample.winner],
                "pick": sample.names[pick],
                "pick_team": sample.teams[pick],
                "winner_prob": float(field_probs[sample.winner]),
                "pick_prob": float(field_probs[pick]),
                "hit": bool(pick == sample.winner),
            }
        )
    return rows


def fit(
    samples: list[RaceSample],
    scaler: Scaler,
    circuit_map: dict[str, int],
    *,
    eval_samples: list[RaceSample] | None = None,
    max_epochs: int = 300,
    batch_size: int = 32,
    learning_rate: float = 1e-3,
    weight_decay: float = 1e-4,
    patience: int = 50,
    seed: int = 42,
    verbose: bool = True,
) -> tuple[RaceNet, dict]:
    known = make_arrays(samples, scaler, circuit_map, variant="known")
    prior = make_arrays(samples, scaler, circuit_map, variant="prior")

    eval_known = eval_prior = None
    if eval_samples:
        eval_known = make_arrays(eval_samples, scaler, circuit_map, variant="known")
        eval_prior = make_arrays(eval_samples, scaler, circuit_map, variant="prior")

    model = RaceNet(
        n_features=N_FEATURES,
        n_circuits=len(circuit_map) + 1,
        hidden=64,
        embed_dim=8,
    )
    optimizer = optim.AdamW(
        learning_rate=learning_rate, weight_decay=weight_decay
    )
    loss_and_grad = nn.value_and_grad(model, total_loss)
    rng = np.random.default_rng(seed)

    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    checkpoint = PROCESSED_DIR / "_best.safetensors"
    best_loss = float("inf")
    best_epoch = 0
    bad_epochs = 0
    epoch = 0
    started = time.time()

    for epoch in range(1, max_epochs + 1):
        order = rng.permutation(len(samples))
        running = 0.0
        batches = 0
        for start in range(0, len(samples), batch_size):
            index = order[start : start + batch_size]
            use_prior = rng.random(len(index)) < 0.25
            batch_x = np.where(
                use_prior[:, None, None],
                prior["x"][index],
                known["x"][index],
            )
            loss, grads = loss_and_grad(
                model,
                mx.array(batch_x),
                mx.array(known["mask"][index]),
                mx.array(known["winner"][index]),
                mx.array(known["circuit"][index]),
                mx.array(known["rank_idx"][index]),
                mx.array(known["rank_valid"][index]),
            )
            optimizer.update(model, grads)
            mx.eval(model.parameters(), optimizer.state, loss)
            running += float(loss.item())
            batches += 1

        train_loss = running / max(batches, 1)
        val_loss = train_loss
        if eval_known is not None and eval_prior is not None:
            known_loss, _ = _evaluation_loss(model, eval_known)
            prior_loss, _ = _evaluation_loss(model, eval_prior)
            val_loss = 0.5 * (known_loss + prior_loss)

            if val_loss < best_loss - 1e-5:
                best_loss = val_loss
                best_epoch = epoch
                bad_epochs = 0
                model.save_weights(str(checkpoint))
            else:
                bad_epochs += 1

        if verbose and (epoch == 1 or epoch % 25 == 0):
            elapsed = time.time() - started
            print(
                f"  epoch {epoch:3d}/{max_epochs}  train {train_loss:.4f}  "
                f"val {val_loss:.4f}  best {best_loss:.4f}  {elapsed:5.1f}s"
            )

        if eval_known is not None and bad_epochs >= patience:
            if verbose:
                print(f"  early stop at epoch {epoch}")
            break

    if eval_known is not None:
        model.load_weights(str(checkpoint))
    checkpoint.unlink(missing_ok=True)

    info = {
        "epochs_run": epoch,
        "best_epoch": best_epoch or epoch,
        "best_val": best_loss if best_loss < float("inf") else train_loss,
    }
    return model, info


def _save_artifacts(
    model: RaceNet,
    scaler: Scaler,
    circuit_map: dict[str, int],
    metrics: dict,
) -> None:
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    model.save_weights(str(MODEL_PATH))
    scaler.save(SCALER_PATH)
    CIRCUITS_PATH.write_text(json.dumps(circuit_map, indent=2))
    CONFIG_PATH.write_text(
        json.dumps(
            {
                "n_features": N_FEATURES,
                "n_circuits": len(circuit_map) + 1,
                "hidden": 64,
                "embed_dim": 8,
                "features": FEATURE_NAMES,
                "pl_k": PL_K,
            },
            indent=2,
        )
    )
    METRICS_PATH.write_text(json.dumps(metrics, indent=2))


def _print_metrics(label: str, values: dict[str, float]) -> None:
    print(
        f"  {label:<22} top1 {values['top1']:5.1%}  "
        f"top3 {values['top3']:5.1%}  "
        f"MRR {values.get('mrr', 0):.3f}  "
        f"logloss {values.get('log_loss', 0):.3f}"
    )


def run_training(
    start_season: int = START_SEASON,
    *,
    max_epochs: int = 300,
    verbose: bool = True,
) -> dict:
    print(f"Loading F1 data from {start_season} to present ...")
    races = load_races(start_season)
    samples = build_samples(races)
    print(f"  {len(races)} races, {sum(len(s.drivers) for s in samples)} driver-race rows")

    train = [s for s in samples if s.season <= TRAIN_END]
    val = [s for s in samples if s.season in VAL_SEASONS]
    test = [s for s in samples if s.season >= TEST_START]
    if not train or not val or not test:
        # Fall back to a time-ordered split if the calendar has moved on.
        cutoff = max(1, int(len(samples) * 0.8))
        train, val, test = samples[:cutoff], samples[cutoff:], samples[cutoff:]

    print(
        f"  train {len(train)} races (through {train[-1].season}), "
        f"validate {len(val)} races ({val[0].season}-{val[-1].season}), "
        f"test {len(test)} races ({test[0].season}+)"
    )

    scaler = fit_scaler(train)
    circuit_map = build_circuit_map(train)

    print("Training holdout model ...")
    model, info = fit(
        train,
        scaler,
        circuit_map,
        eval_samples=val,
        max_epochs=max_epochs,
        verbose=verbose,
    )

    test_known = make_arrays(test, scaler, circuit_map, variant="known")
    test_prior = make_arrays(test, scaler, circuit_map, variant="prior")
    metrics_known = _metrics(model, test_known, test)
    metrics_prior = _metrics(model, test_prior, test)

    pole = _baseline(test, FEATURE_NAMES.index("grid_pos"))
    quali = _baseline(test, FEATURE_NAMES.index("quali_pos"))
    form = _baseline(test, FEATURE_NAMES.index("driver_finish_5"))
    random_top1 = float(
        np.mean([1.0 / len(s.drivers) for s in test if s.winner is not None])
    )

    print(f"\nBacktest {test[0].season} ({len(test)} races)")
    _print_metrics("model pre-quali", metrics_prior)
    _print_metrics("model post-quali", metrics_known)
    _print_metrics("baseline: pole", {**pole, "mrr": 0, "log_loss": 0})
    _print_metrics("baseline: quali", {**quali, "mrr": 0, "log_loss": 0})
    _print_metrics("baseline: form", {**form, "mrr": 0, "log_loss": 0})
    print(f"  baseline: random       top1 {random_top1:5.1%}")

    backtest = _backtest_rows(model, test_prior, test)
    if backtest:
        print("\nRace-by-race picks (pre-qualifying model):")
        for row in backtest:
            mark = "hit " if row["hit"] else "miss"
            print(
                f"  R{row['round']:02d} {row['race'][:24]:<24} "
                f"winner {row['winner'][:20]:<20} "
                f"pick {row['pick'][:20]:<20} "
                f"p(win) {row['winner_prob']:5.1%}  {mark}"
            )

    # Refit on every available race for the actual next-race prediction.
    print("\nRefitting on all data for the next race ...")
    scaler_all = fit_scaler(samples)
    circuit_map_all = build_circuit_map(samples)
    final_epochs = max(30, info["best_epoch"])
    final_model, _ = fit(
        samples,
        scaler_all,
        circuit_map_all,
        eval_samples=None,
        max_epochs=final_epochs,
        verbose=False,
    )

    last = samples[-1]
    metrics = {
        "trained_through": {"season": last.season, "round": last.round},
        "races": len(races),
        "train_races": len(train),
        "val_races": len(val),
        "test_races": len(test),
        "best_epoch": info["best_epoch"],
        "holdout": {
            "pre_quali": metrics_prior,
            "post_quali": metrics_known,
        },
        "baselines": {
            "pole": pole,
            "quali": quali,
            "form": form,
            "random_top1": random_top1,
        },
        "backtest": backtest,
    }
    _save_artifacts(final_model, scaler_all, circuit_map_all, metrics)
    print(f"\nSaved model + metrics to {PROCESSED_DIR}")
    return metrics
