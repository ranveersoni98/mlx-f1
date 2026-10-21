"""Scratch hyperparameter/feature experiments (not part of the package)."""

from __future__ import annotations

import time
from pathlib import Path

import mlx.core as mx
import mlx.nn as nn
import mlx.optimizers as optim
import numpy as np

from mlx_f1.data import PROCESSED_DIR, load_races
from mlx_f1.features import (
    FEATURE_NAMES,
    build_circuit_map,
    build_samples,
    fit_scaler,
    make_arrays,
)
from mlx_f1.model import RaceNet, loss_terms, probabilities
from mlx_f1.train import TEST_START, TRAIN_END, VAL_SEASONS

RACES = load_races()
SAMPLES = build_samples(RACES)
TRAIN = [s for s in SAMPLES if s.season <= TRAIN_END]
VAL = [s for s in SAMPLES if s.season in VAL_SEASONS]
TEST = [s for s in SAMPLES if s.season >= TEST_START]

SCALER = fit_scaler(TRAIN)
CIRCUITS = build_circuit_map(TRAIN)


def ids(*names: str) -> list[int]:
    return [FEATURE_NAMES.index(name) for name in names]


CORE = ids(
    "grid_pos",
    "quali_pos",
    "quali_gap",
    "grid_penalty",
    "quali_known",
    "driver_finish_5",
    "driver_quali_5",
    "driver_dnf_10",
    "team_finish_5",
    "team_quali_5",
    "team_dnf_10",
    "circuit_driver_finish",
    "circuit_team_finish",
    "teammate_quali_delta",
    "driver_season_points",
    "team_season_points",
)

MINIMAL = ids(
    "grid_pos",
    "quali_pos",
    "quali_gap",
    "grid_penalty",
    "quali_known",
    "driver_finish_5",
    "driver_quali_5",
    "team_finish_5",
    "team_quali_5",
    "circuit_driver_finish",
    "circuit_team_finish",
    "teammate_quali_delta",
)

ALL = list(range(len(FEATURE_NAMES)))
SETS = {"all": ALL, "core": CORE, "minimal": MINIMAL}


def metrics(model, arrays, samples):
    x = mx.array(arrays["x"])
    probs = np.asarray(
        probabilities(model, x, mx.array(arrays["mask"]), mx.array(arrays["circuit"]))
    )
    top1 = top3 = 0
    rr = 0.0
    ll = 0.0
    count = 0
    for i, sample in enumerate(samples):
        if sample.winner is None:
            continue
        row = probs[i, : len(sample.drivers)]
        order = np.argsort(-row)
        rank = int(np.where(order == sample.winner)[0][0]) + 1
        top1 += rank == 1
        top3 += rank <= 3
        rr += 1.0 / rank
        ll -= float(np.log(max(row[sample.winner], 1e-12)))
        count += 1
    return {
        "top1": top1 / count,
        "top3": top3 / count,
        "mrr": rr / count,
        "logloss": ll / count,
    }


def sliced(arrays, indices):
    out = dict(arrays)
    out["x"] = arrays["x"][:, :, indices]
    return out


def run(
    name,
    indices,
    hidden=64,
    weight_decay=1e-4,
    pl_weight=0.3,
    lr=1e-3,
    max_epochs=250,
    patience=40,
    seed=42,
    verbose=False,
):
    known = sliced(make_arrays(TRAIN, SCALER, CIRCUITS, variant="known"), indices)
    prior = sliced(make_arrays(TRAIN, SCALER, CIRCUITS, variant="prior"), indices)
    val_known = sliced(make_arrays(VAL, SCALER, CIRCUITS, variant="known"), indices)
    val_prior = sliced(make_arrays(VAL, SCALER, CIRCUITS, variant="prior"), indices)
    test_known = sliced(make_arrays(TEST, SCALER, CIRCUITS, variant="known"), indices)
    test_prior = sliced(make_arrays(TEST, SCALER, CIRCUITS, variant="prior"), indices)

    model = RaceNet(len(indices), len(CIRCUITS) + 1, hidden=hidden, embed_dim=8)
    optimizer = optim.AdamW(learning_rate=lr, weight_decay=weight_decay)

    def loss_fn(model, x, mask, winner, circuit_idx, rank_idx, rank_valid):
        return loss_terms(
            model,
            x,
            mask,
            winner,
            circuit_idx,
            rank_idx,
            rank_valid,
            pl_weight=pl_weight,
        )[0]

    loss_and_grad = nn.value_and_grad(model, loss_fn)
    rng = np.random.default_rng(seed)

    best = -1.0
    best_epoch = 0
    bad = 0
    checkpoint = str(PROCESSED_DIR / f"_exp_{name.replace(' ', '_')}.safetensors")
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    start = time.time()

    for epoch in range(1, max_epochs + 1):
        order = rng.permutation(len(TRAIN))
        for begin in range(0, len(TRAIN), 32):
            index = order[begin : begin + 32]
            use_prior = rng.random(len(index)) < 0.25
            batch_x = np.where(
                use_prior[:, None, None], prior["x"][index], known["x"][index]
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

        vk = metrics(model, val_known, VAL)
        vp = metrics(model, val_prior, VAL)
        score = 0.5 * (vk["mrr"] + vp["mrr"]) + 0.5 * (vk["top1"] + vp["top1"])
        if score > best:
            best = score
            best_epoch = epoch
            bad = 0
            model.save_weights(checkpoint)
        else:
            bad += 1
        if bad >= patience:
            break
        if verbose and epoch % 25 == 0:
            print(
                f"  {name} epoch {epoch} val {vk['top1']:.3f}/{vp['top1']:.3f} "
                f"mrr {vk['mrr']:.3f}/{vp['mrr']:.3f}"
            )

    model.load_weights(checkpoint)
    tk = metrics(model, test_known, TEST)
    tp = metrics(model, test_prior, TEST)
    print(
        f"{name:<34} ep {best_epoch:3d}  "
        f"post {tk['top1']:5.1%}/{tk['top3']:5.1%} mrr {tk['mrr']:.3f} ll {tk['logloss']:.3f}  |  "
        f"pre {tp['top1']:5.1%}/{tp['top3']:5.1%} mrr {tp['mrr']:.3f} ll {tp['logloss']:.3f}  "
        f"({time.time() - start:.0f}s)"
    )
    return model, tk, tp


if __name__ == "__main__":
    configs = [
        ("all h64 wd1e-4 pl0.3", "all", 64, 1e-4, 0.3),
        ("core h64 wd1e-4 pl0.3", "core", 64, 1e-4, 0.3),
        ("minimal h64 wd1e-4 pl0.3", "minimal", 64, 1e-4, 0.3),
        ("core h32 wd1e-3 pl0.3", "core", 32, 1e-3, 0.3),
        ("core h64 wd1e-3 pl1.0", "core", 64, 1e-3, 1.0),
        ("minimal h32 wd1e-3 pl0.3", "minimal", 32, 1e-3, 0.3),
        ("all h64 wd1e-3 pl0.3", "all", 64, 1e-3, 0.3),
        ("core h16 wd1e-3 pl0.3", "core", 16, 1e-3, 0.3),
    ]
    for label, set_name, hidden, wd, plw in configs:
        run(
            label,
            SETS[set_name],
            hidden=hidden,
            weight_decay=wd,
            pl_weight=plw,
        )
