"""Second round: specialized pre/post-qualifying models."""

from __future__ import annotations

import time

import mlx.core as mx
import mlx.nn as nn
import mlx.optimizers as optim
import numpy as np

from mlx_f1.data import PROCESSED_DIR
from mlx_f1.features import (
    FEATURE_NAMES,
    build_circuit_map,
    fit_scaler,
    make_arrays,
)
from mlx_f1.model import RaceNet, loss_terms, probabilities
from experiments import ALL, CIRCUITS, SCALER, TEST, TRAIN, VAL, ids, metrics


def sliced(arrays, indices):
    out = dict(arrays)
    out["x"] = arrays["x"][:, :, indices]
    return out


POST_MINIMAL = ids(
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
POST_DNF = POST_MINIMAL + ids("driver_dnf_10", "team_dnf_10")

PRE_NAMES = [
    name
    for name in FEATURE_NAMES
    if name
    not in ("grid_pos", "quali_pos", "quali_gap", "grid_penalty", "quali_known")
]
PRE_ALL = [FEATURE_NAMES.index(name) for name in PRE_NAMES]
PRE_CORE = ids(
    "driver_finish_5",
    "driver_quali_5",
    "driver_points_5",
    "driver_gap_5",
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
PRE_LEAN = ids(
    "driver_finish_5",
    "driver_quali_5",
    "driver_gap_5",
    "team_finish_5",
    "team_quali_5",
    "circuit_driver_finish",
    "circuit_team_finish",
    "teammate_quali_delta",
)
PRE_FORM = ids(
    "driver_finish_5",
    "driver_quali_5",
    "driver_points_5",
    "driver_gap_5",
    "driver_dnf_10",
    "team_finish_5",
    "team_quali_5",
    "team_dnf_10",
    "teammate_quali_delta",
)


def run(
    name,
    indices,
    variant,
    hidden=64,
    weight_decay=1e-4,
    pl_weight=0.3,
    lr=1e-3,
    max_epochs=250,
    patience=40,
    seed=42,
):
    train = sliced(make_arrays(TRAIN, SCALER, CIRCUITS, variant=variant), indices)
    val = sliced(make_arrays(VAL, SCALER, CIRCUITS, variant=variant), indices)
    test = sliced(make_arrays(TEST, SCALER, CIRCUITS, variant=variant), indices)

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
    checkpoint = str(PROCESSED_DIR / f"_exp2_{name.replace(' ', '_')}.safetensors")
    start = time.time()

    for epoch in range(1, max_epochs + 1):
        order = rng.permutation(len(TRAIN))
        for begin in range(0, len(TRAIN), 32):
            index = order[begin : begin + 32]
            loss, grads = loss_and_grad(
                model,
                mx.array(train["x"][index]),
                mx.array(train["mask"][index]),
                mx.array(train["winner"][index]),
                mx.array(train["circuit"][index]),
                mx.array(train["rank_idx"][index]),
                mx.array(train["rank_valid"][index]),
            )
            optimizer.update(model, grads)
            mx.eval(model.parameters(), optimizer.state, loss)

        vm = metrics(model, val, VAL)
        score = vm["top1"] + vm["mrr"]
        if score > best:
            best = score
            best_epoch = epoch
            bad = 0
            model.save_weights(checkpoint)
        else:
            bad += 1
        if bad >= patience:
            break

    model.load_weights(checkpoint)
    tm = metrics(model, test, TEST)
    print(
        f"{name:<36} {variant:<6} ep {best_epoch:3d}  "
        f"top1 {tm['top1']:5.1%}  top3 {tm['top3']:5.1%}  "
        f"mrr {tm['mrr']:.3f}  ll {tm['logloss']:.3f}  ({time.time() - start:.0f}s)"
    )
    return model, tm


if __name__ == "__main__":
    configs = [
        ("post minimal known-only", POST_MINIMAL, "known", 64, 1e-4, 0.3),
        ("post minimal+dnf known", POST_DNF, "known", 64, 1e-4, 0.3),
        ("post minimal known h32", POST_MINIMAL, "known", 32, 1e-3, 0.3),
        ("pre all", PRE_ALL, "prior", 64, 1e-3, 0.3),
        ("pre core", PRE_CORE, "prior", 64, 1e-3, 0.3),
        ("pre lean", PRE_LEAN, "prior", 64, 1e-3, 0.3),
        ("pre form", PRE_FORM, "prior", 64, 1e-3, 0.3),
        ("pre core h32", PRE_CORE, "prior", 32, 1e-3, 0.3),
        ("pre all h32", PRE_ALL, "prior", 32, 1e-3, 0.3),
    ]
    for label, idxs, variant, hidden, wd, plw in configs:
        run(label, idxs, variant, hidden=hidden, weight_decay=wd, pl_weight=plw)
