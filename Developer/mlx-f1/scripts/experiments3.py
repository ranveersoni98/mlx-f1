"""Third round: feature ablations and hyperparameters for the pre-quali model."""

from __future__ import annotations

from mlx_f1.features import FEATURE_NAMES
from experiments import ids
from experiments2 import PRE_CORE, run

REMOVE = lambda names: [i for i in PRE_CORE if FEATURE_NAMES[i] not in names]

PRE_NO_SEASON = REMOVE({"driver_season_points", "team_season_points"})
PRE_NO_CIRCUIT = REMOVE({"circuit_driver_finish", "circuit_team_finish"})
PRE_NO_TEAMMATE = REMOVE({"teammate_quali_delta"})
PRE_NO_DNF = REMOVE({"driver_dnf_10", "team_dnf_10"})
PRE_PLUS = PRE_CORE + ids("driver_last_finish", "driver_exp", "round_frac")

if __name__ == "__main__":
    ablations = [
        ("pre core", PRE_CORE),
        ("pre no season pts", PRE_NO_SEASON),
        ("pre no circuit", PRE_NO_CIRCUIT),
        ("pre no teammate", PRE_NO_TEAMMATE),
        ("pre no dnf", PRE_NO_DNF),
        ("pre plus last/exp/round", PRE_PLUS),
    ]
    for label, idxs in ablations:
        run(label, idxs, "prior", hidden=32, weight_decay=1e-3, pl_weight=0.3)

    print()
    for hidden, wd, pl in [
        (16, 1e-3, 0.3),
        (24, 1e-3, 0.3),
        (32, 1e-4, 0.3),
        (32, 1e-2, 0.3),
        (32, 1e-3, 0.0),
        (32, 1e-3, 0.15),
        (32, 1e-3, 0.5),
        (48, 1e-3, 0.3),
        (32, 1e-3, 0.3),
    ]:
        run(
            f"pre core h{hidden} wd{wd:g} pl{pl:g}",
            PRE_CORE,
            "prior",
            hidden=hidden,
            weight_decay=wd,
            pl_weight=pl,
        )
