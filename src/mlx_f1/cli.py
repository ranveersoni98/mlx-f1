"""Command-line interface for mlx-f1."""

from __future__ import annotations

import argparse

from .data import START_SEASON, load_races
from .predict import run_prediction
from .train import run_training


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="mlx-f1",
        description="Predict Formula 1 race winners with an MLX model.",
    )
    subparsers = parser.add_subparsers(dest="command")

    fetch = subparsers.add_parser("fetch", help="download and cache F1 data")
    fetch.add_argument(
        "--start", type=int, default=START_SEASON, help="first season to load"
    )

    train = subparsers.add_parser(
        "train", help="train the model and run a season backtest"
    )
    train.add_argument(
        "--start", type=int, default=START_SEASON, help="first season to load"
    )
    train.add_argument(
        "--epochs", type=int, default=300, help="maximum training epochs"
    )
    train.add_argument("--quiet", action="store_true", help="hide epoch logs")

    predict = subparsers.add_parser(
        "predict", help="predict the winner of the next Grand Prix"
    )
    predict.add_argument("--json", action="store_true", help="print JSON output")

    return parser


def main(argv: list[str] | None = None) -> None:
    parser = build_parser()
    args = parser.parse_args(argv)

    if args.command == "fetch":
        races = load_races(args.start)
        rows = sum(len(race.entries) for race in races)
        print(f"Cached {len(races)} races ({rows} driver-race rows) from {args.start}.")
    elif args.command == "train":
        run_training(
            args.start,
            max_epochs=args.epochs,
            verbose=not args.quiet,
        )
    elif args.command == "predict":
        run_prediction(as_json=args.json)
    else:
        parser.print_help()
