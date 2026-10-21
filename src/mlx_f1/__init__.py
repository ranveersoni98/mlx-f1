"""mlx-f1: predict Formula 1 race winners with MLX."""

__version__ = "0.1.0"


def main() -> None:
    from .cli import main as _main

    _main()
