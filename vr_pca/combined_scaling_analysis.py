#!/usr/bin/env python3
"""Aggregate scaling experiments into summary tables."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Iterable

import pandas as pd

_ROOT = Path(__file__).resolve().parents[1]
_SRC = _ROOT / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))


def summarise_frame(frame: pd.DataFrame) -> pd.DataFrame:
    metrics = frame.groupby(["solver", "width", "samples"], as_index=False).agg(
        eigenvalue_mean=("eigenvalue", "mean"),
        eigenvalue_std=("eigenvalue", "std"),
        elapsed_mean=("elapsed", "mean"),
        elapsed_std=("elapsed", "std"),
        iterations_mean=("iterations", "mean"),
    )
    return metrics.sort_values(["solver", "width", "samples"]).reset_index(drop=True)


def main(argv: Iterable[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("inputs", nargs="+", type=Path, help="input CSV files")
    parser.add_argument("--output", type=Path, default=None, help="optional summary CSV path")
    args = parser.parse_args(list(argv) if argv is not None else None)

    frames = []
    for path in args.inputs:
        frame = pd.read_csv(path)
        frame["source"] = path.stem
        frames.append(frame)
    data = pd.concat(frames, ignore_index=True)

    summary = summarise_frame(data)
    print(summary.to_string(index=False))

    if args.output is not None:
        summary.to_csv(args.output, index=False)
        print(f"Saved summary to {args.output}")


if __name__ == "__main__":  # pragma: no cover
    main()
