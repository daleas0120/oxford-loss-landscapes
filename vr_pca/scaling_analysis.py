#!/usr/bin/env python3
"""Run a grid of VR-PCA scaling experiments and persist the results."""

from __future__ import annotations

import argparse
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable, List

import pandas as pd
import torch
import torch.nn as nn

_ROOT = Path(__file__).resolve().parents[1]
_SRC = _ROOT / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from oxford_loss_landscapes.hessian.vrpca import VRPCAConfig

if __package__ is None or __package__ == "":  # pragma: no cover
    from pathlib import Path as _Path

    sys.path.append(str(_Path(__file__).resolve().parent.parent))
    from vr_pca.hessian_scaling import (  # type: ignore
        ExperimentResult,
        make_data,
        make_model,
        run_eigsh,
        run_vrpca,
    )
else:  # pragma: no cover
    from .hessian_scaling import ExperimentResult, make_data, make_model, run_eigsh, run_vrpca


@dataclass
class ExperimentSpec:
    width: int
    samples: int


def run_spec(spec: ExperimentSpec, repeats: int, config: VRPCAConfig, seed: int) -> List[dict]:
    results: List[dict] = []
    for repeat in range(repeats):
        torch.manual_seed(seed + repeat)
        model = make_model(spec.width, spec.width)
        data = make_data(spec.samples, spec.width, seed + repeat)
        criterion = nn.MSELoss()

        eigsh_result = run_eigsh(model, data["inputs"], data["targets"], criterion)
        vrpca_result = run_vrpca(model, data["inputs"], data["targets"], criterion, config=config)

        for record in (eigsh_result, vrpca_result):
            entry = asdict(record)
            entry.update(asdict(spec))
            entry["repeat"] = repeat
            results.append(entry)
    return results


def main(argv: Iterable[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--widths", type=int, nargs="+", default=[32, 64, 128], help="hidden layer widths")
    parser.add_argument("--samples", type=int, nargs="+", default=[256, 512], help="dataset sizes")
    parser.add_argument("--repeats", type=int, default=3, help="number of repeats per configuration")
    parser.add_argument("--epochs", type=int, default=10, help="VR-PCA epochs")
    parser.add_argument("--batch-size", type=int, default=128, help="VR-PCA batch size")
    parser.add_argument("--output", type=Path, default=Path("vrpca_scaling.csv"), help="output CSV path")
    parser.add_argument("--seed", type=int, default=0, help="base random seed")
    args = parser.parse_args(list(argv) if argv is not None else None)

    config = VRPCAConfig(batch_size=args.batch_size, epochs=args.epochs)
    all_results: List[dict] = []

    for width in args.widths:
        for samples in args.samples:
            spec = ExperimentSpec(width=width, samples=samples)
            all_results.extend(run_spec(spec, args.repeats, config=config, seed=args.seed))

    df = pd.DataFrame(all_results)
    df.to_csv(args.output, index=False)
    print(f"Saved results to {args.output}")


if __name__ == "__main__":  # pragma: no cover
    main()
