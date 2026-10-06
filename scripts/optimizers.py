"""Appendix optimizer plots: four real architectures, four optimizers, three seeds."""

import argparse
import fcntl
import os
from pathlib import Path

import torch

from neuralzip._experiment_inputs import write_new
from neuralzip._experiment_optimizers import (
    MODELS,
    OPTIMIZERS,
    check_saved,
    export_curves,
    load_data,
    run_one,
    verify_prefix,
)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--models", nargs="+", choices=MODELS, default=MODELS)
    p.add_argument("--optimizers", nargs="+", choices=OPTIMIZERS, default=OPTIMIZERS)
    p.add_argument("--seeds", nargs="+", type=int, default=[17, 29, 43])
    p.add_argument("--steps", type=int, default=1000)
    p.add_argument(
        "--gpu",
        type=int,
        default=0,
        help="Visible CUDA index; select physical GPU with CUDA_VISIBLE_DEVICES",
    )
    p.add_argument(
        "--cache", type=Path, default=Path(".cache/experiments/optimizer-data")
    )
    p.add_argument("--output", type=Path, default=Path("results/reruns/optimizers"))
    p.add_argument(
        "--check",
        action="store_true",
        help="Verify all published raw histograms; no training",
    )
    p.add_argument(
        "--export-saved",
        action="store_true",
        help="Write mean/SD CSV curves from saved traces; no training",
    )
    p.add_argument(
        "--verify-published",
        action="store_true",
        help="Require exact step-0..N histograms and initial state",
    )
    args = p.parse_args()
    if args.check or args.export_saved:
        check_saved(args.models, args.optimizers, args.seeds)
        if args.export_saved:
            export_curves(
                args.models, args.optimizers, args.seeds, args.output, saved=True
            )
        return
    assert 0 < args.steps <= 1000
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    torch.cuda.set_device(args.gpu)
    torch.set_num_threads(4)
    torch.set_num_interop_threads(4)
    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    args.output.mkdir(parents=True, exist_ok=True)
    with (args.output / "training.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        for key in args.models:
            data = load_data(key, args.cache, f"cuda:{args.gpu}")
            for seed in args.seeds:
                for kind in args.optimizers:
                    run_one(args.output, key, kind, seed, args.steps, data)
                    if args.verify_published:
                        proof = verify_prefix(args.output, key, kind, seed)
                        write_new(
                            args.output
                            / f"{key}-{kind}-seed{seed}"
                            / "published-prefix-check.json",
                            proof,
                        )
            del data
            torch.cuda.empty_cache()
        export_curves(args.models, args.optimizers, args.seeds, args.output)


if __name__ == "__main__":
    main()
