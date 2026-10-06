"""Serial, explicitly selected GPU launcher. Never signals another process."""

import argparse
import fcntl
import json
import os
import subprocess
import sys
import time
from pathlib import Path

MODELS = ("gemma3", "parakeet", "dinov3-convnext", "qwen35", "dinov3-vit", "siglip2")


def wait_for_gpu(gpu):
    waiting = False
    while True:
        result = subprocess.run(
            [
                "nvidia-smi",
                "-i",
                str(gpu),
                "--query-compute-apps=pid",
                "--format=csv,noheader,nounits",
            ],
            check=True,
            capture_output=True,
            text=True,
        )
        if not result.stdout.strip():
            if waiting:
                print(
                    f"GPU {gpu} is free; beginning the next check/control.", flush=True
                )
            return
        if not waiting:
            print(
                f"Waiting for the existing GPU {gpu} compute job. No signals or queue changes.",
                flush=True,
            )
            waiting = True
        time.sleep(30)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--candidate", type=Path, required=True)
    parser.add_argument("--baseline", type=Path, required=True)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--inputs", type=Path, required=True)
    parser.add_argument("--dfloat", type=Path, required=True)
    parser.add_argument("--dependencies", type=Path, required=True)
    parser.add_argument("--cache", type=Path, required=True)
    parser.add_argument("--phase", choices=("pilot", "campaigns", "ablations"), required=True)
    parser.add_argument("--ablation-root", type=Path)
    parser.add_argument("--reference-root", type=Path)
    parser.add_argument("--gpu", type=int, choices=(0, 1), default=1)
    args = parser.parse_args()
    if args.phase == "ablations":
        assert args.ablation_root and args.reference_root, "Frozen setups and completed GPU campaigns required"
    args.root.mkdir(parents=True, exist_ok=True)
    lock = (args.root / "launcher.lock").open("a")
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    environment = os.environ.copy()
    environment.update(
        CUDA_VISIBLE_DEVICES=str(args.gpu),
        HF_HUB_OFFLINE="1",
        TRANSFORMERS_OFFLINE="1",
        HF_HUB_CACHE=str(args.cache),
        OMP_NUM_THREADS="8",
        MKL_NUM_THREADS="8",
        PYTHONDONTWRITEBYTECODE="1",
    )
    scripts = Path(__file__).parent
    output = args.root / "results" / args.candidate.name

    def run(source, arguments):
        wait_for_gpu(args.gpu)
        environment["PYTHONPATH"] = os.pathsep.join(
            map(str, (source, args.data_root, args.dfloat, args.dependencies))
        )
        print(
            json.dumps({"source": source.name, "arguments": list(map(str, arguments))}),
            flush=True,
        )
        subprocess.run(
            [sys.executable, *map(str, arguments)],
            env=environment,
            cwd=source,
            check=True,
        )

    check_path = output / "correctness.json"
    output.mkdir(parents=True, exist_ok=True)
    if args.phase != "ablations" and not check_path.exists():
        run(
            args.candidate,
            [
                scripts / "check_packed_gpu.py",
                "--reference",
                args.root / "baseline-cpu.json",
                "--output",
                check_path,
            ],
        )
    modules_path = output / "module-correctness.json"
    if args.phase != "ablations" and not modules_path.exists():
        run(
            args.candidate,
            [
                scripts / "check_gpu_weight_modules.py",
                "--cuda",
                "--output",
                modules_path,
            ],
        )
    campaigns = ("pilot",) if args.phase == "pilot" else ("1", "2")
    for campaign in campaigns:
        models = MODELS[:3] if campaign == "pilot" else MODELS
        for model in models:
            variants = [
                ("baseline", "own"),
                ("baseline", "transfer"),
                ("candidate", "own"),
                ("candidate", "transfer"),
            ]
            if args.phase == "ablations":
                variants = [(v, m) for v in ("baseline", "candidate") for m in ("sharing", "packing")]
            if campaign == "2":
                variants = variants[2:] + variants[:2]
            controls = [] if args.phase == "ablations" else [("control", "bf16"), ("control", "dfloat11")]
            for variant, method in [*controls, *variants]:
                path = output / campaign / f"{model}-{variant}-{method}.json"
                if path.exists():
                    continue
                source = args.candidate if variant == "candidate" else args.baseline
                run(
                    source,
                    [
                        scripts / "bench_packed_gpu.py",
                        model,
                        method,
                        "--variant",
                        variant,
                        "--campaign",
                        campaign,
                        "--data-root",
                        args.data_root,
                        "--inputs",
                        args.inputs,
                        "--output",
                        output,
                        "--gpu",
                        args.gpu,
                        *(["--ablation-root", args.ablation_root, "--reference-root", args.reference_root]
                          if args.phase == "ablations" else []),
                    ],
                )
    if args.phase in ("campaigns", "ablations"):
        run(
            args.candidate,
            [
                scripts / "check_packed_gpu_results.py",
                output,
                "--baseline-cpu",
                args.root / "baseline-cpu.json",
                *(["--ablations", "--reference-root", args.reference_root]
                  if args.phase == "ablations" else []),
            ],
        )
    print(json.dumps({"phase_finished": args.phase, "output": str(output)}), flush=True)


if __name__ == "__main__":
    main()
