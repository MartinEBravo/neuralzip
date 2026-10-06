"""Pinned official DFloat11 CPU comparison, without importing its CUDA package."""

import argparse
import contextlib
import os
from itertools import islice
from pathlib import Path

import dfloat11_utils as dfloat
import numpy as np
import torch

from neuralzip._experiment_cpu import configure
from neuralzip._experiment_inputs import (
    catalogue,
    evidence,
    file_sha,
    prepare,
    tensor_manifest,
    write_new,
)
from neuralzip.utils import time_call

OFFICIAL_COMMIT = "457733886ce6ebc6d8dda1621fad1ffa2661e028"
UTILS_SHA = "c50216fda9da2a516131ce84afec6003f566f6e0f88cb4fa8178bc18f83851fe"


def encode_tensor(tensor):
    _, frequencies = dfloat.get_codec(tensor.flatten())
    if len(frequencies) == 1:
        return None
    codec, _, table = dfloat.get_32bit_codec(frequencies)
    try:
        luts = dfloat.get_luts(table)
    except UnboundLocalError as error:
        if "curr_val" not in str(error):
            raise
        return None
    return (
        codec,
        luts,
        dfloat.encode_weights(
            [tensor.flatten()], codec, bytes_per_thread=8, threads_per_block=512
        ),
    )


def benchmark(key, out, cache, inputs, reference=None):
    assert file_sha(dfloat.__file__) == UTILS_SHA, (
        "Official DFloat11 CPU source differs"
    )
    named = prepare(key, cache, inputs)
    original = tensor_manifest(named)
    seconds = 0.0
    raw = stored = 0
    fallback = []
    with open(os.devnull, "w") as quiet:
        for name, tensor in named:
            raw += tensor.numel()
            with contextlib.redirect_stdout(quiet), contextlib.redirect_stderr(quiet):
                encoded, elapsed = time_call(encode_tensor, tensor)
            seconds += elapsed
            if encoded is None:
                stored += tensor.numel()
                fallback.append(name)
                continue
            codec, luts, (payload, other, positions, gaps, splits) = encoded
            stored += sum(
                t.numel() * t.element_size()
                for t in (payload, positions, gaps, splits, luts)
            )
            values = np.fromiter(
                islice(
                    codec.decode_streaming(payload.numpy().tobytes()), tensor.numel()
                ),
                dtype=np.uint16,
                count=tensor.numel(),
            )
            other = other.numpy().astype(np.uint16)
            restored = (values << 7) | (other & 127) | ((other & 128) << 8)
            assert np.array_equal(
                restored, tensor.flatten().view(torch.int16).numpy().view(np.uint16)
            ), name
    assert tensor_manifest(named) == original
    row = dict(
        model=key,
        method="dfloat11",
        input=catalogue()["inputs"][key],
        encode_seconds=seconds,
        exponent_ratio_pct=100 * stored / raw,
        exponent_raw_bytes=raw,
        exponent_compressed_bytes=stored,
        raw_fallback_tensors=fallback,
        roundtrip_exact=True,
        input_unchanged=True,
        official_commit=OFFICIAL_COMMIT,
        official_utils_sha256=UTILS_SHA,
        threads=torch.get_num_threads(),
        affinity=sorted(os.sched_getaffinity(0)),
        protocol="One full tensorwise encode, including frequency/LUT/restarts; validation outside timing",
    )
    if reference:
        old = evidence(reference)
        for field in (
            "exponent_raw_bytes",
            "exponent_compressed_bytes",
            "raw_fallback_tensors",
        ):
            assert row[field] == old[field], (key, field)
        row["published_compressed_bytes_exact"] = True
    write_new(Path(out) / f"{key}-dfloat11.json", row)
    print(key, "DFloat11", row["exponent_ratio_pct"], seconds, flush=True)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--model", required=True)
    for name in ("output", "cache", "inputs"):
        p.add_argument("--" + name, type=Path, required=True)
    p.add_argument("--threads", type=int, default=12)
    p.add_argument("--cpus", nargs="+", type=int, default=list(range(12)))
    p.add_argument("--reference")
    args = p.parse_args()
    configure(args.threads, args.cpus)
    benchmark(args.model, args.output, args.cache, args.inputs, args.reference)


if __name__ == "__main__":
    main()
