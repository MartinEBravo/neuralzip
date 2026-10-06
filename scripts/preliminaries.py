"""Figures 1--3: BF16 field entropy, n-gram entropy and random-chunk JS distance."""

import argparse
import csv
import hashlib
from pathlib import Path

import numpy as np
import torch
from scipy.spatial.distance import jensenshannon

from neuralzip._experiment_inputs import (
    catalogue,
    evidence,
    prepare,
    tensor_manifest,
    write_new,
)

CHUNK_VALUES = (256 << 10) // 2


def entropy(counts):
    counts = np.asarray(counts, dtype=float)
    probabilities = counts[counts > 0] / counts.sum()
    return float(-np.dot(probabilities, np.log2(probabilities)))


def sample_plan(sizes, seed):
    selected = sorted(sorted(sizes, key=lambda t: (-t[1], t[0]))[:12])
    assert len(selected) == 12 and all(n // CHUNK_VALUES >= 4 for _, n in selected)
    rng = np.random.default_rng(seed)
    chunks = [
        (name, int(i))
        for name, n in selected
        for i in rng.choice(n // CHUNK_VALUES, 4, replace=False)
    ]
    rng.shuffle(chunks)
    return chunks


def measure(key, named):
    named = tuple(sorted(named))
    original = tensor_manifest(named)
    counts = [np.zeros(n, dtype=np.int64) for n in (2, 256, 128)]
    sampled = []
    elements = sum(t.numel() for _, t in named)
    for _, tensor in named:
        bits = tensor.reshape(-1).view(torch.uint16).numpy()
        exponents = ((bits >> 7) & 255).astype(np.uint8)
        for target, values in zip(
            counts, (bits >> 15, (bits >> 7) & 255, bits & 127), strict=True
        ):
            target += np.bincount(values, minlength=len(target))
        take = min(len(bits), int((8 << 20) * len(bits) / elements)) // 8 * 8
        if take:
            width = max(8, take // 64 * 8)
            for start in np.linspace(
                0, len(bits) - width, min(8, take // width), dtype=int
            ):
                start = int(start) // 8 * 8
                sampled.append(exponents[start : start + width].copy())
    ngrams = []
    for n in (1, 2, 4, 8):
        tuples = np.concatenate([s.view(np.dtype(f"<u{n}")) for s in sampled])
        _, frequency = np.unique(tuples, return_counts=True)
        ngrams.append(
            dict(
                n=n,
                bits_per_exponent=entropy(frequency) / n,
                tuples=len(tuples),
                unique_symbols=len(frequency),
            )
        )
    seed = [
        20261003,
        int.from_bytes(hashlib.sha256(key.encode()).digest()[:4], "little"),
    ]
    plan = sample_plan([(n, t.numel()) for n, t in named], seed)
    tensors = dict(named)
    chunks = []
    histograms = []
    for name, index in plan:
        bits = (
            tensors[name]
            .reshape(-1)[index * CHUNK_VALUES : (index + 1) * CHUNK_VALUES]
            .view(torch.uint16)
            .numpy()
        )
        histograms.append(np.bincount((bits >> 7) & 255, minlength=256).tolist())
        chunks.append(
            dict(
                tensor=name,
                source_chunk_index=index,
                bf16_sha256=hashlib.sha256(bits.tobytes()).hexdigest(),
            )
        )
    probs = np.asarray(histograms, dtype=float) / CHUNK_VALUES
    matrix = jensenshannon(probs[:, None, :], probs[None, :, :], base=2, axis=-1)
    assert tensor_manifest(named) == original
    return dict(
        key=key,
        field_counts=[c.tolist() for c in counts],
        field_entropy=dict(zip(("sign", "exponent", "mantissa"), map(entropy, counts))),
        ngrams=ngrams,
        sampled_exponents=sum(map(len, sampled)),
        rng_seed=seed,
        chunks=chunks,
        histogram_counts=histograms,
        js_distance=matrix.tolist(),
        input_unchanged=True,
    )


def check_saved(keys):
    random = evidence("preliminaries/random-js.json")
    for key in keys:
        row = evidence(f"preliminaries/{key}-statistics.json")
        assert all(
            abs(entropy(counts) - row["field_entropy"][name]) < 1e-12
            for counts, name in zip(
                row["field_counts"], ("sign", "exponent", "mantissa"), strict=True
            )
        )
        js = next(r for r in random["models"] if r["key"] == key)
        probs = np.asarray(js["histogram_counts"], dtype=float) / CHUNK_VALUES
        matrix = jensenshannon(probs[:, None, :], probs[None, :, :], base=2, axis=-1)
        assert np.array_equal(matrix, np.asarray(js["js_distance"]))
        expected = sample_plan(
            [(n["name"], int(np.prod(n["shape"]))) for n in js["input_tensors"]],
            js["rng_seed"],
        )
        assert expected == [
            (c["tensor"], c["source_chunk_index"]) for c in js["chunks"]
        ]
    print(
        "PASS: full-field entropies, random sampling identities and all JS matrix entries"
    )


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--models", nargs="+", choices=catalogue()["six"], default=catalogue()["six"]
    )
    p.add_argument("--output", type=Path, default=Path("results/reruns/preliminaries"))
    p.add_argument("--cache", type=Path, default=Path(".cache/experiments/hf"))
    p.add_argument("--inputs", type=Path, default=Path(".cache/experiments/inputs"))
    p.add_argument("--check", action="store_true")
    p.add_argument(
        "--export-saved",
        action="store_true",
        help="Export verified Figures 1--3 coordinates; no weights or rerun",
    )
    p.add_argument("--verify-published", action="store_true")
    args = p.parse_args()
    if args.check or args.export_saved:
        check_saved(args.models)
        if args.check and not args.export_saved:
            return
    if not args.export_saved:
        torch.set_num_threads(1)
        torch.set_num_interop_threads(1)
    args.output.mkdir(parents=True, exist_ok=True)
    fields, ngrams = [], []
    for key in args.models:
        if args.export_saved:
            row = dict(evidence(f"preliminaries/{key}-statistics.json"))
            js = next(
                r
                for r in evidence("preliminaries/random-js.json")["models"]
                if r["key"] == key
            )
            row.update(
                {
                    name: js[name]
                    for name in (
                        "rng_seed",
                        "chunks",
                        "histogram_counts",
                        "js_distance",
                    )
                }
            )
        else:
            row = measure(key, prepare(key, args.cache, args.inputs))
        if args.verify_published:
            old = evidence(f"preliminaries/{key}-statistics.json")
            assert (
                row["field_counts"] == old["field_counts"]
                and row["sampled_exponents"] == old["sampled_exponents"]
            )
            for actual, expected in zip(row["ngrams"], old["ngrams"], strict=True):
                for field in ("n", "tuples", "unique_symbols"):
                    assert actual[field] == expected[field], field
                # BLAS reduction order can differ by a few ulps; integer populations must match exactly.
                assert (
                    abs(actual["bits_per_exponent"] - expected["bits_per_exponent"])
                    < 1e-12
                )
            js = next(
                r
                for r in evidence("preliminaries/random-js.json")["models"]
                if r["key"] == key
            )
            assert (
                row["chunks"] == js["chunks"]
                and row["histogram_counts"] == js["histogram_counts"]
            )
            assert np.allclose(
                row["js_distance"], js["js_distance"], rtol=0, atol=1e-12
            )
            row["published_populations_exact"] = True
        write_new(args.output / f"{key}-statistics.json", row)
        fields.append(dict(model=key, **row["field_entropy"]))
        ngrams.extend(dict(model=key, **r) for r in row["ngrams"])
        with (args.output / f"js-{key}.csv").open("x") as stream:
            writer = csv.writer(stream)
            writer.writerow(("row", "column", "distance"))
            for i, r in enumerate(row["js_distance"]):
                writer.writerows((i, j, v) for j, v in enumerate(r))
    for filename, values in (
        ("zero_order_entropy.csv", fields),
        ("ngram_exponent_entropy.csv", ngrams),
    ):
        with (args.output / filename).open("x") as stream:
            writer = csv.DictWriter(stream, fieldnames=list(values[0]))
            writer.writeheader()
            writer.writerows(values)


if __name__ == "__main__":
    main()
