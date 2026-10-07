"""Shared measured CPU protocol for the three compression entry points."""

import ast
import copy
import functools
import gc
import hashlib
import json
import math
import os
import statistics
import struct
import sys
import time
import types
from contextlib import ExitStack, contextmanager
from pathlib import Path
from unittest.mock import patch

import numpy as np
import torch

try:
    from zipnn import ZipNN
except ImportError:
    ZipNN = None

from neuralzip import huff0
from _experiment_inputs import (
    catalogue,
    evidence,
    exact,
    file_sha,
    prepare,
    tensor_manifest,
    write_new,
)
from _experiment_transfer import BankCompressor, build_bank, exponents
from neuralzip.compressor import NeuralZipCompressor
from neuralzip.payloads import CHUNK_METADATA_BYTES, CodebookPayload
from neuralzip.setup import NeuralZipSetup
from neuralzip.utils import time_call

CHUNK, WARMUPS = 256 << 10, 4
PHASES = (
    "extraction",
    "histograms",
    "js_grouping",
    "sampling",
    "vocabulary_pruning",
    "token_counting",
    "huffman",
    "mode_selection",
    "assembly",
    "other",
)


def configure(threads=12, cpus=None):
    torch.set_num_threads(threads)
    torch.set_num_interop_threads(1)
    if cpus is not None:
        assert hasattr(os, "sched_setaffinity"), "Affinity is available only on Linux"
        os.sched_setaffinity(0, set(cpus))


def summarize(values):
    q1, _, q3 = (
        statistics.quantiles(values, n=4, method="inclusive")
        if len(values) > 1
        else (values[0],) * 3
    )
    return dict(
        median=statistics.median(values),
        q1=q1,
        q3=q3,
        minimum=min(values),
        maximum=max(values),
    )


def forbidden(*args, **kwargs):
    raise AssertionError("Encoding attempted to learn or construct a codebook")


def strict_codebooks(self, groups, assignments, parents):
    books, chunks, tensors, report = NeuralZipSetup._build_codebooks(
        self, groups, assignments, {}
    )
    assert not tensors
    for key, ids in chunks.items():
        assert ids and len(set(ids)) == 1
        tensors[key[:-1]] = ids[0]
    return books, chunks, tensors, report


class StrictSetup(NeuralZipSetup):
    _build_codebooks = strict_codebooks


def own_setup(named, method, output):
    options = {}
    if method == "sharing":
        options["enable_packed_codebooks"] = False
    elif method == "packing":
        options["enable_shared_codebooks"] = False
    elif method.startswith("js-"):
        options["cluster_js_threshold"] = float(method[3:])
    if method.startswith("level-"):
        level = method.removeprefix("level-")
        options["cluster_map"] = {
            n: "global" if level == "global" else n for n, _ in named
        }
        cls = StrictSetup
    else:
        cls = NeuralZipSetup
    gc.collect()
    if method == "new":
        timer = PhaseTimer()
        fitted = profiled_class(timer)(**options)
        with ExitStack() as stack:
            for name, phase in (
                ("count_mixed_tokens", "token_counting"),
                ("build_codebook_from_counts", "huffman"),
                ("build_mixed_codebook", "huffman"),
            ):
                stack.enter_context(
                    patch.object(huff0, name, timer.wrap(phase, getattr(huff0, name)))
                )

            def build():
                with timer.phase("other"):
                    return fitted.build(dict(named))

            setup, elapsed = time_call(build)
        timer.seconds["other"] += elapsed - math.fsum(timer.seconds.values())
        # Observational subclass is not a dependency of a saved setup.
        setup.__class__ = NeuralZipSetup
        setup.setup_report = type(NeuralZipSetup().setup_report)(
            **vars(setup.setup_report)
        )
        write_new(
            output.with_name(output.stem + "-phases.json"),
            dict(
                phase_seconds=timer.seconds,
                phase_calls=timer.calls,
                total_seconds=elapsed,
                same_setup_execution=True,
                profiler_ast_stripping_exact=True,
            ),
        )
    else:
        setup, elapsed = time_call(cls(**options).build, dict(named))
        setup.__class__ = NeuralZipSetup
    setup.save(output)
    return setup, elapsed


def gather_source(named, alias=False):
    setup = NeuralZipSetup(cluster_map={n: n for n, _ in named}).build(dict(named))
    counts = np.zeros(256, dtype=np.int64)
    for _, tensor in named:
        counts += np.bincount(exponents(tensor), minlength=256)
    book = huff0.build_codebook_from_counts(counts, prior_count=1, max_bits=12)
    global_id = len(setup.codebooks)
    setup.codebooks += (
        CodebookPayload(global_id, 1, "bfloat16", codebook=book.codebook),
    )
    parents = {}
    for key, ids in setup.chunk_codebooks.items():
        assert len(set(ids)) == 1
        parents[key[:3]] = ids[0]
    if alias:
        for (name, dtype, stream), value in list(parents.items()):
            if name.startswith("language_model.model."):
                parents[(name.removeprefix("language_model."), dtype, stream)] = value
    setup.tensor_codebooks, setup.chunk_codebooks = parents, {}
    setup.global_codebooks = {("bfloat16", 1): global_id}
    used = set(parents.values()) | {global_id}
    setup.codebooks = tuple(b for b in setup.codebooks if b.codebook_id in used)
    setup.packing_report = ()
    return setup


def transfer_setup(key, cache, inputs, out, gate_bits=3.0):
    donor = catalogue()["donors"][key]
    destination = out / "sources" / f"{donor}-gate{gate_bits:g}.setup.pkl"
    metadata = destination.with_suffix(".json")
    if destination.exists():
        assert file_sha(destination) == json.loads(metadata.read_text())["setup_sha256"]
        return NeuralZipSetup.load(destination)
    named = prepare(donor, cache, inputs)
    original = tensor_manifest(named)
    gathered, seconds = time_call(
        gather_source, named, donor == "six-donor-gemma3" or donor == "gemma3-4b"
    )
    bank, bank_seconds = build_bank(named, gathered, gate_bits)
    assert tensor_manifest(named) == original and not bank.chunk_codebooks
    destination.parent.mkdir(parents=True, exist_ok=True)
    bank.save(destination)
    write_new(
        metadata,
        dict(
            donor=donor,
            input=catalogue()["inputs"][donor],
            setup_sha256=file_sha(destination),
            source_only=True,
            target_codebook_fit=False,
            gathered_setup_seconds=seconds,
            bank_setup_seconds=bank_seconds,
            gate_bits=gate_bits,
        ),
    )
    return bank


def frozen_gpu_setup(codec, setup, named, path):
    """Freeze the actual CPU routes; the GPU exporter must not reroute or learn."""
    codec.prepare_storage_transfer_target(named)
    frozen = copy.deepcopy(setup)
    frozen.chunk_codebooks, frozen.tensor_codebooks = {}, {}
    for name, tensor in named:
        for stream in (0, 1):
            key = codec._transfer_plan_key(name, tensor, stream)
            ids = tuple(m.get("codebook_id") for m in codec._transfer_chunk_modes[key])
            frozen.tensor_codebooks[key] = ids[0]
            frozen.chunk_codebooks[(*key, tensor.numel())] = ids
    frozen.save(path)


def zipnn_sizes(frames, named, parser):
    exponent = 0
    for frame, (_, tensor) in zip(frames, named, strict=True):
        header = int(parser._retrieve_header(frame))
        assert (
            int(parser.original_len) == 2 * tensor.numel()
            and int(parser._bit_reorder) == 1
        )
        chunks = math.ceil(int(parser.original_len) / int(parser.compression_chunk))
        offsets = struct.unpack_from(f"<{2 * chunks}Q", frame, header + 2 * chunks)
        assert header + 18 * chunks + offsets[chunks - 1] + offsets[-1] == len(frame)
        exponent += offsets[-1] + 9 * chunks
    raw = sum(t.numel() * t.element_size() for _, t in named)
    return dict(
        raw_bytes=raw,
        compressed_bytes=sum(map(len, frames)),
        exponent_raw_bytes=raw // 2,
        exponent_compressed_bytes=exponent,
    )


def benchmark(
    key,
    method,
    out,
    cache,
    inputs,
    repetitions=9,
    threads=12,
    gate_bits=3.0,
    reference=None,
):
    """Same full-encode and cold-decode boundaries as the recorded paper run."""
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    path = out / f"{key}-{method}.json"
    assert not path.exists(), (
        "Never overwrite a result; choose a fresh output directory"
    )
    named = prepare(key, cache, inputs)
    original = tensor_manifest(named)
    row = dict(
        model=key,
        method=method,
        input=catalogue()["inputs"][key],
        samples=[],
        setup_samples=[],
        threads=threads,
        affinity=sorted(os.sched_getaffinity(0))
        if hasattr(os, "sched_getaffinity")
        else None,
        repetitions=repetitions,
        warmups=WARMUPS,
        runner_sha256=file_sha(__file__),
    )
    setup = None
    if method == "zipnn":
        assert ZipNN is not None, "Install zipnn==0.5.4 for the paper protocol"

        def make_zipnn():
            return ZipNN(
                method="HUFFMAN",
                input_format="torch",
                threads=threads,
                compression_threshold=0.95,
                compression_chunk=CHUNK,
            )

        codec = make_zipnn()

        def encode():
            frames, copying, native = [], 0.0, 0.0
            for _, tensor in named:
                copied, seconds = time_call(tensor.reshape(-1).clone)
                copying += seconds
                frame, seconds = time_call(codec.compress, copied)
                native += seconds
                frames.append(frame)
            return frames, native, copying

        def decode(encoded):
            decoder = make_zipnn()
            return {
                n: decoder.decompress(f).reshape(t.shape)
                for f, (n, t) in zip(encoded[0], named, strict=True)
            }
    else:
        if method == "transfer":
            setup = transfer_setup(key, cache, inputs, out, gate_bits)
            row.update(
                source_only=True,
                target_codebook_fit=False,
                target_routing_charged_in_encode=True,
                gate_bits=gate_bits,
            )
            codec = BankCompressor(setup, threads=threads)
        else:
            setup, elapsed = own_setup(named, method, out / f"{key}-{method}.setup.pkl")
            row["setup_samples"] = [elapsed]
            codec = NeuralZipCompressor(setup, threads=threads)
        row["setup_seconds"] = dict(
            median=row["setup_samples"][0] if row["setup_samples"] else 0.0
        )
        row["setup_codebooks"] = len(setup.codebooks)
        row["packing_report"] = setup.packing_report
        row["setup_report"] = vars(setup.setup_report)
        row["setup_pickle_bytes"] = (
            (out / f"{key}-{method}.setup.pkl").stat().st_size
            if method != "transfer"
            else 0
        )
        codec._add_global_codebooks()
        codec.prepare_storage_transfer_target(named)

        def encode():
            return codec.encode(named)

        def decode(encoded):
            return NeuralZipCompressor(threads=threads).decode(encoded[0], encoded[1])

    for repeat in range(-WARMUPS, repetitions):
        gc.collect()
        with ExitStack() as stack:
            if setup:
                for name in (
                    "count_mixed_tokens",
                    "build_codebook_from_counts",
                    "build_mixed_codebook",
                    "build_codebook_from_header",
                ):
                    stack.enter_context(patch.object(huff0, name, forbidden))
            encoded, seconds = time_call(encode)
        restored, decoding = time_call(decode, encoded)
        assert set(restored) == {n for n, _ in named} and all(
            exact(t, restored[n]) for n, t in named
        )
        if method == "zipnn":
            sizes = zipnn_sizes(encoded[0], named, codec)
            checksum = hashlib.sha256(b"".join(encoded[0])).hexdigest()
            native, copying = encoded[1:]
        else:
            sizes, native = encoded[2:]
            copying = 0.0
            digest = hashlib.sha256(memoryview(encoded[0].numpy()))
            digest.update(json.dumps(encoded[1], sort_keys=True).encode())
            checksum = digest.hexdigest()
            row["archive_codebooks"] = len(encoded[1]["codebooks"])
            books = encoded[1]["codebooks"]
            row["exponent_codebook_metadata_bytes"] = sum(
                CodebookPayload(
                    b["id"],
                    b["stream"],
                    b["dtype"],
                    b["mode"],
                    b"\0" * b["codebook"][1],
                    b"\0" * b["vocab"][1],
                ).metadata_bytes
                for b in books
            )
            chunks = [c for t in encoded[1]["tensors"] for c in t["groups"][1]]
            row["exponent_payload_bytes"] = sum(c["payload"][1] for c in chunks)
            row["exponent_chunk_metadata_bytes"] = sum(
                CHUNK_METADATA_BYTES[c["mode"]] for c in chunks
            )
        if "archive_sha256" in row:
            assert checksum == row["archive_sha256"] and sizes == row["sizes"]
        else:
            row.update(archive_sha256=checksum, sizes=sizes)
        if repeat >= 0:
            row["samples"].append(
                dict(
                    encode_seconds=seconds,
                    cold_decode_seconds=decoding,
                    codec_seconds=native,
                    input_copy_seconds=copying,
                )
            )
        del encoded, restored
    assert tensor_manifest(named) == original, "Input weights changed"
    for metric in row["samples"][0]:
        row[metric] = summarize([s[metric] for s in row["samples"]])
    row.update(
        roundtrip_bitwise=True,
        input_unchanged=True,
        encode_never_fits=method != "zipnn",
        exponent_ratio_pct=100
        * sizes["exponent_compressed_bytes"]
        / sizes["exponent_raw_bytes"],
        full_ratio_pct=100 * sizes["compressed_bytes"] / sizes["raw_bytes"],
        setup_source_sha256=file_sha(sys.modules[NeuralZipSetup.__module__].__file__),
        compressor_source_sha256=file_sha(
            sys.modules[NeuralZipCompressor.__module__].__file__
        ),
        native_binary_sha256=file_sha(huff0._huff0_cpp.__file__),
    )
    if reference:
        old = evidence(reference)
        assert (
            old["input"]["bf16_manifest_sha256"] == row["input"]["bf16_manifest_sha256"]
        )
        assert (
            row["sizes"] == old["sizes"]
            and row["archive_sha256"] == old["archive_sha256"]
        ), (key, method, "Published compressed bytes differ")
        row["published_compressed_bytes_exact"] = True
    if method == "transfer":
        frozen_gpu_setup(codec, setup, named, out / f"{key}-foreign.setup.pkl")
    write_new(path, row)
    print(
        json.dumps(
            dict(
                model=key,
                method=method,
                ratio_pct=row["exponent_ratio_pct"],
                seconds=row["encode_seconds"]["median"],
                published_exact=row.get("published_compressed_bytes_exact"),
            )
        ),
        flush=True,
    )
    return row


class PhaseTimer:
    def __init__(self):
        self.seconds = dict.fromkeys(PHASES, 0.0)
        self.calls = dict.fromkeys(PHASES, 0)
        self.stack = []

    @contextmanager
    def phase(self, name):
        frame = [time.perf_counter(), 0.0]
        self.stack.append(frame)
        try:
            yield
        finally:
            elapsed = time.perf_counter() - frame[0]
            assert self.stack.pop() is frame
            self.seconds[name] += elapsed - frame[1]
            self.calls[name] += 1
            if self.stack:
                self.stack[-1][1] += elapsed

    def wrap(self, name, function):
        @functools.wraps(function)
        def timed(*args, **kwargs):
            with self.phase(name):
                return function(*args, **kwargs)

        return timed


def profiled_class(timer):
    """Insert observational boundaries; stripping them recovers the exact source AST."""
    source = Path(sys.modules[NeuralZipSetup.__module__].__file__)
    original = ast.parse(source.read_text())
    tree = copy.deepcopy(original)

    def block(name, body):
        call = ast.Call(
            ast.Attribute(ast.Name("_phase_timer", ast.Load()), "phase", ast.Load()),
            [ast.Constant(name)],
            [],
        )
        return ast.With([ast.withitem(call, None)], body)

    def target(statement):
        return (
            ast.unparse(statement.targets[0])
            if isinstance(statement, ast.Assign)
            else None
        )

    cls = next(
        node
        for node in tree.body
        if isinstance(node, ast.ClassDef) and node.name == "NeuralZipSetup"
    )
    methods = {
        node.name: node for node in cls.body if isinstance(node, ast.FunctionDef)
    }
    cluster = methods["_cluster_chunks"]
    loop = next(node for node in cluster.body if isinstance(node, ast.For))
    parent_index = next(
        i for i, node in enumerate(loop.body) if target(node) == "parents[key]"
    )
    inner = next(node for node in loop.body if isinstance(node, ast.For))
    assert [target(node) for node in inner.body[:3]] == [
        "part",
        "counts",
        "distribution",
    ]
    group_index = next(
        i for i, node in enumerate(inner.body) if target(node) == "group"
    )
    inner.body = [
        block("histograms", inner.body[:3]),
        block("js_grouping", inner.body[3:group_index]),
        block("sampling", inner.body[group_index:]),
    ]
    loop.body = [
        block("extraction", loop.body[:parent_index]),
        block("histograms", [loop.body[parent_index]]),
        *loop.body[parent_index + 1 :],
    ]
    for name, phase in (
        ("_cluster_chunks", "js_grouping"),
        ("_build_codebooks", "assembly"),
        ("_prepare_codebook", "vocabulary_pruning"),
        ("_compare_codebooks", "mode_selection"),
    ):
        methods[name].body = [block(phase, methods[name].body)]

    class Strip(ast.NodeTransformer):
        def visit_With(self, node):
            node = self.generic_visit(node)
            call = node.items[0].context_expr
            if (
                isinstance(call, ast.Call)
                and isinstance(call.func, ast.Attribute)
                and isinstance(call.func.value, ast.Name)
                and call.func.value.id == "_phase_timer"
            ):
                return node.body
            return node

    assert ast.dump(Strip().visit(copy.deepcopy(tree))) == ast.dump(original), (
        "Profiler changed computation"
    )
    module = types.ModuleType("ablation_profile")
    module._phase_timer = timer
    sys.modules[module.__name__] = module
    exec(compile(ast.fix_missing_locations(tree), str(source), "exec"), module.__dict__)
    return module.NeuralZipSetup
