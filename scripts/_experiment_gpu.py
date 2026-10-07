"""One fresh six-suite GPU control; run in an isolated, pinned source environment.

Pinned model construction and inputs are bundled with the reproduction scripts.
"""

import argparse
import ast
import contextlib
import copy
import gc
import hashlib
import json
import os
import subprocess
from collections import defaultdict
from functools import partial
from pathlib import Path
from statistics import median
from types import SimpleNamespace

import torch

import _experiment_inputs as reference

try:
    from dfloat11 import dfloat11 as official
except ImportError:
    official = None

from neuralzip.gpu import inference
from neuralzip.setup import NeuralZipSetup
from neuralzip.utils import time_call

SETUP_LABELS = {
    "own": "new",
    "transfer": "foreign",
    "sharing": "sharing",
    "packing": "packing",
}
OFFICIAL_COMMIT = "457733886ce6ebc6d8dda1621fad1ffa2661e028"
OFFICIAL_SHA = {
    "dfloat11.py": "647ca864ca5e708c13329d900c808b73e7488f9d156f44efa5f5ec3e6982b303",
    "dfloat11_utils.py": "c50216fda9da2a516131ce84afec6003f566f6e0f88cb4fa8178bc18f83851fe",
    "decode.cu": "9b9a3c7c491bcdd4a85949da4c337ac1ce7f9de627eea3d18cd335878d186950",
    "decode.ptx": "cc9b78f6a14c9118abf46a733390f14ef66a9c2eb4db04ded9b74183a02665cf",
}
PROTOCOL = {
    "physical_gpu": 1,
    "threads": 8,
    "affinity": list(range(16)),
    "batch": 1,
    "attention": "eager",
    "cache": False,
    "offload": False,
    "warmups": 20,
    "trials": 5,
    "forwards_per_trial": 50,
}
FUNCTIONAL_ADAPTER = "multihead-attention-out-proj-v1"


def sha(data):
    return hashlib.sha256(data).hexdigest()


def file_sha(path):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(partial(stream.read, 1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def source_hashes():
    root = Path(inference.__file__).parent.parent
    return {
        str(path.relative_to(root)): sha(path.read_bytes())
        for path in sorted(root.rglob("*"))
        if path.is_file() and path.suffix in (".py", ".cu", ".cpp", ".h", ".so")
    }


def export_identity(exported):
    """Hash original payloads and restart metadata, not runtime-only descriptors."""
    groups = []
    for group in exported.groups:
        streams = []
        for stream in group.encoded.streams:
            streams.append(
                sha(stream)
                if isinstance(stream, bytes)
                else {
                    "bits": stream.bit_length,
                    "count": stream.symbol_count,
                    "bitstream": sha(stream.bitstream),
                    "gaps": sha(stream.packed_gaps),
                    "positions": list(stream.output_positions),
                }
            )
        groups.append(
            {
                "block": group.block_name,
                "codebook_id": group.codebook_id,
                "lengths": list(group.codebook.code_lengths),
                "values": list(group.codebook.code_values),
                "vocab": group.codebook.vocab.hex(),
                "streams": streams,
                "offset": group.workspace_offset,
                "numel": group.encoded.numel,
            }
        )
    bindings = [
        {
            "name": b.tensor_name,
            "module": b.module_name,
            "shape": list(b.shape),
            "offset": b.workspace_offset,
            "numel": b.numel,
        }
        for b in exported.bindings
    ]
    return sha(
        json.dumps({"groups": groups, "bindings": bindings}, sort_keys=True).encode()
    )


def decode_hook_module(model, block_name):
    """Same approved functional-consumer adapter for every compressed control."""
    parent_name, _, child_name = block_name.rpartition(".")
    parent = model.get_submodule(parent_name)
    if child_name == "out_proj" and isinstance(parent, torch.nn.MultiheadAttention):
        return parent
    return model.get_submodule(block_name)


def correct_functional_hooks(runtime):
    targets = {}
    for index, block_name in enumerate(runtime.export.block_order):
        block = runtime.model.get_submodule(block_name)
        target = decode_hook_module(runtime.model, block_name)
        if target is not block:
            # Remove only this runtime's own hook, not pre-existing user hooks.
            runtime._hook_handles[index].remove()
            runtime._hook_handles[index] = target.register_forward_pre_hook(
                partial(runtime._decode_block, block_name), prepend=True
            )
            targets[block_name] = block_name.rpartition(".")[0]
    return targets


def decode_functional_dfloat(block, _consumer, args):
    decode_dfloat(block, args)


def decode_dfloat(block, _args):
    """Unmodified official kernel; restore original Linear/Conv weight shapes."""
    device = block.luts.device
    count = block.sign_mantissa.numel()
    n_bytes = block.encoded_exponent.numel()
    reconstructed = official.TensorManager.allocate_bfloat16(device, count)
    with official.cp.cuda.Device(device.index):
        official._decode(
            grid=((n_bytes + 4095) // 4096,),
            block=(512,),
            shared_mem=block.shared_mem_size,
            args=[
                block.luts.data_ptr(),
                block.encoded_exponent.data_ptr(),
                block.sign_mantissa.data_ptr(),
                block.output_positions.data_ptr(),
                block.gaps.data_ptr(),
                reconstructed.data_ptr(),
                block.luts.shape[0],
                n_bytes,
                count,
            ],
        )
    for module, shape, weight in zip(
        block.weight_injection_modules,
        block.weight_shapes,
        torch.tensor_split(reconstructed, block.split_positions),
        strict=True,
    ):
        module.weight = weight.view(shape)


def attach_dfloat(model, candidates):
    blocks = defaultdict(list)
    for candidate in candidates:
        blocks[candidate.block_name].append(candidate)
    hooks, fallback = [], []
    for name, members in blocks.items():
        block = model.get_submodule(name)
        weights = [member.parameter.detach().reshape(-1).cpu() for member in members]
        _, counter = official.get_codec(torch.cat(weights))
        if len(counter) == 1:
            fallback.extend(
                {"name": m.tensor_name, "reason": "singleton alphabet"} for m in members
            )
            continue
        codec, _, table = official.get_32bit_codec(counter)
        try:
            luts = official.get_luts(table)
        except UnboundLocalError as error:
            if "curr_val" not in str(error):
                raise
            fallback.extend(
                {"name": m.tensor_name, "reason": "official LUT has no first entry"}
                for m in members
            )
            continue
        encoded, raw, positions, gaps, splits = official.encode_weights(
            weights, codec, 8, 512
        )
        for key, tensor in (
            ("luts", luts),
            ("encoded_exponent", encoded),
            ("sign_mantissa", raw),
            ("output_positions", positions.view(torch.uint8)),
            ("gaps", gaps),
        ):
            block.register_buffer(key, tensor)
        block.split_positions = splits.tolist()
        block.shared_mem_size = (
            512 * 4 + 4 + int(torch.diff(positions.long()).max()) * 2
        )
        block.weight_injection_modules = [
            model.get_submodule(m.module_name) for m in members
        ]
        block.weight_shapes = [tuple(m.parameter.shape) for m in members]
        for module in block.weight_injection_modules:
            delattr(module, "weight")
        target = decode_hook_module(model, name)
        hook = (
            decode_dfloat
            if target is block
            else partial(decode_functional_dfloat, block)
        )
        target.register_forward_pre_hook(hook, prepend=True)
        hooks.append(partial(decode_dfloat, block, ()))
    return hooks, fallback


def forward_many(model, inputs):
    for _ in range(PROTOCOL["forwards_per_trial"]):
        model(**inputs)
    torch.cuda.synchronize()


def decode_many(calls):
    for call in calls:
        call()


def check_exclusive_gpu(gpu):
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
    pids = {int(value) for value in result.stdout.split()}
    assert not pids - {os.getpid()}, (
        "Another GPU job started; this control is not valid"
    )


@torch.inference_mode()
def benchmark(args):
    assert os.environ.get("CUDA_VISIBLE_DEVICES") == str(args.gpu), (
        "Physical GPU selection differs"
    )
    PROTOCOL["physical_gpu"] = args.gpu
    assert (args.variant == "control") == (args.method in ("bf16", "dfloat11"))
    destination = (
        args.output / args.campaign / f"{args.model}-{args.variant}-{args.method}.json"
    )
    assert not destination.exists(), "Never overwrite a measurement"
    destination.parent.mkdir(parents=True, exist_ok=True)
    assert official is not None, (
        "Install the pinned official DFloat11 experiment dependency"
    )
    for name, digest in OFFICIAL_SHA.items():
        assert sha(Path(official.__file__).with_name(name).read_bytes()) == digest, name
    os.sched_setaffinity(0, set(PROTOCOL["affinity"]))
    torch.set_num_threads(PROTOCOL["threads"])
    torch.set_num_interop_threads(1)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.backends.cuda.matmul.allow_tf32 = False
    spec = dict(reference.catalogue()["inputs"][args.model]["source"], key=args.model)
    named = dict(reference.prepare(args.model, args.cache, args.inputs))
    input_path = args.inputs / f"{args.model}.pt"
    manifest = reference.tensor_manifest(named.items())
    expected_manifest = reference.catalogue()["inputs"][args.model][
        "bf16_manifest_sha256"
    ]
    model, ignored = reference.load_model(spec, named, str(args.cache))
    if args.model == "parakeet":
        state = model.state_dict()
        assert all(
            n in state and reference.exact(t, state[n]) for n, t in named.items()
        )
        del state
    _, candidates = inference._gpu_candidates(model)
    candidates = inference._unique_candidates(model, candidates)
    expected = {c.tensor_name: c.parameter.detach().clone() for c in candidates}
    selected = {n: reference.fingerprint(t) for n, t in expected.items()}
    row = {
        "model": args.model,
        "method": args.method,
        "variant": args.variant,
        "campaign": args.campaign,
        "protocol": PROTOCOL,
        "pinned_model": spec,
        "manifest_sha256": expected_manifest,
        "input_file_sha256": file_sha(input_path),
        "selected_weight_sha256": selected,
        "ignored_checkpoint_keys": list(ignored),
        "source_hashes": source_hashes(),
        "runner_sha256": sha(Path(__file__).read_bytes()),
        "functional_consumer_adapter": FUNCTIONAL_ADAPTER,
        "reference_sha256": file_sha(Path(reference.__file__)),
        "official_commit": OFFICIAL_COMMIT,
        "official_sha256": OFFICIAL_SHA,
        "torch_version": torch.__version__,
        "cuda_version": torch.version.cuda,
        "cupy_version": official.cp.__version__,
        "gpu_name": torch.cuda.get_device_name(0),
        "gpu_uuid": str(torch.cuda.get_device_properties(0).uuid),
        "runtime_metadata_bytes": 0,
    }
    calls = []
    if args.method in SETUP_LABELS:
        label = SETUP_LABELS[args.method]
        path = args.setups / f"{args.model}-{label}.setup.pkl"
        frozen_setup_sha = file_sha(path)
        row["setup_sha256"] = sha(path.read_bytes())
        assert row["setup_sha256"] == frozen_setup_sha
        setup = copy.deepcopy(NeuralZipSetup.load(path))
        by_digest = defaultdict(list)
        for item in manifest:
            by_digest[(tuple(item["shape"]), item["sha256"])].append(item["name"])
        mapping = {}
        for name, tensor in expected.items():
            sources = by_digest[(tuple(tensor.shape), selected[name])]
            assert name in sources or len(sources) == 1, name
            mapping[name] = name if name in sources else sources[0]
        if any(target != source for target, source in mapping.items()):
            setup.tensor_codebooks = {
                (target, *key[1:]): v
                for target, source in mapping.items()
                for key, v in setup.tensor_codebooks.items()
                if key[0] == source
            }
            setup.chunk_codebooks = {
                (target, *key[1:]): v
                for target, source in mapping.items()
                for key, v in setup.chunk_codebooks.items()
                if key[0] == source
            }
        frozen = (
            setup.codebooks,
            dict(setup.tensor_codebooks),
            dict(setup.chunk_codebooks),
        )
        exported = inference.export_model_for_gpu(model, setup)
        assert frozen == (
            setup.codebooks,
            setup.tensor_codebooks,
            setup.chunk_codebooks,
        ), "Target fit"
        row.update(
            payload_sha256=export_identity(exported),
            compressed_names=sorted(b.tensor_name for b in exported.bindings),
            packed_chunks=sum(bool(g.codebook.vocab) for g in exported.groups),
            groups=len(exported.groups),
        )
        if args.method == "sharing":
            assert row["packed_chunks"] == 0, (
                "Sharing-only unexpectedly packs exponents"
            )
        elif any(book.vocab for book in setup.codebooks):
            assert row["packed_chunks"] > 0, "Packing was silently discarded"
        runtime = inference.NeuralZipRuntime.attach(model, exported, device="cuda:0")
        row["functional_hook_targets"] = correct_functional_hooks(runtime)
        for workspace in runtime.workspaces.values():
            workspace.view(torch.uint8).fill_(0xA5)
        for name, actual in runtime.materialized_weights():
            assert reference.exact(actual, expected[name]), name
        for workspace in runtime.workspaces.values():
            workspace.view(torch.uint8).fill_(0xA5)
        calls = [
            partial(runtime._decode_block, block) for block in exported.block_order
        ]
        row["runtime_metadata_bytes"] = sum(
            t.numel() * t.element_size()
            for batches in runtime._prepared_batches.values()
            for batch in batches
            for t in batch.descriptor_tensors
        )
        del setup, exported
    elif args.method == "dfloat11":
        with (
            open(os.devnull, "w") as quiet,
            contextlib.redirect_stdout(quiet),
            contextlib.redirect_stderr(quiet),
        ):
            calls, row["dfloat_raw_fallbacks"] = attach_dfloat(model, candidates)
        model.cuda()
        for call in calls:
            call()
            block = call.args[0]
            for candidate in candidates:
                module = model.get_submodule(candidate.module_name)
                if any(module is member for member in block.weight_injection_modules):
                    assert reference.exact(
                        module.weight, expected[candidate.tensor_name]
                    ), candidate.tensor_name
        del call, block
    else:
        model.cuda()
    for candidate in candidates:
        module = model.get_submodule(candidate.module_name)
        # NeuralZip workspaces are block-reused; compressed weights were checked above.
        needs_check = (
            args.method == "bf16"
            or (
                args.method == "dfloat11"
                and candidate.tensor_name
                in {item["name"] for item in row["dfloat_raw_fallbacks"]}
            )
            or (
                args.method in SETUP_LABELS
                and candidate.tensor_name not in row["compressed_names"]
            )
        )
        if needs_check:
            assert reference.exact(module.weight, expected[candidate.tensor_name]), (
                candidate.tensor_name
            )
    if spec.get("kind") == "torchvision":
        forward = model.forward
        model.forward = lambda __pixels__: forward(__pixels__)
    inputs = reference.inputs_for(args.model, model)
    row["forward_input_sha256"] = {
        n: reference.fingerprint(t)
        for n, t in inputs.items()
        if isinstance(t, torch.Tensor)
    }
    output = reference.outputs_for(model(**inputs))
    assert output and all(
        torch.isfinite(t).all() for t in output.values() if t.is_floating_point()
    )
    reference_folder = (
        args.reference_root / args.campaign
        if args.reference_root
        else destination.parent
    )
    reference_path = reference_folder / f"{args.model}-reference.pt"
    if args.method == "bf16":
        assert not reference_path.exists()
        torch.save(output, reference_path)
    else:
        if args.reference_root:
            reference_record_path = reference_folder / f"{args.model}-control-bf16.json"
            reference_record = json.loads(reference_record_path.read_text())
            for field in (
                "pinned_model",
                "manifest_sha256",
                "input_file_sha256",
                "selected_weight_sha256",
                "forward_input_sha256",
                "protocol",
                "gpu_uuid",
                "torch_version",
                "cuda_version",
                "cupy_version",
                "official_sha256",
                "functional_consumer_adapter",
            ):
                assert row[field] == reference_record[field], field
            row["bf16_reference_record_sha256"] = sha(
                reference_record_path.read_bytes()
            )
            row["bf16_reference_output_sha256"] = file_sha(reference_path)
        baseline = torch.load(reference_path, weights_only=True, map_location="cpu")
        assert output.keys() == baseline.keys() and all(
            reference.exact(t, baseline[n]) for n, t in output.items()
        )
        del baseline
    assert all(
        reference.fingerprint(named[item["name"]]) == item["sha256"]
        for item in manifest
    )
    del output, expected, named, candidates
    gc.collect()
    for _ in range(PROTOCOL["warmups"]):
        model(**inputs)
    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()
    trials = []
    for _ in range(PROTOCOL["trials"]):
        check_exclusive_gpu(args.gpu)
        torch.cuda.synchronize()
        _, seconds = time_call(forward_many, model, inputs)
        trials.append(seconds * 1000 / PROTOCOL["forwards_per_trial"])
        check_exclusive_gpu(args.gpu)
    row.update(
        inference_trials_ms=trials,
        inference_ms=median(trials),
        allocated_gpu_bytes=torch.cuda.memory_allocated(),
        peak_gpu_bytes=torch.cuda.max_memory_allocated(),
        reserved_gpu_bytes=torch.cuda.memory_reserved(),
    )
    if calls:
        for _ in range(PROTOCOL["warmups"]):
            decode_many(calls)
        start, stop = (
            torch.cuda.Event(enable_timing=True),
            torch.cuda.Event(enable_timing=True),
        )
        decode_trials = []
        for _ in range(PROTOCOL["trials"]):
            start.record()
            for _ in range(PROTOCOL["forwards_per_trial"]):
                decode_many(calls)
            stop.record()
            stop.synchronize()
            decode_trials.append(
                start.elapsed_time(stop) / PROTOCOL["forwards_per_trial"]
            )
        row.update(decode_trials_ms=decode_trials, decode_ms=median(decode_trials))
    assert row["forward_input_sha256"] == {
        n: reference.fingerprint(t)
        for n, t in inputs.items()
        if isinstance(t, torch.Tensor)
    }
    if args.method in SETUP_LABELS:
        assert sha(path.read_bytes()) == row["setup_sha256"], (
            "Frozen setup file changed"
        )
    row.update(
        weight_bitwise=True,
        output_bitwise=True,
        input_unchanged=True,
        setup_frozen=True,
        full_decode_each_forward=True,
        profiling_in_timing=False,
    )
    # The encoder/export AST is also captured independently of runtime changes.
    tree = ast.parse(Path(inference.__file__).read_text())
    row["exporter_ast_sha256"] = sha(
        ast.dump(
            ast.Module(
                body=[
                    n
                    for n in tree.body
                    if isinstance(n, ast.FunctionDef)
                    and n.name
                    in ("_assemble_export", "encode_gpu_native", "export_model_for_gpu")
                ],
                type_ignores=[],
            ),
            include_attributes=False,
        ).encode()
    )
    with destination.open("x") as stream:
        json.dump(row, stream, indent=2)
        stream.write("\n")
    print(
        json.dumps(
            {
                "complete": destination.name,
                "inference_ms": row["inference_ms"],
                "decode_ms": row.get("decode_ms"),
                "allocated_gpu_bytes": row["allocated_gpu_bytes"],
                "peak_gpu_bytes": row["peak_gpu_bytes"],
            }
        ),
        flush=True,
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True)
    parser.add_argument(
        "--method",
        choices=("bf16", "own", "transfer", "sharing", "packing", "dfloat11"),
        required=True,
    )
    parser.add_argument("--gpu", type=int, required=True)
    for name in ("cache", "inputs", "setups", "output"):
        parser.add_argument("--" + name, type=Path, required=True)
    parser.add_argument("--campaign", default="1")
    args = parser.parse_args()
    benchmark(
        SimpleNamespace(
            **vars(args),
            variant="control" if args.method in ("bf16", "dfloat11") else "candidate",
            reference_root=None if args.method == "bf16" else args.output,
        )
    )


if __name__ == "__main__":
    main()
