import argparse
import copy
import hashlib
import json
import tempfile
from pathlib import Path

import numpy as np
import torch
from torch import nn
from transformers import LlamaConfig, LlamaForCausalLM

from neuralzip import _huff0_cpp, huff0, neuralzip
from neuralzip.compressor import NeuralZipCompressor
from neuralzip.dtypes import DTYPES
from neuralzip.gpu import decoder
from neuralzip.gpu.inference import (
    GPUCodebook,
    GPUWeightGroup,
    LinearWeightBinding,
    NeuralZipModelExport,
    NeuralZipRuntime,
    encode_gpu_native,
    export_model_for_gpu,
)
from neuralzip.gpu.pretrained import _export_stats
from neuralzip.setup import NeuralZipSetup


class Model(nn.Module):
    def __init__(self, dtype):
        super().__init__()
        self.layers = nn.ModuleList(
            [
                nn.Linear(257, 513, bias=False, dtype=dtype),
                nn.Linear(513, 17, bias=False, dtype=dtype),
            ]
        )

    def forward(self, inputs):
        for layer in self.layers:
            inputs = layer(inputs)
        return inputs


def exact(a, b):
    return torch.equal(
        a.cpu().contiguous().view(torch.uint8), b.cpu().contiguous().view(torch.uint8)
    )


class DirectDecoder(decoder.CuPyDecoder):
    """Exercise the hardware-capacity fallback without requiring a small GPU."""

    def _staging(self, device_index, max_elements):
        return False, decoder.shared_memory_bytes(1, 0)


@torch.inference_mode()
def check_segments(max_bits, direct):
    # A complete canonical tree with exactly 258 leaves and a requested depth.
    # Token 256 is the common packed token; all rare codes/tails are also used.
    chain = max_bits - 8
    remaining = 258 - chain
    lengths = list(range(2, chain + 1))
    lengths += [max_bits - 1] * (256 - remaining) + [max_bits] * (2 * remaining - 256)
    lengths.insert(256, 1)
    vocab = bytes((127, 127, 125, 126))
    book = huff0.build_codebook_from_header(b"NZM1" + bytes(lengths), vocab=vocab)
    exported_lengths, values, longest = _huff0_cpp.export_code_table(
        book.codebook, max_bits=56, vocab=vocab
    )
    assert longest == max_bits
    symbols = np.array(
        [127] * 70000 + list(range(256)) * 3 + [125, 126, 126], dtype=np.uint8
    )
    for dtype, spec in DTYPES.items():
        # Include nonfinite and arbitrary raw bits: decoding is a bit-preserving operation.
        stream_id = spec.exponent_stream if spec.exponent_stream is not None else 1
        reordered = np.arange(len(symbols) * spec.element_size, dtype=np.uint32).astype(
            np.uint8
        )
        reordered[stream_id :: spec.element_size] = symbols
        raw = spec.reorder_bytes(reordered.tobytes(), restore=True)
        original = torch.frombuffer(bytearray(raw), dtype=dtype).reshape(1, -1)
        model = nn.Module()
        model.layers = nn.ModuleList(
            [nn.Linear(original.numel(), 1, bias=False, dtype=dtype)]
        )
        model.layers[0].weight.copy_(original)
        encoded = encode_gpu_native(
            raw, book.codebook, dtype=spec, stream_id=stream_id, vocab=vocab
        )
        binding = LinearWeightBinding(
            "layers.0", "layers.0.weight", tuple(original.shape), original.numel(), 0
        )
        export = NeuralZipModelExport(
            (
                GPUWeightGroup(
                    "layers.0",
                    0,
                    GPUCodebook(tuple(exported_lengths), tuple(values), vocab),
                    encoded,
                    (binding,),
                    0,
                ),
            )
        )
        original_decoder = decoder._DECODER
        try:
            if direct:
                decoder._DECODER = DirectDecoder()
            runtime = NeuralZipRuntime.attach(model, export, device="cuda:0")
            for _ in range(2):
                for workspace in runtime.workspaces.values():
                    workspace.view(torch.uint8).fill_(0xA5)
                assert exact(
                    dict(runtime.materialized_weights())["layers.0.weight"], original
                )
        finally:
            decoder._DECODER = original_decoder


@torch.inference_mode()
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--cpu-only", action="store_true")
    parser.add_argument("--reference", type=Path)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    torch.manual_seed(42)
    cpu = {}
    gpu = {}
    for dtype, spec in DTYPES.items():
        model = Model(dtype).eval()
        for parameter in model.parameters():
            # Compressible finite values with alternating local exponent regimes.
            pattern = torch.tensor(
                [0.25, 0.25, 0.25, 0.25, 0.5, 0.5, 1, 1], dtype=dtype
            )
            parameter.copy_(
                pattern.repeat((parameter.numel() + 7) // 8)[
                    : parameter.numel()
                ].view_as(parameter)
            )
        named = dict(model.named_parameters())
        setup = NeuralZipSetup(chunk_size=8192).build(named)
        encoded, metadata, _, _ = NeuralZipCompressor(setup, threads=1).encode(
            named.items()
        )
        decoded = NeuralZipCompressor().decode(encoded, metadata)
        assert all(exact(decoded[name], tensor) for name, tensor in named.items())
        cpu[spec.name] = hashlib.sha256(
            encoded.numpy().tobytes() + json.dumps(metadata, sort_keys=True).encode()
        ).hexdigest()
        if args.cpu_only:
            continue
        expected = {name: tensor.clone() for name, tensor in named.items()}
        baseline = copy.deepcopy(model).cuda()
        inputs = torch.ones(1, 257, dtype=dtype, device="cuda")
        output = baseline(inputs).cpu()
        del baseline
        # A raw fragment between compressed chunks must also be restored.
        key = next(iter(setup.chunk_codebooks))
        ids = list(setup.chunk_codebooks[key])
        ids[len(ids) // 2] = None
        setup.chunk_codebooks[key] = tuple(ids)
        exported = export_model_for_gpu(model, setup)
        assert {binding.tensor_name for binding in exported.bindings} == set(named)
        if spec.exponent_stream is not None:
            assert any(group.codebook.vocab for group in exported.groups), (
                "Packed route was not exercised"
            )
        assert any(group.codebook_id == -1 for group in exported.groups)
        stats = _export_stats(model, exported, setup, 0.0)
        assert stats.raw_bytes == sum(
            p.numel() * p.element_size() for p in expected.values()
        )
        runtime = NeuralZipRuntime.attach(model, exported, device="cuda:0")
        for workspace in runtime.workspaces.values():
            workspace.view(torch.uint8).fill_(0xA5)
        restored = dict(runtime.materialized_weights())
        assert all(exact(restored[name], tensor) for name, tensor in expected.items())
        assert all(
            "weight" not in model.get_submodule(binding.module_name)._parameters
            for binding in exported.bindings
        )
        for workspace in runtime.workspaces.values():
            workspace.view(torch.uint8).fill_(0xA5)
        assert exact(model(inputs), output), "Forward did not reconstruct exact weights"
        gpu[spec.name] = {
            "chunks": len(exported.groups),
            "packed_chunks": sum(bool(g.codebook.vocab) for g in exported.groups),
            "weight_roundtrip_bitwise": True,
            "output_bitwise": True,
        }
        del runtime, restored, exported, model, inputs, output
        torch.cuda.empty_cache()
    if args.reference:
        assert json.loads(args.reference.read_text())["cpu"] == cpu, (
            "CPU archives changed"
        )
    if not args.cpu_only:
        for max_bits in (12, 32, 33, 56):
            for direct in (False, True):
                check_segments(max_bits, direct)
        gpu["segment_boundaries"] = {
            "code_bits": [12, 32, 33, 56],
            "dtypes": list(cpu),
            "staged_and_direct": True,
            "odd_tails": True,
            "bitwise": True,
        }
        # Rare symbols force the suffix LUT beyond its 12-bit prefix. Odd tail included.
        vocab = bytes((127, 127, 125, 126))
        counts = [1 << (i // 8) for i in range(258)]
        counts[256] = 1 << 52
        book = huff0.build_mixed_codebook(counts, vocab)
        lengths, values, longest = _huff0_cpp.export_code_table(
            book.codebook, max_bits=56, vocab=vocab
        )
        assert longest > 12
        symbols = np.array([127] * 70000 + list(range(256)) * 3 + [126], dtype=np.uint8)
        spec = DTYPES[torch.bfloat16]
        reordered = np.empty(len(symbols) * 2, dtype=np.uint8)
        reordered[::2] = np.arange(len(symbols), dtype=np.uint32).astype(np.uint8)
        reordered[1::2] = symbols
        raw = spec.reorder_bytes(reordered.tobytes(), restore=True)
        original = torch.frombuffer(bytearray(raw), dtype=torch.bfloat16).reshape(1, -1)
        model = nn.Module()
        model.layers = nn.ModuleList(
            [nn.Linear(original.numel(), 1, bias=False, dtype=torch.bfloat16)]
        )
        model.layers[0].weight.copy_(original)
        encoded = encode_gpu_native(
            raw, book.codebook, dtype=spec, stream_id=1, vocab=vocab
        )
        binding = LinearWeightBinding(
            "layers.0", "layers.0.weight", tuple(original.shape), original.numel(), 0
        )
        export = NeuralZipModelExport(
            (
                GPUWeightGroup(
                    "layers.0",
                    0,
                    GPUCodebook(tuple(lengths), tuple(values), vocab),
                    encoded,
                    (binding,),
                    0,
                ),
            )
        )
        runtime = NeuralZipRuntime.attach(model, export, device="cuda:0")
        assert exact(dict(runtime.materialized_weights())["layers.0.weight"], original)
        gpu["long_codes"] = {
            "max_code_bits": longest,
            "odd_tail": True,
            "bitwise": True,
        }
        # The public wrapper must also preserve packed inference after HF save/load.
        config = LlamaConfig(
            vocab_size=32,
            hidden_size=128,
            intermediate_size=256,
            num_hidden_layers=2,
            num_attention_heads=8,
            num_key_value_heads=2,
        )
        config._attn_implementation = "eager"
        reference = LlamaForCausalLM(config).bfloat16().eval()
        for module in reference.modules():
            if isinstance(module, nn.Linear):
                pattern = torch.tensor([0.25, 0.25, 0.5, 0.5], dtype=torch.bfloat16)
                module.weight.copy_(
                    pattern.repeat(module.weight.numel() // 4).view_as(module.weight)
                )
        expected_state = {
            name: tensor.clone() for name, tensor in reference.state_dict().items()
        }
        setup = neuralzip.setup(reference, chunk_size=8192)
        tokens = torch.arange(16, device="cuda").reshape(1, -1)
        expected_logits = reference.cuda()(tokens, use_cache=False).logits.cpu()
        with tempfile.TemporaryDirectory(prefix="nzip-packed-wrapper-") as folder:
            original_path, compressed_path = (
                Path(folder) / "original",
                Path(folder) / "compressed",
            )
            reference.cpu().save_pretrained(original_path)
            neuralzip(setup, device="cuda:0")
            for source in (original_path, compressed_path):
                restored = LlamaForCausalLM.from_pretrained(
                    source, dtype=torch.bfloat16, attn_implementation="eager"
                )
                assert any(
                    group.codebook.vocab
                    for group in restored.neuralzip_runtime.export.groups
                )
                actual = dict(restored.state_dict())
                actual.update(restored.neuralzip_runtime.materialized_weights())
                assert set(actual) == set(expected_state)
                assert all(
                    exact(actual[name], tensor)
                    for name, tensor in expected_state.items()
                )
                for workspace in restored.neuralzip_runtime.workspaces.values():
                    workspace.view(torch.uint8).fill_(0xA5)
                assert exact(
                    restored(tokens, use_cache=False).logits.cpu(), expected_logits
                )
                if source == original_path:
                    restored.save_pretrained(compressed_path)
        gpu["public_hf_wrapper"] = {
            "packed": True,
            "ordinary_load": True,
            "compressed_reload": True,
            "weights_bitwise": True,
            "output_bitwise": True,
        }
    root = decoder.KERNEL_PATH.parent.parent.parent
    files = ("gpu/decoder.py", "gpu/inference.py", "gpu/cuda/decode.cu")
    result = {
        "cpu": cpu,
        "gpu": gpu,
        "runtime_sha256": {
            name: hashlib.sha256((root / name).read_bytes()).hexdigest()
            for name in files
        },
        "native_sha256": hashlib.sha256(
            Path(_huff0_cpp.__file__).read_bytes()
        ).hexdigest(),
    }
    if args.output:
        with args.output.open("x") as stream:
            json.dump(result, stream, indent=2)
            stream.write("\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
