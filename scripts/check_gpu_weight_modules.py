"""Run discovery checks, or pass --cuda for bitwise compressed-inference checks."""

import argparse
import copy
import hashlib
import json
from collections import Counter
from pathlib import Path

import torch
from torch import nn
from torchvision import models
from transformers import GPT2Config, GPT2LMHeadModel
from transformers.pytorch_utils import Conv1D

from neuralzip.gpu.inference import (
    NeuralZipRuntime,
    _binding_module,
    _gpu_candidates,
    _unique_candidates,
    export_model_for_gpu,
)
from neuralzip.setup import NeuralZipSetup


class Block(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = nn.Conv2d(32, 32, 3, padding=1, groups=32)
        self.linear = nn.Linear(32, 32)

    def forward(self, inputs):
        return self.linear(self.conv(inputs).permute(0, 2, 3, 1)).permute(0, 3, 1, 2)


class Nested(nn.Module):
    def __init__(self):
        super().__init__()
        stage = nn.Module()
        stage.downsample = nn.Conv2d(32, 32, 1)
        stage.blocks = nn.ModuleList([Block(), Block()])
        self.stages = nn.ModuleList([stage])

    def forward(self, inputs):
        for stage in self.stages:
            inputs = stage.downsample(inputs)
            for block in stage.blocks:
                inputs = block(inputs)
        return inputs


class FunctionalAttention(nn.Module):
    def __init__(self):
        super().__init__()
        self.before = nn.Linear(32, 32)
        self.attention = nn.MultiheadAttention(32, 4, batch_first=True)

    def forward(self, inputs):
        inputs = self.before(inputs)
        return self.attention(inputs, inputs, inputs, need_weights=False)[0]


def exact(left, right):
    return (
        left.shape == right.shape
        and left.dtype == right.dtype
        and torch.equal(
            left.detach().cpu().contiguous().reshape(-1).view(torch.uint8),
            right.detach().cpu().contiguous().reshape(-1).view(torch.uint8),
        )
    )


@torch.inference_mode()
def check(name, model, inputs, cuda):
    model = model.eval().bfloat16()
    if name in ("conv2d", "sequential", "nested", "multihead_attention"):
        for index, module in enumerate(model.modules(), 1):
            if isinstance(module, (nn.Linear, nn.Conv2d)):
                pattern = torch.tensor(
                    [0.03125, 0.03125, -0.0625, -0.0625], dtype=torch.bfloat16
                )
                if name == "multihead_attention":
                    pattern *= index  # Different blocks must overwrite shared storage.
                module.weight.copy_(
                    pattern.repeat((module.weight.numel() + 3) // 4)[
                        : module.weight.numel()
                    ].reshape_as(module.weight)
                )
    blocks, candidates = _gpu_candidates(model)
    candidates = _unique_candidates(model, candidates)
    counts = Counter(id(p) for _, p in model.named_parameters(remove_duplicate=False))
    expected = {
        f"{n}.weight" if n else "weight": m.weight.clone()
        for n, m in model.named_modules(remove_duplicate=False)
        if isinstance(m, (nn.Linear, nn.Conv2d, Conv1D)) and counts[id(m.weight)] == 1
    }
    assert expected and {c.tensor_name for c in candidates} == set(expected), name
    assert len(candidates) == len(expected), name
    assert not any(a != b and b.startswith(a + ".") for a in blocks for b in blocks), (
        blocks
    )
    if name == "nested":
        assert blocks == (
            "stages.0.downsample",
            "stages.0.blocks.0",
            "stages.0.blocks.1",
        ), blocks
    row = {"candidates": len(candidates), "discovery": True}
    if cuda:
        inputs = {
            key: value.cuda().to(torch.bfloat16)
            if value.is_floating_point()
            else value.cuda()
            for key, value in inputs.items()
        }
        reference = copy.deepcopy(model).cuda()
        output = reference(**inputs)
        output = (output if isinstance(output, torch.Tensor) else output.logits).cpu()
        del reference
        uses = []
        if name == "multihead_attention":

            def check_projection(consumer, _args):
                assert exact(
                    consumer.out_proj.weight, expected["attention.out_proj.weight"]
                )
                uses.append(True)

            def reject_child_forward(_module, _args):
                raise AssertionError("The projection must be consumed functionally")

            model.attention.register_forward_pre_hook(check_projection)
            model.attention.out_proj.register_forward_pre_hook(reject_child_forward)
        setup = NeuralZipSetup().build(model)
        exported = export_model_for_gpu(model, setup)
        assert exported.groups, name
        if name == "multihead_attention":
            assert any(
                g.block_name == "attention.out_proj" and g.codebook.vocab
                for g in exported.groups
            ), "Functional projection must remain packed, not excluded"
        for binding in exported.bindings:
            assert _binding_module(model, binding).weight.shape == binding.shape
        runtime = NeuralZipRuntime.attach(model, exported, device="cuda:0")
        for workspace in runtime.workspaces.values():
            workspace.view(torch.uint8).fill_(0xA5)
        restored = dict(runtime.materialized_weights())
        assert all(exact(tensor, expected[key]) for key, tensor in restored.items()), (
            name
        )
        assert set(restored) == {
            binding.tensor_name for binding in exported.bindings
        }, name
        for _ in range(2):
            for workspace in runtime.workspaces.values():
                workspace.view(torch.uint8).fill_(0xA5)
            actual = model(**inputs)
            actual = actual if isinstance(actual, torch.Tensor) else actual.logits
            assert exact(actual, output), name
        row.update(
            weights_bitwise=True,
            outputs_bitwise=True,
            compressed=len(restored),
            packed_chunks=sum(bool(group.codebook.vocab) for group in exported.groups),
        )
        if name == "multihead_attention":
            assert len(uses) == 2
            row["functional_consumer_bitwise"] = True
    print(json.dumps({name: row}), flush=True)
    return row


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cuda", action="store_true")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    torch.manual_seed(7)
    torch.set_num_threads(1)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.backends.cuda.matmul.allow_tf32 = False
    rows = {}
    for name in ("resnet18", "convnext_tiny", "efficientnet_b0"):
        rows[name] = check(
            name,
            getattr(models, name)(weights=None),
            {"x": torch.randn(1, 3, 224, 224)},
            args.cuda,
        )
    rows["conv2d"] = check(
        "conv2d",
        nn.Conv2d(32, 32, 3, padding=2, dilation=2, groups=4, padding_mode="reflect"),
        {"input": torch.randn(1, 32, 16, 16)},
        args.cuda,
    )
    rows["sequential"] = check(
        "sequential",
        nn.Sequential(
            nn.Conv2d(3, 32, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 32, 3, padding=1, groups=32),
        ),
        {"input": torch.randn(1, 3, 16, 16)},
        args.cuda,
    )
    rows["nested"] = check(
        "nested", Nested(), {"inputs": torch.randn(1, 32, 16, 16)}, args.cuda
    )
    rows["multihead_attention"] = check(
        "multihead_attention",
        FunctionalAttention(),
        {"inputs": torch.randn(1, 17, 32)},
        args.cuda,
    )
    rows["gpt2"] = check(
        "gpt2",
        GPT2LMHeadModel(
            GPT2Config(vocab_size=64, n_positions=32, n_embd=32, n_layer=2, n_head=4)
        ),
        {"input_ids": torch.arange(16).reshape(1, -1)},
        args.cuda,
    )
    if args.output:
        root = Path(export_model_for_gpu.__code__.co_filename).parent.parent
        rows["_runtime_sha256"] = {
            name: hashlib.sha256((root / name).read_bytes()).hexdigest()
            for name in ("gpu/inference.py", "gpu/decoder.py", "gpu/cuda/decode.cu")
        }
        with args.output.open("x") as stream:
            json.dump(rows, stream, indent=2)
            stream.write("\n")


if __name__ == "__main__":
    main()
