import json
from dataclasses import replace

import pytest
import torch
from safetensors import safe_open
from safetensors.torch import save_file

from neuralzip.gpu_checkpoint import (
    DEFAULT_FILENAME,
    FORMAT_VERSION,
    GPUCheckpointError,
    load_gpu_export,
    save_gpu_export,
)
from neuralzip.gpu_export import GPUCodebook, encode_bfloat16
from neuralzip.gpu_inference import (
    BF16WeightGroup,
    LinearWeightBinding,
    NeuralZipModelExport,
)


def _identity_codebook():
    return GPUCodebook.from_codes([8] * 256, range(256))


def _group(block_name, tensor_name, codebook, *, seed, module_kind="linear"):
    module_name = tensor_name.removesuffix(".weight")
    words = tuple((seed + index * 257) & 0xFFFF for index in range(16))
    raw = b"".join(word.to_bytes(2, "little") for word in words)
    binding = LinearWeightBinding(
        block_name=block_name,
        module_name=module_name,
        tensor_name=tensor_name,
        shape=(4, 4),
        numel=16,
        workspace_offset=0,
        group_offset=0,
        requires_grad=False,
        module_kind=module_kind,
    )
    return BF16WeightGroup(
        block_name=block_name,
        codebook_id=7,
        codebook=codebook,
        encoded=encode_bfloat16(raw, codebook),
        bindings=(binding,),
        workspace_offset=0,
    )


def _export():
    codebook = _identity_codebook()
    return NeuralZipModelExport(
        groups=(
            _group("model.layers.0", "model.layers.0.proj.weight", codebook, seed=3),
            _group(
                "model.layers.1",
                "model.layers.1.token_embed.weight",
                codebook,
                seed=11,
                module_kind="embedding",
            ),
        ),
        block_order=("model.layers.0", "model.layers.1"),
        resident_weight_names=("model.embed_tokens.weight",),
        resident_raw_bytes=128,
    )


def _read(path):
    with safe_open(str(path), framework="pt", device="cpu") as archive:
        tensors = {
            name: archive.get_tensor(name).clone()
            for name in archive.keys()  # noqa: SIM118 - safe_open is not iterable.
        }
        metadata = archive.metadata()
    assert metadata is not None
    return tensors, metadata, json.loads(metadata["neuralzip.manifest"])


def _rewrite(path, tensors, manifest, *, canonical=True):
    if canonical:
        source = json.dumps(
            manifest,
            allow_nan=False,
            ensure_ascii=True,
            separators=(",", ":"),
            sort_keys=True,
        )
    else:
        source = json.dumps(manifest, indent=2)
    save_file(tensors, str(path), metadata={"neuralzip.manifest": source})


def test_directory_roundtrip_is_bitwise_exact_and_deduplicates_luts(tmp_path):
    exported = _export()

    checkpoint = save_gpu_export(exported, tmp_path / "archive")
    restored = load_gpu_export(tmp_path / "archive")
    tensors, metadata, manifest = _read(checkpoint)

    assert checkpoint == tmp_path / "archive" / DEFAULT_FILENAME
    assert restored == exported
    assert restored.groups[1].bindings[0].module_kind == "embedding"
    assert metadata.keys() == {"neuralzip.manifest"}
    assert manifest["version"] == FORMAT_VERSION
    lut_names = [name for name in tensors if name.endswith(".lut")]
    assert lut_names == ["codebooks.000000.lut"]
    assert {entry["lut_tensor"] for entry in manifest["codebooks"]} == {
        "codebooks.000000.lut"
    }
    assert not list(checkpoint.parent.glob(f".{checkpoint.name}.*.tmp"))


def test_explicit_safetensors_file_roundtrips(tmp_path):
    target = tmp_path / "custom.safetensors"

    assert save_gpu_export(_export(), target) == target
    assert load_gpu_export(target) == _export()


@pytest.mark.parametrize("target", ["", "model.bin", "model.json"])
def test_paths_are_strict(tmp_path, target):
    path = target if not target else tmp_path / target
    with pytest.raises((TypeError, ValueError)):
        save_gpu_export(_export(), path)


def test_payload_corruption_is_detected_by_digest(tmp_path):
    checkpoint = save_gpu_export(_export(), tmp_path / "model.safetensors")
    tensors, _metadata, manifest = _read(checkpoint)
    tensors["groups.000000.bitstream"][0] ^= 1
    _rewrite(checkpoint, tensors, manifest)

    with pytest.raises(GPUCheckpointError, match="SHA-256"):
        load_gpu_export(checkpoint)


def test_missing_and_unexpected_tensors_are_rejected(tmp_path):
    checkpoint = save_gpu_export(_export(), tmp_path / "model.safetensors")
    tensors, _metadata, manifest = _read(checkpoint)
    tensors.pop("groups.000000.packed_gaps")
    tensors["attacker.payload"] = torch.zeros(1, dtype=torch.uint8)
    _rewrite(checkpoint, tensors, manifest)

    with pytest.raises(GPUCheckpointError, match="do not match"):
        load_gpu_export(checkpoint)


def test_noncanonical_or_unknown_manifest_fields_are_rejected(tmp_path):
    checkpoint = save_gpu_export(_export(), tmp_path / "model.safetensors")
    tensors, _metadata, manifest = _read(checkpoint)
    _rewrite(checkpoint, tensors, manifest, canonical=False)
    with pytest.raises(GPUCheckpointError, match="not canonical"):
        load_gpu_export(checkpoint)

    manifest["unexpected"] = True
    _rewrite(checkpoint, tensors, manifest)
    with pytest.raises(GPUCheckpointError, match="unexpected"):
        load_gpu_export(checkpoint)


def test_unsupported_format_version_is_rejected(tmp_path):
    checkpoint = save_gpu_export(_export(), tmp_path / "model.safetensors")
    tensors, _metadata, manifest = _read(checkpoint)
    manifest["version"] += 1
    _rewrite(checkpoint, tensors, manifest)

    with pytest.raises(GPUCheckpointError, match="Unsupported"):
        load_gpu_export(checkpoint)


def test_binding_paths_and_offsets_are_validated(tmp_path):
    checkpoint = save_gpu_export(_export(), tmp_path / "model.safetensors")
    tensors, _metadata, manifest = _read(checkpoint)
    binding = manifest["export"]["groups"][0]["bindings"][0]
    binding["tensor_name"] = "model.layers.9.proj.weight"
    _rewrite(checkpoint, tensors, manifest)
    with pytest.raises(GPUCheckpointError, match="tensor path"):
        load_gpu_export(checkpoint)

    binding["tensor_name"] = "model.layers.0.proj.weight"
    binding["workspace_offset"] = 1
    _rewrite(checkpoint, tensors, manifest)
    with pytest.raises(GPUCheckpointError, match="structure is inconsistent"):
        load_gpu_export(checkpoint)


def test_save_rejects_a_binding_outside_its_block(tmp_path):
    exported = _export()
    bad_binding = replace(
        exported.groups[0].bindings[0],
        module_name="model.layers.9.proj",
        tensor_name="model.layers.9.proj.weight",
    )
    bad_group = replace(exported.groups[0], bindings=(bad_binding,))
    bad_export = replace(exported, groups=(bad_group, exported.groups[1]))

    with pytest.raises(GPUCheckpointError, match="inside its transformer block"):
        save_gpu_export(bad_export, tmp_path / "model.safetensors")


def test_lut_corruption_is_detected_even_with_forged_tensor_digest(tmp_path):
    checkpoint = save_gpu_export(_export(), tmp_path / "model.safetensors")
    tensors, _metadata, manifest = _read(checkpoint)
    lut = tensors["codebooks.000000.lut"]
    lut[0] ^= 1
    raw = lut.view(torch.uint8).reshape(-1).numpy().tobytes()
    descriptor = manifest["tensors"]["codebooks.000000.lut"]
    import hashlib

    descriptor["sha256"] = hashlib.sha256(raw).hexdigest()
    _rewrite(checkpoint, tensors, manifest)

    with pytest.raises(GPUCheckpointError, match="LUT is inconsistent"):
        load_gpu_export(checkpoint)
