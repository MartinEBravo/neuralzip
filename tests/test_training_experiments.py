import pandas as pd
import torch

from neuralzip.experiment_scenarios import compress_training_checkpoints


def test_checkpoint_compression_fits_each_checkpoint_separately(tmp_path):
    checkpoint_dir = tmp_path / "checkpoints"
    checkpoint_dir.mkdir()
    base = torch.linspace(-1.0, 1.0, 4096, dtype=torch.float32).reshape(64, 64)
    for epoch in range(2):
        torch.save(
            {
                "epoch": epoch,
                "architecture": "resnet18",
                "model_state": {
                    "layer.weight": (base + epoch * 1e-3).contiguous(),
                    "layer.bias": torch.zeros(64, dtype=torch.float32),
                },
            },
            checkpoint_dir / f"resnet18_cifar100_epoch_{epoch:03d}.pt",
        )

    results, summary = compress_training_checkpoints(
        checkpoint_dir,
        tmp_path / "compression",
        architecture="resnet18",
        max_codebook_sample_bytes=1 << 20,
        verify=True,
    )

    assert len(results) == 4
    assert set(results["method"]) == {"ZipNN", "NeuralZip"}
    assert set(results["setup_scope"]) == {"per-checkpoint"}
    assert results["roundtrip_exact"].all()
    assert (results["exponent_raw_bytes"] > 0).all()
    assert (
        results.loc[results["method"] == "NeuralZip", "setup_seconds"] > 0
    ).all()
    assert set(summary["checkpoints"]) == {2}
    assert set(summary["method"]) == {"ZipNN", "NeuralZip"}

    saved = pd.read_csv(
        tmp_path / "compression" / "checkpoint_compression_results.csv"
    )
    assert len(saved) == len(results)
