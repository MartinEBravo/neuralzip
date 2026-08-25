"""Backward-compatible imports for the historical experiment notebooks.

Reusable model experiments now live in :mod:`neuralzip.model_experiments`.
The checkpoint helper is isolated in :mod:`neuralzip.checkpoint_experiments`.
Training and federated orchestration belong to the final command-line runners
under ``examples/`` and are intentionally no longer duplicated in the library.
"""

from .checkpoint_experiments import compress_training_checkpoints
from .model_experiments import (
    DEFAULT_METHODS,
    FULL_CHECKPOINT_METHODS,
    MappingModel,
    _compress_mapping,
    _encoder_path_for_method,
    _ensure_dir,
    _load_diffusers_unet_mapping,
    _load_failure_row,
    _load_safetensors_mapping,
    _load_transformers_module,
    _load_torchvision_mapping,
    _load_vision_diffusion_mapping,
    _model_methods_completed,
    _model_spec_value,
    _normalise_dtype,
    _resolve_torchvision_weights,
    _result_exponent_metrics,
    _tensor_mapping_from_module,
    _upsert_model_compressor_row,
    run_hf_model_suite_experiment,
    run_lossy_lossless_experiment,
    run_moe_experiment,
    run_vision_diffusion_experiment,
)


def _removed_training_pipeline(name: str, replacement: str):
    raise RuntimeError(
        f"{name} was a duplicated historical pipeline and has been removed. "
        f"Run {replacement} instead; it owns the maintained CLI and output schema."
    )


def run_checkpoint_experiment_suite(*args, **kwargs):
    """Direct callers to the maintained checkpoint runner."""

    del args, kwargs
    return _removed_training_pipeline(
        "run_checkpoint_experiment_suite",
        "examples/run_final_checkpoint_suite.py",
    )


def run_distributed_experiment_suite(*args, **kwargs):
    """Direct callers to the maintained federated runners."""

    del args, kwargs
    return _removed_training_pipeline(
        "run_distributed_experiment_suite",
        "examples/run_final_federated_suite.py or "
        "examples/run_final_federated_language_suite.py",
    )


__all__ = (
    "DEFAULT_METHODS",
    "FULL_CHECKPOINT_METHODS",
    "MappingModel",
    "compress_training_checkpoints",
    "run_checkpoint_experiment_suite",
    "run_distributed_experiment_suite",
    "run_hf_model_suite_experiment",
    "run_lossy_lossless_experiment",
    "run_moe_experiment",
    "run_vision_diffusion_experiment",
)
