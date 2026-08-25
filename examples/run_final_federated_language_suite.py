"""Federated GPT-2/BERT compression trajectories for the thesis.

The default experiment complements the existing ResNet federated suite with
two language objectives over the real WikiText-2 raw training split:

* GPT-2, randomly initialized, trained with causal language modeling;
* BERT Base, randomly initialized, trained with deterministic 15% masking.

WikiText articles are reconstructed from their top-level headings and assigned
whole to ten simulated clients.  Each round performs one token-equivalent local
batch per client and FedAvg.  ZipNN source and NeuralZip encode exactly the same
unique FP32 storages; tied output/embedding parameters are transmitted once and
restored as aliases.  NeuralZip's clustering, mode selection, vocabularies, and
Huffman tables are prepared only on round 0 and remain frozen.

Every completed round is checkpointed before the CSVs are atomically refreshed,
so an interrupted run can resume only when its complete protocol fingerprint
matches.  Synthetic data and reduced model configurations are available solely
behind ``--smoke``::

    python examples/run_final_federated_language_suite.py \
        --models gpt2 bert-base --smoke --clients 2 --rounds 1 \
        --gpt2-sequence-length 16 --bert-sequence-length 16 \
        --tokens-per-client-batch 32 --threads 2 --device cpu \
        --output-dir /tmp/neuralzip-language-federated-smoke --overwrite

Run the two final models independently on different GPUs by passing one value
to ``--models`` in each process.  Do not place two processes for the same model
in the same output directory.
"""

from __future__ import annotations

import argparse
import csv
import fcntl
import hashlib
import json
import os
import platform
import random
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Mapping, Sequence

import torch
import transformers
from transformers import (
    AutoTokenizer,
    BertConfig,
    BertForMaskedLM,
    GPT2Config,
    GPT2LMHeadModel,
)

from neuralzip.hashing import sha256_file as _sha256_file
from neuralzip.system import (
    cpu_affinity as _cpu_affinity,
    parse_cpu_set as _parse_cpu_set_value,
)

try:
    from examples.federated_common import (
        COMPRESSOR_NEURALZIP,
        COMPRESSOR_SRC,
        DTYPE_NAME,
        EncodedState,
        FederatedCodecProtocol,
        NeuralZipStateCodec,
        ZipNNSrcStateCodec,
        _assert_same_tensors,
        _clone_tensor_map,
        _encoded_metrics,
        _write_csv_atomic,
        _write_json_atomic,
        build_round_rows,
        build_summary_rows,
        new_cumulative_metrics,
    )
except ModuleNotFoundError:  # Direct ``python examples/<script>.py`` execution.
    from federated_common import (  # type: ignore
        COMPRESSOR_NEURALZIP,
        COMPRESSOR_SRC,
        DTYPE_NAME,
        EncodedState,
        FederatedCodecProtocol,
        NeuralZipStateCodec,
        ZipNNSrcStateCodec,
        _assert_same_tensors,
        _clone_tensor_map,
        _encoded_metrics,
        _write_csv_atomic,
        _write_json_atomic,
        build_round_rows,
        build_summary_rows,
        new_cumulative_metrics,
    )


DATASET_NAME = "WikiText-2"
DATASET_REPOSITORY = "Salesforce/wikitext"
DATASET_FILENAME = "wikitext-2-raw-v1/train-00000-of-00001.parquet"
GPT2_TOKENIZER = "openai-community/gpt2"
BERT_TOKENIZER = "google-bert/bert-base-uncased"
MODEL_GPT2 = "gpt2"
MODEL_BERT = "bert-base"
PROTOCOL_VERSION = 3


def _parse_cpu_set(value: str) -> tuple[int, ...]:
    """Parse Linux CPU-list syntax such as ``0-3,8,10-11``."""

    try:
        cpus = _parse_cpu_set_value(value)
    except ValueError as error:
        raise argparse.ArgumentTypeError(
            f"Invalid CPU set {value!r}; expected for example 0-11 or 0-3,8-11."
        ) from error
    return tuple(sorted(cpus))


def _set_process_affinity(cpus: Sequence[int]) -> None:
    """Set affinity for every current thread, including native codec workers."""

    if not hasattr(os, "sched_setaffinity"):
        raise RuntimeError("CPU isolation requires Linux sched_setaffinity support.")
    cpu_set = set(int(cpu) for cpu in cpus)
    task_dir = Path("/proc/self/task")
    tids = [int(path.name) for path in task_dir.iterdir()] if task_dir.exists() else [0]
    for tid in tids:
        try:
            os.sched_setaffinity(tid, cpu_set)
        except ProcessLookupError:
            # A short-lived runtime thread may disappear between listing and set.
            continue


class TimingIsolation:
    """Serialize CPU-heavy codec phases and move them onto fixed timing CPUs."""

    def __init__(
        self,
        *,
        lock_path: Path | None,
        timing_cpus: Sequence[int] | None,
        background_cpus: Sequence[int] | None,
    ) -> None:
        self.lock_path = Path(lock_path).resolve() if lock_path is not None else None
        self.timing_cpus = (
            tuple(int(cpu) for cpu in timing_cpus) if timing_cpus is not None else None
        )
        self.background_cpus = (
            tuple(int(cpu) for cpu in background_cpus)
            if background_cpus is not None
            else None
        )
        affinity = _cpu_affinity()
        self.initial_cpus = tuple(affinity) if affinity is not None else None
        self._handle = None
        if self.lock_path is not None:
            self.lock_path.parent.mkdir(parents=True, exist_ok=True)
            self._handle = self.lock_path.open("a+b")

    @property
    def enabled(self) -> bool:
        return self._handle is not None or self.timing_cpus is not None

    def activate_background(self) -> None:
        if self.background_cpus is not None:
            _set_process_affinity(self.background_cpus)

    def __enter__(self) -> "TimingIsolation":
        if self._handle is not None:
            fcntl.flock(self._handle.fileno(), fcntl.LOCK_EX)
        try:
            if self.timing_cpus is not None:
                _set_process_affinity(self.timing_cpus)
        except BaseException:
            if self._handle is not None:
                fcntl.flock(self._handle.fileno(), fcntl.LOCK_UN)
            raise
        return self

    def __exit__(self, exc_type, exc_value, traceback) -> None:
        try:
            restore = self.background_cpus or self.initial_cpus
            if self.timing_cpus is not None and restore is not None:
                _set_process_affinity(restore)
        finally:
            if self._handle is not None:
                fcntl.flock(self._handle.fileno(), fcntl.LOCK_UN)


@dataclass(frozen=True)
class StateLayout:
    """Canonical state names and exact aliases for shared tensor storages."""

    names: tuple[str, ...]
    canonical_names: tuple[str, ...]
    floating_canonical_names: tuple[str, ...]
    canonical_for_name: Mapping[str, str]
    aliases: Mapping[str, str]


@dataclass(frozen=True)
class TokenizerSpec:
    identifier: str
    revision: str | None
    vocab_sha256: str
    vocab_size: int
    special_token_ids: tuple[int, ...]
    mask_token_id: int | None
    tokenizer: object | None


@dataclass
class ClientCorpus:
    """Fixed-length token sequences owned by one document-partitioned client."""

    input_ids: torch.Tensor
    document_count: int

    def batch(
        self,
        *,
        batch_size: int,
        seed: int,
        round_index: int,
        client_index: int,
        local_batch_index: int,
    ) -> torch.Tensor:
        if self.input_ids.ndim != 2 or self.input_ids.shape[0] == 0:
            raise RuntimeError(f"Client {client_index} has no token sequences.")
        generator = torch.Generator().manual_seed(
            _derived_seed(
                seed,
                "batch",
                round_index,
                client_index,
                local_batch_index,
            )
        )
        sample_count = int(self.input_ids.shape[0])
        if sample_count >= int(batch_size):
            indices = torch.randperm(sample_count, generator=generator)[:batch_size]
        else:
            indices = torch.randint(
                sample_count,
                (int(batch_size),),
                generator=generator,
            )
        return self.input_ids.index_select(0, indices).to(torch.long)


class FedAvgAccumulator:
    """Memory-bounded equal-client FedAvg over canonical tensor storages."""

    def __init__(self, layout: StateLayout):
        self.layout = layout
        self.count = 0
        self.values: dict[str, torch.Tensor] = {}

    def add(self, state: Mapping[str, torch.Tensor]) -> None:
        if tuple(state.keys()) != self.layout.names:
            raise AssertionError("Client state names changed before FedAvg.")
        for name in self.layout.canonical_names:
            value = state[name].detach().cpu()
            if value.is_floating_point():
                if name not in self.values:
                    self.values[name] = value.to(torch.float32).clone()
                else:
                    self.values[name].add_(value.to(torch.float32))
            elif name not in self.values:
                self.values[name] = value.clone()
        self.count += 1

    def finish(self) -> dict[str, torch.Tensor]:
        if self.count <= 0:
            raise ValueError("FedAvg requires at least one client state.")
        canonical = {}
        for name in self.layout.canonical_names:
            value = self.values[name]
            if value.is_floating_point():
                value = value.div(float(self.count))
            canonical[name] = value
        return _expand_canonical_state(canonical, self.layout)


def _derived_seed(seed: int, *parts: object) -> int:
    material = ":".join([str(int(seed)), *(str(part) for part in parts)])
    digest = hashlib.sha256(material.encode("utf-8")).digest()
    return int.from_bytes(digest[:8], "little") & ((1 << 63) - 1)


def _seed_everything(seed: int) -> None:
    random.seed(int(seed))
    torch.manual_seed(int(seed))
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(int(seed))


def _storage_signature(name: str, tensor: torch.Tensor) -> tuple[object, ...]:
    """Identify exact tensor aliases, not merely overlapping tensor views."""

    if tensor.numel() == 0:
        return ("empty", name)
    storage = tensor.untyped_storage()
    return (
        tensor.device.type,
        tensor.device.index,
        int(storage.data_ptr()),
        int(tensor.storage_offset()),
        tuple(tensor.shape),
        tuple(tensor.stride()),
        tensor.dtype,
    )


def _state_layout(model: torch.nn.Module) -> StateLayout:
    state = model.state_dict()
    signature_owner: dict[tuple[object, ...], str] = {}
    canonical_for_name: dict[str, str] = {}
    aliases: dict[str, str] = {}
    canonical_names: list[str] = []
    floating_names: list[str] = []
    for name, value in state.items():
        signature = _storage_signature(name, value)
        canonical = signature_owner.get(signature)
        if canonical is None:
            signature_owner[signature] = name
            canonical = name
            canonical_names.append(name)
            if value.is_floating_point():
                floating_names.append(name)
        else:
            aliases[name] = canonical
        canonical_for_name[name] = canonical
    return StateLayout(
        names=tuple(state.keys()),
        canonical_names=tuple(canonical_names),
        floating_canonical_names=tuple(floating_names),
        canonical_for_name=canonical_for_name,
        aliases=aliases,
    )


def _validate_model_layout(model: torch.nn.Module, layout: StateLayout) -> None:
    state = model.state_dict()
    if tuple(state.keys()) != layout.names:
        raise AssertionError("Model state names changed after initialization.")
    for alias, canonical in layout.aliases.items():
        alias_value = state[alias]
        canonical_value = state[canonical]
        if _storage_signature(alias, alias_value) != _storage_signature(
            canonical, canonical_value
        ):
            raise AssertionError(f"Tied storage was broken: {alias} -> {canonical}.")


def _snapshot_model_state(
    model: torch.nn.Module,
    layout: StateLayout,
) -> dict[str, torch.Tensor]:
    """Copy each canonical storage to CPU once and recreate exact aliases."""

    _validate_model_layout(model, layout)
    source = model.state_dict()
    canonical = {
        name: source[name].detach().cpu().contiguous().clone()
        for name in layout.canonical_names
    }
    return _expand_canonical_state(canonical, layout)


def _expand_canonical_state(
    canonical: Mapping[str, torch.Tensor],
    layout: StateLayout,
) -> dict[str, torch.Tensor]:
    if tuple(canonical.keys()) != layout.canonical_names:
        missing = set(layout.canonical_names).difference(canonical)
        extra = set(canonical).difference(layout.canonical_names)
        raise AssertionError(
            f"Canonical state mismatch; missing={sorted(missing)} extra={sorted(extra)}."
        )
    expanded = {
        name: canonical[layout.canonical_for_name[name]] for name in layout.names
    }
    for alias, owner in layout.aliases.items():
        if expanded[alias].data_ptr() != expanded[owner].data_ptr():
            raise AssertionError(f"Failed to restore tied alias {alias} -> {owner}.")
    return expanded


def _canonical_state(
    state: Mapping[str, torch.Tensor],
    layout: StateLayout,
) -> dict[str, torch.Tensor]:
    return {name: state[name] for name in layout.canonical_names}


def _unique_floating_state(
    state: Mapping[str, torch.Tensor],
    layout: StateLayout,
) -> dict[str, torch.Tensor]:
    return {name: state[name] for name in layout.floating_canonical_names}


def _merge_decoded_unique(
    base: Mapping[str, torch.Tensor],
    decoded: Mapping[str, torch.Tensor],
    layout: StateLayout,
) -> dict[str, torch.Tensor]:
    if tuple(decoded.keys()) != layout.floating_canonical_names:
        raise AssertionError("Decoded unique floating state changed tensor names.")
    canonical = {}
    for name in layout.canonical_names:
        source = decoded[name] if name in decoded else base[name]
        canonical[name] = source.detach().cpu().contiguous().clone()
    return _expand_canonical_state(canonical, layout)


def _resolve_wikitext_parquet(data_dir: Path, *, offline: bool) -> Path:
    if data_dir.is_file():
        if data_dir.suffix != ".parquet":
            raise ValueError(f"Expected a Parquet file, observed {data_dir}.")
        return data_dir
    expected = data_dir / DATASET_FILENAME
    if expected.exists():
        return expected
    candidates = sorted(data_dir.glob("**/train-*.parquet")) if data_dir.exists() else []
    raw_candidates = [path for path in candidates if "wikitext-2-raw-v1" in str(path)]
    if len(raw_candidates) == 1:
        return raw_candidates[0]
    if offline:
        raise FileNotFoundError(
            f"Real WikiText-2 was not found at {expected}; --offline forbids download."
        )
    try:
        from huggingface_hub import hf_hub_download
    except ImportError as error:
        raise ImportError(
            "huggingface-hub is required to download the real WikiText-2 Parquet."
        ) from error
    downloaded = hf_hub_download(
        repo_id=DATASET_REPOSITORY,
        repo_type="dataset",
        filename=DATASET_FILENAME,
        local_dir=str(data_dir),
    )
    path = Path(downloaded)
    if not path.exists():
        raise FileNotFoundError(f"WikiText download did not create {path}.")
    return path


_ARTICLE_HEADING = re.compile(r"^=\s+[^=].*?\s+=$")


def _reconstruct_wikitext_documents(lines: Sequence[str]) -> list[str]:
    """Reconstruct articles without joining tokens across document boundaries."""

    documents: list[str] = []
    current: list[str] = []
    for row in lines:
        for line in str(row).splitlines() or [""]:
            stripped = line.strip()
            if _ARTICLE_HEADING.fullmatch(stripped) and current:
                document = "\n".join(current).strip()
                if document:
                    documents.append(document)
                current = []
            if stripped or current:
                current.append(line)
    if current:
        document = "\n".join(current).strip()
        if document:
            documents.append(document)
    return documents


def _load_real_documents(data_dir: Path, *, offline: bool) -> tuple[list[str], dict[str, object]]:
    parquet_path = _resolve_wikitext_parquet(data_dir, offline=offline)
    try:
        import pyarrow.parquet as pq
    except ImportError as error:
        raise ImportError(
            "pyarrow is required to read the real WikiText-2 Parquet."
        ) from error
    table = pq.read_table(parquet_path, columns=["text"])
    documents = _reconstruct_wikitext_documents(table.column("text").to_pylist())
    if len(documents) < 10:
        raise RuntimeError(
            f"Only {len(documents)} WikiText documents were reconstructed; expected many."
        )
    return documents, {
        "dataset_repository": DATASET_REPOSITORY,
        "dataset_file": str(parquet_path.resolve()),
        "dataset_file_sha256": _sha256_file(parquet_path),
        "document_count": len(documents),
        "reconstruction": "top-level WikiText headings; no cross-document sequences",
    }


def _partition_documents(
    documents: Sequence[str],
    *,
    clients: int,
    seed: int,
) -> list[list[str]]:
    """Greedily balance whole documents by UTF-8 bytes across clients."""

    if len(documents) < int(clients):
        raise ValueError("Document partition requires at least one article per client.")
    ranked = sorted(
        enumerate(documents),
        key=lambda item: (
            -len(item[1].encode("utf-8")),
            _derived_seed(seed, "document", item[0]),
        ),
    )
    partitions: list[list[tuple[int, str]]] = [[] for _ in range(int(clients))]
    loads = [0] * int(clients)
    for document_index, document in ranked:
        client = min(range(int(clients)), key=lambda index: (loads[index], index))
        partitions[client].append((document_index, document))
        loads[client] += len(document.encode("utf-8"))
    return [
        [document for _, document in sorted(partition)] for partition in partitions
    ]


def _tokenizer_spec(model_name: str, *, offline: bool) -> TokenizerSpec:
    identifier = GPT2_TOKENIZER if model_name == MODEL_GPT2 else BERT_TOKENIZER
    tokenizer = AutoTokenizer.from_pretrained(
        identifier,
        use_fast=True,
        local_files_only=bool(offline),
    )
    vocabulary = tokenizer.get_vocab()
    encoded_vocabulary = json.dumps(
        sorted((str(token), int(index)) for token, index in vocabulary.items()),
        ensure_ascii=False,
        separators=(",", ":"),
    ).encode("utf-8")
    revision = tokenizer.init_kwargs.get("_commit_hash")
    return TokenizerSpec(
        identifier=identifier,
        revision=str(revision) if revision else None,
        vocab_sha256=hashlib.sha256(encoded_vocabulary).hexdigest(),
        vocab_size=len(tokenizer),
        special_token_ids=tuple(sorted(set(int(value) for value in tokenizer.all_special_ids))),
        mask_token_id=(
            int(tokenizer.mask_token_id)
            if tokenizer.mask_token_id is not None
            else None
        ),
        tokenizer=tokenizer,
    )


def _synthetic_tokenizer_spec(model_name: str) -> TokenizerSpec:
    if model_name == MODEL_BERT:
        special_ids = (0, 101, 102, 103)
        mask_id = 103
    else:
        special_ids = (0,)
        mask_id = None
    return TokenizerSpec(
        identifier="synthetic-smoke-tokenizer",
        revision=None,
        vocab_sha256=hashlib.sha256(model_name.encode("utf-8")).hexdigest(),
        vocab_size=257,
        special_token_ids=special_ids,
        mask_token_id=mask_id,
        tokenizer=None,
    )


def _tokenize_partition(
    documents: Sequence[str],
    *,
    model_name: str,
    sequence_length: int,
    tokenizer_spec: TokenizerSpec,
) -> torch.Tensor:
    tokenizer = tokenizer_spec.tokenizer
    if tokenizer is None:
        raise RuntimeError("A real tokenizer is required for real WikiText-2.")
    sequences: list[list[int]] = []
    content_length = (
        int(sequence_length) if model_name == MODEL_GPT2 else int(sequence_length) - 2
    )
    for document in documents:
        token_ids = tokenizer.encode(document, add_special_tokens=False)
        for start in range(0, len(token_ids) - content_length + 1, content_length):
            chunk = token_ids[start : start + content_length]
            if model_name == MODEL_BERT:
                cls_token_id = tokenizer.cls_token_id
                sep_token_id = tokenizer.sep_token_id
                if cls_token_id is None or sep_token_id is None:
                    raise RuntimeError(
                        "BERT tokenizer must define cls_token_id and sep_token_id."
                    )
                # Transformers 5.14 no longer exposes
                # build_inputs_with_special_tokens on BertTokenizer.  Construct
                # the canonical single-sequence input explicitly, matching the
                # checkpoint runner: [CLS] content [SEP].
                chunk = [int(cls_token_id), *chunk, int(sep_token_id)]
            if len(chunk) == int(sequence_length):
                sequences.append([int(value) for value in chunk])
    if not sequences:
        return torch.empty((0, int(sequence_length)), dtype=torch.int32)
    return torch.tensor(sequences, dtype=torch.int32)


def _real_client_corpora(
    documents: Sequence[str],
    *,
    clients: int,
    seed: int,
    model_name: str,
    sequence_length: int,
    tokenizer_spec: TokenizerSpec,
) -> tuple[list[ClientCorpus], str]:
    document_partitions = _partition_documents(documents, clients=clients, seed=seed)
    corpora = []
    assignment_digest = hashlib.sha256()
    for client_index, partition in enumerate(document_partitions):
        assignment_digest.update(f"client={client_index}\n".encode("utf-8"))
        for document in partition:
            assignment_digest.update(hashlib.sha256(document.encode("utf-8")).digest())
        sequences = _tokenize_partition(
            partition,
            model_name=model_name,
            sequence_length=sequence_length,
            tokenizer_spec=tokenizer_spec,
        )
        if sequences.shape[0] == 0:
            raise RuntimeError(
                f"Client {client_index} has no full {sequence_length}-token sequence."
            )
        corpora.append(ClientCorpus(sequences, len(partition)))
    return corpora, assignment_digest.hexdigest()


def _synthetic_client_corpora(
    *,
    clients: int,
    seed: int,
    model_name: str,
    sequence_length: int,
    tokenizer_spec: TokenizerSpec,
) -> tuple[list[ClientCorpus], str]:
    corpora = []
    for client_index in range(int(clients)):
        generator = torch.Generator().manual_seed(
            _derived_seed(seed, "synthetic", model_name, client_index)
        )
        sequences = torch.randint(
            104 if model_name == MODEL_BERT else 1,
            tokenizer_spec.vocab_size,
            (32, int(sequence_length)),
            generator=generator,
            dtype=torch.int32,
        )
        if model_name == MODEL_BERT:
            sequences[:, 0] = 101
            sequences[:, -1] = 102
        corpora.append(ClientCorpus(sequences, 1))
    digest = hashlib.sha256(
        f"synthetic:{seed}:{clients}:{model_name}:{sequence_length}".encode("utf-8")
    ).hexdigest()
    return corpora, digest


def _build_model(
    model_name: str,
    *,
    tokenizer_spec: TokenizerSpec,
    sequence_length: int,
    smoke: bool,
) -> torch.nn.Module:
    if model_name == MODEL_GPT2:
        if smoke:
            config = GPT2Config(
                vocab_size=tokenizer_spec.vocab_size,
                n_positions=max(32, int(sequence_length)),
                n_ctx=max(32, int(sequence_length)),
                n_embd=32,
                n_layer=2,
                n_head=2,
                bos_token_id=0,
                eos_token_id=0,
            )
        else:
            config = GPT2Config(vocab_size=tokenizer_spec.vocab_size)
        config._attn_implementation = "eager"
        return GPT2LMHeadModel(config)
    if model_name == MODEL_BERT:
        if smoke:
            config = BertConfig(
                vocab_size=tokenizer_spec.vocab_size,
                hidden_size=32,
                num_hidden_layers=2,
                num_attention_heads=2,
                intermediate_size=64,
                max_position_embeddings=max(32, int(sequence_length)),
            )
        else:
            config = BertConfig(vocab_size=tokenizer_spec.vocab_size)
        config._attn_implementation = "eager"
        return BertForMaskedLM(config)
    raise ValueError(f"Unsupported language model: {model_name}.")


def _bert_mask(
    input_ids: torch.Tensor,
    *,
    tokenizer_spec: TokenizerSpec,
    seed: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    if tokenizer_spec.mask_token_id is None:
        raise ValueError("BERT masking requires a mask token ID.")
    generator = torch.Generator().manual_seed(int(seed))
    eligible = torch.ones_like(input_ids, dtype=torch.bool)
    for token_id in tokenizer_spec.special_token_ids:
        eligible.logical_and_(input_ids.ne(int(token_id)))
    selected = torch.rand(input_ids.shape, generator=generator).lt(0.15) & eligible
    if not bool(selected.any()):
        candidate = eligible.nonzero(as_tuple=False)
        if candidate.numel() == 0:
            raise RuntimeError("BERT batch contains no maskable token.")
        selected[tuple(candidate[0].tolist())] = True

    labels = input_ids.clone()
    labels[~selected] = -100
    masked = input_ids.clone()
    decisions = torch.rand(input_ids.shape, generator=generator)
    replace_with_mask = selected & decisions.lt(0.8)
    replace_randomly = selected & decisions.ge(0.8) & decisions.lt(0.9)
    masked[replace_with_mask] = int(tokenizer_spec.mask_token_id)
    random_words = torch.randint(
        tokenizer_spec.vocab_size,
        input_ids.shape,
        generator=generator,
    )
    masked[replace_randomly] = random_words[replace_randomly]
    return masked, labels


def _train_one_client(
    model: torch.nn.Module,
    model_name: str,
    corpus: ClientCorpus,
    *,
    client_index: int,
    round_index: int,
    local_batches: int,
    batch_size: int,
    tokenizer_spec: TokenizerSpec,
    device: torch.device,
    learning_rate: float,
    weight_decay: float,
    seed: int,
) -> float:
    model.train()
    client_seed = _derived_seed(seed, "train", round_index, client_index)
    _seed_everything(client_seed)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(learning_rate),
        weight_decay=float(weight_decay),
    )
    weighted_loss = 0.0
    examples = 0
    for local_batch_index in range(int(local_batches)):
        input_ids = corpus.batch(
            batch_size=batch_size,
            seed=seed,
            round_index=round_index,
            client_index=client_index,
            local_batch_index=local_batch_index,
        )
        optimizer.zero_grad(set_to_none=True)
        if model_name == MODEL_GPT2:
            device_input = input_ids.to(device=device, non_blocking=True)
            output = model(input_ids=device_input, labels=device_input)
        else:
            masked, labels = _bert_mask(
                input_ids,
                tokenizer_spec=tokenizer_spec,
                seed=_derived_seed(
                    seed,
                    "mask",
                    round_index,
                    client_index,
                    local_batch_index,
                ),
            )
            output = model(
                input_ids=masked.to(device=device, non_blocking=True),
                attention_mask=torch.ones_like(masked).to(
                    device=device, non_blocking=True
                ),
                labels=labels.to(device=device, non_blocking=True),
            )
        loss = output.loss
        if loss is None or not bool(torch.isfinite(loss.detach()).item()):
            raise FloatingPointError(
                f"Non-finite loss for {model_name}, round {round_index}, "
                f"client {client_index}."
            )
        loss.backward()
        optimizer.step()
        weighted_loss += float(loss.detach().cpu()) * int(input_ids.shape[0])
        examples += int(input_ids.shape[0])
    optimizer.zero_grad(set_to_none=True)
    del optimizer
    return weighted_loss / max(examples, 1)


def _append_safe_csv(path: Path, rows: list[dict[str, object]]) -> None:
    """Create an empty-header-free file only through the atomic full rewrite."""

    if rows:
        _write_csv_atomic(path, rows)


def _save_torch_atomic(path: Path, payload: Mapping[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    torch.save(dict(payload), temporary)
    os.replace(temporary, path)


def _read_csv(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        return []
    with path.open(newline="") as handle:
        return list(csv.DictReader(handle))


def _validate_checkpoint_rows(
    rows: Sequence[Mapping[str, object]],
    *,
    completed_rounds: int,
) -> None:
    if len(rows) != int(completed_rounds) * 2:
        raise RuntimeError(
            "Resume checkpoint has an incomplete per-round result table."
        )
    expected = {COMPRESSOR_SRC, COMPRESSOR_NEURALZIP}
    for round_index in range(1, int(completed_rounds) + 1):
        observed = {
            str(row["compressor"])
            for row in rows
            if int(row["round"]) == round_index
        }
        if observed != expected:
            raise RuntimeError(
                f"Resume checkpoint round {round_index} has compressors {observed}."
            )


def _protocol_fingerprint(payload: Mapping[str, object]) -> str:
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode(
        "utf-8"
    )
    return hashlib.sha256(encoded).hexdigest()


def _run_model(
    args: argparse.Namespace,
    *,
    model_name: str,
    documents: Sequence[str] | None,
    dataset_metadata: Mapping[str, object],
) -> None:
    model_dir = Path(args.output_dir) / model_name
    round_path = model_dir / "federated_round_results.csv"
    summary_path = model_dir / "federated_summary.csv"
    metadata_path = model_dir / "run_metadata.json"
    checkpoint_path = model_dir / "resume_state.pt"

    if args.overwrite:
        for path in (round_path, summary_path, metadata_path, checkpoint_path):
            if path.exists():
                path.unlink()
    elif not args.resume and any(
        path.exists() for path in (round_path, summary_path, metadata_path, checkpoint_path)
    ):
        raise FileExistsError(
            f"{model_dir} already contains a run; use --resume or --overwrite."
        )
    elif args.resume and not checkpoint_path.exists() and any(
        path.exists() for path in (round_path, summary_path, metadata_path)
    ):
        raise RuntimeError(
            f"{model_dir} has result files but no resume checkpoint; refusing unsafe resume."
        )

    if args.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError(f"CUDA device requested but unavailable: {device}.")

    timing_isolation = TimingIsolation(
        lock_path=args.timing_lock_path,
        timing_cpus=args.timing_cpus,
        background_cpus=args.background_cpus,
    )
    timing_isolation.activate_background()

    sequence_length = (
        int(args.gpt2_sequence_length)
        if model_name == MODEL_GPT2
        else int(args.bert_sequence_length)
    )
    if int(args.tokens_per_client_batch) < sequence_length:
        raise ValueError(
            "--tokens-per-client-batch must be at least the selected sequence length."
        )
    batch_size = int(args.tokens_per_client_batch) // sequence_length
    actual_tokens = batch_size * sequence_length

    tokenizer_spec = (
        _synthetic_tokenizer_spec(model_name)
        if args.smoke
        else _tokenizer_spec(model_name, offline=args.offline)
    )
    if args.smoke:
        corpora, assignment_sha256 = _synthetic_client_corpora(
            clients=args.clients,
            seed=args.seed,
            model_name=model_name,
            sequence_length=sequence_length,
            tokenizer_spec=tokenizer_spec,
        )
    else:
        if documents is None:
            raise AssertionError("Real mode requires reconstructed WikiText documents.")
        corpora, assignment_sha256 = _real_client_corpora(
            documents,
            clients=args.clients,
            seed=args.seed,
            model_name=model_name,
            sequence_length=sequence_length,
            tokenizer_spec=tokenizer_spec,
        )

    _seed_everything(args.seed)
    initial_model = _build_model(
        model_name,
        tokenizer_spec=tokenizer_spec,
        sequence_length=sequence_length,
        smoke=args.smoke,
    )
    layout = _state_layout(initial_model)
    if not layout.aliases:
        raise AssertionError(
            f"{model_name} unexpectedly has no tied storage to deduplicate."
        )
    codec_protocol = FederatedCodecProtocol(
        threads=args.threads,
        chunk_size=args.chunk_size,
        threshold=args.compression_threshold,
    )
    # State materialization and setup are not reported as encode time, but they
    # are serialized because their memory traffic would otherwise perturb a
    # concurrent model's timed codec section.
    with timing_isolation:
        initial_state = _snapshot_model_state(initial_model, layout)
        initial_floating = _unique_floating_state(initial_state, layout)
        codec_protocol.prepare_once(initial_floating)
    codecs = codec_protocol.codecs
    fresh_setup_seconds = dict(codec_protocol.setup_seconds)
    fresh_setup_bytes = dict(codec_protocol.setup_bytes)

    cpu_affinity = (
        list(args.timing_cpus)
        if args.timing_cpus is not None
        else _cpu_affinity()
    )
    device_name = (
        torch.cuda.get_device_name(device) if device.type == "cuda" else None
    )
    protocol = {
        "protocol_version": PROTOCOL_VERSION,
        "model": model_name,
        "model_configuration": "smoke-reduced" if args.smoke else "official-default",
        "dataset_file_sha256": dataset_metadata["dataset_file_sha256"],
        "document_assignment_sha256": assignment_sha256,
        "tokenizer": tokenizer_spec.identifier,
        "tokenizer_revision": tokenizer_spec.revision,
        "tokenizer_vocab_sha256": tokenizer_spec.vocab_sha256,
        "clients": int(args.clients),
        "local_batches": int(args.local_batches),
        "sequence_length": sequence_length,
        "batch_size": batch_size,
        "tokens_per_client_batch": actual_tokens,
        "learning_rate": float(args.learning_rate),
        "weight_decay": float(args.weight_decay),
        "seed": int(args.seed),
        "device": str(device),
        "device_name": device_name,
        "host": platform.node(),
        "cpu_affinity": cpu_affinity,
        "timing_serialized": args.timing_lock_path is not None,
        "timing_cpus": (
            list(args.timing_cpus) if args.timing_cpus is not None else None
        ),
        "background_cpus": (
            list(args.background_cpus) if args.background_cpus is not None else None
        ),
        "threads": int(args.threads),
        "chunk_size": int(args.chunk_size),
        "compression_threshold": float(args.compression_threshold),
        "automatic_gc_during_encode": False,
        "torch": torch.__version__,
        "transformers": transformers.__version__,
        "smoke": bool(args.smoke),
    }
    fingerprint = _protocol_fingerprint(protocol)

    completed_rounds = 0
    all_rows: list[dict[str, object]] = []
    cumulative: dict[str, dict[str, float | int]] = new_cumulative_metrics()
    setup_seconds = fresh_setup_seconds
    setup_bytes = fresh_setup_bytes
    server_state = initial_state

    if args.resume and checkpoint_path.exists():
        with timing_isolation:
            checkpoint = torch.load(
                checkpoint_path, map_location="cpu", weights_only=False
            )
        if checkpoint.get("protocol_fingerprint") != fingerprint:
            raise RuntimeError(
                "Resume protocol fingerprint differs from this command; use a fresh "
                "output directory or --overwrite."
            )
        completed_rounds = int(checkpoint["completed_rounds"])
        all_rows = list(checkpoint["round_rows"])
        _validate_checkpoint_rows(all_rows, completed_rounds=completed_rounds)
        cumulative = checkpoint["cumulative"]
        setup_seconds = checkpoint["setup_seconds"]
        setup_bytes = checkpoint["setup_bytes"]
        if int(fresh_setup_bytes[COMPRESSOR_NEURALZIP]) != int(
            setup_bytes[COMPRESSOR_NEURALZIP]
        ):
            raise RuntimeError(
                "The deterministically rebuilt NeuralZip round-0 plan has a "
                "different metadata size; refusing unsafe resume."
            )
        with timing_isolation:
            canonical_server = checkpoint["server_canonical_state"]
            server_state = _expand_canonical_state(canonical_server, layout)
        existing_rows = _read_csv(round_path)
        if len(existing_rows) != len(all_rows):
            print(
                f"[{model_name}] repairing incremental CSV from the atomic checkpoint",
                flush=True,
            )
        _append_safe_csv(round_path, all_rows)
        _write_csv_atomic(
            summary_path,
            build_summary_rows(
                prefix={
                    "architecture": model_name,
                    "dataset": DATASET_NAME,
                },
                dimensions={
                    "completed_rounds": int(completed_rounds),
                    "clients": int(args.clients),
                    "local_batches": int(args.local_batches),
                    "batch_size": int(batch_size),
                    "sequence_length": int(sequence_length),
                    "tokens_per_client_batch": int(batch_size)
                    * int(sequence_length),
                    "dtype": DTYPE_NAME,
                },
                clients=args.clients,
                cumulative=cumulative,
                setup_seconds=setup_seconds,
                setup_bytes=setup_bytes,
            ),
        )
        print(f"[{model_name}] resume from round {completed_rounds}", flush=True)

    # Warm both complete paths on the same round-0 state.  On resume this also
    # validates the deterministically rebuilt frozen plan before any new row.
    with timing_isolation:
        codec_protocol.warmup(initial_floating, context=model_name)
        del initial_floating

    metadata = {
        **protocol,
        **dataset_metadata,
        "protocol_fingerprint": fingerprint,
        "target_rounds": int(args.rounds),
        "dtype": DTYPE_NAME,
        "task": (
            "causal language modeling"
            if model_name == MODEL_GPT2
            else "masked language modeling"
        ),
        "mask_probability": None if model_name == MODEL_GPT2 else 0.15,
        "random_initialization": True,
        "pretrained_tokenizer_only": not bool(args.smoke),
        "document_partition": "whole reconstructed articles, byte-balanced",
        "documents_per_client": [corpus.document_count for corpus in corpora],
        "sequences_per_client": [int(corpus.input_ids.shape[0]) for corpus in corpora],
        "unique_state_tensors": len(layout.canonical_names),
        "unique_floating_tensors": len(layout.floating_canonical_names),
        "tied_aliases_elided": dict(layout.aliases),
        "compressors": [COMPRESSOR_SRC, COMPRESSOR_NEURALZIP],
        "neuralzip_plan": "built once on random initialization (round 0) and frozen",
        "trajectory": (
            "NeuralZip-decoded canonical FP32 storages after bitwise equality "
            "with ZipNN-src; exact tied aliases restored before training and FedAvg"
        ),
        "time_metric": (
            "downlink encode plus maximum client uplink encode per round; "
            "decode, training, alias restoration, setup, and automatic cyclic "
            "GC excluded"
        ),
        "automatic_gc_during_encode": False,
        "ratio_metric": (
            "exact exponent-group payload plus group metadata divided by raw "
            "exponent symbols over the same unique-storage messages"
        ),
        "resume": (
            "server canonical state, cumulative metrics, and all complete round rows "
            "atomically checkpointed; resume requires an exact protocol fingerprint"
        ),
        "setup": {
            name: {"seconds": setup_seconds[name], "bytes": setup_bytes[name]}
            for name in codecs
        },
        "host": platform.node(),
        "python": platform.python_version(),
    }
    _write_json_atomic(metadata_path, metadata)

    if completed_rounds >= int(args.rounds):
        print(
            f"[{model_name}] already complete: {completed_rounds}/{args.rounds} rounds",
            flush=True,
        )
        return

    with timing_isolation:
        training_model = initial_model.to(device=device, dtype=torch.float32)
    del initial_state

    for round_index in range(completed_rounds + 1, int(args.rounds) + 1):
        order = codec_protocol.encoding_order(round_index)
        # Keep all CPU-heavy state handling inside the same global critical
        # section as the codec calls.  Only each adapter's internal encode
        # timer is reported; serialization itself remains outside that timer.
        with timing_isolation:
            floating_server = _unique_floating_state(server_state, layout)
            downlink_metrics, decoded_floating_server = (
                codec_protocol.encode_verified(
                    floating_server,
                    order=order,
                    context=f"{model_name} round {round_index}",
                    phase="downlink",
                )
            )
            decoded_server_state = _merge_decoded_unique(
                server_state,
                decoded_floating_server,
                layout,
            )
            del decoded_floating_server, floating_server

        uplink_metrics: dict[str, list[EncodedState]] = {name: [] for name in codecs}
        fedavg = FedAvgAccumulator(layout)
        client_losses = []
        for client_index, corpus in enumerate(corpora):
            # Loading the full server state is also memory/PCIe intensive, so
            # isolate it from the other process's measured encode.
            with timing_isolation:
                training_model.load_state_dict(decoded_server_state, strict=True)
            client_loss = _train_one_client(
                training_model,
                model_name,
                corpus,
                client_index=client_index,
                round_index=round_index,
                local_batches=args.local_batches,
                batch_size=batch_size,
                tokenizer_spec=tokenizer_spec,
                device=device,
                learning_rate=args.learning_rate,
                weight_decay=args.weight_decay,
                seed=args.seed,
            )
            client_losses.append(client_loss)
            with timing_isolation:
                client_state = _snapshot_model_state(training_model, layout)
                floating_client = _unique_floating_state(client_state, layout)
                measurements, decoded_floating_client = (
                    codec_protocol.encode_verified(
                        floating_client,
                        order=codec_protocol.encoding_order(
                            round_index,
                            client_index=client_index,
                        ),
                        context=(
                            f"{model_name} round {round_index} client "
                            f"{client_index}"
                        ),
                        phase="uplink",
                    )
                )
                for name, encoded in measurements.items():
                    uplink_metrics[name].append(encoded)
                canonical_client = _merge_decoded_unique(
                    client_state,
                    decoded_floating_client,
                    layout,
                )
                fedavg.add(canonical_client)
                del (
                    client_state,
                    floating_client,
                    decoded_floating_client,
                    canonical_client,
                )
        with timing_isolation:
            server_state = fedavg.finish()
        del fedavg, decoded_server_state

        round_rows = build_round_rows(
            prefix={
                "architecture": model_name,
                "dataset": DATASET_NAME,
                "round": round_index,
                "clients": int(args.clients),
                "local_batches": int(args.local_batches),
                "batch_size": batch_size,
                "sequence_length": sequence_length,
                "tokens_per_client_batch": actual_tokens,
                "dtype": DTYPE_NAME,
            },
            suffix={
                "train_loss_mean": sum(client_losses) / len(client_losses),
                "roundtrip_ok": True,
            },
            downlinks=downlink_metrics,
            uplinks=uplink_metrics,
            clients=args.clients,
            cumulative=cumulative,
            setup_seconds=setup_seconds,
            setup_bytes=setup_bytes,
        )
        all_rows.extend(round_rows)
        completed_rounds = round_index
        summary_rows = build_summary_rows(
            prefix={
                "architecture": model_name,
                "dataset": DATASET_NAME,
            },
            dimensions={
                "completed_rounds": int(completed_rounds),
                "clients": int(args.clients),
                "local_batches": int(args.local_batches),
                "batch_size": int(batch_size),
                "sequence_length": int(sequence_length),
                "tokens_per_client_batch": int(batch_size)
                * int(sequence_length),
                "dtype": DTYPE_NAME,
            },
            clients=args.clients,
            cumulative=cumulative,
            setup_seconds=setup_seconds,
            setup_bytes=setup_bytes,
        )
        with timing_isolation:
            _save_torch_atomic(
                checkpoint_path,
                {
                    "protocol_fingerprint": fingerprint,
                    "completed_rounds": completed_rounds,
                    "server_canonical_state": _canonical_state(server_state, layout),
                    "round_rows": all_rows,
                    "cumulative": cumulative,
                    "setup_seconds": setup_seconds,
                    "setup_bytes": setup_bytes,
                },
            )
        _write_csv_atomic(round_path, all_rows)
        _write_csv_atomic(summary_path, summary_rows)
        print(
            f"{model_name} round={round_index}/{args.rounds} "
            f"loss={sum(client_losses) / len(client_losses):.4f} "
            + " ".join(
                f"{row['compressor']}={row['encoding_critical_seconds']:.3f}s/"
                f"{row['exponent_ratio_pct']:.3f}%"
                for row in round_rows
            ),
            flush=True,
        )


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--models",
        nargs="+",
        choices=(MODEL_GPT2, MODEL_BERT),
        default=(MODEL_GPT2, MODEL_BERT),
    )
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=Path("examples/outputs/datasets/wikitext"),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("examples/outputs/final_federated_wikitext_200x10"),
    )
    parser.add_argument("--clients", type=int, default=10)
    parser.add_argument("--rounds", type=int, default=200)
    parser.add_argument("--local-batches", type=int, default=1)
    parser.add_argument("--tokens-per-client-batch", type=int, default=2048)
    parser.add_argument("--gpt2-sequence-length", type=int, default=256)
    parser.add_argument("--bert-sequence-length", type=int, default=128)
    parser.add_argument("--learning-rate", type=float, default=5e-5)
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--threads", type=int, default=12)
    parser.add_argument(
        "--timing-lock-path",
        type=Path,
        help=(
            "Shared flock file used to serialize CPU-heavy codec/state phases "
            "across concurrently trained models."
        ),
    )
    parser.add_argument(
        "--timing-cpus",
        type=_parse_cpu_set,
        help="CPUs used only while holding the timing lock, e.g. 0-11.",
    )
    parser.add_argument(
        "--background-cpus",
        type=_parse_cpu_set,
        help=(
            "Disjoint CPUs used for GPU training and orchestration outside the "
            "timing lock, e.g. 12-15,28-31 on relela-06."
        ),
    )
    parser.add_argument("--chunk-size", type=int, default=256 << 10)
    parser.add_argument("--compression-threshold", type=float, default=0.95)
    parser.add_argument("--offline", action="store_true")
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument(
        "--resume",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    for field in (
        "clients",
        "rounds",
        "local_batches",
        "tokens_per_client_batch",
        "gpt2_sequence_length",
        "bert_sequence_length",
        "threads",
        "chunk_size",
    ):
        if int(getattr(args, field)) <= 0:
            parser.error(f"--{field.replace('_', '-')} must be positive")
    if args.gpt2_sequence_length > 1024 and not args.smoke:
        parser.error("GPT-2 Small supports at most 1024 positions")
    if args.bert_sequence_length > 512 and not args.smoke:
        parser.error("BERT Base supports at most 512 positions")
    if args.bert_sequence_length < 3:
        parser.error("BERT sequence length must leave room for [CLS] and [SEP]")
    if args.learning_rate <= 0:
        parser.error("--learning-rate must be positive")
    if args.weight_decay < 0:
        parser.error("--weight-decay must be non-negative")
    if not 0 < args.compression_threshold <= 1:
        parser.error("--compression-threshold must be in (0, 1]")
    if args.chunk_size & (args.chunk_size - 1):
        parser.error("--chunk-size must be a power of two")
    if args.background_cpus is not None and args.timing_cpus is None:
        parser.error("--background-cpus requires --timing-cpus")
    if args.timing_cpus is not None and len(args.timing_cpus) < args.threads:
        parser.error("--timing-cpus must provide at least --threads CPUs")
    if (
        args.timing_cpus is not None
        and args.background_cpus is not None
        and set(args.timing_cpus).intersection(args.background_cpus)
    ):
        parser.error("--timing-cpus and --background-cpus must be disjoint")
    return args


def main() -> None:
    args = _parse_args()
    torch.set_num_threads(max(1, int(args.threads)))
    try:
        torch.set_num_interop_threads(1)
    except RuntimeError:
        pass
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    documents: list[str] | None
    if args.smoke:
        documents = None
        dataset_metadata = {
            "dataset_repository": None,
            "dataset_file": None,
            "dataset_file_sha256": hashlib.sha256(b"synthetic-smoke").hexdigest(),
            "document_count": int(args.clients),
            "reconstruction": "synthetic smoke mode only",
        }
    else:
        documents, dataset_metadata = _load_real_documents(
            Path(args.data_dir), offline=args.offline
        )
    for model_name in args.models:
        _run_model(
            args,
            model_name=model_name,
            documents=documents,
            dataset_metadata=dataset_metadata,
        )


if __name__ == "__main__":
    main()
