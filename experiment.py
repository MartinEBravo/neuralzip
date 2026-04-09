from __future__ import annotations

from collections import defaultdict
from pathlib import Path
import sys

import numpy as np
from tqdm.auto import tqdm

from constants import get_model_label
from entropy import cluster_blocks, summarize_field_entropy
from loaders import (
    SafetensorEntries,
    discover_safetensor_entries,
    iter_tensors_from_safetensors,
    iter_tensors_from_transformers,
    resolve_safetensor_source,
)
from quantization import float_quantize_tensor, normalize_prequant_bits
from tensors import (
    FLOAT_LAYOUTS,
    BlockInfo,
    ScanSummary,
    block_name_from_tensor_name,
    concat_arrays,
    is_known_non_parameter_buffer,
    iter_tensor_exponent_chunks,
    natural_sort_key,
    pack_binary_stream,
    pack_unsigned_stream,
    summarize_dtypes,
    tensor_field_stats,
    tensor_exponent_stream,
    tensor_mantissa_stream,
    tensor_sign_stream,
)
from compression import (
    NeuralZipArchive,
    NeuralZipBlock,
    NeuralZipCluster,
    ZipNNArchive,
    build_canonical_codebook_from_counts,
    build_arithmetic_model_from_counts,
    decode_neuralzip_archive_exponent_streams,
    decode_zipnn_archive_exponent_streams,
    decode_varint,
    encode_varint,
    uint32_words_to_bytes,
)

try:
    from bitarray import bitarray
except ImportError as exc:  # pragma: no cover - dependency error path
    raise RuntimeError(
        "Real codecs require the 'bitarray' package. Install it with: pip install bitarray"
    ) from exc

try:
    import constriction
except ImportError as exc:  # pragma: no cover - dependency error path
    raise RuntimeError(
        "NeuralZip arithmetic coding requires the 'constriction' package. Install it with: pip install constriction"
    ) from exc

ZIPNN_SUBMODULE_ROOT = Path(__file__).resolve().parent / "external" / "zipnn"
if str(ZIPNN_SUBMODULE_ROOT) not in sys.path:
    sys.path.insert(0, str(ZIPNN_SUBMODULE_ROOT))

try:
    from zipnn import ZipNN as UpstreamZipNN
except ImportError as exc:  # pragma: no cover - dependency error path
    raise RuntimeError(
        "The upstream ZipNN submodule is required for ZipNN comparisons. "
        "Build it under Codebase/external/zipnn before running the pipeline."
    ) from exc

class ModelCompressionExperiment:
    def __init__(
        self,
        model_name: str,
        loader: str,
        hf_token: str | None,
        local_files_only: bool,
        trust_remote_code: bool,
        block_grouping: str = "module",
        prequant_bits: int | None = None,
    ):
        self.model_name = model_name
        self.loader_preference = loader
        self.hf_token = hf_token
        self.local_files_only = local_files_only
        self.trust_remote_code = trust_remote_code
        self.block_grouping = block_grouping
        self.prequant_bits = normalize_prequant_bits(prequant_bits)
        self.loader_name, self.loader_details = self._resolve_loader()

    def _resolve_loader(self) -> tuple[str, tuple[Path, SafetensorEntries] | None]:
        if self.loader_preference == "transformers":
            return "transformers", None

        try:
            source_path = resolve_safetensor_source(
                model_name=self.model_name,
                hf_token=self.hf_token,
                local_files_only=self.local_files_only,
            )
            entries = discover_safetensor_entries(source_path)
            return "safetensors", (source_path, entries)
        except Exception:
            if self.loader_preference == "safetensors":
                raise
            return "transformers", None

    def _raw_tensor_iter(self):
        if self.loader_name == "safetensors":
            if self.loader_details is None:
                raise RuntimeError("Safetensors loader details were not initialized.")
            _, entries = self.loader_details
            return iter_tensors_from_safetensors(entries)
        return iter_tensors_from_transformers(
            model_name=self.model_name,
            hf_token=self.hf_token,
            trust_remote_code=self.trust_remote_code,
        )

    def iter_tensors(self):
        for name, tensor in self._raw_tensor_iter():
            yield name, float_quantize_tensor(tensor, self.prequant_bits)

    def _tensor_progress(self, desc: str):
        total = None
        if self.loader_name == "safetensors" and self.loader_details is not None:
            _, entries = self.loader_details
            total = sum(len(keys) for _, keys in entries)
        model_label = get_model_label(self.model_name)
        if self.prequant_bits is not None:
            model_label = f"{model_label} Q{self.prequant_bits}"
        return tqdm(
            self.iter_tensors(),
            total=total,
            desc=f"{model_label} | {desc}",
            unit="tensor",
            dynamic_ncols=True,
        )

    @staticmethod
    def _is_supported_tensor(tensor) -> bool:
        return tensor.numel() > 0 and tensor.is_floating_point() and tensor.dtype in FLOAT_LAYOUTS

    def _should_skip_tensor(self, name: str, tensor) -> bool:
        return is_known_non_parameter_buffer(name, tensor) or not self._is_supported_tensor(tensor)

    def _iter_valid_tensors(self, desc: str):
        with self._tensor_progress(desc) as progress:
            for name, tensor in progress:
                if self._should_skip_tensor(name, tensor):
                    del tensor
                    continue
                yield name, tensor
                del tensor

    def scan(self) -> ScanSummary:
        payload_bits = 0
        original_bits = 0
        sign_counts = np.zeros(2, dtype=np.int64)
        global_exponent_counts = np.zeros(256, dtype=np.int64)
        mantissa_counts_by_bits: dict[int, np.ndarray] = {}
        dtype_weights: dict[str, int] = defaultdict(int)
        grouped_counts: dict[str, np.ndarray] = {}
        grouped_weights: dict[str, int] = defaultdict(int)
        grouped_tensors: dict[str, list[str]] = defaultdict(list)

        for name, tensor in self._iter_valid_tensors("Field stats / ZipNN / NeuralZip scan"):
            stats = tensor_field_stats(tensor)
            if stats is None:
                continue

            payload_bits += stats.body_bits
            original_bits += stats.original_bits
            dtype_weights[stats.dtype_name] += stats.num_weights
            sign_counts += stats.sign_counts
            global_exponent_counts += stats.exponent_counts
            if stats.mantissa_bits not in mantissa_counts_by_bits:
                mantissa_counts_by_bits[stats.mantissa_bits] = stats.mantissa_counts.astype(
                    np.int64, copy=True
                )
            else:
                mantissa_counts_by_bits[stats.mantissa_bits] += stats.mantissa_counts

            block_name = block_name_from_tensor_name(name, self.block_grouping)
            if block_name not in grouped_counts:
                grouped_counts[block_name] = stats.exponent_counts.astype(np.int64, copy=True)
            else:
                grouped_counts[block_name] += stats.exponent_counts
            grouped_weights[block_name] += int(stats.num_weights)
            grouped_tensors[block_name].append(name)

        if not grouped_counts:
            raise RuntimeError(f"No supported floating-point tensors were found in {self.model_name}.")

        block_infos = [
            BlockInfo(
                name=block_name,
                exponent_counts=grouped_counts[block_name],
                num_weights=grouped_weights[block_name],
                tensor_names=tuple(grouped_tensors[block_name]),
            )
            for block_name in sorted(grouped_counts, key=natural_sort_key)
        ]

        return ScanSummary(
            block_infos=block_infos,
            original_bits=original_bits,
            payload_bits=payload_bits,
            dtype_summary=summarize_dtypes(dtype_weights),
            num_blocks=len(block_infos),
            loader=self.loader_name,
            field_entropies=(
                summarize_field_entropy("sign", [sign_counts]),
                summarize_field_entropy("exponent", [global_exponent_counts]),
                summarize_field_entropy("mantissa", list(mantissa_counts_by_bits.values())),
            ),
        )

    def _tensor_to_block_map(self, summary: ScanSummary) -> dict[str, str]:
        tensor_to_block: dict[str, str] = {}
        for block in summary.block_infos:
            for tensor_name in block.tensor_names:
                tensor_to_block[tensor_name] = block.name
        return tensor_to_block

    def _encoded_tensor_bitarrays(
        self,
        summary: ScanSummary,
        desc: str,
        codebooks_by_block: dict[str, dict[int, bitarray]],
    ) -> dict[str, bitarray]:
        tensor_to_block = self._tensor_to_block_map(summary)
        encoded_tensors: dict[str, bitarray] = {}

        for name, tensor in self._iter_valid_tensors(desc):
            block_name = tensor_to_block.get(name)
            if block_name is None:
                raise RuntimeError(f"Unexpected tensor {name!r} during {desc}.")

            code = codebooks_by_block[block_name]
            body_bits = bitarray(endian="big")
            for exponents in iter_tensor_exponent_chunks(tensor):
                body_bits.encode(code, exponents)
            encoded_tensors[name] = body_bits

        expected_tensors = sum(len(block.tensor_names) for block in summary.block_infos)
        if len(encoded_tensors) != expected_tensors:
            raise RuntimeError(f"{desc} did not cover every tensor in the scan summary.")
        return encoded_tensors

    def zipnn_archive(self, summary: ScanSummary) -> ZipNNArchive:
        global_counts = np.zeros(256, dtype=np.int64)
        for block in summary.block_infos:
            global_counts += block.exponent_counts

        code, codebook = build_canonical_codebook_from_counts(global_counts)
        encoded_tensors = self._encoded_tensor_bitarrays(
            summary=summary,
            desc="ZipNN encode",
            codebooks_by_block={block.name: code for block in summary.block_infos},
        )

        body_bits = bitarray(endian="big")
        block_symbol_counts: list[int] = []
        for block in summary.block_infos:
            block_symbol_counts.append(block.num_weights)
            for tensor_name in block.tensor_names:
                body_bits.extend(encoded_tensors.pop(tensor_name))

        if encoded_tensors:
            raise RuntimeError("ZipNN encoding left unread tensor bodies.")

        return ZipNNArchive.build(
            block_symbol_counts=block_symbol_counts,
            codebook=codebook,
            body=body_bits.tobytes(),
        )

    def _neuralzip_membership(
        self,
        summary: ScanSummary,
        cluster_threshold: float,
        max_clusters: int | None,
    ) -> tuple[list[list[int]], dict[str, int]]:
        clusters = cluster_blocks(
            summary.block_infos,
            threshold=cluster_threshold,
            max_clusters=max_clusters,
        )
        membership = {
            summary.block_infos[block_idx].name: cluster_idx
            for cluster_idx, member_ids in enumerate(clusters)
            for block_idx in member_ids
        }
        return clusters, membership

    def neuralzip_archive(
        self,
        summary: ScanSummary,
        cluster_threshold: float,
        max_clusters: int | None,
    ) -> NeuralZipArchive:
        clusters, membership = self._neuralzip_membership(
            summary=summary,
            cluster_threshold=cluster_threshold,
            max_clusters=max_clusters,
        )

        archive_clusters: list[NeuralZipCluster] = []
        cluster_models: list[object | None] = []
        block_encoders: dict[str, object | None] = {}

        for member_ids in clusters:
            cluster_counts = np.zeros(256, dtype=np.int64)
            for block_idx in member_ids:
                cluster_counts += summary.block_infos[block_idx].exponent_counts
            archive_clusters.append(NeuralZipCluster(symbol_counts=cluster_counts.copy()))
            if np.count_nonzero(cluster_counts) <= 1:
                cluster_models.append(None)
            else:
                cluster_models.append(build_arithmetic_model_from_counts(cluster_counts))
            for block_idx in member_ids:
                block_encoders[summary.block_infos[block_idx].name] = constriction.stream.queue.RangeEncoder()

        tensor_to_block = self._tensor_to_block_map(summary)
        for name, tensor in self._iter_valid_tensors("NeuralZip encode"):
            block_name = tensor_to_block.get(name)
            if block_name is None:
                raise RuntimeError(f"Unexpected tensor {name!r} during NeuralZip encode.")
            cluster_idx = membership[block_name]
            model = cluster_models[cluster_idx]
            if model is not None:
                encoder = block_encoders[block_name]
                if encoder is None:
                    raise RuntimeError(f"Missing arithmetic encoder for block {block_name!r}.")
                for exponents in iter_tensor_exponent_chunks(tensor):
                    encoder.encode(exponents.astype(np.int32, copy=False), model)

        archive_blocks: list[NeuralZipBlock] = []
        for block in summary.block_infos:
            cluster_idx = membership[block.name]
            model = cluster_models[cluster_idx]
            if model is None:
                body = b""
            else:
                encoder = block_encoders[block.name]
                if encoder is None:
                    raise RuntimeError(f"Missing final arithmetic encoder for block {block.name!r}.")
                body = uint32_words_to_bytes(encoder.get_compressed())
            archive_blocks.append(
                NeuralZipBlock(
                    exponent_count=block.num_weights,
                    cluster_idx=cluster_idx,
                    body=body,
                )
            )

        return NeuralZipArchive.build(blocks=archive_blocks, clusters=archive_clusters)

    def _expected_block_streams(self, summary: ScanSummary, desc: str) -> list[np.ndarray]:
        tensor_streams: dict[str, np.ndarray] = {}
        for name, tensor in self._iter_valid_tensors(desc):
            tensor_streams[name] = tensor_exponent_stream(tensor).copy()

        expected_blocks: list[np.ndarray] = []
        for block in summary.block_infos:
            parts: list[np.ndarray] = []
            for tensor_name in block.tensor_names:
                if tensor_name not in tensor_streams:
                    raise RuntimeError(f"Missing tensor {tensor_name!r} during {desc}.")
                parts.append(tensor_streams.pop(tensor_name))
            expected_blocks.append(concat_arrays(parts))

        if tensor_streams:
            raise RuntimeError(f"{desc} produced unexpected extra tensors.")
        return expected_blocks

    def _verify_block_streams(
        self,
        summary: ScanSummary,
        decoded_blocks: list[np.ndarray],
        desc: str,
    ) -> None:
        expected_blocks = self._expected_block_streams(summary, desc)
        if len(expected_blocks) != len(decoded_blocks):
            raise RuntimeError(f"{desc} decoded a different number of blocks than expected.")

        for block_idx, (block, expected, actual) in enumerate(
            zip(summary.block_infos, expected_blocks, decoded_blocks)
        ):
            if not np.array_equal(expected, actual):
                raise RuntimeError(f"{desc} failed at block {block_idx}: {block.name}.")

    def _verify_zipnn_archive(self, summary: ScanSummary, archive: ZipNNArchive) -> None:
        decoded_blocks = decode_zipnn_archive_exponent_streams(archive.blob)
        self._verify_block_streams(summary, decoded_blocks, "ZipNN verify")

    def _verify_neuralzip_archive(
        self,
        summary: ScanSummary,
        archive: NeuralZipArchive,
    ) -> None:
        decoded_blocks = decode_neuralzip_archive_exponent_streams(archive.blob)
        self._verify_block_streams(summary, decoded_blocks, "NeuralZip verify")

    def _upstream_zipnn_codec(self) -> UpstreamZipNN:
        return UpstreamZipNN(input_format="byte", method="HUFFMAN", threads=0)

    def _global_exponent_stream(self) -> np.ndarray:
        parts: list[np.ndarray] = []
        for _, tensor in self._iter_valid_tensors("ZipNN encode"):
            parts.append(tensor_exponent_stream(tensor).copy())
        return concat_arrays(parts)

    def _raw_side_streams(self, desc: str) -> tuple[np.ndarray, dict[int, list[np.ndarray]]]:
        sign_parts: list[np.ndarray] = []
        mantissa_parts_by_bits: dict[int, list[np.ndarray]] = defaultdict(list)
        for _, tensor in self._iter_valid_tensors(desc):
            sign_parts.append(tensor_sign_stream(tensor).copy())
            mantissa_values, mantissa_bits = tensor_mantissa_stream(tensor)
            mantissa_parts_by_bits[mantissa_bits].append(mantissa_values.copy())
        return concat_arrays(sign_parts), mantissa_parts_by_bits

    def _build_neuralzip_full_side_blob(self) -> tuple[bytes, tuple[bytes, dict[int, bytes]]]:
        codec = self._upstream_zipnn_codec()
        sign_stream, mantissa_parts_by_bits = self._raw_side_streams("NeuralZipFull side streams")
        sign_bytes = pack_binary_stream(sign_stream)
        encoded_sign = codec.compress(sign_bytes) if sign_bytes else b""

        mantissa_payloads: dict[int, bytes] = {}
        encoded_mantissas: dict[int, bytes] = {}
        for mantissa_bits in sorted(mantissa_parts_by_bits):
            stream = concat_arrays(mantissa_parts_by_bits[mantissa_bits])
            payload = pack_unsigned_stream(stream, mantissa_bits)
            mantissa_payloads[mantissa_bits] = payload
            encoded_mantissas[mantissa_bits] = codec.compress(payload) if payload else b""

        blob = bytearray(b"NZF1")
        blob.extend(encode_varint(sign_stream.size))
        blob.extend(encode_varint(len(sign_bytes)))
        blob.extend(encode_varint(len(encoded_sign)))
        blob.extend(encoded_sign)
        blob.extend(encode_varint(len(encoded_mantissas)))
        for mantissa_bits, encoded_payload in encoded_mantissas.items():
            payload = mantissa_payloads[mantissa_bits]
            blob.append(int(mantissa_bits))
            blob.extend(encode_varint(len(payload)))
            blob.extend(encode_varint(len(encoded_payload)))
            blob.extend(encoded_payload)
        return bytes(blob), (sign_bytes, mantissa_payloads)

    def _verify_neuralzip_full_side_blob(
        self,
        blob: bytes,
        expected_sign_bytes: bytes,
        expected_mantissa_payloads: dict[int, bytes],
    ) -> None:
        if not blob.startswith(b"NZF1"):
            raise RuntimeError("NeuralZipFull side blob has an invalid header.")
        codec = self._upstream_zipnn_codec()
        offset = 4
        _, offset = decode_varint(blob, offset)
        sign_bytes_len, offset = decode_varint(blob, offset)
        sign_encoded_len, offset = decode_varint(blob, offset)
        sign_encoded = blob[offset : offset + sign_encoded_len]
        offset += sign_encoded_len
        restored_sign = codec.decompress(sign_encoded) if sign_encoded else b""
        if len(restored_sign) != sign_bytes_len or restored_sign != expected_sign_bytes:
            raise RuntimeError("NeuralZipFull sign stream verification failed.")

        group_count, offset = decode_varint(blob, offset)
        seen_bits: set[int] = set()
        for _ in range(group_count):
            mantissa_bits = blob[offset]
            offset += 1
            payload_len, offset = decode_varint(blob, offset)
            encoded_len, offset = decode_varint(blob, offset)
            encoded_payload = blob[offset : offset + encoded_len]
            offset += encoded_len
            restored_payload = codec.decompress(encoded_payload) if encoded_payload else b""
            expected_payload = expected_mantissa_payloads.get(mantissa_bits)
            if expected_payload is None:
                raise RuntimeError(f"NeuralZipFull verification saw unexpected mantissa width {mantissa_bits}.")
            if len(restored_payload) != payload_len or restored_payload != expected_payload:
                raise RuntimeError(f"NeuralZipFull mantissa verification failed for {mantissa_bits}-bit mantissas.")
            seen_bits.add(mantissa_bits)

        if seen_bits != set(expected_mantissa_payloads):
            raise RuntimeError("NeuralZipFull verification did not cover every mantissa stream.")
        if offset != len(blob):
            raise RuntimeError("NeuralZipFull side blob has trailing bytes.")

    def _verify_upstream_zipnn(self, encoded: bytes, expected_stream: np.ndarray) -> None:
        codec = self._upstream_zipnn_codec()
        restored = codec.decompress(encoded)
        restored_stream = np.frombuffer(restored, dtype=np.uint8)
        if not np.array_equal(restored_stream, expected_stream):
            raise RuntimeError("ZipNN verify failed on the exponent stream.")

    def zipnn_bits(self, summary: ScanSummary, verify: bool = False) -> int:
        codec = self._upstream_zipnn_codec()
        exponent_stream = self._global_exponent_stream()
        encoded = codec.compress(exponent_stream.tobytes())
        if verify:
            self._verify_upstream_zipnn(encoded, exponent_stream)
        return summary.payload_bits + len(encoded) * 8

    def neuralzip_bits(
        self,
        summary: ScanSummary,
        cluster_threshold: float,
        max_clusters: int | None,
        verify: bool = False,
    ) -> int:
        archive = self.neuralzip_archive(
            summary=summary,
            cluster_threshold=cluster_threshold,
            max_clusters=max_clusters,
        )
        if verify:
            self._verify_neuralzip_archive(summary, archive)
        return summary.payload_bits + archive.compressed_bits

    def neuralzip_full_bits(
        self,
        summary: ScanSummary,
        cluster_threshold: float,
        max_clusters: int | None,
        verify: bool = False,
    ) -> int:
        archive = self.neuralzip_archive(
            summary=summary,
            cluster_threshold=cluster_threshold,
            max_clusters=max_clusters,
        )
        side_blob, (sign_bytes, mantissa_payloads) = self._build_neuralzip_full_side_blob()
        if verify:
            self._verify_neuralzip_archive(summary, archive)
            self._verify_neuralzip_full_side_blob(side_blob, sign_bytes, mantissa_payloads)
        return archive.compressed_bits + len(side_blob) * 8
