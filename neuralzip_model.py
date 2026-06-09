import numpy as np
import torch
import zstandard as zstd
from scipy.spatial.distance import jensenshannon
from tqdm import tqdm

from bitpacking import bits_required, packed_byte_length
from compression_algorithms.arithmetic import ARITHMETIC_CODER
from compression_algorithms.huffman import huffman_code_lengths
from compression_codecs import CodecMixin
from float_utils import extract_float_parts, get_float_format


class NeuralZipMixin:
    FIELD_NAMES = ("sign", "exponent", "mantissa")

    def initialize_neuralzip(self, named_params):
        self._named_params = list(named_params)
        self._original_bits = self.total_original_bits()
        self._exp_offsets = None
        self._exp_codes = None
        self._exp_block_size = None
        self._exp_total_bits = None
        self._exp_ngram = None
        return self

    @property
    def named_params(self):
        return tuple(self._named_params)

    @property
    def original_bits(self):
        return self._original_bits

    def iter_tensors(self):
        yield from self._named_params

    def named_parameters(self):
        yield from self.iter_tensors()

    def _validate_field(self, field):
        if field not in self.FIELD_NAMES:
            choices = ", ".join(self.FIELD_NAMES)
            raise ValueError(f"Unsupported field {field!r}. Expected one of: {choices}.")

    def split_fields(self, tensor):
        """Return sign, exponent, and mantissa arrays for one floating tensor."""
        return extract_float_parts(tensor)

    def iter_tensor_field_parts(self):
        for name, tensor in self.iter_tensors():
            sign, exponent, mantissa = self.split_fields(tensor)
            yield name, tensor, sign, exponent, mantissa

    def iter_field_parts(self):
        for name, _, sign, exponent, mantissa in self.iter_tensor_field_parts():
            yield name, sign, exponent, mantissa

    def iter_field_values(self, field):
        self._validate_field(field)
        field_index = self.FIELD_NAMES.index(field)
        for _, sign, exponent, mantissa in self.iter_field_parts():
            yield (sign, exponent, mantissa)[field_index]

    def iter_model_field_values(self, field):
        yield from self.iter_field_values(field)

    def collect_field(self, field, max_symbols=0, dtype=np.uint32):
        chunks = []
        seen = 0
        for values in self.iter_field_values(field):
            remaining = (
                None
                if max_symbols is None or max_symbols <= 0
                else max_symbols - seen
            )
            if remaining is not None and remaining <= 0:
                break
            if remaining is not None:
                values = values[:remaining]
            chunks.append(values.astype(dtype, copy=False))
            seen += values.size

        if not chunks:
            return np.array([], dtype=dtype)
        return np.concatenate(chunks).astype(dtype, copy=False)

    def collect_exponents(self):
        return [
            values.astype(np.uint8, copy=False)
            for values in self.iter_field_values("exponent")
        ]

    def named_exponents(self):
        return [
            (name, exponent.astype(np.uint8, copy=False))
            for name, _, exponent, _ in self.iter_field_parts()
        ]

    def field_bits_per_value(self, field, tensor=None):
        self._validate_field(field)
        if field == "sign":
            return 1
        if tensor is None:
            try:
                _, tensor = next(iter(self._named_params))
            except StopIteration:
                return 0
        fmt = get_float_format(tensor.dtype)
        if field == "mantissa":
            return fmt["mantissa_bits"]
        return fmt["total_bits"] - fmt["mantissa_bits"] - 1

    def total_original_bits(self):
        return sum(
            tensor.numel() * get_float_format(tensor.dtype)["total_bits"]
            for _, tensor in self.iter_tensors()
        )

    def log_bits(self, name, total_bits):
        mbits = total_bits / 1e6
        mbytes = total_bits / 8e6
        ratio = 100 * (total_bits / self._original_bits) if self._original_bits else 0.0
        print(
            f"[{name}] {mbits:8.2f} Mbits | {mbytes:8.2f} MB | {ratio:5.2f}% of original"
        )

    def print_model_size(self):
        total_bits = self._original_bits
        self.log_bits("Original", total_bits)
        return total_bits

    def parse_context_order(self, method, default=2):
        _, _, order = method.partition(":")
        return int(order) if order else default

    def cluster_layers(self, layer_arrays, threshold=0.02):
        """Group layers with similar exponent histograms (Jensen-Shannon)."""
        hists = []
        for arr in layer_arrays:
            _, exps, _ = self.split_fields(arr)
            hist = np.bincount(exps, minlength=256).astype(np.float64)
            hist /= hist.sum() if hist.sum() > 0 else 1
            hists.append(hist)

        clusters = []
        for idx, hist in enumerate(hists):
            placed = False
            for cluster in clusters:
                if jensenshannon(hist, cluster["centroid"]) < threshold:
                    cluster["idx"].append(idx)
                    cluster["centroid"] = (cluster["centroid"] + hist) / 2
                    placed = True
                    break
            if not placed:
                clusters.append({"idx": [idx], "centroid": hist})
        return [cluster["idx"] for cluster in clusters]

    def build_huffman_index(self, exps, block_size=4096, ngram=1):
        if ngram > 1:
            exps = self.pack_ngrams_uint(exps, n=ngram).astype(np.uint32)

        if exps.size == 0:
            return np.array([], dtype=np.uint64), {}, 0

        unique, counts = np.unique(exps, return_counts=True)
        lengths = huffman_code_lengths(unique, counts)
        symbol_ids = np.searchsorted(unique, exps)

        offsets = []
        bit_offset = 0
        for idx, symbol_id in enumerate(symbol_ids):
            if idx % block_size == 0:
                offsets.append(bit_offset)
            bit_offset += int(lengths[symbol_id])

        codes = dict(zip(unique.astype(int).tolist(), lengths.astype(int).tolist()))
        return np.array(offsets, dtype=np.uint64), codes, bit_offset

    def build_global_exponent_index(self, block_size=4096, ngram=1):
        exponents = []
        mantissa_bits = 0
        sign_bits = 0
        for _, tensor, sign, exps, mant in self.iter_tensor_field_parts():
            exponents.append(exps)
            sign_bits += sign.size
            mantissa_bits += mant.size * self.field_bits_per_value("mantissa", tensor)
        exps = (
            np.concatenate(exponents)
            if exponents
            else np.array([], dtype=np.uint32)
        )
        offsets, codes, total_bits_exp = self.build_huffman_index(
            exps,
            block_size=block_size,
            ngram=ngram,
        )

        self._exp_offsets = offsets
        self._exp_codes = codes
        self._exp_block_size = block_size
        self._exp_total_bits = total_bits_exp
        self._exp_ngram = ngram

        total_bits = total_bits_exp + mantissa_bits + sign_bits
        self.log_bits(
            f"Exp bitstream indexed (block={block_size}, n={ngram})",
            total_bits,
        )
        return total_bits

    def exp_block_start_bit(self, position):
        if self._exp_offsets is None:
            raise RuntimeError(
                "Index not built. Call build_global_exponent_index first."
            )
        block = position // self._exp_block_size
        if block >= len(self._exp_offsets):
            raise IndexError("position out of range for exponent stream")
        return self._exp_offsets[block]

    def global_huffman_compress(self, verbose=1, label="Global Huffman"):
        """Compress the exponent field with one global Huffman model."""
        exponents = []
        mantissa_bits = 0
        sign_bits = 0
        for _, tensor, sign, exps, mant in self.iter_tensor_field_parts():
            exponents.append(exps)
            sign_bits += sign.size
            mantissa_bits += mant.size * self.field_bits_per_value("mantissa", tensor)
        exps = (
            np.concatenate(exponents)
            if exponents
            else np.array([], dtype=np.uint32)
        )
        bits_exp = self.compute_zipnn_bits(exps, method="huffman")
        total_bits = bits_exp + mantissa_bits + sign_bits
        if verbose:
            self.log_bits(label, total_bits)
        return total_bits

    def zipnn_compress(self, verbose=1):
        """Classic ZipNN-style baseline: global Huffman over exponent bytes."""
        return self.global_huffman_compress(verbose=verbose, label="ZipNN")

    def _cluster_exponent_streams(
        self,
        clusters,
        layer_arrays,
        method,
        ngram,
        compress_aux_fields,
        aux_method,
        validate,
    ):
        cluster_lookup = np.zeros(len(layer_arrays), dtype=np.uint32)
        for cluster_id, cluster_ids in enumerate(clusters):
            for block_id in cluster_ids:
                cluster_lookup[block_id] = cluster_id

        total_bytes = 16
        cluster_id_bits = bits_required(len(clusters))
        if cluster_id_bits:
            total_bytes += packed_byte_length(cluster_lookup.size, cluster_id_bits)

        cluster_streams = [[] for _ in clusters]
        for cluster_ids in tqdm(clusters, desc="NeuralZip clustered"):
            for idx in cluster_ids:
                arr = layer_arrays[idx]
                sign, exps_raw, mant_raw = self.split_fields(arr)
                fmt = get_float_format(arr.dtype)
                sign_size, mantissa_size = self.encode_aux_field_sizes(
                    sign,
                    mant_raw,
                    fmt["mantissa_bits"],
                    compress_aux_fields=compress_aux_fields,
                    aux_method=aux_method,
                    validate=validate,
                )

                total_bytes += 4 + 1 + 4 + 4 + sign_size + mantissa_size

                exps_model, exps_tail = self.prepare_exponent_symbols(
                    exps_raw,
                    method,
                    ngram,
                )
                total_bytes += 4 + exps_tail.size
                cluster_streams[cluster_lookup[idx]].append(
                    {
                        "packed": exps_model.astype(np.uint32, copy=False),
                        "tail": exps_tail.astype(np.uint8, copy=False),
                        "raw": exps_raw.astype(np.uint32, copy=False),
                    }
                )

        return total_bytes, cluster_streams

    def _encode_cluster_stream(self, cluster_stream, method, ngram, validate):
        if method.startswith("ctx-huffman-best"):
            order = self.parse_context_order(method, default=2)
            raw_parts = [part["raw"] for part in cluster_stream if part["raw"].size]
            raw_stream = (
                np.concatenate(raw_parts).astype(np.uint32, copy=False)
                if raw_parts
                else np.array([], dtype=np.uint32)
            )
            context_payload = self.encode_context_huffman_payload(
                raw_stream,
                order=order,
            )
            context_size = self.cluster_payload_byte_length(context_payload)

            packed_parts = []
            tail_size = 0
            for part in cluster_stream:
                packed, tail = self.pack_ngrams_uint_with_tail(part["raw"], n=3)
                if packed.size:
                    packed_parts.append(packed)
                tail_size += tail.size
            packed_stream = (
                np.concatenate(packed_parts).astype(np.uint32, copy=False)
                if packed_parts
                else np.array([], dtype=np.uint32)
            )
            trigram_payload = self.encode_huffman_payload(packed_stream)
            trigram_size = self.cluster_payload_byte_length(trigram_payload) + tail_size
            return 1 + min(context_size, trigram_size)

        packed_parts = [
            part["packed"] for part in cluster_stream if part["packed"].size
        ]
        packed_stream = (
            np.concatenate(packed_parts).astype(np.uint32, copy=False)
            if packed_parts
            else np.array([], dtype=np.uint32)
        )
        payload = self.encode_exponent_payload(
            packed_stream,
            method=method,
            validate=validate,
        )
        total_bytes = self.cluster_payload_byte_length(payload)

        if validate:
            self._validate_cluster_stream(cluster_stream, payload, packed_stream, method, ngram)
        return total_bytes

    def _validate_cluster_stream(self, cluster_stream, payload, packed_stream, method, ngram):
        if method == "arith" and packed_stream.size:
            decoded_ids = ARITHMETIC_CODER.decode(
                payload.encoded,
                payload.counts,
                packed_stream.size,
            )
            decoded_stream = payload.values[decoded_ids]
        else:
            decoded_stream = np.array([], dtype=np.uint32)

        offset = 0
        for part in cluster_stream:
            packed_len = part["packed"].size
            decoded_packed = np.array([], dtype=np.uint32)
            if packed_len:
                decoded_packed = decoded_stream[offset : offset + packed_len]
            offset += packed_len

            if ngram > 1 and decoded_packed.size:
                decoded_exps = np.concatenate(
                    [
                        self.unpack_ngrams_uint(decoded_packed, n=ngram),
                        part["tail"].astype(np.uint32),
                    ]
                )
            elif ngram > 1:
                decoded_exps = part["tail"].astype(np.uint32)
            else:
                decoded_exps = decoded_packed.astype(np.uint32, copy=False)

            if not np.array_equal(decoded_exps, part["raw"]):
                raise ValueError("NeuralZip block exponent reconstruction failed.")

    def build_gpu_inference_plan(self, ngram=3, block_size=4096, threshold=0.02):
        """Build a metadata-only layout for future NeuralZip GPU inference tests.

        The plan mirrors the DFloat11 serving idea: keep compressed exponent
        payloads resident, decode only the tensor needed by the current module,
        reconstruct a temporary floating-point buffer, run the matmul, and
        discard the buffer. This method does not launch a CUDA kernel; it
        exposes the cluster codebooks, tensor offsets, and tails needed by one.
        """
        if ngram < 1:
            raise ValueError("ngram must be >= 1.")
        if block_size < 1:
            raise ValueError("block_size must be >= 1.")

        layer_arrays = [param for _, param in self.iter_tensors()]
        clusters = self.cluster_layers(layer_arrays, threshold=threshold)
        cluster_lookup = np.zeros(len(layer_arrays), dtype=np.uint32)
        for cluster_id, tensor_ids in enumerate(clusters):
            for tensor_id in tensor_ids:
                cluster_lookup[tensor_id] = cluster_id

        tensor_parts = []
        for tensor_id, (name, param) in enumerate(self.iter_tensors()):
            sign, exps, mant = self.split_fields(param)
            fmt = get_float_format(param.dtype)
            if ngram > 1:
                symbols, tail = self.pack_ngrams_uint_with_tail(exps, n=ngram)
            else:
                symbols = exps.astype(np.uint32, copy=False)
                tail = np.array([], dtype=np.uint8)

            tensor_parts.append(
                {
                    "name": name,
                    "shape": tuple(param.shape),
                    "dtype": str(param.dtype).replace("torch.", ""),
                    "numel": int(param.numel()),
                    "cluster": int(cluster_lookup[tensor_id]),
                    "symbols": symbols.astype(np.uint32, copy=False),
                    "tail": tail.astype(np.uint8, copy=False),
                    "sign_bits": int(sign.size),
                    "mantissa_bits": int(mant.size * fmt["mantissa_bits"]),
                }
            )

        tensor_records = [None] * len(tensor_parts)
        cluster_records = []
        total_exponent_bits = 0
        total_codebook_bytes = 0
        total_offset_bytes = 0
        total_tail_bytes = 0
        total_sign_bits = 0
        total_mantissa_bits = 0

        for cluster_id, tensor_ids in enumerate(clusters):
            stream_parts = []
            symbol_offset = 0
            tail_offset = 0
            for tensor_id in tensor_ids:
                part = tensor_parts[tensor_id]
                tensor_records[tensor_id] = {
                    "name": part["name"],
                    "shape": part["shape"],
                    "dtype": part["dtype"],
                    "numel": part["numel"],
                    "cluster": cluster_id,
                    "symbol_offset": int(symbol_offset),
                    "symbol_count": int(part["symbols"].size),
                    "tail_offset": int(tail_offset),
                    "tail_count": int(part["tail"].size),
                    "sign_bits": part["sign_bits"],
                    "mantissa_bits": part["mantissa_bits"],
                }
                symbol_offset += int(part["symbols"].size)
                tail_offset += int(part["tail"].size)
                if part["symbols"].size:
                    stream_parts.append(part["symbols"])
                total_tail_bytes += int(part["tail"].size)
                total_sign_bits += part["sign_bits"]
                total_mantissa_bits += part["mantissa_bits"]

            stream = (
                np.concatenate(stream_parts).astype(np.uint32, copy=False)
                if stream_parts
                else np.array([], dtype=np.uint32)
            )
            if stream.size:
                values, counts = np.unique(stream, return_counts=True)
                lengths = huffman_code_lengths(values, counts)
                symbol_ids = np.searchsorted(values, stream)
                block_offsets = []
                bit_offset = 0
                for idx, symbol_id in enumerate(symbol_ids):
                    if idx % block_size == 0:
                        block_offsets.append(bit_offset)
                    bit_offset += int(lengths[symbol_id])
            else:
                values = np.array([], dtype=np.uint32)
                counts = np.array([], dtype=np.uint32)
                lengths = np.array([], dtype=np.uint32)
                block_offsets = []
                bit_offset = 0

            total_exponent_bits += bit_offset
            codebook_bytes = 4 + 8 + values.nbytes + lengths.nbytes
            offset_bytes = len(block_offsets) * 8
            total_codebook_bytes += codebook_bytes
            total_offset_bytes += offset_bytes

            cluster_records.append(
                {
                    "cluster": cluster_id,
                    "tensor_ids": [int(idx) for idx in tensor_ids],
                    "symbol_count": int(stream.size),
                    "encoded_bits": int(bit_offset),
                    "codebook_bytes": int(codebook_bytes),
                    "offset_bytes": int(offset_bytes),
                    "values": values.astype(np.uint32, copy=False).tolist(),
                    "code_lengths": lengths.astype(np.uint32, copy=False).tolist(),
                    "block_offsets": [int(offset) for offset in block_offsets],
                }
            )

        estimated_total_bits = (
            total_exponent_bits
            + total_sign_bits
            + total_mantissa_bits
            + 8 * (total_codebook_bytes + total_offset_bytes + total_tail_bytes)
        )

        return {
            "format": "neuralzip-gpu-inference-plan",
            "ngram": int(ngram),
            "block_size": int(block_size),
            "cluster_threshold": float(threshold),
            "decode_policy": "decode tensor on demand, run compute, discard temporary weights",
            "estimated_total_bits": int(estimated_total_bits),
            "estimated_total_bytes": int(packed_byte_length(estimated_total_bits, 1)),
            "exponent_bits": int(total_exponent_bits),
            "sign_bits": int(total_sign_bits),
            "mantissa_bits": int(total_mantissa_bits),
            "metadata_bytes": int(total_codebook_bytes + total_offset_bytes),
            "tail_bytes": int(total_tail_bytes),
            "clusters": cluster_records,
            "tensors": tensor_records,
        }

    def _neuralzip_clustered(
        self,
        layer_arrays,
        method,
        ngram,
        validate,
        compress_aux_fields,
        aux_method,
    ):
        clusters = self.cluster_layers(layer_arrays)
        total_bytes, cluster_streams = self._cluster_exponent_streams(
            clusters,
            layer_arrays,
            method,
            ngram,
            compress_aux_fields,
            aux_method,
            validate,
        )
        for cluster_stream in cluster_streams:
            total_bytes += self._encode_cluster_stream(
                cluster_stream,
                method,
                ngram,
                validate,
            )
        return total_bytes * 8

    def _neuralzip_per_tensor(
        self,
        layer_arrays,
        method,
        ngram,
        validate,
        compress_aux_fields,
        aux_method,
    ):
        total_bytes = 0
        for _, param in tqdm(
            self.iter_tensors(),
            desc="NeuralZip",
            total=len(layer_arrays),
        ):
            if param.numel() == 0:
                continue
            sign, exps, mant = self.split_fields(param)
            fmt = get_float_format(param.dtype)
            sign_size, mantissa_size = self.encode_aux_field_sizes(
                sign,
                mant,
                fmt["mantissa_bits"],
                compress_aux_fields=compress_aux_fields,
                aux_method=aux_method,
                validate=validate,
            )
            total_bytes += 4 + 1 + 4 + 4 + sign_size + mantissa_size

            exps_model, exps_tail = self.prepare_exponent_symbols(exps, method, ngram)
            payload = self.encode_exponent_payload(
                exps_model.astype(np.uint32, copy=False),
                method=method,
                validate=validate,
            )
            total_bytes += self.cluster_payload_byte_length(payload)
            total_bytes += 4 + exps_tail.size

            if method == "arith" and validate:
                decoded_ids = ARITHMETIC_CODER.decode(
                    payload.encoded,
                    payload.counts,
                    exps_model.size,
                )
                decoded_stream = payload.values[decoded_ids]
                if ngram > 1 and decoded_stream.size:
                    decoded_exps = np.concatenate(
                        [
                            self.unpack_ngrams_uint(decoded_stream, n=ngram),
                            exps_tail.astype(np.uint32),
                        ]
                    )
                elif ngram > 1:
                    decoded_exps = exps_tail.astype(np.uint32)
                else:
                    decoded_exps = decoded_stream.astype(np.uint32, copy=False)
                if not np.array_equal(decoded_exps, exps):
                    raise ValueError("NeuralZip exponent reconstruction failed.")
        return total_bytes * 8

    def neuralzip_compress(
        self,
        verbose=5,
        ngram=1,
        method="huffman",
        cluster=False,
        exp="",
        validate=False,
        compress_aux_fields=False,
        aux_method="huffman",
    ):
        layer_arrays = [param for _, param in self.iter_tensors()]

        if cluster:
            total_bits = self._neuralzip_clustered(
                layer_arrays,
                method,
                ngram,
                validate,
                compress_aux_fields,
                aux_method,
            )
        else:
            total_bits = self._neuralzip_per_tensor(
                layer_arrays,
                method,
                ngram,
                validate,
                compress_aux_fields,
                aux_method,
            )

        if verbose:
            fields = f"all-fields:{aux_method}" if compress_aux_fields else "exp-only"
            self.log_bits(
                f"NeuralZip ({exp}) [n={ngram}, {method}, cluster={cluster}, {fields}]",
                total_bits,
            )
        return total_bits

    def zstd_compress(self, level=19, verbose=1):
        raw = b"".join(
            param.detach().cpu().contiguous().view(torch.uint8).numpy().tobytes()
            for _, param in self.iter_tensors()
        )
        cctx = zstd.ZstdCompressor(level=level, threads=8)
        compressed = cctx.compress(raw)
        bits = len(compressed) * 8
        if verbose:
            self.log_bits(f"Zstd/level={level}", bits)
        return bits


class NeuralZipModel(NeuralZipMixin, CodecMixin):
    """Concrete NeuralZip API over a sequence of named torch tensors."""

    def __init__(self, named_params):
        self.initialize_neuralzip(named_params)
