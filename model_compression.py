from dataclasses import dataclass

import numpy as np
import torch
import zstandard as zstd
from scipy.spatial.distance import jensenshannon
from tqdm import tqdm
from transformers import AutoModelForCausalLM

from arithmetic import ARITHMETIC_CODER
from bitpacking import bits_required, packed_byte_length, pack_values, unpack_values
from float_utils import extract_float_parts, get_float_format
from huffman import build_huffman_tree, generate_huffman_codes


@dataclass
class ClusterPayload:
    values: np.ndarray
    counts: np.ndarray
    encoded: bytes


class HuggingFaceModel:
    def __init__(self, model_name, dtype="auto"):
        dtype_map = {
            "auto": "auto",
            "float16": torch.float16,
            "bfloat16": torch.bfloat16,
            "float32": torch.float32,
        }
        self._model = AutoModelForCausalLM.from_pretrained(
            model_name, dtype=dtype_map[dtype]
        )
        self._named_params = list(self._model.named_parameters())
        self._original_bits = sum(
            param.numel() * get_float_format(param.dtype)["total_bits"]
            for _, param in self._named_params
        )

        self._exp_offsets = None
        self._exp_codes = None
        self._exp_block_size = None
        self._exp_total_bits = None
        self._exp_ngram = None

    def log_bits(self, name, total_bits):
        mbits = total_bits / 1e6
        mbytes = total_bits / 8e6
        ratio = 100 * (total_bits / self._original_bits)
        print(
            f"[{name}] {mbits:8.2f} Mbits | {mbytes:8.2f} MB | {ratio:5.2f}% of original"
        )

    def print_model_size(self):
        total_bits = self._original_bits
        self.log_bits("Original", total_bits)
        return total_bits

    def compute_arithmetic_bits(self, arr, order=1):
        """Adaptive arithmetic coding bits estimation (order-1)."""
        if arr.size < 2:
            return 0.0
        A = 256
        eps = 1
        arr = arr.astype(np.uint8)
        counts = np.ones((A, A)) * eps
        totals = np.ones(A) * (A * eps)
        bits = 0.0
        prev = int(arr[0])
        for x in arr[1:]:
            p = counts[prev, x] / totals[prev]
            bits += -np.log2(p)
            counts[prev, x] += 1
            totals[prev] += 1
            prev = x
        return bits

    def compute_sigma_huffman_bits(self, arr):
        arr = arr.astype(np.uint8)
        if arr.size < 2:
            return 0.0

        A = 256
        counts = np.zeros((A, A), dtype=np.int64)
        prev = int(arr[0])
        for x in arr[1:]:
            x = int(x)
            counts[prev, x] += 1
            prev = x

        total_bits = 0.0
        for prev in range(A):
            row = counts[prev]
            row_sum = int(row.sum())
            if row_sum == 0:
                continue

            symbols = np.nonzero(row)[0]
            freqs = row[symbols]

            root = build_huffman_tree(symbols.tolist(), freqs.tolist())
            codes = generate_huffman_codes(root, code="", huffman_codes={})

            for s, f in zip(symbols, freqs):
                total_bits += int(f) * len(codes[int(s)])

        return total_bits

    def compute_zipnn_bits(self, arr, method="huffman"):
        """
        Compute expected bits using:
        - 'huffman': global order-0 Huffman.
        - 'arith': adaptive order-1 arithmetic model.
        - 'sigma': order-1 Huffman per context.
        """
        if arr.size == 0:
            return 0.0
        if method == "arith":
            return self.compute_arithmetic_bits(arr)
        if method == "sigma":
            return self.compute_sigma_huffman_bits(arr)

        unique, counts = np.unique(arr, return_counts=True)
        probs = counts / counts.sum()
        bits = np.sum(counts * np.ceil(-np.log2(probs)))
        return bits

    def pack_ngrams_uint(self, exps, n=2):
        """Pack consecutive exponent symbols into 16-24-bit ngrams."""
        L = (len(exps) // n) * n
        x = exps[:L].reshape(-1, n).astype(np.uint32)
        packed = np.zeros(x.shape[0], dtype=np.uint32)
        for i in range(n):
            packed |= (x[:, i] & 0xFF) << (8 * i)
        return packed

    def pack_ngrams_uint_with_tail(self, exps, n=2):
        L = (len(exps) // n) * n
        packed = self.pack_ngrams_uint(exps, n=n)
        tail = exps[L:].astype(np.uint8, copy=False)
        return packed, tail

    def unpack_ngrams_uint(self, packed, n=2):
        unpacked = np.zeros(packed.size * n, dtype=np.uint32)
        for i in range(n):
            unpacked[i::n] = (packed >> (8 * i)) & 0xFF
        return unpacked

    def encode_signs(self, sign, validate=False):
        payload_size = packed_byte_length(sign.size, 1)
        if validate:
            payload = pack_values(sign, 1)
            decoded = unpack_values(payload, sign.size, 1)
            if not np.array_equal(decoded, sign):
                raise ValueError("Sign round-trip reconstruction failed.")
            return payload
        return payload_size

    def encode_mantissa(self, mantissa, bits_per_value, validate=False):
        payload_size = packed_byte_length(mantissa.size, bits_per_value)
        if validate:
            payload = pack_values(mantissa, bits_per_value)
            decoded = unpack_values(payload, mantissa.size, bits_per_value)
            if not np.array_equal(decoded, mantissa):
                raise ValueError("Mantissa round-trip reconstruction failed.")
            return payload
        return payload_size

    def encode_cluster_payload(self, exponents, validate=False):
        if exponents.size == 0:
            return ClusterPayload(
                values=np.array([], dtype=np.uint32),
                counts=np.array([], dtype=np.uint32),
                encoded=b"",
            )

        values, counts = np.unique(exponents, return_counts=True)
        symbol_ids = np.searchsorted(values, exponents)
        encoded = ARITHMETIC_CODER.encode(symbol_ids, counts)
        if validate:
            decoded_ids = ARITHMETIC_CODER.decode(encoded, counts, symbol_ids.size)
            decoded = values[decoded_ids]
            if not np.array_equal(
                decoded.astype(np.uint32, copy=False),
                exponents.astype(np.uint32, copy=False),
            ):
                raise ValueError("Exponent payload round-trip reconstruction failed.")
        return ClusterPayload(
            values=values.astype(np.uint32),
            counts=counts.astype(np.uint32),
            encoded=encoded,
        )

    def encode_huffman_payload(self, exponents):
        if exponents.size == 0:
            return ClusterPayload(
                values=np.array([], dtype=np.uint32),
                counts=np.array([], dtype=np.uint32),
                encoded=b"",
            )

        values, counts = np.unique(exponents, return_counts=True)
        root = build_huffman_tree(values.tolist(), counts.tolist())
        codes = generate_huffman_codes(root)
        bit_count = sum(
            int(count) * len(codes[int(value)])
            for value, count in zip(values, counts)
        )
        encoded = bytes(packed_byte_length(bit_count, 1))
        return ClusterPayload(
            values=values.astype(np.uint32),
            counts=counts.astype(np.uint32),
            encoded=encoded,
        )

    def encode_exponent_payload(self, exponents, method, validate=False):
        if method == "arith":
            return self.encode_cluster_payload(exponents, validate=validate)
        if method == "huffman":
            if validate:
                raise ValueError("Huffman payload validation is not implemented.")
            return self.encode_huffman_payload(exponents)
        raise ValueError(f"Unsupported NeuralZip method: {method}")

    def cluster_payload_byte_length(self, payload):
        size = 4 + 8 + len(payload.encoded)
        if payload.values.size:
            size += payload.values.nbytes + payload.counts.nbytes
        return size

    def cluster_layers(self, layer_arrays, threshold=0.02):
        """Group layers with similar exponent histograms (Jensen-Shannon)."""
        hists = []
        for arr in layer_arrays:
            _, exps, _ = extract_float_parts(arr)
            h = np.bincount(exps, minlength=256).astype(np.float64)
            h /= h.sum() if h.sum() > 0 else 1
            hists.append(h)
        clusters = []
        for i, h in enumerate(hists):
            placed = False
            for c in clusters:
                if jensenshannon(h, c["centroid"]) < threshold:
                    c["idx"].append(i)
                    c["centroid"] = (c["centroid"] + h) / 2
                    placed = True
                    break
            if not placed:
                clusters.append({"idx": [i], "centroid": h})
        return [c["idx"] for c in clusters]

    def build_huffman_index(self, exps, block_size=4096, ngram=1):
        if ngram > 1:
            exps = self.pack_ngrams_uint(exps, n=ngram).astype(np.uint32)

        unique, counts = np.unique(exps, return_counts=True)
        root = build_huffman_tree(unique.tolist(), counts.tolist())
        codes = generate_huffman_codes(root)

        offsets = []
        bit_offset = 0
        for i, sym in enumerate(exps):
            if i % block_size == 0:
                offsets.append(bit_offset)
            bit_offset += len(codes[sym])

        return np.array(offsets, dtype=np.uint64), codes, bit_offset

    def build_global_exponent_index(self, block_size=4096, ngram=1):
        exponents = []
        mantissa_bits = 0
        sign_bits = 0
        for _, param in self._named_params:
            sign, exps, mant = extract_float_parts(param)
            exponents.append(exps)
            sign_bits += sign.size
            mantissa_bits += mant.size * get_float_format(param.dtype)["mantissa_bits"]
        exps = np.concatenate(exponents)
        offsets, codes, total_bits_exp = self.build_huffman_index(
            exps, block_size=block_size, ngram=ngram
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
        b = position // self._exp_block_size
        if b >= len(self._exp_offsets):
            raise IndexError("position out of range for exponent stream")
        return self._exp_offsets[b]

    def zipnn_compress(self, verbose=1):
        """Classic ZipNN: compress only exponent field globally."""
        exponents = []
        mantissa_bits = 0
        sign_bits = 0
        for _, param in self._named_params:
            sign, exps, mant = extract_float_parts(param)
            exponents.append(exps)
            sign_bits += sign.size
            mantissa_bits += mant.size * get_float_format(param.dtype)["mantissa_bits"]
        exps = np.concatenate(exponents)
        bits_exp = self.compute_zipnn_bits(exps, method="huffman")
        total_bits = bits_exp + mantissa_bits + sign_bits
        if verbose:
            self.log_bits("ZipNN", total_bits)
        return total_bits

    def neuralzip_compress(
        self,
        verbose=5,
        ngram=1,
        method="huffman",
        cluster=False,
        exp="",
        validate=False,
    ):
        total_bits = 0
        layer_arrays = [param for _, param in self._named_params]

        if cluster:
            clusters = self.cluster_layers(layer_arrays)
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
                    sign, exps_raw, mant_raw = extract_float_parts(arr)
                    fmt = get_float_format(arr.dtype)
                    sign_size = self.encode_signs(sign, validate=validate)
                    mantissa_size = self.encode_mantissa(
                        mant_raw, fmt["mantissa_bits"], validate=validate
                    )
                    if validate:
                        sign_size = len(sign_size)
                        mantissa_size = len(mantissa_size)

                    total_bytes += 4 + 1 + 4 + 4
                    total_bytes += sign_size + mantissa_size

                    if method == "sigma":
                        exps_model = exps_raw.astype(np.uint8)
                        exps_tail = np.array([], dtype=np.uint8)
                    else:
                        if ngram > 1:
                            exps_model, exps_tail = self.pack_ngrams_uint_with_tail(
                                exps_raw, n=ngram
                            )
                        else:
                            exps_model = exps_raw
                            exps_tail = np.array([], dtype=np.uint8)

                    total_bytes += 4 + exps_tail.size
                    cluster_streams[cluster_lookup[idx]].append(
                        {
                            "packed": exps_model.astype(np.uint32, copy=False),
                            "tail": exps_tail.astype(np.uint8, copy=False),
                            "raw": exps_raw.astype(np.uint32, copy=False),
                        }
                    )

            for cluster_stream in cluster_streams:
                packed_parts = [
                    part["packed"] for part in cluster_stream if part["packed"].size
                ]
                packed_stream = (
                    np.concatenate(packed_parts).astype(np.uint32, copy=False)
                    if packed_parts
                    else np.array([], dtype=np.uint32)
                )
                payload = self.encode_exponent_payload(
                    packed_stream, method=method, validate=validate
                )
                total_bytes += self.cluster_payload_byte_length(payload)
                if method == "arith" and validate and packed_stream.size:
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
                    if not validate:
                        offset += part["packed"].size
                        continue
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
                        raise ValueError(
                            "NeuralZip block exponent reconstruction failed."
                        )

            total_bits = total_bytes * 8
        else:
            total_bytes = 0
            for name, param in tqdm(
                self._named_params, desc="NeuralZip", total=len(layer_arrays)
            ):
                if param.numel() == 0:
                    continue
                sign, exps, mant = extract_float_parts(param)
                fmt = get_float_format(param.dtype)
                sign_size = self.encode_signs(sign, validate=validate)
                mantissa_size = self.encode_mantissa(
                    mant, fmt["mantissa_bits"], validate=validate
                )
                if validate:
                    sign_size = len(sign_size)
                    mantissa_size = len(mantissa_size)
                total_bytes += 4 + 1 + 4 + 4 + sign_size + mantissa_size

                if method == "sigma":
                    exps_model = exps.astype(np.uint8)
                    exps_tail = np.array([], dtype=np.uint8)
                else:
                    if ngram > 1:
                        exps_model, exps_tail = self.pack_ngrams_uint_with_tail(
                            exps, n=ngram
                        )
                    else:
                        exps_model = exps
                        exps_tail = np.array([], dtype=np.uint8)

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
            total_bits = total_bytes * 8

        if verbose:
            self.log_bits(
                f"NeuralZip ({exp}) [n={ngram}, {method}, cluster={cluster}]",
                total_bits,
            )
        return total_bits

    def zstd_compress(self, level=19, verbose=1):
        raw = b"".join(
            param.detach().cpu().contiguous().view(torch.uint8).numpy().tobytes()
            for _, param in self._named_params
        )
        cctx = zstd.ZstdCompressor(level=level, threads=8)
        compressed = cctx.compress(raw)
        bits = len(compressed) * 8
        if verbose:
            self.log_bits(f"Zstd/level={level}", bits)
        return bits
