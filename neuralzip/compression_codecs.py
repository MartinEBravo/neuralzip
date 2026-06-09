from dataclasses import dataclass

import numpy as np

from .bitpacking import bits_required, packed_byte_length, pack_values, unpack_values
from .algorithms.arithmetic import ARITHMETIC_CODER
from .algorithms.huffman import (
    decode_huffman_stream,
    encode_huffman_stream,
    huffman_code_lengths,
)


@dataclass
class ClusterPayload:
    values: np.ndarray
    counts: np.ndarray
    encoded: bytes


def empty_payload():
    return ClusterPayload(
        values=np.array([], dtype=np.uint32),
        counts=np.array([], dtype=np.uint32),
        encoded=b"",
    )


def weighted_code_bits(counts, lengths):
    return int(np.sum(counts.astype(np.uint64) * lengths.astype(np.uint64)))


class CodecMixin:
    def compute_arithmetic_bits(self, arr, order=1):
        """Adaptive arithmetic coding bits estimation (order-1)."""
        if arr.size < 2:
            return 0.0
        alphabet = 256
        eps = 1
        arr = arr.astype(np.uint8)
        counts = np.ones((alphabet, alphabet)) * eps
        totals = np.ones(alphabet) * (alphabet * eps)
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

        alphabet = 256
        counts = np.zeros((alphabet, alphabet), dtype=np.int64)
        prev = int(arr[0])
        for x in arr[1:]:
            x = int(x)
            counts[prev, x] += 1
            prev = x

        total_bits = 0.0
        for row in counts:
            if int(row.sum()) == 0:
                continue
            symbols = np.nonzero(row)[0]
            freqs = row[symbols]
            lengths = huffman_code_lengths(symbols, freqs)
            total_bits += weighted_code_bits(freqs, lengths)

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

        values, counts = np.unique(arr, return_counts=True)
        lengths = huffman_code_lengths(values, counts)
        return weighted_code_bits(counts, lengths)

    def pack_ngrams_uint(self, exps, n=2):
        """Pack consecutive exponent symbols into 16-24-bit ngrams."""
        length = (len(exps) // n) * n
        groups = exps[:length].reshape(-1, n).astype(np.uint32)
        packed = np.zeros(groups.shape[0], dtype=np.uint32)
        for i in range(n):
            packed |= (groups[:, i] & 0xFF) << (8 * i)
        return packed

    def pack_ngrams_uint_with_tail(self, exps, n=2):
        length = (len(exps) // n) * n
        packed = self.pack_ngrams_uint(exps, n=n)
        tail = exps[length:].astype(np.uint8, copy=False)
        return packed, tail

    def unpack_ngrams_uint(self, packed, n=2):
        unpacked = np.zeros(packed.size * n, dtype=np.uint32)
        for i in range(n):
            unpacked[i::n] = (packed >> (8 * i)) & 0xFF
        return unpacked

    def encode_raw_values(self, values, bits_per_value, validate=False, label="Field"):
        payload_size = packed_byte_length(values.size, bits_per_value)
        if validate:
            payload = pack_values(values, bits_per_value)
            decoded = unpack_values(payload, values.size, bits_per_value)
            if not np.array_equal(decoded, values):
                raise ValueError(f"{label} round-trip reconstruction failed.")
            return payload
        return payload_size

    def encode_signs(self, sign, validate=False):
        return self.encode_raw_values(sign, 1, validate=validate, label="Sign")

    def encode_mantissa(self, mantissa, bits_per_value, validate=False):
        return self.encode_raw_values(
            mantissa,
            bits_per_value,
            validate=validate,
            label="Mantissa",
        )

    def encode_field_size(self, values, bits_per_value, method, validate=False):
        raw_size = packed_byte_length(values.size, bits_per_value)
        if values.size == 0:
            return 1

        payload = self.encode_exponent_payload(
            values.astype(np.uint32, copy=False),
            method=method,
            validate=validate,
        )
        compressed_size = self.cluster_payload_byte_length(payload)
        return 1 + min(raw_size, compressed_size)

    def encode_aux_field_sizes(
        self,
        sign,
        mantissa,
        mantissa_bits,
        compress_aux_fields=False,
        aux_method="huffman",
        validate=False,
    ):
        if compress_aux_fields:
            sign_size = self.encode_field_size(
                sign,
                bits_per_value=1,
                method=aux_method,
                validate=aux_method == "arith" and validate,
            )
            mantissa_size = self.encode_field_size(
                mantissa,
                bits_per_value=mantissa_bits,
                method=aux_method,
                validate=aux_method == "arith" and validate,
            )
            return sign_size, mantissa_size

        sign_size = self.encode_signs(sign, validate=validate)
        mantissa_size = self.encode_mantissa(
            mantissa,
            mantissa_bits,
            validate=validate,
        )
        if validate:
            sign_size = len(sign_size)
            mantissa_size = len(mantissa_size)
        return sign_size, mantissa_size

    def prepare_exponent_symbols(self, exponents, method, ngram):
        if method == "sigma":
            return exponents.astype(np.uint8), np.array([], dtype=np.uint8)
        if ngram > 1:
            return self.pack_ngrams_uint_with_tail(exponents, n=ngram)
        return exponents, np.array([], dtype=np.uint8)

    def encode_cluster_payload(self, exponents, validate=False):
        if exponents.size == 0:
            return empty_payload()

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

    def encode_huffman_payload(self, exponents, validate=False):
        if exponents.size == 0:
            return empty_payload()

        values, counts = np.unique(exponents, return_counts=True)
        lengths = huffman_code_lengths(values, counts)
        bit_count = weighted_code_bits(counts, lengths)
        encoded = b""
        if validate:
            encoded = encode_huffman_stream(
                exponents,
                values.astype(np.uint32, copy=False),
                lengths,
            )
            decoded = decode_huffman_stream(
                encoded,
                values=values.astype(np.uint32, copy=False),
                lengths=lengths,
                symbol_count=exponents.size,
            )
            if not np.array_equal(
                decoded.astype(np.uint32, copy=False),
                exponents.astype(np.uint32, copy=False),
            ):
                raise ValueError("Huffman payload round-trip reconstruction failed.")

        payload = ClusterPayload(
            values=values.astype(np.uint32),
            counts=counts.astype(np.uint32),
            encoded=encoded,
        )
        payload.encoded_bit_length = bit_count
        payload.encoded_byte_length = (
            len(encoded) if validate else packed_byte_length(bit_count, 1)
        )
        payload.code_lengths = lengths
        return payload

    def serialize_symbols(self, symbols):
        if symbols.size == 0:
            return b"", 1

        max_symbol = int(symbols.max())
        width = max(1, packed_byte_length(bits_required(max_symbol + 1), 1))
        if width > 4:
            raise ValueError(f"Cannot serialize {width}-byte symbols.")

        data = symbols.astype(np.uint32, copy=False)
        chunks = [(data >> (8 * i)).astype(np.uint8) for i in range(width)]
        return np.column_stack(chunks).ravel().tobytes(), width

    def encode_zipnn_huffman_payload(self, exponents, validate=False):
        if exponents.size == 0:
            payload = empty_payload()
            payload.fixed_size = 0
            return payload

        try:
            from zipnn import ZipNN
        except ImportError as exc:
            raise ImportError(
                "The official zipnn package is required for method='zipnn-huffman'."
            ) from exc

        stream, width = self.serialize_symbols(exponents.astype(np.uint32, copy=False))
        original_length = len(stream)
        if len(stream) % 2:
            stream += b"\x00"

        compressor = ZipNN(
            method="HUFFMAN",
            input_format="byte",
            bytearray_dtype="float16",
            is_streaming=False,
        )
        raw = bytearray(stream)
        encoded = compressor.compress(raw)
        if validate:
            decoded = bytes(compressor.decompress(encoded))[:original_length]
            if decoded != stream[:original_length]:
                raise ValueError("ZipNN Huffman payload round-trip failed.")

        payload = empty_payload()
        payload.fixed_size = 1 + 8 + 1 + len(encoded)
        payload.symbol_width = width
        return payload

    def huffman_bits_from_counts(self, counts):
        symbols = np.nonzero(counts)[0]
        if symbols.size == 0:
            return 0
        if symbols.size == 1:
            return int(counts[symbols[0]])
        freqs = counts[symbols]
        lengths = huffman_code_lengths(symbols, freqs)
        return weighted_code_bits(freqs, lengths)

    def encode_context_huffman_payload(self, exponents, order=2, alphabet_size=256):
        if exponents.size == 0:
            return empty_payload()

        if order <= 0 or exponents.size <= order:
            return self.encode_huffman_payload(exponents, validate=False)

        data = exponents.astype(np.uint64, copy=False)
        prefix_bits = order * bits_required(alphabet_size)
        contexts = np.zeros(data.size - order, dtype=np.uint64)
        for i in range(order):
            contexts = (contexts * alphabet_size) + data[i : data.size - order + i]
        symbols = data[order:]
        pair_keys = contexts * alphabet_size + symbols
        pair_keys, pair_counts = np.unique(pair_keys, return_counts=True)

        payload_bits = prefix_bits
        model_bytes = 8
        context_ids = pair_keys // alphabet_size
        next_symbols = (pair_keys % alphabet_size).astype(np.uint32, copy=False)
        boundaries = np.flatnonzero(np.diff(context_ids)) + 1
        starts = np.r_[0, boundaries]
        ends = np.r_[boundaries, pair_keys.size]
        for start, end in zip(starts, ends):
            counts = np.zeros(alphabet_size, dtype=np.uint32)
            counts[next_symbols[start:end]] = pair_counts[start:end].astype(np.uint32)
            used = np.count_nonzero(counts)
            model_bytes += 4 + 2 + used * (1 + 4)
            payload_bits += self.huffman_bits_from_counts(counts)

        payload = empty_payload()
        context_size = model_bytes + packed_byte_length(payload_bits, 1)
        base_payload = self.encode_huffman_payload(exponents, validate=False)
        base_size = self.cluster_payload_byte_length(base_payload)
        payload.fixed_size = 1 + min(context_size, base_size)
        return payload

    def encode_exponent_payload(self, exponents, method, validate=False):
        if method == "arith":
            return self.encode_cluster_payload(exponents, validate=validate)
        if method == "huffman":
            return self.encode_huffman_payload(exponents, validate=validate)
        if method == "zipnn-huffman":
            return self.encode_zipnn_huffman_payload(exponents, validate=validate)
        if method.startswith("ctx-huffman"):
            if validate:
                raise ValueError("Context Huffman validation is not implemented.")
            _, _, order = method.partition(":")
            return self.encode_context_huffman_payload(
                exponents,
                order=int(order) if order else 2,
            )
        raise ValueError(f"Unsupported NeuralZip method: {method}")

    def cluster_payload_byte_length(self, payload):
        fixed_size = getattr(payload, "fixed_size", None)
        if fixed_size is not None:
            return fixed_size
        model_bytes = getattr(payload, "model_bytes", None)
        if model_bytes is not None:
            return model_bytes + getattr(
                payload,
                "encoded_byte_length",
                len(payload.encoded),
            )
        if payload.values.size:
            lengths = getattr(payload, "code_lengths", None)
            if lengths is not None:
                return (
                    4
                    + 8
                    + payload.values.nbytes
                    + lengths.nbytes
                    + getattr(payload, "encoded_byte_length", len(payload.encoded))
                )
            size = 4 + 8 + getattr(
                payload,
                "encoded_byte_length",
                len(payload.encoded),
            )
            size += payload.values.nbytes + payload.counts.nbytes
            return size
        return 4 + 8 + getattr(payload, "encoded_byte_length", len(payload.encoded))
