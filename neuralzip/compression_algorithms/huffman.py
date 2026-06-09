import ctypes
from pathlib import Path

import numpy as np


class NativeHuffmanCoder:
    def __init__(self, library_path):
        self._lib = ctypes.CDLL(str(library_path))
        self._lib.huffman_code_lengths.argtypes = [
            ctypes.POINTER(ctypes.c_uint32),
            ctypes.POINTER(ctypes.c_uint32),
            ctypes.c_size_t,
            ctypes.POINTER(ctypes.c_uint8),
        ]
        self._lib.huffman_code_lengths.restype = ctypes.c_int
        self._lib.huffman_encode.argtypes = [
            ctypes.POINTER(ctypes.c_uint32),
            ctypes.c_size_t,
            ctypes.POINTER(ctypes.c_uint32),
            ctypes.POINTER(ctypes.c_uint8),
            ctypes.c_size_t,
            ctypes.POINTER(ctypes.POINTER(ctypes.c_uint8)),
            ctypes.POINTER(ctypes.c_size_t),
        ]
        self._lib.huffman_encode.restype = ctypes.c_int
        self._lib.huffman_decode.argtypes = [
            ctypes.POINTER(ctypes.c_uint8),
            ctypes.c_size_t,
            ctypes.POINTER(ctypes.c_uint32),
            ctypes.POINTER(ctypes.c_uint8),
            ctypes.c_size_t,
            ctypes.c_size_t,
            ctypes.POINTER(ctypes.POINTER(ctypes.c_uint32)),
        ]
        self._lib.huffman_decode.restype = ctypes.c_int
        self._lib.huffman_free.argtypes = [ctypes.c_void_p]
        self._lib.huffman_free.restype = None

    def code_lengths(self, values, counts):
        value_arr = np.ascontiguousarray(values, dtype=np.uint32)
        count_arr = np.ascontiguousarray(counts, dtype=np.uint32)
        out = np.zeros(value_arr.size, dtype=np.uint8)
        rc = self._lib.huffman_code_lengths(
            value_arr.ctypes.data_as(ctypes.POINTER(ctypes.c_uint32)),
            count_arr.ctypes.data_as(ctypes.POINTER(ctypes.c_uint32)),
            value_arr.size,
            out.ctypes.data_as(ctypes.POINTER(ctypes.c_uint8)),
        )
        if rc != 0:
            raise RuntimeError(f"Native Huffman code-length build failed with code {rc}.")
        return out

    def encode(self, symbols, values, lengths):
        symbol_arr = np.ascontiguousarray(symbols, dtype=np.uint32)
        value_arr = np.ascontiguousarray(values, dtype=np.uint32)
        length_arr = np.ascontiguousarray(lengths, dtype=np.uint8)
        out_ptr = ctypes.POINTER(ctypes.c_uint8)()
        out_size = ctypes.c_size_t()
        rc = self._lib.huffman_encode(
            symbol_arr.ctypes.data_as(ctypes.POINTER(ctypes.c_uint32)),
            symbol_arr.size,
            value_arr.ctypes.data_as(ctypes.POINTER(ctypes.c_uint32)),
            length_arr.ctypes.data_as(ctypes.POINTER(ctypes.c_uint8)),
            value_arr.size,
            ctypes.byref(out_ptr),
            ctypes.byref(out_size),
        )
        if rc != 0:
            raise RuntimeError(f"Native Huffman encode failed with code {rc}.")
        try:
            return ctypes.string_at(out_ptr, out_size.value)
        finally:
            self._lib.huffman_free(out_ptr)

    def decode(self, encoded, values, lengths, symbol_count):
        byte_arr = np.frombuffer(encoded, dtype=np.uint8)
        value_arr = np.ascontiguousarray(values, dtype=np.uint32)
        length_arr = np.ascontiguousarray(lengths, dtype=np.uint8)
        out_ptr = ctypes.POINTER(ctypes.c_uint32)()
        rc = self._lib.huffman_decode(
            byte_arr.ctypes.data_as(ctypes.POINTER(ctypes.c_uint8)),
            byte_arr.size,
            value_arr.ctypes.data_as(ctypes.POINTER(ctypes.c_uint32)),
            length_arr.ctypes.data_as(ctypes.POINTER(ctypes.c_uint8)),
            value_arr.size,
            symbol_count,
            ctypes.byref(out_ptr),
        )
        if rc != 0:
            raise RuntimeError(f"Native Huffman decode failed with code {rc}.")
        try:
            result = np.ctypeslib.as_array(out_ptr, shape=(symbol_count,)).copy()
        finally:
            self._lib.huffman_free(out_ptr)
        return result.astype(np.uint32, copy=False)


def load_huffman_coder():
    library_path = Path(__file__).with_name("huffman_coder.so")
    if not library_path.exists():
        raise FileNotFoundError(
            f"Missing native Huffman coder: {library_path}. "
            "Build huffman_coder.so before running experiments."
        )
    return NativeHuffmanCoder(library_path)


HUFFMAN_CODER = load_huffman_coder()


def huffman_code_lengths(symbols, counts):
    return HUFFMAN_CODER.code_lengths(symbols, counts)


def encode_huffman_stream(symbols, values, lengths):
    return HUFFMAN_CODER.encode(symbols, values, lengths)


def decode_huffman_stream(encoded, values, lengths, symbol_count):
    if symbol_count == 0:
        return np.array([], dtype=np.uint32)
    return HUFFMAN_CODER.decode(encoded, values, lengths, symbol_count)
