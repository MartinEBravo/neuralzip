import ctypes
from pathlib import Path

import numpy as np


class NativeArithmeticCoder:
    def __init__(self, library_path):
        self._lib = ctypes.CDLL(str(library_path))
        self._lib.arithmetic_encode.argtypes = [
            ctypes.POINTER(ctypes.c_uint32),
            ctypes.c_size_t,
            ctypes.POINTER(ctypes.c_uint32),
            ctypes.c_size_t,
            ctypes.POINTER(ctypes.POINTER(ctypes.c_uint8)),
            ctypes.POINTER(ctypes.c_size_t),
        ]
        self._lib.arithmetic_encode.restype = ctypes.c_int
        self._lib.arithmetic_decode.argtypes = [
            ctypes.POINTER(ctypes.c_uint8),
            ctypes.c_size_t,
            ctypes.POINTER(ctypes.c_uint32),
            ctypes.c_size_t,
            ctypes.c_size_t,
            ctypes.POINTER(ctypes.POINTER(ctypes.c_uint32)),
        ]
        self._lib.arithmetic_decode.restype = ctypes.c_int
        self._lib.arithmetic_free.argtypes = [ctypes.c_void_p]
        self._lib.arithmetic_free.restype = None

    def encode(self, symbols, counts):
        symbol_arr = np.ascontiguousarray(symbols, dtype=np.uint32)
        count_arr = np.ascontiguousarray(counts, dtype=np.uint32)
        out_ptr = ctypes.POINTER(ctypes.c_uint8)()
        out_size = ctypes.c_size_t()
        rc = self._lib.arithmetic_encode(
            symbol_arr.ctypes.data_as(ctypes.POINTER(ctypes.c_uint32)),
            symbol_arr.size,
            count_arr.ctypes.data_as(ctypes.POINTER(ctypes.c_uint32)),
            count_arr.size,
            ctypes.byref(out_ptr),
            ctypes.byref(out_size),
        )
        if rc != 0:
            raise RuntimeError(f"Native arithmetic encode failed with code {rc}.")
        try:
            return ctypes.string_at(out_ptr, out_size.value)
        finally:
            self._lib.arithmetic_free(out_ptr)

    def decode(self, data, counts, symbol_count):
        byte_arr = np.frombuffer(data, dtype=np.uint8)
        count_arr = np.ascontiguousarray(counts, dtype=np.uint32)
        out_ptr = ctypes.POINTER(ctypes.c_uint32)()
        rc = self._lib.arithmetic_decode(
            byte_arr.ctypes.data_as(ctypes.POINTER(ctypes.c_uint8)),
            byte_arr.size,
            count_arr.ctypes.data_as(ctypes.POINTER(ctypes.c_uint32)),
            count_arr.size,
            symbol_count,
            ctypes.byref(out_ptr),
        )
        if rc != 0:
            raise RuntimeError(f"Native arithmetic decode failed with code {rc}.")
        try:
            result = np.ctypeslib.as_array(out_ptr, shape=(symbol_count,)).copy()
        finally:
            self._lib.arithmetic_free(out_ptr)
        return result.astype(np.uint32, copy=False)


def load_arithmetic_coder():
    library_path = Path(__file__).with_name("arithmetic_coder.so")
    if not library_path.exists():
        raise FileNotFoundError(
            f"Missing native arithmetic coder: {library_path}. "
            "Build arithmetic_coder.so before running experiments."
        )
    return NativeArithmeticCoder(library_path)


ARITHMETIC_CODER = load_arithmetic_coder()
