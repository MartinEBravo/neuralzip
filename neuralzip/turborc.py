import ctypes
import os
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

DEFAULT_METHOD = "rccs"

_METHODS = {
    "rccs": ("rccsenc", "rccsdec"),
}


class TurboRangeUnavailable(RuntimeError):
    pass


@dataclass(frozen=True)
class TurboRangePayload:
    payload: bytes
    method: str
    symbol_count: int


def _normalize_method(method):
    key = str(method or DEFAULT_METHOD).strip().lower()
    key = key.replace("-", "").replace("_", "")
    key = key.removeprefix("turborc")
    if key not in _METHODS:
        choices = ", ".join(sorted(_METHODS))
        raise ValueError(
            f"Unsupported TurboRC method {method!r}. Expected one of: {choices}."
        )
    return key


def _candidate_library_paths(path=None):
    if path:
        yield path
    for env_var in ("NEURALZIP_TURBORC_LIB", "TURBORC_LIB"):
        value = os.environ.get(env_var)
        if value:
            yield value
    yield "/tmp/turborc-build/libturborc_min.so"
    yield "/usr/local/lib/libturborc_min.so"
    yield "/usr/local/lib/libturborc.so"


@lru_cache(maxsize=8)
def _load_library(path=None):
    tried = []
    for candidate in _candidate_library_paths(path):
        candidate_path = Path(candidate).expanduser()
        tried.append(str(candidate_path))
        if candidate_path.exists():
            return ctypes.CDLL(str(candidate_path))
    locations = "\n".join(f"  - {item}" for item in tried)
    raise TurboRangeUnavailable(
        "TurboRC shared library not found. Build Turbo-Range-Coder and set "
        "NEURALZIP_TURBORC_LIB to the resulting .so.\n"
        f"Tried:\n{locations}"
    )


def _function_pair(method, *, library_path=None):
    method = _normalize_method(method)
    library = _load_library(library_path)
    encoder_name, decoder_name = _METHODS[method]
    try:
        encoder = getattr(library, encoder_name)
        decoder = getattr(library, decoder_name)
    except AttributeError as exc:
        raise TurboRangeUnavailable(
            f"TurboRC library does not export {encoder_name}/{decoder_name}."
        ) from exc

    pointer = ctypes.POINTER(ctypes.c_ubyte)
    signature = [pointer, ctypes.c_size_t, pointer]
    encoder.argtypes = signature
    decoder.argtypes = signature
    encoder.restype = ctypes.c_size_t
    decoder.restype = ctypes.c_size_t
    return method, encoder, decoder


def _buffer(data):
    if not isinstance(data, (bytes, bytearray, memoryview)):
        data = bytes(data)
    data = bytes(data)
    return data, ctypes.create_string_buffer(data, len(data))


def encode(data, *, method=DEFAULT_METHOD, library_path=None):
    data, input_buffer = _buffer(data)
    if not data:
        return TurboRangePayload(
            payload=b"",
            method=_normalize_method(method),
            symbol_count=0,
        )

    method, encoder, _ = _function_pair(method, library_path=library_path)
    output_capacity = max(len(data) * 2 + 4096, 4096)
    output_buffer = ctypes.create_string_buffer(output_capacity)
    pointer = ctypes.POINTER(ctypes.c_ubyte)
    written = int(
        encoder(
            ctypes.cast(input_buffer, pointer),
            len(data),
            ctypes.cast(output_buffer, pointer),
        )
    )
    if written <= 0 or written > output_capacity:
        raise TurboRangeUnavailable(
            f"TurboRC {method} produced invalid payload size {written}."
        )
    return TurboRangePayload(
        payload=output_buffer.raw[:written],
        method=method,
        symbol_count=len(data),
    )


def compress(
    data,
    *,
    method=DEFAULT_METHOD,
    min_saving_bytes=0,
    max_output_bytes=None,
    library_path=None,
):
    encoded = encode(data, method=method, library_path=library_path)
    if max_output_bytes is not None and len(encoded.payload) > int(max_output_bytes):
        return None
    if len(encoded.payload) + int(min_saving_bytes) >= encoded.symbol_count:
        return None
    return encoded


def decompress(payload, *, library_path=None):
    if not isinstance(payload, TurboRangePayload):
        raise TypeError("Expected a TurboRangePayload.")
    if payload.symbol_count == 0:
        return b""

    _, _, decoder = _function_pair(payload.method, library_path=library_path)
    _, input_buffer = _buffer(payload.payload)
    output_buffer = ctypes.create_string_buffer(int(payload.symbol_count))
    pointer = ctypes.POINTER(ctypes.c_ubyte)
    decoder(
        ctypes.cast(input_buffer, pointer),
        int(payload.symbol_count),
        ctypes.cast(output_buffer, pointer),
    )
    return output_buffer.raw[: int(payload.symbol_count)]
