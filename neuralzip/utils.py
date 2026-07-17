import torch

from . import huff0


FLOAT_LAYOUTS = {
    torch.float16: {"exponent_bits": 5, "mantissa_bits": 10},
    torch.bfloat16: {"exponent_bits": 8, "mantissa_bits": 7},
    torch.float32: {"exponent_bits": 8, "mantissa_bits": 23},
    torch.float64: {"exponent_bits": 11, "mantissa_bits": 52},
}


def _as_byte_stream(data):
    if torch.is_tensor(data):
        return data.detach().cpu().contiguous().view(torch.uint8).reshape(-1)
    if isinstance(data, bytes):
        return torch.tensor(tuple(data), dtype=torch.uint8)
    if isinstance(data, bytearray):
        return torch.tensor(tuple(data), dtype=torch.uint8)
    return torch.as_tensor(data, dtype=torch.uint8).reshape(-1)


def _tensor_bits(tensor):
    tensor = tensor.detach().cpu().contiguous()
    element_size = tensor.element_size()
    bytes_view = tensor.view(torch.uint8).reshape(-1, element_size).to(torch.int64)
    bits = torch.zeros(bytes_view.shape[0], dtype=torch.int64)
    for byte_index in range(element_size):
        bits |= bytes_view[:, byte_index] << (8 * byte_index)
    return bits


def split_fields(tensor, bit_width=None):
    """Return sign, exponent, and mantissa integer tensors for a float tensor."""

    if not torch.is_tensor(tensor) or not tensor.is_floating_point():
        raise TypeError("split_fields expects a floating-point torch tensor.")
    if tensor.dtype not in FLOAT_LAYOUTS:
        raise TypeError(f"Unsupported floating dtype: {tensor.dtype}.")

    actual_bit_width = tensor.element_size() * 8
    if bit_width is not None and int(bit_width) != actual_bit_width:
        raise ValueError(
            f"bit_width={bit_width} does not match tensor dtype width "
            f"{actual_bit_width}."
        )

    layout = FLOAT_LAYOUTS[tensor.dtype]
    exponent_bits = layout["exponent_bits"]
    mantissa_bits = layout["mantissa_bits"]

    bits = _tensor_bits(tensor)
    sign = (bits >> (exponent_bits + mantissa_bits)) & 0x1
    exponent = (bits >> mantissa_bits) & ((1 << exponent_bits) - 1)
    mantissa = bits & ((1 << mantissa_bits) - 1)
    return sign.reshape(tensor.shape), exponent.reshape(tensor.shape), mantissa.reshape(tensor.shape)


def partition_into_byte_groups(data, element_size=None):
    """Split a byte stream into ZipNN-style streams by byte offset."""

    if torch.is_tensor(data):
        element_size = data.element_size() if element_size is None else int(element_size)
        stream = _as_byte_stream(data)
    else:
        if element_size is None:
            raise ValueError("element_size is required for raw byte streams.")
        element_size = int(element_size)
        stream = _as_byte_stream(data)

    if element_size <= 0:
        raise ValueError("element_size must be positive.")
    if stream.numel() % element_size != 0:
        raise ValueError("byte stream length must be divisible by element_size.")

    return tuple(
        stream[byte_offset::element_size].contiguous().numpy().tobytes()
        for byte_offset in range(element_size)
    )


def entropic_encode(data, method="huffman", *, min_saving_bytes=16):
    """Compress byte-valued data with the selected entropy coder."""

    method_key = method.lower().replace("-", "_")
    if method_key != "huffman":
        raise ValueError("Only method='huffman' is implemented.")
    stream = _as_byte_stream(data).numpy().tobytes()
    return huff0.compress(stream, min_saving_bytes=min_saving_bytes)


def is_compressible(data, *, min_saving_bytes=16):
    """Return True when Huffman beats raw storage plus codebook overhead."""

    return entropic_encode(data, min_saving_bytes=min_saving_bytes) is not None
