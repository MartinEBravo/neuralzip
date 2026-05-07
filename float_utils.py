import numpy as np

from formats import FLOAT_FORMATS


def get_float_format(dtype):
    try:
        return FLOAT_FORMATS[dtype]
    except KeyError as exc:
        raise TypeError(f"Unsupported floating dtype: {dtype}") from exc


def extract_float_parts(tensor):
    tensor = tensor.detach().cpu().contiguous().flatten()
    fmt = get_float_format(tensor.dtype)
    bits = tensor.view(fmt["view_dtype"]).numpy().astype(np.uint32, copy=False)
    sign = (bits >> fmt["sign_shift"]) & 0x1
    exponents = (bits >> fmt["exp_shift"]) & fmt["exp_mask"]
    mantissa = bits & fmt["mantissa_mask"]
    return (
        sign.astype(np.uint32),
        exponents.astype(np.uint32),
        mantissa.astype(np.uint32),
    )
