import torch

FLOAT_FORMATS = {
    torch.float16: {
        "view_dtype": torch.uint16,
        "sign_shift": 15,
        "exp_shift": 10,
        "exp_mask": 0x1F,
        "mantissa_mask": 0x3FF,
        "total_bits": 16,
        "mantissa_bits": 10,
    },
    torch.bfloat16: {
        "view_dtype": torch.uint16,
        "sign_shift": 15,
        "exp_shift": 7,
        "exp_mask": 0xFF,
        "mantissa_mask": 0x7F,
        "total_bits": 16,
        "mantissa_bits": 7,
    },
    torch.float32: {
        "view_dtype": torch.uint32,
        "sign_shift": 31,
        "exp_shift": 23,
        "exp_mask": 0xFF,
        "mantissa_mask": 0x7FFFFF,
        "total_bits": 32,
        "mantissa_bits": 23,
    },
}
