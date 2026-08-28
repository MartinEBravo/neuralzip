"""Tensor identity helpers shared by codecs and experiment runners."""

import torch


def bitwise_equal(left, right):
    return (
        left.shape == right.shape
        and left.dtype == right.dtype
        and torch.equal(
            left.detach().cpu().contiguous().reshape(-1).view(torch.uint8),
            right.detach().cpu().contiguous().reshape(-1).view(torch.uint8),
        )
    )
