"""Pick the torch device for inference."""

from __future__ import annotations


def resolve_device(device: str) -> str:
    """Return ``device`` unchanged, or the best available one when it is ``"auto"``."""
    if device != "auto":
        return device
    import torch

    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"
