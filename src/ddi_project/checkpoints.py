from __future__ import annotations

import io
import pickle
from pathlib import Path

import torch


def load_checkpoint(path: str | Path, device: torch.device):
    """Load torch checkpoints and notebook pickle checkpoints on CPU or GPU."""
    checkpoint = Path(path)
    try:
        return torch.load(checkpoint, map_location=device, weights_only=False)
    except RuntimeError:
        original_loader = torch.storage._load_from_bytes

        def load_from_bytes(buffer: bytes):
            return torch.load(io.BytesIO(buffer), map_location=device, weights_only=False)

        torch.storage._load_from_bytes = load_from_bytes
        try:
            with checkpoint.open("rb") as handle:
                return pickle.load(handle)
        finally:
            torch.storage._load_from_bytes = original_loader
