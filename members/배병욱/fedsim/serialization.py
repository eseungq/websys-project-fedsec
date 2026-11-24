"""REST 본문용 텐서 직렬화 (torch.save, 읽을 때 weights_only=True)."""
import io

import torch


def to_bytes(tensor):
    buf = io.BytesIO()
    torch.save(tensor.detach().cpu().contiguous(), buf)
    return buf.getvalue()


def from_bytes(raw):
    t = torch.load(io.BytesIO(raw), weights_only=True)
    if not isinstance(t, torch.Tensor):
        raise ValueError("텐서가 아닌 본문")
    return t
