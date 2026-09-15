import torch


def subsequent_mask(size: int) ->torch.Tensor:
    shape = (1, size, size)
    return torch.tril(torch.ones(shape, dtype=torch.bool))

def make_src_mask(src: torch.Tensor, pad_idx: int) ->torch.Tensor:
    return (src != pad_idx).unsqueeze(-2)

def make_tgt_mask(tgt: torch.Tensor, pad_idx: int) -> torch.Tensor:
    padding_mask = (tgt != pad_idx).unsqueeze(-2)
    size = tgt.size(-1)
    casual_mask = subsequent_mask(size).to(tgt.device)
    return padding_mask & casual_mask