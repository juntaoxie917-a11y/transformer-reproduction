import torch


def greedy_decode(
    model,
    src,
    src_mask,
    bos_id: int,
    eos_id: int,
    pad_id: int,
    max_len: int,
    make_tgt_mask
):
    """
    Greedy decoding for sequence generation.

    Args:
        model: The sequence-to-sequence model.
        src: The source input tensor.
        src_mask: The source mask tensor.
        bos_id: The beginning-of-sequence token ID.
        eos_id: The end-of-sequence token ID.
        pad_id: The padding token ID.
        max_len: The maximum length of the generated sequence.
        make_tgt_mask: A function to create the target mask.

    Returns:
        generated:
            Generated target token IDs.
            Shape: (B, generated_length)
    """

    model.eval()

    batch_size = src.size(0)
    device = src.device

    # Initialize the generated sequence with the BOS token
    generated = torch.full(
        (batch_size, 1),
        bos_id,
        dtype=torch.long,
        device=device
    )
    
    finished = torch.zeros(
        batch_size,
        dtype=torch.bool,
        device=device
    )

    with torch.no_grad():
        for _ in range(max_len - 1):
            tgt_mask = make_tgt_mask(generated, pad_id)

            hidden = model(src, generated, src_mask, tgt_mask)
            logits = model.generatedor(hidden)

            next_token_logits = logits[:, -1, :]
            next_token = torch.argmax(
                next_token_logits,
                dim=-1
            )

            generated = torch.cat(
                [
                    generated,
                    next_token.unsqueeze(1)
                ],
                dim=-1
            )

            finished |= (next_token == eos_id)

            if finished.all():
                break

    return generated


def strip_special_tokens(
    ids,
    bos_id: int,
    eos_id: int,
    pad_id: int
):
    result = []

    for token_id in ids:
        if token_id == bos_id:
            continue
        if token_id == eos_id:
            break
        if token_id == pad_id:
            continue
        result.append(token_id)

    return result
