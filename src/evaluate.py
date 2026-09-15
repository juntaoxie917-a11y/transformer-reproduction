import torch
from sacrebleu.metrics import BLEU
from pathlib import Path

from src.decode import greedy_decode


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


def evaluate_bleu(
    model,
    dataloader,
    tokenizer,
    device: torch.device,
    make_src_mask,
    make_tgt_mask,
    max_len: int = 100
):
    """
    Generate translations for the entire dataset and compute the BLEU score.

    Returns:
        bleu_score: The BLEU score for the generated translations.
        predictions: A list of generated translations (as strings).
        references: A list of reference translations (as strings).
    """

    model.eval()

    predictions = []
    references = []

    with torch.no_grad():
        for src, tgt in dataloader:
            src = src.to(device)
            src_mask = make_src_mask(
                src,
                tokenizer.pad_id
            )

            generated = greedy_decode(
                model=model,
                src=src,
                src_mask=src_mask,
                bos_id=tokenizer.bos_id,
                eos_id=tokenizer.eos_id,
                pad_id=tokenizer.pad_id,
                max_len=max_len,
                make_tgt_mask=make_tgt_mask
            )

            for pred_ids, ref_ids in zip(generated, tgt):
                pred_ids = strip_special_tokens(
                    pred_ids.tolist(),
                    bos_id=tokenizer.bos_id,
                    eos_id=tokenizer.eos_id,
                    pad_id=tokenizer.pad_id
                )

                ref_ids = strip_special_tokens(
                    ref_ids.tolist(),
                    bos_id=tokenizer.bos_id,
                    eos_id=tokenizer.eos_id,
                    pad_id=tokenizer.pad_id
                )

                pred_text = tokenizer.decode(pred_ids)
                ref_text = tokenizer.decode(ref_ids)

                predictions.append(pred_text)
                references.append(ref_text)

    bleu = BLEU()
    score = bleu.corpus_score(predictions, [references])

    return score.score, predictions, references


def save_translations(
    path,
    predictions,
    references,
    sources=None
):
    path = Path(path)
    path.parent.mkdir(parent=True, exist_ok=True)

    with path.open("w", encoding="utf-8") as f:
        for i, (pred, ref) in enumerate(zip(predictions, references)):
            f.write(f"Example {i + 1}\n")

            if sources is not None:
                f.write(f"Source:     {sources[i]}\n")

            f.write(f"Prediction: {pred}\n")
            f.write(f"Reference:  {ref}\n")
            f.write("\n")