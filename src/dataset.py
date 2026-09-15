"""
`dataset.py` is responsible for reading the sentence pairs, tokenizing 
them and padding batches.
"""


from pathlib import Path
import torch
from torch.utils.data import Dataset, DataLoader
from src.tokenizer import Tokenizer


class TranslationDataset(Dataset):
    def __init__(
        self,
        src_path: str,
        tgt_path: str,
        tokenizer: Tokenizer,
    ):
        self.tokenizer = tokenizer
        self.src_sentences = self._read_lines(src_path)
        self.tgt_sentences = self._read_lines(tgt_path)

        if len(self.src_sentences) != len(self.tgt_sentences):
            raise ValueError(
                "Source and target files contain different numbers " \
                "of sentences."
            )

    @staticmethod
    def _read_lines(path: str) -> list[str]:
        path = Path(path)
        with path.open("r", encoding="utf-8") as f:
            return [line.strip() for line in f]

    def __len__(self) -> int:
        return len(self.src_sentences)

    def __getitem__(self, idx: int):
        src_text = self.src_sentences[idx]
        tgt_text = self.tgt_sentences[idx]

        src_ids = self.tokenizer.encode(
            src_text,
            add_bos=False,
            add_eos=True
        )
        tgt_ids = self.tokenizer.encode(
            tgt_text,
            add_bos=True,
            add_eos=True
        )

        return (
            torch.tensor(src_ids, dtype=torch.long),
            torch.tensor(tgt_ids, dtype=torch.long)
        )


# To make sure that the samples have the same length, we need a function 
# to pad them, so that these samples can be stacked into a batch.
# Note that the function below is not a synthetic function, but a 
# closure function that returns a function.
# Padding is performed within each batch, not globally.

def make_collate_fn(pad_idx: int):

    def collate_fn(batch):
        src_sequences, tgt_sequences = zip(*batch)

        src_batch = torch.nn.utils.rnn.pad_sequence(
            src_sequences,
            batch_first=True,
            padding_value=pad_idx
        )

        tgt_batch = torch.nn.utils.rnn.pad_sequence(
            tgt_sequences,
            batch_first=True,
            padding_value=pad_idx
        )

        return src_batch, tgt_batch

    return collate_fn


def build_dataloader(
    src_path: str,
    tgt_path: str,
    tokenizer: Tokenizer,
    batch_size: int,
    shuffle: bool = True,  # Train / Eval
):
    dataset = TranslationDataset(
        src_path=src_path,
        tgt_path=tgt_path,
        tokenizer=tokenizer
    )

    collate_fn = make_collate_fn(pad_idx=tokenizer.pad_id)

    return DataLoader(
        dataset=dataset,
        batch_size=batch_size,
        shuffle=shuffle,       # True / False
        collate_fn=collate_fn  # Padding
    )