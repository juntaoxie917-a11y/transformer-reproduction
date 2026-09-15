import sentencepiece as spm


class Tokenizer:
    def __init__(self, model_path: str):
        self.sp = spm.SentencePieceProcessor()
        self.sp.load(model_path)

    @property
    def pad_id(self) -> int:
        return self.sp.pad_id()

    @property
    def bos_id(self) -> int:
        return self.sp.bos_id()

    @property
    def eos_id(self) -> int:
        return self.sp.eos_id()

    @property
    def unk_id(self) -> int:
        return self.sp.unk_id()

    @property
    def vocab_size(self) -> int:
        return self.sp.vocab_size()

    def encode(
        self,
        text: str,
        add_bos: bool = False,
        add_eos: bool = False
    ) -> list[int]:
        ids = self.sp.encode(text, out_type=int)
        if add_bos:
            ids = [self.bos_id] + ids
        if add_eos:
            ids = ids + [self.eos_id]
        return ids

    def decode(self, ids: list[int]) -> str:
        return self.sp.decode(ids)