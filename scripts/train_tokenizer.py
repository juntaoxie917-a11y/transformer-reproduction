import shutil
from src.paths import DATA_DIR, TOKENIZER_WORK_DIR, TOKENIZER_PATH, ensure_artifact_dirs
import sentencepiece as spm


# ------------------------------------------------------------
# Paths
# ------------------------------------------------------------

TOKENIZER_DIR = TOKENIZER_WORK_DIR
TOKENIZER_DIR.mkdir(parents=True, exist_ok=True)

SRC_PATH = DATA_DIR / "train.en"
TGT_PATH = DATA_DIR / "train.de"

COMBINED_PATH = TOKENIZER_DIR / "train_shared.txt"
MODEL_PREFIX = TOKENIZER_DIR / "bpe"


# ------------------------------------------------------------
# Configuration
# ------------------------------------------------------------

VOCAB_SIZE = 8000


# ------------------------------------------------------------
# Build shared English-German training corpus
# ------------------------------------------------------------

def build_shared_corpus():
    """
    Combine English and German training sentences into one file.
    """
    with COMBINED_PATH.open("w", encoding="utf-8") as out_file:
        with SRC_PATH.open("r", encoding="utf-8") as src_file:
            for line in src_file:
                line = line.strip()
                if line:
                    out_file.write(line + "\n")

        with TGT_PATH.open("r", encoding="utf-8") as tgt_file:
            for line in tgt_file:
                line = line.strip()
                if line:
                    out_file.write(line + "\n")

    print(f"Shared corpus saved to {COMBINED_PATH}")


# ------------------------------------------------------------
# Train SentencePiece BPE tokenizer
# ------------------------------------------------------------

def train_tokenizer():
    """
    Train a SentencePiece BPE tokenizer on the combined corpus.
    """

    spm.SentencePieceTrainer.train(
        input=str(COMBINED_PATH),
        model_prefix=str(MODEL_PREFIX),
        vocab_size=VOCAB_SIZE,

        model_type="bpe",
        
        pad_id=0,
        bos_id=1,
        eos_id=2,
        unk_id=3,

        pad_piece="<pad>",
        bos_piece="<bos>",
        eos_piece="<eos>",
        unk_piece="<unk>",

        # Useful for reproducibility / clarity
        character_coverage=1.0,
    )

    print("Tokenizer training finished.")
    print(f"Model: {MODEL_PREFIX}.model")
    print(f"Vocabulary: {MODEL_PREFIX}.vocab")


# ------------------------------------------------------------
# Simple sanity test
# ------------------------------------------------------------

def test_tokenizer():
    tokenizer = spm.SentencePieceProcessor(
        model_file=str(MODEL_PREFIX) + ".model"
    )

    sentence = "Attention is all you need."

    ids = tokenizer.encode(
        sentence,
        out_type=int,
        add_bos=True,
        add_eos=True
    )

    pieces = tokenizer.encode(
        sentence,
        out_type=str
    )

    decoded = tokenizer.decode(ids)

    print("\nSanity check:")
    print("Sentence:", sentence)
    print("Pieces:  ", pieces)
    print("IDs:     ", ids)
    print("Decoded: ", decoded)

    print("\nSpecial token IDs:")
    print("PAD:", tokenizer.pad_id())
    print("BOS:", tokenizer.bos_id())
    print("EOS:", tokenizer.eos_id())
    print("UNK:", tokenizer.unk_id())

    print("Vocabulary size:", tokenizer.vocab_size())


# ------------------------------------------------------------
# Main
# ------------------------------------------------------------

def main():
    ensure_artifact_dirs()
    if TOKENIZER_PATH.exists():
        print(f"Reusing saved tokenizer: {TOKENIZER_PATH}")
        return

    if not SRC_PATH.exists():
        raise FileNotFoundError(
            f"{SRC_PATH} does not exist. "
            "Run scripts/prepare_data.py first."
        )

    if not TGT_PATH.exists():
        raise FileNotFoundError(
            f"{TGT_PATH} does not exist. "
            "Run scripts/prepare_data.py first."
        )

    build_shared_corpus()
    train_tokenizer()
    test_tokenizer()
    shutil.copy2(MODEL_PREFIX.with_suffix(".model"), TOKENIZER_PATH)
    print(f"Persistent tokenizer: {TOKENIZER_PATH}")


if __name__ == "__main__":
    main()