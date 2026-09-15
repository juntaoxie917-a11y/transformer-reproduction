from pathlib import Path
import torch

from config import Config
from src.dataset import build_dataloader
from src.evaluate import evaluate_bleu, save_translations
from src.masks import make_src_mask, make_tgt_mask
from src.model import Encoder, make_model
from src.tokenizer import Tokenizer
from src.train import load_checkpoint
from src.utils import get_device


def main():
    # ------------------------------------------------------------
    # 1. Configuration
    # ------------------------------------------------------------

    config = Config()
    device = get_device()

    print(f"Using device: {device}")

    # ------------------------------------------------------------
    # 2. Paths
    # ------------------------------------------------------------

    data_dir = Path(__file__).parent.parent / "data" / "processed"
    tokenizer_path = Path(__file__).parent.parent / "data" / "tokenizer" / "bpe.model"
    checkpoint_path = Path(__file__).parent.parent / "checkpoints" / "best.pt"
    result_dir = Path(__file__).parent.parent / "results"
    result_dir.mkdir(parent=True, exist_ok=True)

    # ------------------------------------------------------------
    # 3. Load tokenizer
    # ------------------------------------------------------------

    tokenizer = Tokenizer(str(tokenizer_path))

    print(f"Vocabulary size: {tokenizer.vocab_size}")

    # ------------------------------------------------------------
    # 4. Build test DataLoader
    # ------------------------------------------------------------

    test_loader = build_dataloader(
        src_path=str(data_dir / "test.en"),
        tgt_path=str(data_dir / "test.de"),
        tokenizer=tokenizer,
        batch_size=config.batch_size,
        shuffle=False,
    )

    # ------------------------------------------------------------
    # 5. Rebuild model architecture
    #
    # The architecture must be exactly the same as the model
    # used during training.
    # ------------------------------------------------------------

    model = make_model(
        src_vocab=tokenizer.vocab_size,
        tgt_vocab=tokenizer.vocab_size,
        N=config.num_layers,
        d_model=config.d_model,
        d_ff=config.d_ff,
        h=config.num_heads,
        dropout=config.dropout
    )

    model = model.to(device)

    # ------------------------------------------------------------
    # 6. Load trained parameters
    # ------------------------------------------------------------

    checkpoint = load_checkpoint(
        path=checkpoint_path,
        model=model,
        device=device
    )

    print(
        f"Loaded checkpoint from epoch "
        f"{checkpoint['epoch'] + 1}"
    )

    print(
        f"Checkpoint validation loss: "
        f"{checkpoint['val_loss']:.4f}"
    )

    # ------------------------------------------------------------
    # 7. Evaluate BLEU
    # ------------------------------------------------------------

    bleu_score, predictions, references = evaluate_bleu(
        model=model,
        dataloader=test_loader,
        tokenizer=tokenizer,
        device=device,
        make_src_mask=make_src_mask,
        make_tgt_mask=make_tgt_mask,
        max_len=100
    )

    print(f"Test BLEU: {bleu_score:.2f}")

    # ------------------------------------------------------------
    # 8. Save translations
    # ------------------------------------------------------------

    translation_path = result_dir / "translations.txt"

    save_translations(
        path=translation_path,
        predictions=predictions,
        references=references
    )

    print(
        f"Translations saved to: "
        f"{translation_path}"
    )

    print("Evaluation finished.")


if __name__ == "__main__":
    main()
