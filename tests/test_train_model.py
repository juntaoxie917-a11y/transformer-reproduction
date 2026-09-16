from src.paths import DATA_DIR, TOKENIZER_PATH, CHECKPOINT_DIR, RESULT_DIR, ensure_artifact_dirs
from dataclasses import dataclass

import torch
from torch import nn

from src.dataset import build_dataloader
from src.model import make_model
from src.masks import make_src_mask, make_tgt_mask
from src.scheduler import NoamScheduler
from src.tokenizer import Tokenizer
from src.train import train_one_epoch, validate, save_checkpoint
from src.utils import get_device


@dataclass
class Config:
    vocab_size: int = 8000

    d_model: int = 32
    d_ff: int = 128
    num_heads: int = 2
    num_layers: int = 3
    dropout: float = 0.1

    batch_size: int = 16
    warmup_steps: int = 4000
    label_smoothing: float = 0.1

    num_epochs: int = 2

    seed: int = 42


def main():
    # ------------------------------------------------------------
    # 1. Configuration
    # ------------------------------------------------------------

    config = Config()

    device = get_device()
    print(f"Using device: {device}")

    torch.manual_seed(config.seed)

    # ------------------------------------------------------------
    # 2. Paths
    # ------------------------------------------------------------

    data_dir = DATA_DIR
    tokenizer_path = TOKENIZER_PATH
    checkpoint_dir = CHECKPOINT_DIR

    ensure_artifact_dirs()

    # ------------------------------------------------------------
    # 3. Load tokenizer
    # ------------------------------------------------------------

    tokenizer = Tokenizer(str(tokenizer_path))

    print(f"Vocabulary size: {tokenizer.vocab_size}")

    # ------------------------------------------------------------
    # 4. Build DataLoaders
    # ------------------------------------------------------------

    train_loader = build_dataloader(
        src_path=str(data_dir / "train.en"),
        tgt_path=str(data_dir / "train.de"),
        tokenizer=tokenizer,
        batch_size=config.batch_size,
        shuffle=True
    )

    val_loader = build_dataloader(
        src_path=str(data_dir / "validation.en"),
        tgt_path=str(data_dir / "validation.de"),
        tokenizer=tokenizer,
        batch_size=config.batch_size,
        shuffle=False
    )

    # ------------------------------------------------------------
    # 5. Build model
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
    # 6. Loss
    # ------------------------------------------------------------

    criterion = nn.CrossEntropyLoss(
        ignore_index=tokenizer.pad_id,
        label_smoothing=config.label_smoothing
    )

    # ------------------------------------------------------------
    # 7. Optimizer
    #
    # LR is controlled by NoamScheduler.
    # ------------------------------------------------------------

    optimizer = torch.optim.Adam(
        params=model.parameters(),
        lr=0.0,
        betas=(0.9, 0.98),
        eps=1e-9
    )

    # ------------------------------------------------------------
    # 8. Scheduler
    # ------------------------------------------------------------

    scheduler = NoamScheduler(
        optimizer=optimizer,
        d_model=config.d_model,
        warmup_steps=config.warmup_steps
    )

    # ------------------------------------------------------------
    # 9. Training
    # ------------------------------------------------------------

    best_val_loss = float("inf")

    for epoch in range(config.num_epochs):
        train_loss = train_one_epoch(
            model=model,
            data_loader=train_loader,
            criterion=criterion,
            optimizer=optimizer,
            scheduler=scheduler,
            device=device,
            pad_idx=tokenizer.pad_id,
            make_src_mask=make_src_mask,
            make_tgt_mask=make_tgt_mask
        )

        val_loss = validate(
            model=model,
            data_loader=val_loader,
            criterion=criterion,
            device=device,
            pad_idx=tokenizer.pad_id,
            make_src_mask=make_src_mask,
            make_tgt_mask=make_tgt_mask
        )

        print(
            f"Epoch {epoch + 1:02d}/{config.num_epochs:02d} | "
            f"Train Loss: {train_loss:.4f} | "
            f"Val Loss: {val_loss:.4f} | "
            f"LR: {optimizer.param_groups[0]['lr']:.6e}"
        )

        # --------------------------------------------------------
        # 10. Save latest checkpoint
        # --------------------------------------------------------

        save_checkpoint(
            path=checkpoint_dir / "last.pt",
            model=model,
            optimizer=optimizer,
            scheduler=scheduler,
            epoch=epoch,
            val_loss=val_loss,
            config=config
        )

        # --------------------------------------------------------
        # 11. Save best checkpoint
        # --------------------------------------------------------

        if val_loss < best_val_loss:
            best_val_loss = val_loss

            save_checkpoint(
                path=checkpoint_dir / "best.pt",
                model=model,
                optimizer=optimizer,
                scheduler=scheduler,
                epoch=epoch,
                val_loss=val_loss,
                config=config
            )

            print(
                f"New best model saved. "
                f"Validation loss: {best_val_loss:.4f}"
            )

    print("Training finished.")


if __name__ == "__main__":
    main()
