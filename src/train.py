from pathlib import Path
import torch


def train_one_epoch(
    model,
    data_loader,
    criterion,
    optimizer,
    scheduler,
    device,
    pad_idx: int,
    make_src_mask,
    make_tgt_mask
):
    """
    Train the model for one epoch.

    Returns:
        average_loss: average loss over no-padding target tokens
    """

    model.train()

    total_loss = 0.0
    total_tokens = 0

    for src, tgt in data_loader:
        src = src.to(device)
        tgt = tgt.to(device)

        tgt_input = tgt[:, :-1]
        tgt_y = tgt[:, 1:]

        src_mask = make_src_mask(src, pad_idx)
        tgt_mask = make_tgt_mask(tgt_input, pad_idx)

        hidden = model(src, tgt_input, src_mask, tgt_mask)
        logits = model.generator(hidden)

        optimizer.zero_grad()

        vocab_size = logits.size(-1)
        loss = criterion(
            logits.reshape(-1, vocab_size),
            tgt_y.reshape(-1)
        )
        loss.backward()

        scheduler.step()
        optimizer.step()

        num_tokens = (tgt_y != pad_idx).sum().item()
        total_loss += loss.item() * num_tokens
        total_tokens += num_tokens

    return total_loss / total_tokens


def validate(
    model,
    data_loader,
    criterion,
    device,
    pad_idx: int,
    make_src_mask,
    make_tgt_mask
):
    """
    Evaluate the model on a validation dataset.

    No greadient computation and no parameter updates.
    """

    model.eval()

    total_loss = 0.0
    total_tokens = 0

    with torch.no_grad():
        for src, tgt in data_loader:
            src = src.to(device)
            tgt = tgt.to(device)

            tgt_input = tgt[:, :-1]
            tgt_y = tgt[:, 1:]

            src_mask = make_src_mask(src, pad_idx)
            tgt_mask = make_tgt_mask(tgt, pad_idx)

            hidden = model(src, tgt_input, src_mask, tgt_mask)
            logits = model.generator(hidden)

            vocab_size = logits.size(-1)
            loss = criterion(
                logits.reshape(-1, vocab_size),
                tgt_y.reshape(-1)
            )

            num_tokens = (tgt_y != pad_idx).sum().item()
            total_loss += loss.item() * num_tokens
            total_tokens += num_tokens

    return total_loss / total_tokens


def save_checkpoint(
    path,
    model,
    optimizer,
    scheduler,
    epoch: int,
    val_loss: float,
    config = None
):
    """
    Save all information necessary to resume training.
    """

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    checkpoint = {
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "scheduler_state_dict": scheduler.state_dict(),
        "epoch": epoch,
        "val_loss": val_loss
    }

    if config is not None:
        checkpoint["config"] = config

    torch.save(checkpoint, path)


def load_checkpoint(
    path,
    model,
    optimizer=None,
    scheduler=None,
    device="cpu"
):
    """
    Load a previously saved checkpoint.

    Returns:
        checkpoint dictionary
    """

    checkpoint = torch.load(path, map_location=device)

    model.load_state_dict(checkpoint["model_state_dict"])

    if optimizer is not None:
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])

    if scheduler is not None:
        scheduler.load_state_dict(checkpoint["scheduler_state_dict"])

    return checkpoint
