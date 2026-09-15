from dataclasses import dataclass


@dataclass
class Config:
    vocab_size: int = 8000

    d_model: int = 256
    d_ff: int = 1024
    num_heads: int = 4
    num_layers: int = 3
    dropout: float = 0.1

    batch_size: int = 64
    warmup_steps: int = 4000
    label_smoothing: float = 0.1

    num_epochs: int = 20

    seed: int = 42


@dataclass
class ConfigTest:
    vocab_size: int = 8000

    d_model: int = 64
    d_ff: int = 256
    num_heads: int = 2
    num_layers: int = 3
    dropout: float = 0.1

    batch_size: int = 16
    warmup_steps: int = 4000
    label_smoothing: float = 0.1

    num_epochs: int = 2

    seed: int = 42