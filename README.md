# Transformer Reproduction

A readable, from-scratch PyTorch implementation of the encoder-decoder Transformer, together with a scaled-down English-to-German translation experiment inspired by [*Attention Is All You Need*](https://papers.neurips.cc/paper/7181-attention-is-all-you-need.pdf) and [*The Annotated Transformer*](https://nlp.seas.harvard.edu/annotated-transformer/).

This is an educational reproduction project rather than an exact reproduction of the paper's headline results. It was created as a first paper-reproduction exercise by an undergraduate, with guidance from GPT and later code and workflow improvements assisted by Codex. The code favors clarity and traceability over production-level performance, making it most useful to learners who want to follow a complete Transformer training pipeline.

## What this project includes

- A PyTorch implementation of scaled dot-product attention, multi-head attention, sinusoidal positional encoding, encoder and decoder stacks, residual connections, layer normalization, and position-wise feed-forward networks.
- Dynamic padding and causal/padding masks for variable-length translation batches.
- A shared 8,000-token SentencePiece BPE tokenizer for English and German.
- The Adam optimizer settings, label smoothing, Xavier initialization, and Noam learning-rate schedule used by the Transformer training recipe.
- Training and validation loops with persistent best/latest checkpoints and JSON training history.
- Greedy autoregressive decoding and corpus BLEU evaluation with SacreBLEU.
- Local execution on CUDA, Apple Silicon MPS, or CPU, plus a Google Colab workflow with persistent Google Drive artifacts.

## How it differs from the original paper

The project reproduces the main architecture and training ideas, but not the original experiment at full scale:

| Aspect | Original paper | This project |
|---|---|---|
| Main task | WMT 2014 English-German | IWSLT 2017 English-German |
| Model size | Base/big configurations | `d_model=256`, 3 layers, 4 heads |
| Tokenization | WMT-era subword pipeline | Shared SentencePiece BPE |
| Evaluation decoding | Beam search | Greedy decoding |
| Primary goal | State-of-the-art translation | Understandable, lower-cost reproduction |

Consequently, this repository should not be used to compare its BLEU score directly with the paper's WMT results.

## Pipeline

```text
IWSLT 2017
    |
    v
prepare_data.py --> aligned train/validation/test text files
    |
    v
train_tokenizer.py --> shared SentencePiece BPE model
    |
    v
train_model.py --> best.pt, last.pt, training_history.json
    |
    v
evaluate_model.py --> metrics.json, translations.txt
```

## Requirements

- Python 3.11 is the environment used by this repository.
- Internet access is required when downloading IWSLT 2017 and installing dependencies.
- A CUDA GPU is recommended for training. Apple Silicon MPS and CPU are detected automatically, but CPU training may be slow.
- The prepared dataset requires additional local or Colab runtime storage. Generated data and experiment artifacts are intentionally not committed to Git.

## Installation

Clone the repository and enter it:

```bash
git clone https://github.com/juntaoxie917-a11y/transformer-reproduction.git
cd transformer-reproduction
```

### Local setup with Conda

The supplied environment file creates the `transformer-repro` environment and installs the pinned Python dependencies:

```bash
conda env create -f environment.yml
conda activate transformer-repro
```

Alternatively, create your own Python 3.11 environment and install the pip requirements:

```bash
python -m venv .venv
source .venv/bin/activate        # Windows PowerShell: .venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

The full requirements are pinned to the versions used during development. If a pinned PyTorch build is unavailable for your platform, install an appropriate PyTorch build for that platform first and then install the remaining dependencies.

### Google Colab setup

Select a GPU runtime, clone or upload the repository to `/content/transformer-reproduction`, and mount Google Drive:

```python
from google.colab import drive
drive.mount("/content/drive")
```

Then install the lightweight Colab requirements. This file intentionally uses Colab's preinstalled CUDA-enabled PyTorch:

```python
%cd /content/transformer-reproduction
%pip install -r requirements-colab.txt
```

See [Running in Colab with persistent Google Drive artifacts](docs/chat-records/colab-drive.md) for runtime restoration, tokenizer reuse, and storage details.

## Running the experiment

Run all commands from the repository root. The order matters because each stage consumes files produced by the previous one.

### 1. Select an artifact location

For a local run, set a writable artifact root before launching any Python command:

```bash
export TRANSFORMER_ARTIFACT_ROOT="$PWD/local-artifacts"
```

On Windows PowerShell, use:

```powershell
$env:TRANSFORMER_ARTIFACT_ROOT = "$PWD/local-artifacts"
```

In Colab, leave this variable unset. The default is `/content/drive/MyDrive/transformer-reproduction`, and Google Drive must already be mounted. The variable is read when `src.paths` is imported, so set it before running or importing project modules.

### 2. Prepare IWSLT 2017

```bash
python -m scripts.prepare_data
```

This downloads the IWSLT 2017 English-German Parquet splits from Hugging Face and writes aligned files to:

```text
data/processed/
├── train.en
├── train.de
├── validation.en
├── validation.de
├── test.en
└── test.de
```

### 3. Train the tokenizer

```bash
python -m scripts.train_tokenizer
```

This combines the English and German training text, trains an 8,000-token shared BPE vocabulary, and saves the persistent tokenizer as `<artifact-root>/tokenizer/bpe.model`. If that file already exists, the script reuses it rather than replacing it.

### 4. Train the Transformer

```bash
python -m scripts.train_model
```

The script trains for the number of epochs specified in `config.py`, records epoch losses and learning rates, saves the most recent checkpoint, and updates the best checkpoint whenever validation loss improves.

Training does not currently resume automatically. Starting this command again begins a new epoch loop and overwrites the named history/checkpoint files in the selected artifact root.

### 5. Evaluate the best checkpoint

```bash
python -m scripts.evaluate_model
```

Evaluation rebuilds the model from `config.py`, loads `best.pt`, generates translations greedily, and calculates corpus BLEU. Keep the model architecture settings and tokenizer unchanged between training and evaluation.

> **Current evaluation caveat:** this checkout contains two known typographical defects: `src/decode.py` calls `model.generatedor` instead of `model.generator`, and `src/evaluate.py` passes `parent=True` instead of `parents=True` to `Path.mkdir()`. Evaluation will require those corrections. They are documented in more detail in the project code guide linked below.

In a Colab notebook, prefix shell commands with `!`:

```python
!python -m scripts.prepare_data
!python -m scripts.train_tokenizer
!python -m scripts.train_model
!python -m scripts.evaluate_model
```

## Configuration

The main experiment settings live in `config.py`:

| Setting | Default | Meaning |
|---|---:|---|
| `vocab_size` | 8000 | Intended tokenizer vocabulary size |
| `d_model` | 256 | Embedding and hidden-state width |
| `d_ff` | 1024 | Feed-forward inner width |
| `num_heads` | 4 | Attention heads |
| `num_layers` | 3 | Encoder layers and decoder layers |
| `dropout` | 0.1 | Dropout probability |
| `batch_size` | 64 | Sentence pairs per batch |
| `warmup_steps` | 4000 | Linear warmup duration for the Noam schedule |
| `label_smoothing` | 0.1 | Cross-entropy label smoothing |
| `num_epochs` | 20 | Training epochs |
| `seed` | 42 | PyTorch random seed |

The actual model vocabulary dimensions come from the loaded tokenizer. If you change the tokenizer vocabulary size, retrain both tokenizer and model. If you change `d_model`, `d_ff`, `num_heads`, or `num_layers`, old checkpoints will no longer match the rebuilt architecture.

## Output files

With `TRANSFORMER_ARTIFACT_ROOT=/path/to/artifacts`, the persistent outputs are:

```text
/path/to/artifacts/
├── checkpoints/
│   ├── best.pt                   # lowest validation loss
│   └── last.pt                   # most recently completed epoch
├── results/
│   ├── training_history.json     # epoch, train/validation loss, learning rate
│   ├── metrics.json              # test BLEU and checkpoint metadata
│   └── translations.txt          # predictions and references
└── tokenizer/
    └── bpe.model                 # tokenizer required by training/evaluation
```

Tokenizer work files remain under `data/tokenizer/`. Prepared datasets and these work files are disposable runtime artifacts; the persistent `bpe.model` must stay paired with its checkpoints because token IDs define the model's embedding and output rows.

## Project structure

```text
config.py                  experiment hyperparameters
scripts/prepare_data.py    download, filter, and export IWSLT17
scripts/train_tokenizer.py train/reuse the shared SentencePiece tokenizer
scripts/train_model.py     assemble and run the training pipeline
scripts/evaluate_model.py  load the best checkpoint and evaluate BLEU
src/model.py               Transformer components and model factory
src/dataset.py             parallel-text dataset, tokenization, batch padding
src/masks.py               source-padding and causal target masks
src/tokenizer.py           SentencePiece wrapper
src/scheduler.py           Noam learning-rate schedule
src/train.py               epoch loops and checkpoint serialization
src/decode.py              greedy autoregressive decoding
src/evaluate.py            BLEU calculation and translation export
src/paths.py               data and persistent artifact paths
src/utils.py               CUDA/MPS/CPU device selection
tests/test_train_model.py  manually executable small-model training smoke script
notebooks/inspect.ipynb    exploratory inspection of the IWSLT17 source data
```

For class-by-class behavior, tensor shapes, module dependencies, and current implementation caveats, see the [project structure and code guide](docs/project-structure.md).

## Further reading

- To learn how the Transformer and experiment pipeline are implemented, including the role of each file and how the modules interact, read the [project structure and code guide](docs/project-structure.md).
- To follow the project's development process, design discussions, and AI-assisted problem-solving history, browse the [chat records with GPT and Codex](docs/chat-records/).

## Reproducibility notes and limitations

- The PyTorch seed is set, but complete bit-for-bit determinism across hardware and backends is not guaranteed.
- Batches are formed by sentence count rather than a fixed token budget. Long examples can therefore use substantially more memory.
- Data files are loaded into memory by `TranslationDataset`.
- Decoding is greedy and recomputes the model for each output step; beam search and key/value caching are not implemented.
- Checkpoints contain model, optimizer, scheduler, epoch, validation loss, and configuration state, but there is no command-line resume workflow yet.
- `tests/test_train_model.py` is a manual smoke script, not an automated pytest suite, and it uses the full prepared dataset.
- This repository is intended for learning and experimentation, not production translation or research benchmarking.

## References and acknowledgements

- Vaswani et al., [*Attention Is All You Need*](https://papers.neurips.cc/paper/7181-attention-is-all-you-need.pdf), NeurIPS 2017.
- Rush, [*The Annotated Transformer*](https://nlp.seas.harvard.edu/annotated-transformer/), Harvard NLP.
- [IWSLT 2017 dataset](https://huggingface.co/datasets/IWSLT/iwslt2017) distributed through Hugging Face Datasets.
- [SentencePiece](https://github.com/google/sentencepiece) for subword tokenization.
- [SacreBLEU](https://github.com/mjpost/sacrebleu) for corpus BLEU evaluation.

The implementation follows the presentation and code organization of *The Annotated Transformer* more closely than the original paper's implementation. AI assistance was used during learning, implementation guidance, code review, and project refinement; the repository remains an educational work whose behavior should be verified before reuse.

## License

This project is released under the [MIT License](LICENSE). Copyright © 2026 Juntao Xie.
