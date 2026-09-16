# Running in Colab with persistent Google Drive artifacts

## Storage layout

The default persistent root is `/content/drive/MyDrive/transformer-reproduction/`:

```text
transformer-reproduction/
├── checkpoints/
│   ├── best.pt
│   └── last.pt
├── results/
│   ├── training_history.json
│   ├── metrics.json
│   └── translations.txt
└── tokenizer/
    └── bpe.model
```

Keep the repository in `/content/transformer-reproduction`. Prepared data,
SentencePiece's `train_shared.txt` and `bpe.vocab`, and its working model stay
under the runtime repository's `data/` directory. Only `bpe.model` is copied to
Drive. Runtime files must be recreated after Colab deletes the runtime.

## Colab setup and execution

Select a GPU runtime in Colab. Clone or upload this repository to
`/content/transformer-reproduction`, then run these notebook cells in order.

```python
from google.colab import drive
drive.mount('/content/drive')
```

```python
%cd /content/transformer-reproduction
%pip install -r requirements-colab.txt
```

The Colab requirements file uses the runtime's installed PyTorch and adds the
project's data, tokenizer, and evaluation dependencies. The original laptop
requirements and Conda environment files are retained.

```python
!python -m scripts.prepare_data
!python -m scripts.train_tokenizer
!python -m scripts.train_model
!python -m scripts.evaluate_model
```

Use module commands from the repository root so Python can import `src` and
`config`. Training uses CUDA when available through the existing device helper.
The scripts check that `MyDrive` exists before creating artifact directories;
mount Drive first to avoid accidentally saving into temporary runtime storage.

On a new runtime, mount Drive, restore the code, install dependencies, and
prepare data again. Tokenizer training reuses an existing Drive `bpe.model`.
Evaluation reads Drive's `best.pt` and the same saved tokenizer directly.
Keep `config.py` consistent with the checkpoint's model architecture.

Training still starts from epoch one: checkpoint persistence does not enable
automatic resume. A new training run replaces the named checkpoints and training
history, and evaluation replaces its result files. Use a separate artifact root
for separate experiments. Do not replace a tokenizer associated with existing
checkpoints, because its token IDs define the model's vocabulary.

## Local use or separate experiments

Set the environment variable before launching any scripts (or before importing
`src.paths` in a notebook):

```bash
export TRANSFORMER_ARTIFACT_ROOT="$PWD/local-artifacts"
python -m scripts.train_tokenizer
python -m scripts.train_model
```

The same `checkpoints/`, `results/`, and `tokenizer/bpe.model` layout is used.
To reuse an older laptop tokenizer, copy its `data/tokenizer/bpe.model` to the
chosen artifact root's `tokenizer/bpe.model` before training or evaluation.

## Change record

- Added `src/paths.py` as the shared source for runtime and persistent paths,
  directory creation, and an unavailable-Drive check.
- Updated data preparation, tokenizer training, model training, evaluation, and
  the manual training smoke script in `tests/test_train_model.py` to use it.
- Tokenizer training copies only the model to Drive and reuses a saved model.
- Added epoch losses and learning rates in `results/training_history.json` and
  evaluation BLEU/checkpoint metadata in `results/metrics.json`.
- Fixed evaluation's invalid `mkdir(parent=True)` call via shared initialization.
- Fixed validation's target mask to match the shifted decoder input.
- Store checkpoint dataclass configuration as a plain dictionary so new
  checkpoints do not require unpickling a custom configuration class. Older
  checkpoints with custom classes may need migration before loading with
  PyTorch's restricted checkpoint loader.
- Added `requirements-colab.txt` for the Colab setup above.

## Verification

Local syntax, path-routing, missing-Drive, and diff whitespace checks passed.
The validation/checkpoint smoke check could not run because the available local
Python environment has no PyTorch installed. Full GPU
training and actual Google Drive persistence must be verified in a mounted
Colab runtime; they were not run as part of this change.
