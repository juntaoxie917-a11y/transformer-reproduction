# Project structure and code guide

This document describes the repository as it is currently implemented. The project is an English-to-German neural machine translation pipeline built around a from-scratch PyTorch implementation of the encoder-decoder Transformer from *Attention Is All You Need*. It prepares IWSLT 2017 data, trains a shared SentencePiece vocabulary, trains the model, and evaluates a saved checkpoint with corpus BLEU.

## End-to-end structure

```text
IWSLT 2017 on Hugging Face
        |
        v
scripts/prepare_data.py
        |
        +--> data/processed/{train,validation,test}.{en,de}
                         |
                         v
scripts/train_tokenizer.py --> tokenizer/bpe.model
                         |              |
                         +--------------+
                                        v
scripts/train_model.py --> src/dataset.py --> src/model.py
          |                    |                |
          |                    +--> src/masks.py|
          +--> src/scheduler.py                 |
          +--> src/train.py <-------------------+
          |
          +--> checkpoints/{last,best}.pt
          +--> results/training_history.json
                         |
                         v
scripts/evaluate_model.py --> src/evaluate.py --> src/decode.py
          |                                      |
          +--------------------------------------+--> results/metrics.json
                                                   results/translations.txt
```

The files under `scripts/` are entry points and orchestration code. The reusable data, model, optimization, decoding, and evaluation logic is under `src/`. `config.py` supplies the main experiment settings, while `src/paths.py` gives every entry point a consistent storage layout.

## Repository tree

```text
transformer-reproduction/
├── config.py
├── environment.yml
├── requirements.txt
├── requirements-colab.txt
├── LICENSE
├── README.md
├── data/
│   ├── README.md
│   ├── processed/                 # generated parallel-text splits
│   └── tokenizer/                 # generated tokenizer work files
├── docs/
│   ├── colab-drive.md
│   └── project-structure.md       # this document
├── notebooks/
│   └── inspect.ipynb
├── scripts/
│   ├── prepare_data.py
│   ├── train_tokenizer.py
│   ├── train_model.py
│   └── evaluate_model.py
├── src/
│   ├── dataset.py
│   ├── decode.py
│   ├── evaluate.py
│   ├── masks.py
│   ├── model.py
│   ├── paths.py
│   ├── scheduler.py
│   ├── tokenizer.py
│   ├── train.py
│   └── utils.py
├── tests/
│   └── test_train_model.py
├── checkpoints/                  # legacy/local placeholder
└── results/                      # legacy/local placeholder
```

Generated data, tokenizers, Python bytecode, editor settings, and type-checker caches are excluded by `.gitignore`. The tracked `.gitkeep` files retain the otherwise empty top-level `checkpoints/` and `results/` directories, but current scripts write artifacts through `src/paths.py`, not necessarily to those directories.

## Configuration and path management

### `config.py`

This file contains two dataclasses:

- `Config` is used by `scripts/train_model.py` and `scripts/evaluate_model.py`. It defines vocabulary size, Transformer dimensions, dropout, batch size, Noam warmup length, label smoothing, epoch count, and random seed. The current main model uses `d_model=256`, `d_ff=1024`, four heads, and three encoder and decoder layers.
- `ConfigTest` is a smaller configuration intended for lightweight experiments. It is currently not imported anywhere; `tests/test_train_model.py` defines a separate local test configuration instead.

`Config.vocab_size` records the intended vocabulary size, but model construction uses `tokenizer.vocab_size` so that the actual SentencePiece model is authoritative.

### `src/paths.py`

This is the shared path registry.

- `PROJECT_ROOT` resolves the repository root from the location of `src/paths.py`.
- `DATA_DIR` points to `data/processed/`.
- `TOKENIZER_WORK_DIR` points to `data/tokenizer/` for the combined training corpus and intermediate SentencePiece outputs.
- `ARTIFACT_ROOT` comes from `TRANSFORMER_ARTIFACT_ROOT` when set. Otherwise it defaults to `/content/drive/MyDrive/transformer-reproduction` for Colab persistence.
- `CHECKPOINT_DIR`, `RESULT_DIR`, and `TOKENIZER_PATH` are derived from the artifact root.
- `ensure_artifact_dirs()` creates the checkpoint, result, and persistent-tokenizer directories. If the default Google Drive location is selected but Drive is not mounted, it fails before writing.

All four executable scripts import paths from this module. For local execution, set `TRANSFORMER_ARTIFACT_ROOT` before Python imports `src.paths`.

## Data preparation and tokenization

### `scripts/prepare_data.py`

This is the first pipeline stage. At import time it defines the IWSLT 2017 English-German Parquet URLs and loads all three splits with Hugging Face `datasets.load_dataset()`.

- `is_valid_pair(src, tgt)` rejects missing, empty, or whitespace-only source/target pairs.
- `save_split(split_name)` iterates through a dataset split, reads `example["translation"]["en"]` and `["de"]`, strips whitespace, and writes aligned sentences to separate UTF-8 files.
- `main()` saves `train`, `validation`, and `test`.

Its output contract is six files under `data/processed/`: one English and one German file for each split. Alignment is positional: line *n* in an `.en` file corresponds to line *n* in the matching `.de` file.

### `scripts/train_tokenizer.py`

This stage trains one shared subword vocabulary for both languages.

- `build_shared_corpus()` concatenates non-empty lines from `train.en` and `train.de` into `data/tokenizer/train_shared.txt`.
- `train_tokenizer()` calls SentencePiece with BPE, an 8,000-token vocabulary, and fixed special IDs: padding 0, beginning-of-sequence 1, end-of-sequence 2, and unknown 3. It creates the working files `bpe.model` and `bpe.vocab`.
- `test_tokenizer()` prints token pieces, IDs, decoded text, special IDs, and vocabulary size for a sample sentence.
- `main()` creates artifact directories, reuses an existing persistent `TOKENIZER_PATH` if present, checks for prepared training files, runs the three steps above, and copies the trained model to the persistent artifact root.

Only `bpe.model` is required by later stages. The `.vocab` file and shared corpus are training-time inspection artifacts.

### `src/tokenizer.py`

`Tokenizer` is the small application-facing wrapper around `SentencePieceProcessor`.

- The constructor loads a `.model` file.
- `pad_id`, `bos_id`, `eos_id`, `unk_id`, and `vocab_size` expose tokenizer metadata.
- `encode()` converts text into integer token IDs and optionally inserts BOS and EOS.
- `decode()` converts token IDs back to text.

`src/dataset.py` uses `encode()`, while `src/evaluate.py` uses `decode()`. The special-token properties are also used for padding, masking, loss exclusion, and autoregressive stopping.

### `src/dataset.py`

This module turns the aligned text files into padded PyTorch batches.

- `TranslationDataset` loads both entire text files into memory and verifies that they contain the same number of lines.
- `__getitem__()` encodes the source without BOS but with EOS. It encodes the target with both BOS and EOS. This asymmetry supports teacher forcing: the decoder starts from BOS, while EOS marks completion.
- `make_collate_fn(pad_idx)` returns a closure that pads source and target sequences independently to the longest sequence in the current batch.
- `build_dataloader()` combines the dataset and collator into a `DataLoader`; training enables shuffling, whereas validation and test evaluation do not.

The returned tensors have shapes `(batch, source_length)` and `(batch, target_length)`, with lengths determined independently for each batch.

## Transformer implementation

### `src/model.py`

This file implements the Transformer hierarchy from small mathematical operations up to the complete encoder-decoder model.

#### Shared helpers and input representation

- `clones(module, N)` deep-copies a module into an `nn.ModuleList`. Deep copies ensure repeated layers have independent parameters.
- `Embedding` maps token IDs to `d_model` vectors and scales them by `sqrt(d_model)`.
- `PositionalEncoding` precomputes sinusoidal positions up to `max_len=5000`, registers them as a non-trainable buffer, adds the needed prefix to embeddings, and applies dropout.

Source and target inputs each pass through an `Embedding` followed by a separately copied `PositionalEncoding` module.

#### Attention and feed-forward computation

- `attention()` implements scaled dot-product attention: `softmax(QK^T / sqrt(d_k))V`. Masked positions receive a large negative score before softmax. It returns both the result and attention probabilities.
- `MultiHeadedAttention` owns four bias-free projections: query, key, value, and output. It splits the model dimension across heads, calls `attention()`, joins the heads, and applies the output projection. The latest attention probabilities are retained in `self.atten` for possible inspection.
- `PositionwiseFeedForward` applies `Linear(d_model, d_ff)`, ReLU, dropout, and `Linear(d_ff, d_model)` independently at every sequence position.

#### Residual layers and stacks

- `LayerNorm` implements learned feature-wise normalization.
- `SublayerConnection` implements the pre-normalization residual form `x + dropout(sublayer(norm(x)))`.
- `EncoderLayer` contains masked self-attention followed by the feed-forward network, each inside a residual connection.
- `DecoderLayer` contains causal target self-attention, encoder-decoder attention over the source memory, and the feed-forward network.
- `Encoder` applies `N` cloned encoder layers and a final normalization.
- `Decoder` applies `N` cloned decoder layers and a final normalization.

#### Complete model

- `Generator` projects decoder states from `d_model` to the target vocabulary and applies log-softmax.
- `EncoderDecoder` ties the stacks together. `encode()` produces source memory; `decode()` consumes that memory and the shifted target; `forward()` runs both operations.
- `make_model()` constructs all components, clones attention/feed-forward modules where independent weights are needed, and Xavier-initializes every parameter tensor with more than one dimension.

The model deliberately does not call `Generator` inside `EncoderDecoder.forward()`. Training and decoding receive decoder hidden states first and invoke `model.generator(...)` separately.

### `src/masks.py`

Mask tensors use Boolean `True` for visible positions.

- `subsequent_mask(size)` returns a lower-triangular `(1, size, size)` causal mask. A target position can attend only to itself and earlier positions.
- `make_src_mask(src, pad_idx)` returns `(batch, 1, source_length)`, hiding source padding keys.
- `make_tgt_mask(tgt, pad_idx)` combines a target padding mask with the causal mask, producing `(batch, target_length, target_length)`.

`MultiHeadedAttention.forward()` adds a head dimension to either form so masks broadcast across all attention heads.

## Training

### `src/scheduler.py`

`NoamScheduler` implements the learning-rate schedule from the Transformer paper:

```text
factor * d_model^(-0.5) * min(step^(-0.5), step * warmup_steps^(-1.5))
```

The rate rises linearly during warmup and then decays with the inverse square root of the step. `step()` updates every optimizer parameter group. `state_dict()` and `load_state_dict()` preserve the step number in checkpoints and reconstruct the corresponding rate.

### `src/train.py`

This module contains reusable epoch and checkpoint operations.

- `train_one_epoch()` switches to training mode, moves batches to the selected device, and shifts each target into `tgt_input = tgt[:, :-1]` and expected output `tgt_y = tgt[:, 1:]`. It builds masks, obtains decoder states, projects them through the generator, computes token loss, backpropagates, advances the Noam rate, and updates parameters. Its returned loss is weighted by the number of non-padding target tokens in each batch.
- `validate()` repeats the same target shifting and loss calculation in evaluation mode under `torch.no_grad()`, without parameter or scheduler updates.
- `save_checkpoint()` stores model, optimizer, scheduler, zero-based epoch, validation loss, and optionally a plain-dictionary copy of the configuration.
- `load_checkpoint()` restores model weights and, when supplied, optimizer and scheduler states. Evaluation supplies only the model; a future resume-training flow could supply all three.

### `scripts/train_model.py`

This is the training orchestrator:

1. Instantiate `Config`, choose a device, and seed PyTorch.
2. Initialize artifact directories and load the persistent tokenizer.
3. Build shuffled training and ordered validation loaders.
4. Construct a model whose vocabulary dimensions come from the tokenizer.
5. Create label-smoothed cross-entropy loss that ignores padding.
6. Create Adam with the paper's betas and epsilon; initialize its rate to zero because `NoamScheduler` controls it.
7. For each epoch, call `train_one_epoch()` and `validate()`.
8. Rewrite `results/training_history.json` with all completed epochs.
9. Always overwrite `checkpoints/last.pt`; overwrite `best.pt` only after an improved validation loss.

The script always starts a new epoch loop. Although checkpoints contain enough optimizer and scheduler state for resumption, automatic resume is not currently implemented.

### `src/utils.py`

`get_device()` selects CUDA first, then Apple's MPS backend, then CPU. Training and evaluation both use it, so the rest of the code remains device-agnostic.

## Decoding and evaluation

### `src/decode.py`

- `greedy_decode()` begins every target with BOS, then repeatedly builds a causal mask, runs the encoder-decoder with the target prefix, selects the highest-scoring final-position token, and appends it. Generation stops when every batch item has emitted EOS or `max_len` is reached.
- `strip_special_tokens()` removes BOS and padding and truncates at the first EOS.

The implementation recomputes both encoder and decoder computation at every generation step; it does not cache encoder memory or decoder key/value tensors. This is straightforward but slower than cached decoding.

### `src/evaluate.py`

- `strip_special_tokens()` provides the special-token cleanup actually used by this module. It duplicates the helper in `src/decode.py`; the latter is currently not imported.
- `evaluate_bleu()` builds source masks, calls `greedy_decode()` for every test batch, cleans prediction and reference IDs, decodes them to strings, and calculates SacreBLEU corpus BLEU. It returns the numeric score plus both text lists.
- `save_translations()` writes numbered prediction/reference pairs and optionally sources.

### `scripts/evaluate_model.py`

This entry point rebuilds the architecture from `Config`, loads `checkpoints/best.pt`, calls `evaluate_bleu()` over the test split with a maximum output length of 100, and writes:

- `results/translations.txt`: each predicted translation beside its reference.
- `results/metrics.json`: BLEU, one-based checkpoint epoch, and checkpoint validation loss.

The architecture settings in `config.py` must exactly match those used by the checkpoint. The tokenizer must also be the same model used during training because token IDs define embedding and output rows.

## Supporting files

### `tests/test_train_model.py`

Despite its location and name, this is a manually executable training smoke script, not a pytest test: it defines `main()` but no `test_*` function. It mirrors `scripts/train_model.py` with a smaller local `d_model=32` configuration and two epochs. It reads the full prepared train/validation data and writes to the same configured artifact locations, so running it can overwrite `last.pt` and `best.pt`.

### `notebooks/inspect.ipynb`

This exploratory notebook verifies access to the IWSLT 2017 Parquet files, displays split sizes, and prints one translation record. It informed the data-loading path but is not imported or called by the production pipeline.

### `docs/colab-drive.md`

Documents the Colab execution order, persistent Google Drive artifact layout, local override through `TRANSFORMER_ARTIFACT_ROOT`, checkpoint/tokenizer compatibility, and current persistence limitations.

### Dependency and repository metadata

- `requirements.txt` is the pinned full Python environment, including PyTorch, Hugging Face Datasets, SentencePiece, SacreBLEU, Jupyter, and analysis packages.
- `requirements-colab.txt` is intentionally smaller. It relies on Colab's CUDA-enabled PyTorch and installs only the extra runtime libraries needed by the pipeline.
- `environment.yml` reproduces the named Conda environment and its pinned pip packages.
- `.gitignore` excludes generated data, caches, bytecode, VS Code settings, and macOS metadata.
- `LICENSE` contains the repository's license terms.
- `README.md` and `data/README.md` are currently empty placeholders.
- `.vscode/`, `.mypy_cache/`, `__pycache__/`, `.DS_Store`, and generated data/tokenizer files are local development artifacts rather than application modules.

## Important tensor flow

For a batch size `B`, source length `S`, target length `T`, model width `D`, vocabulary size `V`, and number of heads `H`:

```text
src IDs                         (B, S)
  -> source embedding + position
                                (B, S, D)
  -> encoder with src_mask      (B, 1, S)
  -> memory                     (B, S, D)

tgt[:, :-1]                     (B, T-1)
  -> target embedding + position
                                (B, T-1, D)
  -> decoder self-attention with tgt_mask
                                (B, T-1, T-1)
  -> cross-attention over memory
  -> decoder states             (B, T-1, D)
  -> generator                  (B, T-1, V)
  -> compare with tgt[:, 1:]    (B, T-1)
```

Inside multi-head attention, projected tensors have shape `(B, H, length, D/H)`. Source and target padding IDs originate in the SentencePiece model, so the tokenizer is a dependency of batching, masks, loss, and decoding—not merely text preprocessing.

## Known implementation issues and sharp edges

The following are descriptions of the current code, not proposed behavior:

- `src/decode.py` calls `model.generatedor(hidden)`. The model exposes `model.generator`, so evaluation will raise `AttributeError` until this typo is corrected.
- `src/evaluate.py::save_translations()` calls `path.parent.mkdir(parent=True, exist_ok=True)`. `pathlib.Path.mkdir()` expects `parents=True`, so this can raise `TypeError`; the result directory is normally pre-created by `ensure_artifact_dirs()`, but the invalid keyword is still evaluated.
- `Generator` returns log-probabilities, while training uses `CrossEntropyLoss`, which conventionally receives unnormalized logits and applies log-softmax internally. The extra log-softmax is redundant; its numerical output is effectively already normalized, but the interface is misleading.
- Greedy decoding tracks finished sequences but continues appending tokens to an item until every item in the batch has finished. `strip_special_tokens()` truncates each result at its first EOS, so the extra tokens do not appear in final text.
- `src/model.py` imports `Config` but does not use it; `scripts/evaluate_model.py` imports `Encoder` but does not use it.
- `prepare_data.py` loads remote data at module import time rather than inside `main()`. Importing the module therefore performs network/cache work.
- The project reproduces the Transformer architecture and core training recipe on IWSLT17 En-De, but the default dimensions, layer count, dataset, and greedy evaluation are smaller/different from the original paper's full WMT experiments. Results should therefore be treated as an educational reproduction rather than a like-for-like reproduction of the paper's headline numbers.

## Execution order

Run commands from the repository root so imports of `src` and `config` resolve:

```bash
export TRANSFORMER_ARTIFACT_ROOT="$PWD/local-artifacts"  # local runs only
python -m scripts.prepare_data
python -m scripts.train_tokenizer
python -m scripts.train_model
python -m scripts.evaluate_model
```

The dependency chain is strict: prepared parallel files are needed to train the tokenizer; the tokenizer and prepared files are needed for training; and the matching tokenizer, configuration, test files, and best checkpoint are needed for evaluation.
