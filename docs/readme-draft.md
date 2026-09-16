# README-draft

## Introduction

This project aims for reproducing the Transformer architecture and reexcuting scaled-down experiments based on *Attention Is All You Need* (https://papers.neurips.cc/paper/7181-attention-is-all-you-need.pdf) and *The Annotated Transformer* (https://nlp.seas.harvard.edu/annotated-transformer/). It is constructed based on GPT's guidances and is optimized by Codex. It is the first trial of an undergraduate to reproduce a paper, so I believe it is suitable for a naive to refer to and reimplement but not suitable or a sophisticated researcher.

This project has the features below:

- The Transformer architecture is implemented based on *The Annotated Transformer*, not the original architecture introduced in *Attention Is All You Need*;
- The model is trained on **IWSLT2017** dataset, not on **WMT 2014 English-to-German translation task** in the original paper, in order to reduce the computational cost and time consumption;

## Config

### clone

```bash
git clone https://github.com/juntaoxie917-a11y/transformer-reproduction.git
```

### local Computer

(conda + pip)

### Google Colab

(`requirements-colab.txt` + Google Drive)

## Run

```bash
!python -m scripts.prepare_data
...
```

## LICENSE

MIT...