# Chujian 楚简

<div align="center">
    <a href="https://huggingface.co/datasets/chen-yingfa/CHUBS">Dataset</a> | Paper (Upcoming)
</div>

<br>

This repository contains the official code for the paper [(Upcoming link)](https://arxiv.org/abs/). 

Chu bamboo slips (CBS, Chinese: 楚简, pronounced as *chujian*) is a script used during the Spring and Autumn period of Ancient China roughly 2,000 years ago. The study of which hold great significant value for understanding the history and culture of Ancient China. We scraped, processed, annotated, and then released the first large-scale dataset of CBS characters, named CHUBS, with over 100K annotated CBS characters. Additionally, we propose a novel multi-model multi-granularity tokenizer tailored for handling the large number of out-of-vocabulary characters in CBS (characters that have no modern Chinese equivalence).

## Data

All our datasets are provided at <https://huggingface.co/datasets/chen-yingfa/CHUBS>

It contains the following two parts.

1. The main dataset (CHUBS)
2. A small part-of-speech (POS) tagging dataset of CBS text.

And, the file structure is as follows.

```
- glyphs.zip
- pos-tagging-data/
    - dev_examples.json
    - dev_examples_subchars.json
    - test_examples.json
    - test_exampels_subchars.json
    - train_examples.json
    - train_examples_subchars.json
```

CHUBS is contained within the `glyphs.zip` file while the latter is under the directory `pos-tagging-data`. The files with the `_subchars` suffix is the same data examples as the original file but each character is split into its sub-character components (see the paper for more details).

## Experiments

This repo contains three sets of experiments, each under its own directory.

1. [Character Recognition](char-recognition/README.md)
2. [Sub-Character Component Recognition](subchar-recognition/README.md)
3. [POS Tagging](pos-tagging/README.md)

