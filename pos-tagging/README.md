# POS Tagging

This directory contains the code for reproducing the POS tagging experiment in the [paper (upcoming)](). We employ in-context learning with the Openai API to perform this task.

## Data

Download the `pos-tagging-data` directory from 🤗 Datasets: [CHUBS POS Tagging](https://huggingface.co/datasets/chen-yingfa/CHUBS) and move it to the `data` directory.

## Running

To run the code, execute the following command.

```shell
python run.py --data_dir path/to/data
```

where `path/to/data` is the directory containing the `pos-tagging-data` directory.

