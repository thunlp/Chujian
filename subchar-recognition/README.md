# CBS Sub-Character Component Recognition

This directory contains the code that implements the CBS (Chu Bamboo Slip) sub-character component recognition task. Given an image of a CBS character, the objective is the recognize the components that make up the character. This step is useful for handling CBS characters that do not have a modern Chinese equivalence. This task is framed as a multi-label classification task.

## Data

Before running the code, you need to download the CHUBS data from 🤗 Datasets: <https://huggingface.co/datasets/chen-yingfa/CHUBS>

## Model

For ResNet, we fine-tune the `resnet50` provided by the PyTorch library.
For ViT, we fine-tune the public model `vit_base_patch16_224_in21k` at 🤗 HuggingFace, which was pre-trained on ImageNet.

## Training

Execute the following to train either of the baseline models using default arguments.

- For training ViT: `python train_vit.py --data_dir path/to/data`
- For training ResNet: `python train_resnet.py --data_dir path/to/data`

where `path/to/data` contains the [CHUBS dataset](https://huggingface.co/datasets/chen-yingfa/CHUBS) (extracted from the `glyphs.zip` zip file).

For more options, see `args.py` or execute `python -h train_vit.py` or `python -h train_resnet.py`.
