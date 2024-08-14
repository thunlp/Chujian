# Character Recognition

This directory implements the character recognition task for CBS (Chu Bamboo Slip) characters. Given an image of a CBS character, the objective is to classify it into one of the pre-specified set of labels. This is a standard image classification task.

## Data Processing

This task operates on the [CHUBS data](https://huggingface.co/datasets/chen-yingfa/CHUBS) (the `glyphs.zip` directory). Each example consists of an image of a CBS character and its corresponding label. The labels are the names of the directory that the image file belongs to.

Before running this code, we pre-process the data by running, for example, `python merge_classes.py -k 3 --src_dir path/to/data --dst_dir ./data`, where `--src_dir` specifies the directory containing the raw data and `--dst_dir` specifies the directory to save the processed data.

This code will perform the following.

1. Merging some similar classes. For instance, some variations of the same character are labeled differently, but for image classification, we just want to classify them into the same class.
2. Removing classes with less than $k$ examples. In the paper we used 2, 3, 10, 20.

This will generate some JSON files contained the examples used for training, validation, and testing the character recognizer into the directory specified through `--dst_dir`.

## Training

To train the ViT model, execute the following command.

```shell
python train.py --model vit --model_name vit_base_patch16_224_in21k --data_dir path/to/processed/data --device cuda
```

To train a ResNet-50, use:

```shell
python train.py --model resnet --model_name resnet50 --data_dir path/to/processed/data --device cuda
```

For more options, see the `args.py` file or execute `python -h train.py`.