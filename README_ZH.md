# 楚简

<div align="center">
    <a href="https://huggingface.co/datasets/chen-yingfa/CHUBS">🤗 Dataset</a> | 论文 (即将发布) | <a href="./README.md">英文</a>
</div>

<br>

此代码库包含了论文 [Multi-Modal Multi-Granularity Tokenizer for Chu Bamboo Slip（链接待发布）](https://arxiv.org/abs/) 的官方代码。

楚简 (英文: Chu bamboo slip, CBS) 是一种在超过 2000 年前中国春秋时期使用的文字体系。研究楚简对于理解中国古代的历史与文化具有重要意义。我们收集、处理并注释了第一个面向机器学习的大规模的楚简字符数据集，命名为 CHUBS，该数据集包含了超过10万条注释过的楚简字符。此外，我们提出了一种新颖的多模态多粒度分词器，专门用于处理楚简中大量的超出现代汉字范围的字符（即没有现代汉字对应的字符）。

## 数据

我们所有的数据集都提供在 [🤗 Datasets](https://huggingface.co/datasets/chen-yingfa/CHUBS)。

数据集包含以下两部分：

1. 主数据集 （CHUBS）。每个样本是一个楚简字的图片和其对应的标签。标签是图片文件所属目录的名称。
2. 一个楚简的词性标注 (POS tagging) 数据集。

文件结构如下：

```
- glyphs.zip
- pos-tagging-data/
    - dev_examples.json
    - dev_examples_subchars.json
    - test_examples.json
    - test_examples_subchars.json
    - train_examples.json
    - train_examples_subchars.json
```

CHUBS 数据集包含在 `glyphs.zip` 文件中，而后者位于目录 `pos-tagging-data` 中。带有 `_subchars` 后缀的文件跟不带这个后缀的文件中的样例是一样的，但每个字符被拆分为其子字符组件（详细信息见论文）。

## 实验

此代码库包含了三组实验，每组实验位于各自的目录中。

1. [字符识别](char-recognition/README.md)
2. [子字符组件识别](subchar-recognition/README.md)
3. [词性标注](pos-tagging/README.md)