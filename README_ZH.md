# 楚简

<div align="center">
    <a href="https://huggingface.co/datasets/chen-yingfa/CHUBS">🤗 Dataset</a> | 论文 (即将发布) | <a href="./README.md">英文</a>
</div>

<br>

此代码库包含了论文 [Multi-Modal Multi-Granularity Tokenizer for Chu Bamboo Slip（链接待发布）](https://arxiv.org/abs/) 的官方代码。

楚简 (英文: Chu bamboo slip, CBS) 是一种在超过 2000 年前中国春秋时期使用的文字体系。研究楚简对于理解中国古代的历史与文化具有重要意义。我们收集、处理并注释了第一个面向机器学习的大规模的楚简字符数据集，命名为 CHUBS，该数据集包含了超过10万条注释过的楚简字符。此外，我们提出了一种新颖的多模态多粒度分词器，专门用于处理楚简中大量的超出现代汉字范围的字符（即没有现代汉字对应的字符）。

## 数据

我们所有的数据集都提供在 [🤗 Datasets](https://huggingface.co/datasets/chen-yingfa/CHUBS)。

数据的来源如下：

| **英文名**            | **中文名** | **文章数量** | **条数** | **字数量** |
|----------------------------|------------------|-----------------|-------------|------------------|
| Tsinghua University Slips   | 清华简           | 50              | 1,402       | 31,468           |
| Shanghai Museum Slips       | 上博简           | 60              | 881         | 25,795           |
| Baoshan Slips               | 包山简           | 4               | 337         | 12,647           |
| Guodian Slips               | 郭店简           | 18              | 705         | 11,865           |
| Geling Slips                | 葛陵简           | 8               | 743         | 6,209            |
| Zenghouyi Slips             | 曾侯乙简         | 4               | 198         | 6,016            |
| Jiudian Slips               | 九店简           | 2               | 232         | 2,956            |
| Wangshan Slips              | 望山简           | 3               | 273         | 2,218            |
| Changtaiguan Slips          | 长台关简         | 3               | 148         | 1,504            |
| Zidanku Silk                | 子弹库帛         | 7               | 7           | 1,471            |
| Yangtianhu Slips            | 仰天湖简         | 1               | 42          | 335              |
| Wulipai Slips               | 五里牌简         | 1               | 18          | 109              |
| Xiyangpo Slips              | 夕阳坡简         | 1               | 2           | 54               |
| Ynagjiawan Slips            | 杨家湾简         | 1               | 38          | 41               |
| Caojiagang Slips            | 曹家岗简         | 1               | 7           | 34               |
| **总和**                   |                  | **164**         | **5,033**   | **102,722**      |


数据集包含以下两部分：

1. 主数据集 （CHUBS）。每个样本是一个楚简字的图片和其对应的标签。标签是图片文件所属目录的名称。
2. 一个楚简的词性标注（POS tagging）数据集。

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

此代码库包含了三组实验，每组实验的代码位于各自的目录中。

1. [字符识别](char-recognition/README.md)
2. [子字符组件识别](subchar-recognition/README.md)
3. [词性标注](pos-tagging/README.md)
