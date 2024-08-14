from pathlib import Path
import json
from torch.utils.data import Dataset


def dump_json(data: list[dict], fname: Path):
    with open(fname, "w", encoding="utf8") as fout:
        json.dump(data, fout, ensure_ascii=False, indent=2)


def load_json(fname: Path) -> list[dict]:
    with open(fname, "r", encoding="utf8") as fin:
        data = json.load(fin)
    return data


LABELS = [
    "O",
    "B-名词",
    "I-名词",
    "B-动词",
    "I-动词",
    "B-连词",
    "I-连词",
    "B-形容词",
    "I-形容词",
    "B-副词",
    "I-副词",
    "B-数量词",
    "I-数量词",
    "B-语气词",
    "I-语气词",
    "B-代词",
    "I-代词",
    "B-介词",
    "I-介词",
    "B-助词",
    "I-助词",
]
LABEL2ID = {label: i for i, label in enumerate(LABELS)}


def load_examples(fname: Path) -> list[dict]:
    """
    File format:
    [	B-名词
    天	I-名词
    ]	I-名词
    生	B-动词
    民	B-名词
    而	B-连词
    成	B-动词
    大	B-名词
    命	I-名词
    𴺡	O
    (	O
    命	O
    。	O

    命	B-动词
    ...
    """
    examples: list[dict] = []
    with open(fname) as fin:
        chars = []
        labels = []
        for line in fin:
            line = line.strip()
            # print(len(line))
            if len(line) == 0:
                examples.append(
                    {
                        "input": chars,
                        "label": labels,
                    }
                )
                chars = []
                labels = []
            else:
                char, label = line.split("\t")
                chars.append(char)
                labels.append(label)
    if len(chars) > 0:
        examples.append(
            {
                "input": chars,
                "label": labels,
            }
        )
    return examples


def parse_data(examples: list[dict], use_subchars: bool = False):
    parsed = []
    for eid, eg in enumerate(examples):
        orig_chars: list[str] = eg["input"]
        orig_labels: list[str] = eg["label"]
        assert len(orig_chars) == len(orig_labels)

        chars: list[str] = []
        labels: list[str] = []
        n = len(orig_chars)
        i = 0
        while i < n:
            if i < n - 1 and orig_chars[i + 1] in "(（【[":
                # find sub-chars
                j = i + 2
                while j < n and orig_chars[j] not in ")）】]":
                    j += 1
                # if j == n:
                #     raise ValueError(f"Unmatched '（' in {eg = },    {i = },    {eg['input'][i] = }")
                if use_subchars:
                    # Get labels of sub-chars
                    label = orig_labels[i + 1 : j + 1]
                    char = orig_chars[i + 1 : j + 1]
                    orig_label = orig_labels[i]
                    if orig_label[0] == "B":
                        label[0] = orig_label
                else:
                    char = [orig_chars[i]]
                    label = [orig_labels[i]]
                # print(j, n, i)
                i = j + 1
            else:
                # no sub-chars
                char = [orig_chars[i]]
                label = [orig_labels[i]]
                i += 1

            chars += char
            labels += label

        parsed.append(
            {
                "id": eid,
                "input": chars,
                "label": labels,
            }
        )
        # print(use_subchars)
        # print(parsed[-1])
        # exit()
    return parsed


def load_data(data_dir: Path, use_subchars: bool = False):
    fname = data_dir / "chujian_ner_fc_sent2_all.tsv"
    print(f"Loading {fname}")
    examples = load_examples(fname)
    all_labels = set()
    for eg in examples:
        assert all([len(c) == 1 for c in eg["input"]])
        all_labels.update(set(eg["label"]))
    # print(sorted(all_labels))
    examples = parse_data(examples, use_subchars=use_subchars)
    return examples


class ChujianPOSTDataset(Dataset):
    def __init__(
        self, examples: list[dict], label2id: dict[str, int], char2id: dict[str, int]
    ):
        self.examples = examples
        self.label2id = label2id
        self.char2id = char2id

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, idx: int) -> dict:
        example = self.examples[idx]
        input = example["input"]
        label = example["label"]
        label_ids = [self.label2id[this_label] for this_label in label]
        input_ids = [self.char2id[c] for c in input]
        return {
            "input": input,
            "label": label,
            "input_ids": input_ids,
            "label_ids": label_ids,
        }


def extract_entities(
    sentence: list[str],
    labels: list[str],
) -> tuple[list[str], list[str]]:
    """Given an input and label in BIO format, extract all entities and the correponding format"""
    entity_labels = []
    entities = []
    entity = []
    cur_label = None
    for word, label in zip(sentence, labels):
        if label.startswith("B-"):
            if entity:
                entities.append(entity)
                entity_labels.append(cur_label)
                entity = []
            entity.append(word)
            cur_label = label[2:]
        elif label.startswith("I-"):
            entity.append(word)
        else:
            if entity:
                entities.append(entity)
                entity_labels.append(cur_label)
                entity = []
    if entity:
        entities.append(entity)
        entity_labels.append(cur_label)
    return entities, entity_labels


def verbalize_input(input: list[str]) -> str:
    sent = "".join(input)
    return f"### 句子：{sent}"


def verbalize_one(example: dict):
    input = example["input"]
    label = example["label"]
    entities, entity_labels = extract_entities(input, label)
    input_str = verbalize_input(input)
    prompt = f"{input_str}\n"
    prompt += "### 词：\n"
    for entity, label in zip(entities, entity_labels):
        entity_str = "".join(entity)
        prompt += f"- {entity_str}: {label}\n"
    return prompt


def create_prompt(demonstrations: list[dict]):
    # print("Demonstrations:", demonstrations)
    prompt = """以下是一段来自楚简（两千年前战国时代）的一句话。请抽取一句话里面的每一个单词的词性。词性包括

- 名词
- 动词
- 连词
- 形容词
- 副词
- 数量词
- 语气词
- 代词
- 介词
- 助词

例如：
"""
    examples_verbalized = [verbalize_one(demo) for demo in demonstrations]
    prompt += "\n\n".join(examples_verbalized)
    return prompt
