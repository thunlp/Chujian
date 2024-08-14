from pathlib import Path
import random
from typing import Dict
import time

import torch
from torch.utils.data import Dataset
from PIL import Image
from tqdm import tqdm

from utils import load_json


class ChujianPartsDataset(Dataset):
    def parse_img_path(self, path):
        return path.replace('/data/', '/').replace('~/', '/home/jeeves/')

    def __init__(
        self,
        data_file: Path,
        label_to_id: Dict[str, int],
        transform=None,
        shuffle: bool = False,
    ):
        self.data_file = data_file
        self.transform = transform
        self.shuffle = shuffle

        # Load data
        examples = load_json(data_file)
        self.label_to_id = label_to_id
        # self.label_to_cnt = load_json(data_file.parent.parent / "part_cnts.json")
        # self.label_to_id = {label: i for i, label in enumerate(self.label_to_cnt)}
        self.num_classes = len(self.label_to_id)

        # Convert labels to IDs
        for eg in examples:
            label_idxs = [self.label_to_id[x] for x in eg["label"]]
            eg['label'] = label_idxs

        self.examples = [
            (self.parse_img_path(eg["img"]), eg['label']) for eg in examples]

        print("Computing loss weights to balance underrepresented classes")
        self.pos_weight = self.compute_loss_pos_weight()
        print(self.pos_weight)

        if shuffle:
            print("Shuffling examples")
            random.shuffle(self.examples)

        print('Converting labels to one-hot vectors')
        for i in tqdm(range(len(self.examples))):
            img, label_idxs = self.examples[i]
            label_idxs = torch.LongTensor(label_idxs)
            # (num_classes,)
            label = torch.zeros(self.num_classes).scatter_(0, label_idxs, 1)
            self.examples[i] = (img, label)

    def compute_loss_pos_weight(self):
        """
        Compute class weight for weighted loss function, to increase the
        importance of rare classes.

        The pos_weight for a label is the number of negative examples divided
        by the number of positive examples.
        """
        neg = [0] * self.num_classes
        pos = [0] * self.num_classes
        for _, labels in self.examples:
            for i in range(self.num_classes):
                if i in labels:
                    pos[i] += 1
                else:
                    neg[i] += 1
        pos_weight = [
            neg[i] / pos[i] if pos[i] > 0 else 0 for i in range(self.num_classes)
        ]
        pos_weight = torch.FloatTensor(pos_weight)
        pos_weight = torch.sqrt(pos_weight / 2)
        return pos_weight

    def __getitem__(self, idx) -> tuple:
        start_time = time.time()
        img_path, label = self.examples[idx]
        image = Image.open(img_path)
        # print('open time:', time.time() - start_time)
        if self.transform:
            image = self.transform(image)
            # print('transform:', time.time() - start_time)

        return image, label

    def __len__(self) -> int:
        return len(self.examples)
