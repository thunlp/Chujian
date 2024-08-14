from utils import load_json
from pathlib import Path
# import numpy as np


classes_path = Path('data_preprocessing/parts_230719_k20/classes.json')
classes = load_json(classes_path)
print(len(classes))
label_to_id = {label: i for i, label in enumerate(classes)}
id_to_label = {i: label for i, label in enumerate(classes)}
output_dir = Path('result/vit_base_patch16_224_in21k/parts_230719_k20/lr0.0002-gamma0.8-bs64-ep16')
preds_path = output_dir / 'ckpt_15/preds.json'
all_preds = load_json(preds_path)
print(len(all_preds))
num_classes = len(classes)

for i in range(20):
    preds = all_preds[i]
    pred_str = ''
    for j in range(num_classes):
        if preds[j] == 1:
            pred_str += id_to_label[j]
    print(i, pred_str)
