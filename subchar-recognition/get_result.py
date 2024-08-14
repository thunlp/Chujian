from utils import load_json
from pathlib import Path


output_dir = Path(
    # "E:\\donny\\code\\research\chujian\chujian-part-recognition\\result\\old\\vit_base_patch16_224_in21k\\lr0.0001-gamma0.8-bs64-ep16\\ckpt_5"
    "E:\\donny\\code\\research\chujian\chujian-part-recognition\\result\\vit_base_patch16_224_in21k\\lr0.0001-gamma0.8-bs64-ep16\\ckpt_10"
)
all_pred_ids = load_json(output_dir / "preds.json")

label_to_cnt = load_json("data_preprocessing/label_to_cnt.json")
label_to_id = {label: i for i, label in enumerate(label_to_cnt)}
id_to_label = {i: label for label, i in label_to_id.items()}
preds = []
for pred_ids in all_pred_ids:
    this_preds = [id_to_label[i] for i, pred_id in enumerate(pred_ids) if pred_id == 1]
    preds.append(this_preds)

for i, pred in enumerate(preds[500:]):
    print(i, pred)
