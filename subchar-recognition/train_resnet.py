from pathlib import Path
from typing import Tuple

import torch
from torch import nn
from torchvision import transforms

from modeling.resnet import ResNet
from utils import get_param_cnt, set_seed, load_json
from args import Args
from trainer import Trainer
from dataset import ChujianPartsDataset


def load_model(
    args: Args, img_size: Tuple[int, int], num_classes: int
) -> nn.Module:
    # model = SmallVit(model_name, img_size)
    # model = Vit(model_name, img_size, num_classes, pretrained=True)
    return model


def main():
    set_seed(0)
    args = Args().parse_args()
    data_dir = Path(args.data_dir)
    output_dir = Path(
        args.output_dir,
        'resnet',
        data_dir.name,
        f"lr{args.lr}-gamma{args.lr_gamma}" f"-bs{args.batch_size}-ep{args.num_epochs}",
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    args.save(str(output_dir / "args.json"))
    print(args)

    img_size = (224, 224)

    train_transform = transforms.Compose(
        [
            transforms.Resize((256, 256)),
            transforms.RandomCrop(img_size),
            transforms.ToTensor(),
            # Data augmentation
            # transforms.GaussianBlur(kernel_size=3),
            # transforms.RandomAdjustSharpness(sharpness_factor=4),
            # transforms.RandomInvert(),
            # transforms.RandomAutocontrast(),
            # transforms.RandomGrayscale(),
            transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
        ]
    )
    test_transform = transforms.Compose(
        [
            transforms.Resize(img_size),
            transforms.ToTensor(),
            transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
        ]
    )

    # Data
    data_dir = Path(args.data_dir)
    classes_path = data_dir / "classes.json"
    all_classes = load_json(classes_path)
    label_to_id = {label: i for i, label in enumerate(all_classes)}
    train_data = ChujianPartsDataset(
        data_dir / "train.json",
        label_to_id=label_to_id,
        transform=train_transform,
        shuffle=True,
    )
    dev_data = ChujianPartsDataset(
        data_dir / "dev.json",
        label_to_id=label_to_id,
        transform=test_transform,
        shuffle=False,
    )

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f'Training on {device}')
    num_classes = len(all_classes)
    print(f'# classes: {num_classes}')

    print("Loading model...", flush=True)
    model = ResNet(img_size=img_size, num_classes=num_classes)
    print(f"params: {get_param_cnt(model)}")
    model.to(device)

    # Trainer
    loss_fn = nn.BCEWithLogitsLoss(pos_weight=train_data.pos_weight.to(device))
    print("Instantiating trainer...", flush=True)
    trainer = Trainer(
        model,
        output_dir=output_dir,
        loss_fn=loss_fn,
        batch_size=args.batch_size,
        num_epochs=args.num_epochs,
        log_interval=args.log_interval,
        lr=args.lr,
        lr_gamma=args.lr_gamma,
        device=device,
    )

    set_seed(0)
    if "train" in args.mode:
        trainer.train(train_data, dev_data, do_resume=True)
    if "test" in args.mode:
        test_data = ChujianPartsDataset(
            data_dir / "test.json",
            label_to_id=label_to_id,
            transform=test_transform,
            shuffle=False,
        )
        test_output_dir = output_dir / "test"
        trainer.load_ckpt(output_dir / "ckpt_15/ckpt.pt")
        result = trainer.evaluate(test_data, test_output_dir)
        del result["preds"]
        print(result)


if __name__ == "__main__":
    main()
