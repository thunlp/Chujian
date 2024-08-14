from pathlib import Path
import json
from typing import Tuple

from torch import nn
from torchvision import transforms

from dataset import ChujianDataset
from trainer import Trainer
from modeling.vit import Vit
from modeling.resnet import ResNet
from utils import get_param_cnt
from args import parse_args


def load_model(
    model: str,
    model_name: str,
    img_size: Tuple[int, int],
    num_classes: int,
    pretrained: bool
) -> nn.Module:
    if model == 'vit':
        model = Vit(model_name, img_size, num_classes, pretrained=pretrained)
    elif model == 'resnet':
        model = ResNet(model_name, img_size, num_classes, pretrained=pretrained)
    else:
        raise ValueError(f"Invalid model arg: {model}")
    return model


def main():
    args = parse_args()
    print("===== args =====")
    print(json.dumps(args.__dict__, indent=4))
    print("================")

    data_dir = Path(args.data_dir)
    train_path = data_dir / "train.json"
    dev_path = data_dir / "dev.json"
    test_path = data_dir / "test.json"

    # Get number of classes
    glyph_to_count_path = data_dir / 'glyph_to_count_sorted.json'
    with open(glyph_to_count_path, 'r') as f:
        glyph_to_count = json.load(f)
    num_classes = len(glyph_to_count)

    if args.model == 'vit':
        IMG_SIZE = (224, 224)
    elif args.model == 'resnet':
        IMG_SIZE = (64, 64)
    else:
        raise ValueError(f"Invalid model arg: {args.model}")

    output_dir = Path(
        args.output_dir,
        args.model_name,
        f"lr{args.lr}-gamma{args.lr_gamma}"
        f"-bs{args.batch_size}-ep{args.num_epochs}",
    )

    train_transform = transforms.Compose(
        [
            transforms.Resize((256, 256)),
            transforms.RandomCrop(IMG_SIZE),
            transforms.ToTensor(),
            # Data augmentation
            transforms.GaussianBlur(kernel_size=3),
            transforms.RandomAdjustSharpness(sharpness_factor=4),
            # transforms.RandomInvert(),
            # transforms.RandomAutocontrast(),
            transforms.RandomGrayscale(),
            transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
        ]
    )
    test_transform = transforms.Compose(
        [
            transforms.Resize(IMG_SIZE),
            transforms.ToTensor(),
            transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
        ]
    )

    print("Loading model...", flush=True)
    model = load_model(
        args.model, args.model_name, IMG_SIZE, num_classes, args.pretrained)
    print(f"Params: {get_param_cnt(model)}")
    print("Instantiating trainer...", flush=True)
    trainer = Trainer(
        model,
        output_dir,
        batch_size=args.batch_size,
        num_epochs=args.num_epochs,
        log_interval=args.log_interval,
        lr=args.lr,
        lr_gamma=args.lr_gamma,
        device=args.device,
    )

    if "train" in args.mode:
        train_data = ChujianDataset(train_path, train_transform, True)
        dev_data = ChujianDataset(dev_path, test_transform, False)
        trainer.train(train_data, dev_data)
    if "test" in args.mode:
        test_data = ChujianDataset(test_path, test_transform, False)
        test_output_dir = output_dir / "test"
        trainer.load_best_ckpt()
        result = trainer.evaluate(test_data, test_output_dir)
        del result["preds"]
        print(result)


if __name__ == "__main__":
    main()
