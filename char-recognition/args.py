from argparse import Namespace, ArgumentParser


def parse_args() -> Namespace:
    p = ArgumentParser()
    p.add_argument("--lr", type=float, default=0.0001)
    p.add_argument("--lr_gamma", type=float, default=0.8)
    p.add_argument("--batch_size", type=int, default=64)
    p.add_argument("--num_epochs", type=int, default=16)
    p.add_argument("--mode", default="train_test")
    p.add_argument(
        "--data_dir",
        default="../data/glyphs_k-10",
    )
    p.add_argument(
        "--output_dir",
        default="./result/glyphs_k-10",
    )
    p.add_argument("--pretrained", type=bool, default=True)
    p.add_argument(
        "--model_name",
        default="vit_base_patch16_224_in21k",
        choices=["vit_base_patch16_224_in21k", "resnet50"],
    )
    p.add_argument("--model", default="vit", choices=["vit", "resnet"])
    p.add_argument("--log_interval", type=int, default=10)
    p.add_argument("--device", type=str, default="cuda")
    return p.parse_args()
