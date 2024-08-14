from tap import Tap


class Args(Tap):
    mode: str = "train_test"
    log_interval: int = 5

    # Hyperparameters
    lr = 0.001
    lr_gamma: float = 0.8
    batch_size: int = 128
    num_epochs: int = 16

    # Data
    data_dir: str = "../data/parts_230719_k100"
    output_dir: str = "result"

    # Model
    pretrained_name: str = "vit_base_patch16_224_in21k"
