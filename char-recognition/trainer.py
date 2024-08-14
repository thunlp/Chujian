from pathlib import Path
import time
import json

import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset


class Trainer:
    def __init__(
        self,
        model: nn.Module,
        output_dir: Path,
        num_epochs: int = 2,
        batch_size: int = 4,
        lr: float = 0.005,
        lr_gamma: float = 0.7,
        log_interval: int = 10,
        device: str = "cuda",
    ):
        self.model = model
        self.output_dir = output_dir
        self.num_epochs = num_epochs
        self.batch_size = batch_size
        self.lr = lr
        self.log_interval = log_interval
        self.device = device

        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.lr)
        self.scheduler = torch.optim.lr_scheduler.StepLR(
            self.optimizer,
            step_size=1,
            gamma=lr_gamma,
        )
        self.loss_fn = nn.CrossEntropyLoss()
        self.model.to(device)

        output_dir.mkdir(exist_ok=True, parents=True)
        self.train_log_path = output_dir / "train.log"
        self.test_log_path = output_dir / "test.log"
        self.log_file = None

        # Dump training args
        train_args = {
            k: str(vars(self)[k])
            for k in [
                "batch_size",
                "num_epochs",
                "output_dir",
                "lr",
                "log_interval",
                "device",
            ]
        }
        args_file = output_dir / "train_args.json"
        json.dump(train_args, args_file.open("w", encoding='utf8'), indent=4)

    def log(self, *args, **kwargs):
        print(*args, **kwargs)
        print(*args, **kwargs, file=self.log_file)

    def train_epoch(self, train_loader: DataLoader):
        self.model.train()
        self.cur_step = 0
        self.total_loss = 0
        self.epoch_start_time = time.time()
        self.log(f"Start epoch {self.cur_ep}")
        for batch in train_loader:
            self.train_step(batch)
        self.scheduler.step()
        self.log("Epoch done")

    def train_step(self, batch: tuple):
        inputs, labels = batch
        inputs = inputs.to(self.device)
        labels = labels.to(self.device)
        # Forward pass
        logits = self.model(inputs)
        loss = self.loss_fn(logits, labels)
        self.total_loss += loss.item()

        # Backward pass
        loss.backward()
        self.optimizer.step()
        self.optimizer.zero_grad()

        self.cur_step += 1

        if self.cur_step % self.log_interval == 0:
            self.log(
                {
                    "epoch": round(
                        self.cur_ep + self.cur_step / len(self.train_loader), 3
                    ),
                    "step": self.cur_step,
                    "lr": round(self.scheduler.get_last_lr()[0], 6),
                    "loss": round(self.total_loss / self.cur_step, 4),
                    "time": round(time.time() - self.train_start_time),
                    "epoch_time": round(time.time() - self.epoch_start_time),
                },
                flush=True,
            )

    def resume(self):
        ckpt_dirs = self.get_ckpt_dirs()
        if len(ckpt_dirs) == 0:
            raise ValueError("No checkpoint found")
        ckpt_dir = ckpt_dirs[-1]
        self.load_ckpt(ckpt_dir)
        # +1 because the checkpoint is saved at the end of an epoch
        self.cur_ep = int(ckpt_dir.name.split("_")[-1]) + 1
        self.log_file = open(
            self.output_dir / "train.log", "a", encoding='utf8')
        self.train_log_file = self.log_file
        self.log(f"\n====== Resuming from {ckpt_dir} ======")
        self.train_start_time = time.time()

    def get_ckpt_dirs(self) -> list:
        return sorted(self.output_dir.glob("ckpt_*"))

    def has_ckpt(self) -> bool:
        return len(self.get_ckpt_dirs()) > 0

    def train(
        self,
        train_data: Dataset,
        dev_data: Dataset,
        do_resume: bool = True,
    ):
        self.train_loader = DataLoader(
            train_data,
            batch_size=self.batch_size,
            shuffle=True,
        )
        if do_resume and self.has_ckpt():
            self.resume()
        else:
            self.cur_ep = 0
            self.log_file = open(self.train_log_path, "w", encoding='utf8')
            self.train_log_file = self.log_file
            self.cur_ep = 0
            self.train_start_time = time.time()

        self.log("====== Training ======")
        self.log(f"  Num steps: {len(self.train_loader)}")
        self.log(f"  Num examples: {len(train_data)}")
        self.log(f"  Num epochs: {self.num_epochs}")
        self.log(f"  Batch size: {self.batch_size}")
        self.log(f"  Log interval: {self.log_interval}")

        while self.cur_ep < self.num_epochs:
            self.train_epoch(self.train_loader)
            self.save_and_validate(dev_data)
            self.cur_ep += 1
        self.log("====== Training Done ======")
        self.train_log_file.close()

    def save_and_validate(self, dev_data: Dataset):
        '''
        Save current model as a checkpoint to `ckpt_{cur_ep}` then
        evaluate on dev set.
        '''
        dev_dir = self.output_dir / f"ckpt_{self.cur_ep}"
        dev_dir.mkdir(exist_ok=True, parents=True)
        self.save_ckpt(dev_dir)

        result = self.evaluate(dev_data, dev_dir)
        del result["preds"]
        result_file = dev_dir / "result.json"
        json.dump(result, open(result_file, "w", encoding='utf8'), indent=4)

    def save_ckpt(self, ckpt_dir: Path):
        print(f"Saving checkpoint to {ckpt_dir}")
        ckpt_file = ckpt_dir / "ckpt.pt"
        optim_file = ckpt_dir / "optim.pt"
        scheduler_file = ckpt_dir / "scheduler.pt"
        ckpt_dir.mkdir(exist_ok=True, parents=True)
        torch.save(self.model.state_dict(), ckpt_file)
        torch.save(self.optimizer.state_dict(), optim_file)
        torch.save(self.scheduler.state_dict(), scheduler_file)

    def load_ckpt(self, ckpt_dir: Path):
        '''
        Load checkpoint from a checkpoint directory.
        Will set `self.model`, `self.optimizer`, `self.scheduler`.
        '''
        print(f"Loading checkpoint from {ckpt_dir}")
        ckpt_file = ckpt_dir / "ckpt.pt"
        optim_file = ckpt_dir / "optim.pt"
        scheduler_file = ckpt_dir / "scheduler.pt"
        self.model.load_state_dict(torch.load(ckpt_file))
        self.optimizer.load_state_dict(torch.load(optim_file))
        self.scheduler.load_state_dict(torch.load(scheduler_file))

    def get_best_ckpt_dir(self) -> Path:
        '''Load the best checkpoint based on loss.'''
        ckpt_dirs = self.get_ckpt_dirs()
        if len(ckpt_dirs) == 0:
            raise ValueError("No checkpoint found")
        best_ckpt_dir = ckpt_dirs[0]
        best_loss = float("inf")
        for ckpt_dir in ckpt_dirs[1:]:
            result_file = ckpt_dir / "result.json"
            result = json.load(open(result_file, "r", encoding='utf8'))
            if best_ckpt_dir is None or result["loss"] < best_loss:
                best_ckpt_dir = ckpt_dir
                best_loss = result["loss"]
        return best_ckpt_dir

    def load_best_ckpt(self):
        best_ckpt_dir = self.get_best_ckpt_dir()
        self.load_ckpt(best_ckpt_dir)

    def evaluate(
        self,
        dataset: Dataset,
        output_dir: Path,
    ):
        '''
        Perform evaluation on `self.model`.

        NOTE: During testing, make sure you first call `load_best_ckpt`
        to load the checkpoint to evaluate on.
        '''
        eval_batch_size = 4 * self.batch_size
        loader = DataLoader(dataset, batch_size=eval_batch_size, shuffle=False)
        self.model.eval()
        if self.log_file is None or self.log_file.closed:
            self.logging_test = True
            output_dir.mkdir(exist_ok=True, parents=True)
            self.test_log_path = output_dir / "test.log"
            self.test_log_file = open(self.test_log_path, 'w', encoding='utf8')
            self.log_file = self.test_log_file
        else:
            self.logging_test = False
        self.log("====== Evaluating ======")
        self.log(f"Num steps: {len(loader)}")
        self.log(f"Num examples: {len(dataset)}")
        self.log(f"batch_size: {eval_batch_size}")

        total_loss = 0
        all_preds = []
        all_labels = []
        with torch.no_grad():
            for step, batch in enumerate(loader):
                inputs, labels = batch
                logits = self.model(inputs.to(self.device))
                loss = self.loss_fn(logits, labels.to(self.device))
                all_labels += labels.tolist()
                topk_preds = torch.topk(logits, 10, dim=1)  # (B, k)
                all_preds += topk_preds.indices.tolist()

                total_loss += loss.item()

                if (step + 1) % self.log_interval == 0:
                    self.log(
                        {
                            "step": step,
                            "loss": total_loss / (step + 1),
                        }
                    )

        preds_file = output_dir / "preds.json"
        json.dump(all_preds, open(preds_file, "w", encoding='utf8'), indent=4)
        # Compute top-k accuracy
        acc = {}
        for k in [1, 3, 5, 10]:
            acc[k] = 0
            for label, preds in zip(all_labels, all_preds):
                if label in preds[:k]:
                    acc[k] += 1
            acc[k] /= len(all_labels)
        self.log(acc)
        self.log("loss", total_loss / len(loader))
        self.log("====== Evaluation Done ======")

        if self.logging_test:
            self.test_log_file.close()

        return {
            "loss": total_loss / len(loader),
            "preds": all_preds,
            "acc": acc,
        }
