from pathlib import Path
import time
import json
from typing import Union

import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset


def get_metrics(labels, preds):
    assert len(labels) == len(preds)
    n = len(labels)
    true_neg = 0
    true_pos = 0
    false_neg = 0
    false_pos = 0
    for i in range(n):
        if labels[i] == preds[i] == 0:
            true_neg += 1
        if labels[i] == preds[i] == 1:
            true_pos += 1
        if labels[i] == 1 and preds[i] == 0:
            false_neg += 1
        if labels[i] == 0 and preds[i] == 1:
            false_pos += 1
    num_pos = true_pos + false_neg
    num_pos_pred = true_pos + false_pos
    recall = true_pos / num_pos if num_pos > 0 else 0
    prec = true_pos / num_pos_pred if num_pos_pred > 0 else 0
    f1 = 2 * (prec * recall) / (prec + recall) if prec + recall > 0 else 0
    return {"recall": recall, "prec": prec, "f1": f1}


class Trainer:
    def __init__(
        self,
        model: nn.Module,
        loss_fn: nn.Module,
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
        self.loss_fn = loss_fn
        self.model.to(device)

        output_dir.mkdir(exist_ok=True, parents=True)
        self.train_log_path = output_dir / "train.log"
        self.test_log_path = output_dir / "test.log"

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
        self.load_ckpt(ckpt_dir / "ckpt.pt")
        self.cur_ep = int(ckpt_dir.name.split("_")[-1])
        self.log_file = open(self.output_dir / "train.log", "a", encoding="utf8")
        self.train_log_file = self.log_file
        self.log("Resumed from:", ckpt_dir)

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
            num_workers=16,
        )
        if do_resume and self.has_ckpt():
            self.resume()
            self.evaluate(dev_data, self.output_dir / "temp")
        else:
            self.cur_ep = 0
            self.log_file = open(self.train_log_path, "w", encoding="utf8")
            self.train_log_file = self.log_file
        self.train_start_time = time.time()

        self.log("------ Training ------")
        self.log(f"  Num steps: {len(self.train_loader)}")
        self.log(f"  Num examples: {len(train_data)}")  # type: ignore
        self.log(f"  Num epochs: {self.num_epochs}")
        self.log(f"  Batch size: {self.batch_size}")
        self.log(f"  Log interval: {self.log_interval}")
        self.log(f"  Init LR: {self.lr}")

        while self.cur_ep < self.num_epochs:
            self.train_epoch(self.train_loader)
            self.validate(dev_data)
            self.cur_ep += 1
        self.log("------ Training Done ------")
        self.train_log_file.close()

    def validate(self, dev_data: Dataset):
        """Save checkpoint and evaluate on dev set"""
        dev_dir = self.output_dir / f"ckpt_{self.cur_ep}"
        dev_dir.mkdir(exist_ok=True, parents=True)
        self.save_ckpt(dev_dir / "ckpt.pt")

        result = self.evaluate(dev_data, dev_dir)
        del result["preds"]
        result_file = dev_dir / "result.json"
        json.dump(result, open(result_file, "w", encoding="utf8"), indent=4)

    def save_ckpt(self, ckpt_file: Path):
        print(f"Saving checkpoint to {ckpt_file}")
        ckpt_file.parent.mkdir(exist_ok=True, parents=True)
        torch.save(self.model.state_dict(), ckpt_file)

    def load_ckpt(self, path: Union[str, Path]):
        print(f"Loading checkpoint from {path}")
        sd = torch.load(path)
        self.model.load_state_dict(sd)

    def evaluate(
        self,
        dataset: Dataset,
        output_dir: Path,
    ):
        output_dir.mkdir(exist_ok=True, parents=True)
        eval_batch_size = 4 * self.batch_size
        loader = DataLoader(
            dataset, batch_size=eval_batch_size, shuffle=False, num_workers=16)
        self.model.eval()
        # if self.train_log_file.closed:
        self.logging_test = True
        eval_log_path = output_dir / "eval.log"
        self.log_file = open(eval_log_path, "w", encoding="utf8")
        self.log("------ Evaluating ------")
        self.log(f"Num steps: {len(loader)}")
        self.log(f"Num examples: {len(dataset)}")  # type: ignore
        self.log(f"batch_size: {eval_batch_size}")

        total_loss = 0
        all_preds = []
        all_labels = []
        with torch.no_grad():
            for step, batch in enumerate(loader):
                inputs, labels = batch
                logits = self.model(inputs.to(self.device))  # (B, C)
                loss = self.loss_fn(logits, labels.to(self.device))
                all_labels += labels.tolist()
                # topk_preds = torch.topk(logits, 10, dim=1)  # (B, k)

                # Multi-class Binary classification
                batch_preds = torch.sigmoid(logits) > 0.5  # (B, C)
                batch_preds = batch_preds.cpu().long().tolist()
                all_preds += batch_preds

                total_loss += loss.item()

                if (step + 1) % self.log_interval == 0:
                    self.log(
                        {
                            "step": step,
                            "loss": total_loss / (step + 1),
                        }
                    )

        preds_file = output_dir / "preds.json"
        json.dump(all_preds, open(preds_file, "w", encoding="utf8"), indent=4)
        # Compute top-k accuracy
        recall = 0
        prec = 0
        f1 = 0
        for pred, label in zip(all_preds, all_labels):
            metrics = get_metrics(label, pred)
            recall += metrics["recall"]
            prec += metrics["prec"]
            f1 += metrics["f1"]
        recall /= len(all_preds)
        prec /= len(all_preds)
        f1 /= len(all_preds)
        self.log({
            "recall": recall,
            "precision": prec,
            "f1": f1,
        })

        if self.logging_test:
            self.log_file.close()
        self.log_file = self.train_log_file

        return {
            "loss": total_loss / len(loader),
            "preds": all_preds,
            "recall": recall,
            "precision": prec,
            "f1": f1,
        }
