from typing import Any, Dict, Tuple
from lightning import LightningModule
from torchmetrics import MaxMetric, MeanMetric
from torchmetrics.classification.accuracy import Accuracy
from collections import Counter
import re
import string
import torch


class SolarModule(LightningModule):
    def __init__(
        self,
        net: torch.nn.Module,
        optimizer: torch.optim.Optimizer,
        scheduler: torch.optim.lr_scheduler,
        compile: bool,
    ) -> None:
        super().__init__()

        # this line allows to access init params with 'self.hparams' attribute
        # also ensures init params will be stored in ckpt
        self.save_hyperparameters(logger=False)

        self.net = net

        # loss function
        self.criterion = MyLoss()

        # metric objects for calculating and averaging accuracy across batches
        self.train_f1 = MeanMetric()
        self.val_f1 = MeanMetric()
        self.test_f1 = MeanMetric()

        # for averaging loss across batches
        self.train_loss = MeanMetric()
        self.val_loss = MeanMetric()
        self.test_loss = MeanMetric()

        # for tracking best so far validation accuracy
        self.val_f1_best = MaxMetric()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)

    def on_train_start(self) -> None:
        self.val_loss.reset()
        self.val_f1.reset()
        self.val_f1_best.reset()

    def model_step(self, batch: Tuple[torch.Tensor, torch.Tensor]) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        x = {k: batch[k] for k in ["input_ids", "attention_mask"]}
        logits = self.forward(x)
        loss = self.criterion(logits, batch)

        start_pred = logits["start_logits"].argmax(dim=-1)
        end_pred = logits["end_logits"].argmax(dim=-1)
        preds = [
            self.net.tokenizer.decode(x[s:e], skip_special_tokens=True)
            for x, s, e in zip(batch["input_ids"].detach().cpu(), start_pred.detach().cpu(), end_pred.detach().cpu())
        ]
        return loss, preds, [answer["text"][0] for answer in batch["answers"]]

    def training_step(self, batch: Tuple[torch.Tensor, torch.Tensor], batch_idx: int) -> torch.Tensor:
        loss, preds, targets = self.model_step(batch)

        # update and log metrics
        self.train_loss(loss)
        f1 = f1_score(preds, targets)
        self.train_f1(f1)
        self.log("train/loss", self.train_loss, on_step=True, on_epoch=True, prog_bar=True)
        self.log("train/f1", self.train_f1, on_step=True, on_epoch=True, prog_bar=True)

        # return loss or backpropagation will fail
        return loss

    def on_train_epoch_end(self) -> None:
        pass

    def validation_step(self, batch: Tuple[torch.Tensor, torch.Tensor], batch_idx: int) -> None:
        loss, preds, targets = self.model_step(batch)

        # update and log metrics
        self.val_loss(loss)
        f1 = f1_score(preds, targets)
        self.val_f1(f1)
        self.log("val/loss", self.val_loss, on_step=False, on_epoch=True, prog_bar=True)
        self.log("val/f1", self.val_f1, on_step=False, on_epoch=True, prog_bar=True)

    def on_validation_epoch_end(self) -> None:
        acc = self.val_f1.compute()
        self.val_f1_best(acc)
        self.log("val/f1_best", self.val_f1_best.compute(), sync_dist=True, prog_bar=True)

    def test_step(self, batch: Tuple[torch.Tensor, torch.Tensor], batch_idx: int) -> None:
        loss, preds, targets = self.model_step(batch)

        # update and log metrics
        self.test_loss(loss)
        self.test_f1(preds, targets)
        self.log("test/loss", self.test_loss, on_step=False, on_epoch=True, prog_bar=True)
        self.log("test/f1", self.test_f1, on_step=False, on_epoch=True, prog_bar=True)

    def on_test_epoch_end(self) -> None:
        pass

    def setup(self, stage: str) -> None:
        if self.hparams.compile and stage == "fit":
            self.net = torch.compile(self.net)

    def configure_optimizers(self) -> Dict[str, Any]:
        optimizer = self.hparams.optimizer(params=self.trainer.model.parameters())
        if self.hparams.scheduler is not None:
            scheduler = self.hparams.scheduler(optimizer=optimizer)
            return {
                "optimizer": optimizer,
                "lr_scheduler": {
                    "scheduler": scheduler,
                    "monitor": "val/loss",
                    "interval": "epoch",
                    "frequency": 1,
                },
            }
        return {"optimizer": optimizer}


class MyLoss(torch.nn.Module):
    def __init__(self):
        super(MyLoss, self).__init__()
        self.criterion = torch.nn.CrossEntropyLoss()

    def forward(self, logits, batch):
        start_loss = self.criterion(logits["start_logits"], batch["start_positions"])
        end_loss = self.criterion(logits["end_logits"], batch["end_positions"])
        loss = (start_loss + end_loss) / 2
        return loss


def normalize_answer(s):
    def remove_(text):
        """불필요한 기호 제거"""
        text = re.sub("'", " ", text)
        text = re.sub('"', " ", text)
        text = re.sub("《", " ", text)
        text = re.sub("》", " ", text)
        text = re.sub("<", " ", text)
        text = re.sub(">", " ", text)
        text = re.sub("〈", " ", text)
        text = re.sub("〉", " ", text)
        text = re.sub("\(", " ", text)
        text = re.sub("\)", " ", text)
        text = re.sub("‘", " ", text)
        text = re.sub("’", " ", text)
        return text

    def white_space_fix(text):
        """연속된 공백일 경우 하나의 공백으로 대체"""
        return " ".join(text.split())

    def remove_punc(text):
        """구두점 제거"""
        exclude = set(string.punctuation)
        return "".join(ch for ch in text if ch not in exclude)

    def lower(text):
        """소문자 전환"""
        return text.lower()

    return white_space_fix(remove_punc(lower(remove_(s))))


def f1_score(predictions, ground_truths):
    result = []
    for prediction, ground_truth in zip(predictions, ground_truths):
        prediction_tokens = normalize_answer(prediction).split()
        ground_truth_tokens = normalize_answer(ground_truth).split()

        # 문자 단위로 f1-score를 계산 합니다.
        prediction_Char = []
        for tok in prediction_tokens:
            now = [a for a in tok]
            prediction_Char.extend(now)

        ground_truth_Char = []
        for tok in ground_truth_tokens:
            now = [a for a in tok]
            ground_truth_Char.extend(now)

        common = Counter(prediction_Char) & Counter(ground_truth_Char)
        num_same = sum(common.values())
        if num_same == 0:
            return 0

        precision = 1.0 * num_same / len(prediction_Char)
        recall = 1.0 * num_same / len(ground_truth_Char)
        f1 = (2 * precision * recall) / (precision + recall)
        result.append(f1)

    return torch.tensor(result).mean()


if __name__ == "__main__":
    _ = MNISTLitModule(None, None, None, None)
