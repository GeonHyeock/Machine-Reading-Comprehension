from typing import Any, Dict, Tuple
from lightning import LightningModule
from torchmetrics import MaxMetric, MeanMetric
from collections import Counter, defaultdict
from datetime import datetime
from pytz import timezone
import re
import os
import string
import torch
import numpy as np
import pandas as pd


class RobertaModule(LightningModule):
    def __init__(
        self,
        net: torch.nn.Module,
        optimizer: torch.optim.Optimizer,
        scheduler: torch.optim.lr_scheduler,
        compile: bool,
        scheduler_monitor: dict,
        train_param: list,
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

        # for averaging loss across batches
        self.train_loss = MeanMetric()
        self.val_loss = MeanMetric()

        # for tracking best so far validation accuracy
        self.val_f1_best = MaxMetric()

        self.valid_result = defaultdict(list)
        self.test_result = defaultdict(list)

        self.train_param = train_param

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)

    def on_train_start(self) -> None:
        self.val_loss.reset()
        self.val_f1.reset()
        self.val_f1_best.reset()

    def post_process(self, logit, batch, top_k=1, n_best=20, max_answer_length=40):
        best_answer = []
        start_logit = torch.softmax(logit["start_logits"].detach().cpu(), dim=1)
        end_logit = torch.softmax(logit["end_logits"].detach().cpu(), dim=1)
        for idx, zips in enumerate(
            zip(
                np.argsort(-start_logit)[:, :n_best],
                np.argsort(-end_logit)[:, :n_best],
                batch["context_position"],
                batch["input_ids"].detach().cpu(),
                [batch["id"][idx][0] for idx in range(len(batch["id"]))],
            )
        ):
            start_indexes, end_indexes, context_position, input_ids, id = zips
            answer = []
            for start_index in start_indexes:
                for end_index in end_indexes:
                    if not (
                        end_index < start_index
                        or context_position[0] > start_index
                        or context_position[1] < end_index
                        or end_index - start_index + 1 > max_answer_length
                    ):
                        answer.append(
                            {
                                "id": id,
                                "input_ids": input_ids[start_index : end_index + 1],
                                "logit_score": start_logit[idx][start_index] + end_logit[idx][end_index],
                            }
                        )
            if len(answer) < top_k:
                answer += [{"id": id, "input_ids": input_ids[0], "logit_score": -1} for _ in range(top_k - len(answer))]
            answer = sorted(answer, key=lambda x: -x["logit_score"])[:top_k]
            best_answer.append(answer)
        return best_answer

    def model_step(self, batch: Tuple[torch.Tensor, torch.Tensor]) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        x = {k: batch[k] for k in ["input_ids", "attention_mask"]}
        logits = self.forward(x)
        loss = self.criterion(logits, batch)
        preds = self.post_process(logits, batch)
        return loss, preds

    def test_model_step(self, batch: Tuple[torch.Tensor, torch.Tensor]) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        x = {k: batch[k] for k in ["input_ids", "attention_mask"]}
        logits = self.forward(x)
        preds = self.post_process(logits, batch)
        return preds

    def training_step(self, batch: Tuple[torch.Tensor, torch.Tensor], batch_idx: int) -> torch.Tensor:
        loss, preds = self.model_step(batch)
        pred_text, target = [], []
        for idx, pred in enumerate(preds):
            target += batch["answers"][idx]
            pred_text.append(self.net.tokenizer.decode(pred[0]["input_ids"], skip_special_tokens=True))

        # update and log metrics
        self.train_loss(loss)
        f1 = f1_score(pred_text, target)
        self.train_f1(f1)
        self.log("train/lr", self.trainer.optimizers[0].param_groups[0]["lr"], on_step=True, prog_bar=False)
        self.log("train/loss", self.train_loss, on_step=True, on_epoch=True, prog_bar=True)
        self.log("train/f1", self.train_f1, on_step=True, on_epoch=True, prog_bar=True)

        # return loss or backpropagation will fail
        return loss

    def on_train_epoch_end(self) -> None:
        if self.current_epoch == self.hparams.scheduler.keywords["start_epoch"] - 1:
            for name, param in self.net.model.named_parameters():
                if "embedding" in name:
                    pass
                elif any([tp in name for tp in self.train_param]):
                    param.requires_grad = True

    def validation_step(self, batch: Tuple[torch.Tensor, torch.Tensor], batch_idx: int) -> None:
        loss, preds = self.model_step(batch)
        for idx, pred in enumerate(preds):
            self.valid_result["id"] += batch["id"][idx]
            self.valid_result["targets"] += batch["answers"][idx]
            self.valid_result["logit_score"].append(float(pred[0]["logit_score"]))
            self.valid_result["answer"].append(self.net.tokenizer.decode(pred[0]["input_ids"], skip_special_tokens=True))

        # update and log metrics
        self.val_loss(loss)
        self.log("val/loss", self.val_loss, on_step=False, on_epoch=True, prog_bar=True)

    def on_validation_epoch_end(self) -> None:
        self.valid_result = pd.DataFrame(self.valid_result)
        idx = self.valid_result.groupby("id")["logit_score"].idxmax()
        self.valid_result = self.valid_result.iloc[idx]

        pred_text, target = self.valid_result["answer"].tolist(), self.valid_result["targets"].tolist()
        f1 = f1_score(pred_text, target)
        self.log("val/f1", f1, sync_dist=True, prog_bar=True)

        self.val_f1_best(f1)
        self.log("val/f1_best", self.val_f1_best.compute(), sync_dist=True, prog_bar=True)

        self.valid_result = defaultdict(list)

    def test_step(self, batch: Tuple[torch.Tensor, torch.Tensor], batch_idx: int) -> None:
        preds = self.test_model_step(batch)
        for pred in preds:
            self.test_result["id"].append(pred["id"])
            self.test_result["logit_score"].append(float(pred["logit_score"]))
            self.test_result["answer"].append(self.net.tokenizer.decode(pred.get("input_ids", 0), skip_special_tokens=True))

    def on_test_epoch_end(self) -> None:
        self.test_result = pd.DataFrame(self.test_result)
        idx = self.test_result.groupby("id")["logit_score"].idxmax()

        self.test_result = self.test_result.iloc[idx]
        now = "_".join(str(datetime.now(timezone("Asia/Seoul"))).split(".")[0].split(" "))
        self.test_result.to_csv(os.path.join("data", f"{now}_result.csv"), index=False)

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
                    "monitor": self.hparams.scheduler_monitor.monitor,
                    "interval": self.hparams.scheduler_monitor.interval,
                    "frequency": self.hparams.scheduler_monitor.frequency,
                },
            }
        return {"optimizer": optimizer}


class MyLoss(torch.nn.Module):
    def __init__(self):
        super(MyLoss, self).__init__()
        self.criterion = torch.nn.CrossEntropyLoss(ignore_index=1)

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
            result.append(0.0)
        else:
            precision = 1.0 * num_same / len(prediction_Char)
            recall = 1.0 * num_same / len(ground_truth_Char)
            f1 = (2 * precision * recall) / (precision + recall)
            result.append(f1)
    result = torch.tensor(result).mean()
    return result


if __name__ == "__main__":
    _ = MNISTLitModule(None, None, None, None)
