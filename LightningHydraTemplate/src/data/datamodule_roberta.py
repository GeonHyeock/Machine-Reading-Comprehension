from typing import Any, Dict, Optional, Tuple
from lightning import LightningDataModule
from torch.utils.data import DataLoader, Dataset
from src.data.components.dataset_roberta import MyDataset, Collate_fn
from transformers import AutoTokenizer
import pandas as pd
import os


class DataModule(LightningDataModule):
    def __init__(
        self,
        data_dir: str = "data/",
        batch_size: int = 64,
        num_workers: int = 0,
        pin_memory: bool = False,
        text_folder: str = "",
        name: str = "",
        train_context_id: list = [0, 1, 2, 3, 4, 5, 6, 7],
        valid_context_id: list = [8, 9],
        fold: str = "",
        max_length: int = 384,
        stride: int = 128,
    ) -> None:
        super().__init__()

        # this line allows to access init params with 'self.hparams' attribute
        # also ensures init params will be stored in ckpt
        self.save_hyperparameters(logger=False)

        # dataframe
        train_df = pd.read_csv(os.path.join(data_dir, text_folder, "train.csv"))
        train_idx = train_df.context_id.apply(lambda x: x % 10 in train_context_id)
        valid_idx = train_df.context_id.apply(lambda x: x % 10 in valid_context_id)
        train_df, valid_df = train_df[train_idx].reset_index(drop=True), train_df[valid_idx].reset_index(drop=True)
        test_df = pd.read_csv(os.path.join(data_dir, text_folder, "test.csv"))

        data_dir = os.path.join(data_dir, text_folder)
        # dataset
        self.tokenizer = AutoTokenizer.from_pretrained(name, padding_side="right")
        self.data_train: Dataset = MyDataset(train_df, self.tokenizer, data_dir, "train", fold, max_length, stride)
        self.data_val: Dataset = MyDataset(valid_df, self.tokenizer, data_dir, "valid", fold, max_length, stride)
        self.data_test: Dataset = MyDataset(test_df, self.tokenizer, data_dir, "test", "", max_length, stride)

        self.batch_size_per_device = batch_size

    def prepare_data(self) -> None:
        pass

    def setup(self, stage: Optional[str] = None) -> None:
        # Divide batch size by the number of devices.
        if self.trainer is not None:
            if self.hparams.batch_size % self.trainer.world_size != 0:
                raise RuntimeError(
                    f"Batch size ({self.hparams.batch_size}) is not divisible by the number of devices ({self.trainer.world_size})."
                )
            self.batch_size_per_device = self.hparams.batch_size // self.trainer.world_size

    def train_dataloader(self) -> DataLoader[Any]:
        """Create and return the train dataloader.

        :return: The train dataloader.
        """
        return DataLoader(
            dataset=self.data_train,
            batch_size=self.batch_size_per_device,
            num_workers=self.hparams.num_workers,
            pin_memory=self.hparams.pin_memory,
            shuffle=True,
            collate_fn=Collate_fn("train"),
        )

    def val_dataloader(self) -> DataLoader[Any]:
        """Create and return the validation dataloader.

        :return: The validation dataloader.
        """
        return DataLoader(
            dataset=self.data_val,
            batch_size=self.batch_size_per_device,
            num_workers=self.hparams.num_workers,
            pin_memory=self.hparams.pin_memory,
            shuffle=False,
            collate_fn=Collate_fn("valid"),
        )

    def test_dataloader(self) -> DataLoader[Any]:
        """Create and return the test dataloader.

        :return: The test dataloader.
        """
        return DataLoader(
            dataset=self.data_test,
            batch_size=self.batch_size_per_device,
            num_workers=self.hparams.num_workers,
            pin_memory=self.hparams.pin_memory,
            shuffle=False,
            collate_fn=Collate_fn("test"),
        )

    def teardown(self, stage: Optional[str] = None) -> None:
        """Lightning hook for cleaning up after `trainer.fit()`, `trainer.validate()`,
        `trainer.test()`, and `trainer.predict()`.

        :param stage: The stage being torn down. Either `"fit"`, `"validate"`, `"test"`, or `"predict"`.
            Defaults to ``None``.
        """
        pass

    def state_dict(self) -> Dict[Any, Any]:
        """Called when saving a checkpoint. Implement to generate and save the datamodule state.

        :return: A dictionary containing the datamodule state that you want to save.
        """
        return {}

    def load_state_dict(self, state_dict: Dict[str, Any]) -> None:
        """Called when loading a checkpoint. Implement to reload datamodule state given datamodule
        `state_dict()`.

        :param state_dict: The datamodule state returned by `self.state_dict()`.
        """
        pass
