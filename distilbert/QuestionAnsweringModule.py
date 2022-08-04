import pytorch_lightning as pl
import pandas as pd
from transformers import T5Tokenizer
from QuestionAnswerDataSet import QuestionAnswerDataset
from torch.utils.data import DataLoader
from typing import Optional


class QuestionAnsweringModule(pl.LightningDataModule):

    def __init__(self, train_df: pd.DataFrame,
                 test_df: pd.DataFrame,
                 val_df: pd.DataFrame,
                 tokenizer: T5Tokenizer,
                 batch_size: int = 2,
                 source_max_len: int = 512,
                 target_max_len: int = 512):

        super().__init__()

        self.val_dataset = None
        self.train_dataset = None
        self.test_dataset = None
        self.batch_size = batch_size
        self.train_df = train_df
        self.test_df = test_df
        self.val_df = val_df
        self.tokenizer = tokenizer
        self.source_max_len = source_max_len
        self.target_max_len = target_max_len

    def setup(self, stage: Optional[str] = None):

        self.train_dataset = QuestionAnswerDataset(self.train_df,
                                  tokenizer=self.tokenizer,
                                  source_text="question",
                                  target_text="answer")

        self.test_dataset = QuestionAnswerDataset(self.test_df,
                                                 tokenizer=self.tokenizer,
                                                 source_text="question",
                                                 target_text="answer")

        self.val_dataset = QuestionAnswerDataset(self.val_df,
                                                 tokenizer=self.tokenizer,
                                                 source_text="question",
                                                 target_text="answer")

        print("Setup done")
        print("TRAIN DATASET", len(self.train_dataset))
        print("TEST DATASET", len(self.test_dataset))
        print("VAL DATASET",len(self.val_dataset))

    def train_dataloader(self):
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=12
        )

    def val_dataloader(self):
        return DataLoader(
            self.val_dataset,
            batch_size=1,
            shuffle=False,
            num_workers=12
        )

    def test_dataloader(self):
        return DataLoader(
            self.val_dataset,
            batch_size=1,
            shuffle=False,
            num_workers=12
        )
