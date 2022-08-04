import pandas as pd
from transformers import DistilBertTokenizer
from torch.utils.data import Dataset


class QuestionAnswerDataset(Dataset):

    def __init__(self, data: pd.DataFrame,
                 tokenizer: DistilBertTokenizer,
                 source_max_len: int = 384,
                 target_max_len: int = 384,
                 target_text: str = "answer",
                 source_text: str = "question"):

        super().__init__()
        self.tokenizer = tokenizer
        self.data = data
        self.source_max_len = source_max_len
        self.target_max_len = target_max_len
        self.target_text = self.data[target_text]
        self.source_text = self.data[source_text]

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index: int):
        try:
            source_text = str(self.source_text[index])
            target_text = str(self.target_text[index])

            # cleaning data so as to ensure data is in string type
            source_text = " ".join(source_text.split())
            target_text = " ".join(target_text.split())

            source_encoding = self.tokenizer(
                source_text,
                max_length=self.source_max_len,
                padding="max_length",
                truncation="only_second",
                return_attention_mask=True,
                add_special_tokens=True,
                return_tensors="pt"
            )

            target_encoding = self.tokenizer(
                target_text,
                max_length=self.target_max_len,
                padding="max_length",
                truncation="only_second",
                return_attention_mask=True,
                add_special_tokens=True,
                return_tensors="pt"
            )

            labels = target_encoding["input_ids"]
            labels[labels == 0] = -100
            #labels[labels == self.tokenizer.pad_token_id] = -100

            return dict(
                question = source_text,
                answer = target_text,
                input_ids= source_encoding["input_ids"].flatten(),
                attention_mask= source_encoding["attention_mask"].flatten(),
                labels=labels.flatten()
            )
        except Exception as e:
            print("Exception:", e)
            return

