import pytorch_lightning as pl
from transformers import DistilBertModel



class QuestionAnsweringModel(pl.LightningModule):

    def __init__(self, model_name):
        super().__init__()
        self.model = DistilBertModel.from_pretrained(model_name, return_dict=True)
        self.save_hyperparameters()

    def forward(self, input_ids, attention_mask, labels=None):
        output = self.model(input_ids=input_ids,
                            attention_mask=attention_mask,
                            labels = labels)

        return output.loss, output.logits

    def training_step(self, batch, batch_idx):
        input_ids = batch["input_ids"]
        attention_mask = batch["attention_mask"]
        labels = batch["labels"]
        loss, outputs = self(input_ids, attention_mask, labels)
        self.log("train_loss", loss, prog_bar=True, logger=True, on_epoch=True, on_step=False,sync_dist=True)
        return loss

    def test_step(self, batch, batch_idx):
        input_ids = batch["input_ids"]
        attention_mask = batch["attention_mask"]
        labels = batch["labels"]
        loss, outputs = self(input_ids, attention_mask, labels)
        self.log("test_loss", loss, prog_bar=True, logger=True, on_epoch=True, on_step=False, sync_dist=True)
        return loss

    def validation_step(self, batch, batch_idx):
        input_ids = batch["input_ids"]
        attention_mask = batch["attention_mask"]
        labels = batch["labels"]
        loss, outputs = self(input_ids, attention_mask, labels)
        self.log("val_loss", loss, prog_bar=True, logger=True, on_epoch=True, on_step=False, sync_dist=True)
        return loss

    def generate_output(self, input_ids):
        return self.model.generate(
                    input_ids=input_ids,
                    max_length=512,
                    num_beams=2,
                    repetition_penalty=2.5,
                    length_penalty=1.0,
                    early_stopping=True)


    def configure_optimizers(self):
        return AdamW(self.parameters(), lr=0.0001)

