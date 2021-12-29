import logging
import typing as t
from pathlib import Path

import pytorch_lightning as pl
import torch
from torch import nn
from torch.nn import CrossEntropyLoss
from torch.optim.lr_scheduler import ExponentialLR
from torchmetrics.functional import f1, precision, recall
from transformers import (BertModel, BertForMaskedLM, BertConfig,
                          BertForSequenceClassification, BertForTokenClassification)

LOGGER = logging.getLogger(__name__)


class BertCentralPooler(nn.Module):

    def __init__(self, config):
        super().__init__()

        self.dense = nn.Linear(config.hidden_size, config.hidden_size)
        self.activation = nn.Tanh()
        self.central_position = config.central_position

    def forward(self, hidden_states):
        # We "pool" the model by simply taking the hidden state corresponding
        # to the first token.
        central_token_tensor = hidden_states[:, self.central_position]
        pooled_output = self.dense(central_token_tensor)
        pooled_output = self.activation(pooled_output)
        return pooled_output


class uBERTa(pl.LightningModule):
    def __init__(self, model_path: t.Optional[Path] = None,
                 config: t.Optional[BertConfig] = None,
                 is_mlm_task: bool = True, token_level: bool = True,
                 opt_kwargs: t.Optional[t.Dict[str, t.Any]] = None,
                 bin_weight: t.Optional[t.Tuple[float, float]] = None,
                 device: str = 'cuda', gamma: float = 0.5,
                 *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.is_mlm_task = is_mlm_task
        self.token_level = token_level
        self.opt_kwargs = {} if opt_kwargs is None else opt_kwargs
        self.config = BertConfig() if config is None else config

        if model_path and model_path.exists():
            self.config = self.config.from_pretrained(model_path)
            LOGGER.debug(f'Loaded config {self.config}')

        if is_mlm_task:
            self.bert = BertForMaskedLM(self.config)
        else:
            if bin_weight is not None:
                self.bert = BertModel(self.config)
                self.dropout = nn.Dropout(0.1)
                self.classifier = nn.Linear(self.config.hidden_size, 2)
            else:
                if token_level:
                    self.bert = BertForTokenClassification(self.config)
                else:
                    self.bert = BertForSequenceClassification(self.config)

        if model_path and model_path.exists():
            self.bert = self.bert.from_pretrained(model_path)

        LOGGER.debug(f'Model def\n{self.bert}')

        if is_mlm_task:
            self.f1 = lambda y_h, y: f1(y_h, y)
            self.prc = lambda y_h, y: precision(y_h, y)
            self.rec = lambda y_h, y: recall(y_h, y)
        else:
            self.f1 = lambda y_h, y: f1(y_h, y, num_classes=2, average='none')[1]
            self.prc = lambda y_h, y: precision(y_h, y, num_classes=2, average='none')[1]
            self.rec = lambda y_h, y: recall(y_h, y, num_classes=2, average='none')[1]

        self.__device = device
        self.gamma = gamma
        self._softmax = nn.Softmax(dim=1)
        self.bin_weigth = (
            torch.tensor(bin_weight, dtype=torch.float).to(device)
            if bin_weight is not None else None)
        self.bert.post_init()

    def forward(self, inp_ids, att_mask, labels, weights=None, **kwargs):
        if self.bin_weigth is None:
            return self.bert(inp_ids, attention_mask=att_mask, labels=labels, **kwargs)
        outputs = self.bert(inp_ids, attention_mask=att_mask, **kwargs)
        # Take either full seq output or pulled output
        x = outputs[0] if self.token_level else outputs[1]
        x = self.dropout(x)
        logits = self.classifier(x)
        loss_fct = CrossEntropyLoss(weight=self.bin_weigth, reduction='mean')
        loss = loss_fct(logits.view(-1, 2), labels.view(-1))
        return {'loss': loss, 'logits': logits, 'outputs': outputs}

    def generic_step(self, batch, step):
        inp_ids, att_mask, labels = batch
        output = self(inp_ids, att_mask, labels)
        if isinstance(output, dict):
            loss, logits = output['loss'], output['logits']
        else:
            loss, logits = output
        mask = labels != -100
        y_prob = self._softmax(logits[mask])
        y_true = labels[mask]

        if step != 'predict':
            self.log(f'{step}_loss', loss, prog_bar=True)
            self.log(f'{step}_f1', self.f1(y_prob, y_true), prog_bar=True)
            self.log(f'{step}_prc', self.prc(y_prob, y_true), prog_bar=True)
            self.log(f'{step}_rec', self.rec(y_prob, y_true), prog_bar=True)
            return output

        return output, y_prob, y_true

    def training_step(self, batch, idx=None):
        return self.generic_step(batch, 'train')

    def validation_step(self, batch, idx=None):
        return self.generic_step(batch, 'val')

    def test_step(self, batch, idx=None):
        return self.generic_step(batch, 'test')

    def predict_step(self, batch, idx=None, **kwargs):
        return self.generic_step(batch, 'predict')

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(self.parameters(), **self.opt_kwargs)
        # scheduler = ReduceLROnPlateau(optimizer, patience=3, mode='min', verbose=True)
        scheduler = ExponentialLR(optimizer, self.gamma)
        return {"optimizer": optimizer,
                "lr_scheduler": {
                    # "monitor": "val_loss",
                    'scheduler': scheduler,
                    'interval': 'epoch',
                    # 'frequency': 5
                }}


if __name__ == '__main__':
    raise RuntimeError
