import logging
import typing as t
from abc import abstractmethod
from pathlib import Path

import pytorch_lightning as pl
import torch
from torch import nn
from torch.nn import CrossEntropyLoss
from torch.optim.lr_scheduler import ExponentialLR
from torchmetrics.functional import f1, precision, recall
from transformers import (
    BertModel, BertForMaskedLM, BertConfig, BertPreTrainedModel,
    DistilBertPreTrainedModel, DistilBertModel, DistilBertConfig,
    DebertaV2PreTrainedModel, DebertaV2Config, DebertaV2Model,
    FunnelForSequenceClassification, FunnelBaseModel
)
from transformers.models.distilbert.modeling_distilbert import TransformerBlock
from transformers.models.funnel.modeling_funnel import FunnelClassificationHead
from transformers.models.bert.modeling_bert import BertLayer

LOGGER = logging.getLogger(__name__)

_T = t.TypeVar('_T')
_Output = t.Union[t.Dict, t.Tuple[_T, t.Any]]


class WeightedBertClassifier(BertPreTrainedModel):
    """
    A classifier based on the BERT model supporting class weights for the `CrossEntropyLoss`.
    """

    def __init__(self, config: BertConfig, weight=None):
        super().__init__(config)
        self.num_labels = config.num_labels
        self.weight = weight
        self.config = config

        self.bert = BertModel(config)

        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.dense = nn.Linear(config.hidden_size, 2)
        self.tanh = nn.Tanh()
        self.loss = CrossEntropyLoss(weight=self.weight)

        if self.config.use_signal:
            self.bert_last = BertLayer(config)

        self.bert.post_init()

    def forward(self, inp_ids, att_mask, labels, signal=None, **kwargs):
        outputs = self.bert(inp_ids, attention_mask=att_mask, **kwargs)
        x = outputs[0]
        if self.config.use_signal and signal is not None:
            x[:, :, -1] += signal
            extended_attention_mask: torch.Tensor = self.bert.get_extended_attention_mask(
                att_mask, inp_ids.size(), inp_ids.device)
            x = self.bert_last(x, attention_mask=extended_attention_mask)[0]
        x = self.dropout(x)
        x = self.dense(x)
        # x = self.dropout(x)
        # x = self.tanh(x)
        loss = self.loss(x.view(-1, 2), labels.view(-1))
        return {'loss': loss, 'logits': x, 'outputs': outputs}


class WeightedDistilBertClassifier(DistilBertPreTrainedModel):
    """
    A classifier based on the DistilBERT model supporting class weights for the `CrossEntropyLoss`.
    """

    def __init__(self, config, weight=None):
        super().__init__(config)
        self.num_labels = config.num_labels
        self.weight = weight
        self.config = config

        self.bert = DistilBertModel(config)
        self.bert_last = TransformerBlock(config)
        self.dropout = nn.Dropout(0.2)
        self.dense1 = nn.Linear(config.hidden_size, 2)
        self.loss = CrossEntropyLoss(weight=self.weight)

        self.bert.post_init()

    def forward(self, inp_ids, att_mask, labels, signal, **kwargs):
        outputs = self.bert(inp_ids, attention_mask=att_mask, **kwargs)
        x = outputs[0]
        if self.config.use_signal:
            if len(signal.shape) == 2:
                x[:, :, -1] += signal
            if len(signal.shape) == 3:
                last_dim = signal.shape[-1]
                x[:, :, -last_dim:] += signal
        x = self.bert_last(x, attn_mask=att_mask)[0]
        x = self.dropout(x)
        x = self.dense1(x)
        if labels is not None:
            loss = self.loss(x.view(-1, 2), labels.view(-1))
        else:
            loss = None
        return {'loss': loss, 'logits': x.detach(), 'outputs': outputs}


class WeightedDeBertaClassifier(DebertaV2PreTrainedModel):
    """
    A classifier based on the DebertaV2 model supporting class weights for the `CrossEntropyLoss`.
    """

    def __init__(self, config: DebertaV2Config, weight=None):
        super().__init__(config)
        self.num_labels = config.num_labels
        self.weight = weight
        self.config = config

        self.bert = DebertaV2Model(config)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        s = config.hidden_size
        self.dense1 = nn.Linear(s, s)
        self.dense2 = nn.Linear(s, 2)
        self.relu = nn.LeakyReLU()
        self.loss = CrossEntropyLoss(weight=self.weight, reduction='mean')

        self.bert.post_init()

    def forward(self, inp_ids, att_mask, labels, signal, **kwargs):
        outputs = self.bert(inp_ids, attention_mask=att_mask, **kwargs)
        x = outputs[0]
        if self.config.use_signal:
            x[:, :, -1] += signal
        x = self.dropout(x)
        x = self.dense1(x)
        x = self.dropout(x)
        x = self.relu(x)
        x = self.dense2(x)
        loss = self.loss(x.view(-1, 2), labels.view(-1))
        return {'loss': loss, 'logits': x.detach(), 'outputs': outputs}


class FunnelClassifier(FunnelForSequenceClassification):
    def __init__(self, config, weight):
        super().__init__(config)
        self.num_labels = config.num_labels
        self.config = config
        self.weight = weight

        self.funnel = FunnelBaseModel(config)
        self.classifier = FunnelClassificationHead(config, config.num_labels)
        # Initialize weights and apply final processing
        self.post_init()

    def forward(self, inp_ids, att_mask, labels, signal):
        outputs = self.funnel(
            inp_ids,
            attention_mask=att_mask)

        last_hidden_state = outputs[0]
        pooled_output = last_hidden_state[:, 0]
        logits = self.classifier(pooled_output)
        loss_fct = CrossEntropyLoss(weight=self.weight)
        loss = loss_fct(logits.view(-1, 2), labels.view(-1))
        return {'loss': loss, 'logits': logits, 'outputs': outputs}


_Model = t.Union[WeightedBertClassifier, WeightedDeBertaClassifier, WeightedDistilBertClassifier]
_Config = t.Union[BertConfig, DistilBertConfig, DebertaV2Config]


class uBERTa_base(pl.LightningModule):
    """
    A base class for the uBERTa models, defining common generic steps to train and evaluate the model.
    The model itself is initialized in subclasses.
    """

    def __init__(self,
                 config,
                 model_path: t.Optional[Path] = None,
                 opt_kwargs: t.Optional[t.Dict[str, t.Any]] = None,
                 scheduler_kwargs: t.Optional[t.Dict[str, t.Any]] = None,
                 gamma: float = 0.5,
                 scheduler=None,
                 scheduler_interval='step',
                 optimizer=None,
                 *args, **kwargs):
        """
        :param config: config supported by a model defined used in a subclass
        :param model_path: a path to a directory with the model
        :param opt_kwargs: keyword args to the optimizer
        :param scheduler_kwargs: keyword args to the scheduler
        :param gamma: a gamma parameter to the default `ExponentialLR` scheduler; if `None`, defaults to 0.9
        :param scheduler: an uninstantiated scheduler; if `None`, will use `ExponentialLR`
        :param scheduler_interval: either "step" or "epoch"
        :param optimizer: an uninstantiated optimizer; if `None`, will use AdamW with default params
        :param args: positional args to pass to the `LightningModule` parent class
        :param kwargs: keyword args to pass to the `LightningModule` parent class
        """
        super().__init__(*args, **kwargs)
        self.opt_kwargs = {} if opt_kwargs is None else opt_kwargs
        self.scheduler_kwargs = dict(gamma=0.9) if scheduler_kwargs is None else scheduler_kwargs
        self.config = config
        self.model_path = model_path
        self.gamma = gamma
        self.scheduler_interval = scheduler_interval
        self.scheduler = scheduler
        self.optimizer = optimizer

    @abstractmethod
    def generic_step(self, batch: t.Sequence[torch.Tensor], step: str) -> t.Any:
        """
        A generic step defines how to unpack a batch and call the forward step.
        It sends the output to the `_common_generic_step` method.
        :param batch: a sequence of tensors
        :param step: a step name, one of the ("train", "test", "val", "predict")
        """
        raise NotImplementedError

    def _common_generic_step(
            self, output: _Output, labels: torch.Tensor, step: str
    ) -> t.Union[_Output, t.Tuple[_Output, torch.Tensor, torch.Tensor]]:
        if isinstance(output, dict):
            loss, logits = output['loss'], output['logits']
        else:
            loss, logits = output
        mask = labels != -100
        y_prob = nn.Softmax(dim=1)(logits[mask])
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
        if self.optimizer is None:
            self.optimizer = torch.optim.AdamW
        if self.scheduler is None:
            self.scheduler = ExponentialLR
        optimizer = self.optimizer(self.parameters(), **self.opt_kwargs)
        scheduler = self.scheduler(optimizer, **self.scheduler_kwargs)
        return [optimizer], [{'scheduler': scheduler, 'interval': self.scheduler_interval}]


class uBERTa_mlm(uBERTa_base):
    """
    A module for the MLM task
    """

    def __init__(self, model=BertForMaskedLM, *args, **kwargs):
        """
        :param model: any transformer model suited for MLM
        :param args: passed to `uBERTa_base`
        :param kwargs: passed to `uBERTa_base`; expected to contain `config`
            with configuration for the used `model`
        """
        super().__init__(*args, **kwargs)

        self.f1 = lambda y_h, y: f1(y_h, y)
        self.prc = lambda y_h, y: precision(y_h, y)
        self.rec = lambda y_h, y: recall(y_h, y)

        if self.model_path:
            self.model = model.from_pretrained(self.model_path)
            LOGGER.debug(f'Loaded model from {self.model_path}')
        else:
            self.model = model(self.config)

        self.save_hyperparameters()

    def generic_step(self, batch: t.Sequence[torch.Tensor], step: str):
        """
        :param batch: a sequence with at least three tensors:
            input_ids, attention_mask, and labels
        :param step: a step name
        :return: a model output; for the "predict" step, also returns probabilities and labels
        """
        inp_ids, att_mask, labels = batch[:3]
        output = self(inp_ids, att_mask, labels)
        return self._common_generic_step(output, labels, step)

    def forward(self, inp_ids, att_mask, labels, **kwargs):
        return self.model(inp_ids, attention_mask=att_mask, labels=labels, **kwargs)


class uBERTa_classifier(uBERTa_base):
    """
    A classifier for token- and sentence-level classification tasks.
    """

    def __init__(self, model=_Model,
                 weight: t.Optional[t.Tuple[float, float]] = None,
                 device: str = 'cuda', *args, **kwargs):
        """
        :param model: any of the weighted models accepting experimental signal values
        :param weight: class weight
        :param device: device to place the weight tensor
        :param args: passed to `uBERTa_base`
        :param kwargs: passed to `uBERTa_base`, `config` is required
        """
        super().__init__(*args, **kwargs)
        self.f1 = lambda y_h, y: f1(y_h, y, num_classes=2, average='none')[1]
        self.prc = lambda y_h, y: precision(y_h, y, num_classes=2, average='none')[1]
        self.rec = lambda y_h, y: recall(y_h, y, num_classes=2, average='none')[1]

        self.weight = (
            torch.tensor(weight, dtype=torch.float).to(device)
            if weight is not None else None)

        if self.model_path:
            self.model = model.from_pretrained(self.model_path, self.weight)
            LOGGER.debug(f'Loaded model from {self.model_path}\n{self.model}')
        else:
            self.model = model(self.config, self.weight)

        self.model.post_init()

    def generic_step(self, batch, step):
        inp_ids, att_mask, labels, signal = batch
        output = self(inp_ids, att_mask, labels, signal)
        return self._common_generic_step(output, labels, step)

    def forward(self, inp_ids, att_mask, labels, signal, **kwargs):
        return self.model(inp_ids, att_mask, labels, signal, **kwargs)


if __name__ == '__main__':
    raise RuntimeError
