import logging
import sys
import typing as t
from itertools import chain
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from scipy.special import softmax
from sklearn.metrics import accuracy_score, roc_auc_score, f1_score, precision_score, recall_score
from sklearn.utils.class_weight import compute_sample_weight, compute_class_weight
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler, TensorDataset
from tqdm.auto import tqdm
from transformers import AdamW, get_linear_schedule_with_warmup
from toolz import curry

from uBERTa.base import Scores, StopSetup, OptSetup, RunSetup

LOGGER = logging.getLogger(__name__)


def setup_logger(
        log_path,
        file_level: int,
        stdout_level: int,
        stderr_level: int
) -> logging.Logger:

    formatter = logging.Formatter(
        '%(asctime)s %(levelname)s [%(module)s--%(funcName)s]: %(message)s')
    logger = logging.getLogger()
    if log_path:
        logging_file = logging.FileHandler(log_path, 'w')
        logging_file.setFormatter(formatter)
        logging_file.setLevel(file_level)
        logger.addHandler(logging_file)

    logging_out = logging.StreamHandler(sys.stdout)
    logging_err = logging.StreamHandler(sys.stderr)
    logging_out.setFormatter(formatter)
    logging_err.setFormatter(formatter)
    logging_out.setLevel(stdout_level)
    logging_err.setLevel(stderr_level)
    logger.addHandler(logging_out)
    logger.addHandler(logging_err)
    logger.setLevel(logging.DEBUG)
    return logger


class EarlyStopping:
    def __init__(self, rounds: int, tolerance: float,
                 score_type: str = 'roc_auc',
                 max_score: float = 1.0):
        self.rounds = rounds
        self.tolerance = tolerance
        self.count = 0
        self.score = 0
        self.score_type = score_type
        self.max_score = max_score

    def __call__(self, eval_scores: Scores) -> t.Tuple[bool, bool]:
        """
        Returns (should_save, should_stop) tuple of booleans
        """
        eval_score = eval_scores._asdict()[self.score_type]
        if self.max_score - eval_score < self.tolerance:
            LOGGER.info(
                f'The model has approached max score {self.max_score}. '
                f'Sending the termination signal')
            self.count = 0
            self.score = eval_score
            return True, True
        diff = eval_score - self.score
        if diff > self.tolerance:
            LOGGER.info(
                f'The model has improved the score by {eval_score}-{self.score}={diff}')
            self.count = 0
            self.score = eval_score
        else:
            self.count += 1
            LOGGER.info(f"The model hasn't improved the score {self.score} "
                        f'for {self.count} rounds out of {self.rounds}')
        return self.count == 0, self.count >= self.rounds


def dump_state(model, tokenizer, optimizer, scheduler, path):
    model.save_pretrained(path)
    tokenizer.save_pretrained(path)
    torch.save(optimizer.state_dict(), path / 'optimizer.pt')
    torch.save(scheduler.state_dict(), path / 'scheduler.pt')


def calc_scores(y_true, y_prob, sample_weight=None):
    y_pred = np.round(y_prob)
    try:  # In case prediction outputs the same classes
        roc_auc = roc_auc_score(
            y_true, y_prob, sample_weight=sample_weight)
    except ValueError as e:
        LOGGER.warning(f'Failed to calculate ROC AUC score due to {e}')
        roc_auc = 0
    return Scores(
        accuracy_score(y_true, y_pred, sample_weight=sample_weight),
        roc_auc,
        f1_score(y_true, y_pred, sample_weight=sample_weight),
        precision_score(y_true, y_pred, sample_weight=sample_weight),
        recall_score(y_true, y_pred, sample_weight=sample_weight)
    )


def calc_scores_mlm(y_true, y_prob, average='macro'):
    y_prob = softmax(y_prob, axis=2)
    y_pred = np.argmax(y_prob, axis=2).flatten()
    y_true = np.concatenate(y_true).flatten()
    mask = y_true != -100
    y_true, y_pred = y_true[mask], y_pred[mask]
    # d1, d2, d3 = y_prob.shape
    return Scores(
        accuracy_score(y_true, y_pred),
        np.nan,
        # roc_auc_score(y_true, y_prob.reshape(d1 * d2, d3), average=average, multi_class='ovo'),
        f1_score(y_true, y_pred, average=average),
        precision_score(y_true, y_pred, average=average),
        recall_score(y_true, y_pred, average=average)
    )


def load_dataset(df: t.Union[Path, pd.DataFrame], tokenizer, y_col, use_sample_weights) -> TensorDataset:
    if isinstance(df, Path):
        df = pd.read_csv(df, sep='\t')
    encoded = [tokenizer.encode_plus(s, add_special_tokens=True) for s in df['Seq']]
    inp_ids, att_masks = map(
        lambda name: torch.tensor([e[name] for e in encoded], dtype=torch.long),
        ['input_ids', 'attention_mask'])

    labels = df[y_col].values
    if use_sample_weights:
        weights = torch.tensor(
            compute_sample_weight("balanced", labels), dtype=torch.float)[:, None]
    else:
        weights = torch.ones((len(labels), 1), dtype=torch.float)

    # This is float to work with 1-sized output loss
    labels = torch.tensor(labels, dtype=torch.float)[:, None]
    return TensorDataset(inp_ids, att_masks, labels, weights)


def _compute_class_weights(labels, vocab_size):
    weights_base = np.zeros(vocab_size)
    actual_labels = np.sort(np.unique(labels))
    class_weights = compute_class_weight('balanced', classes=actual_labels, y=labels)
    LOGGER.debug(f'Obtained weights {class_weights} for classes {actual_labels}')
    weights_base[actual_labels] = class_weights
    return weights_base


def load_fixed_mask_dataset(df, tokenizer, mask_pos, use_weights=False):

    def mask_positions(tokens: t.Iterable[int]):
        return [tokenizer.mask_token_id if i in mask_pos else tok for i, tok in enumerate(tokens)]

    def get_labels(tokens: t.Iterable[int]):
        return [tok if i in mask_pos else -100 for i, tok in enumerate(tokens)]

    if isinstance(df, Path):
        df = pd.read_csv(df, sep='\t')
        LOGGER.debug(f'Loaded dataset with {len(df)} records')

    encoded = [tokenizer.encode_plus(s, add_special_tokens=True) for s in df['Seq']]
    LOGGER.debug('Encoded dataset')
    for e in encoded:
        e['labels'] = get_labels(e['input_ids'])
        e['input_ids'] = mask_positions(e['input_ids'])
    LOGGER.debug('Masked inputs and produced labels')
    inp_ids, att_masks, labels = map(
        lambda name: torch.tensor([_e[name] for _e in encoded], dtype=torch.long),
        ['input_ids', 'attention_mask', 'labels'])
    if use_weights:
        all_labels = np.array(list(chain.from_iterable(
            (filter(lambda x: x != -100, _e['labels']) for _e in encoded))))
        class_weight = torch.tensor(
            _compute_class_weights(all_labels, tokenizer.vocab_size), dtype=torch.float)
        weights = class_weight.expand(len(encoded), -1)
    else:
        weights = torch.ones((len(encoded), tokenizer.vocab_size), dtype=torch.float)
    LOGGER.debug('Obtained class weights')
    return TensorDataset(inp_ids, att_masks, labels, weights)


def train(
    model, tokenizer, 
    run_setup: RunSetup, opt_setup: OptSetup, stop_setup: StopSetup,
    eval_ds: t.Optional[pd.DataFrame] = None, 
    train_ds: t.Optional[pd.DataFrame] = None,
    train_tds: t.Optional[TensorDataset] = None, 
    eval_tds: t.Optional[TensorDataset] = None,
    checkpoint_dir=None, score_fn=calc_scores, device='cuda',
    y_col: str = 'IsPositive', 
    ds_loader: t.Optional[t.Callable] = None,
    use_sample_weights: bool = True, train_save_res_step: int = 100
):
    # train_y = train_ds[y_col].values
    # train_weights = compute_sample_weight('balanced', train_y)
    if ds_loader is None:
        ds_loader = curry(load_dataset)(
            y_col=y_col, use_sample_weights=use_sample_weights)
    if train_tds is None:
        train_tds = ds_loader(train_ds, tokenizer)
    if not train_save_res_step:
        train_save_res_step = 1
    
    train_loader = DataLoader(
        train_tds, sampler=RandomSampler(train_tds), batch_size=run_setup.BatchSize)
    LOGGER.info(f'Prepared train loader with {len(train_loader)} batches')
    train_y = np.concatenate([
        batch[2].numpy() for i, batch in enumerate(train_loader) 
        if i % train_save_res_step == 0])
    LOGGER.info(f'Will use {len(train_y)} train examples for metrics calc')

    total_steps = len(train_loader) * run_setup.Epochs
    warmup_steps = int(total_steps * run_setup.WarmupPerc)

    no_decay = ["bias", "LayerNorm.weight"]
    optimizer_grouped_parameters = [
        {"params": [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)],
         "weight_decay": opt_setup.WeightDecay,
         },
        {"params": [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)],
         "weight_decay": 0.0},
    ]
    optimizer = AdamW(
        optimizer_grouped_parameters,
        lr=opt_setup.LearningRate,
        betas=opt_setup.Betas,
        eps=opt_setup.Epsilon
    )
    scheduler = get_linear_schedule_with_warmup(
        optimizer, warmup_steps, total_steps)

    if eval_tds is None:
        eval_tds = ds_loader(eval_ds, tokenizer)
    eval_loader = DataLoader(
        eval_tds, sampler=SequentialSampler(eval_tds),
        batch_size=run_setup.BatchSize)
    LOGGER.info(f'Prepared eval loader with {len(eval_loader)} batches')
    eval_y = np.concatenate([batch[2].numpy() for batch in eval_loader])
    LOGGER.info(f'Will use {len(eval_y)} examples for evaluation')
    # if use_sample_weights:
    #     eval_weights = compute_sample_weight('balanced', eval_y)
    # else:
    #     eval_weights = None
    eval_weights = None
    
    if stop_setup:
        stopper = EarlyStopping(stop_setup.Rounds, stop_setup.Tolerance, stop_setup.ScoreType)
    else:
        stopper = None

    train_scores_hist, eval_scores_hist = [], []

    model.zero_grad()
    with tqdm(desc='Running epochs', total=run_setup.Epochs) as bar:
        for i_epoch in range(1, run_setup.Epochs + 1):

            train_loss, train_prob = run_epoch(
                model, train_loader, optimizer, scheduler, 
                device, 'train', train_save_res_step)
            train_scores = score_fn(train_y, train_prob)
            train_scores_hist.append(train_scores)
            # TODO: find a way to match weights with samples after shuffling
            # train_scores = None
            LOGGER.info(f'Finished epoch {i_epoch} with loss {train_loss} '
                        f'and (unweighted) scores {train_scores}')

            eval_loss, eval_prob = run_epoch(
                model, eval_loader, optimizer, scheduler, device, 'predict')
            if eval_weights is not None:
                eval_scores = score_fn(eval_y, eval_prob, eval_weights)
            else:
                eval_scores = score_fn(eval_y, eval_prob)
            eval_scores_hist.append(eval_scores)
            LOGGER.info(f'Fininshed evaluation with loss {eval_loss} and scores {eval_scores}')

            bar.set_postfix({'Train loss': train_loss, 'Eval loss': eval_loss})
            bar.update(1)

            if stopper is not None:
                should_save, should_stop = stopper(eval_scores)
                if should_stop:
                    LOGGER.info(f'Early-stopping the training at epoch {i_epoch}')
                    break
                if should_save and checkpoint_dir:
                    checkpoint_dir.mkdir(exist_ok=True, parents=True)
                    dump_state(
                        model, tokenizer, optimizer, scheduler, checkpoint_dir)
                    LOGGER.info(f'Dumped the current state to {checkpoint_dir}')

    return train_scores_hist, eval_scores_hist


def run_epoch(
    model, loader, optimizer, scheduler, device, 
    mode: str = 'train', train_save_res_step: int = 100
):
    batch_iter = tqdm(loader, desc=f'Running {mode} batches')
    total_loss = 0
    predictions = []
    for i, batch in enumerate(batch_iter):

        inp_ids, att_mask, labels, weight = map(
            lambda tens: tens.to(device), batch)
        if mode == 'train':
            model.train()
            outputs = model(
                input_ids=inp_ids, attention_mask=att_mask,
                labels=labels, weight=weight)
        else:
            with torch.no_grad():
                model.eval()
                outputs = model(
                    input_ids=inp_ids, attention_mask=att_mask,
                    labels=labels, weight=weight)
        loss, prob = outputs[:2]
        total_loss += loss.mean().item()
        if mode != 'train' or i % train_save_res_step == 0:
            predictions.append(prob.detach().cpu().numpy())

        if mode == 'train':
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()
            model.zero_grad()

    predictions = np.concatenate(predictions)
    return total_loss, predictions


if __name__ == '__main__':
    raise RuntimeError
