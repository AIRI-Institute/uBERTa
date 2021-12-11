import logging
import sys
import typing as t
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import accuracy_score, roc_auc_score, f1_score, precision_score, recall_score
from sklearn.utils.class_weight import compute_sample_weight
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler, TensorDataset
from tqdm.auto import tqdm
from transformers import AdamW, get_linear_schedule_with_warmup

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
    def __init__(self, rounds: int, tolerance: float):
        self.rounds = rounds
        self.tolerance = tolerance
        self.count = 0
        self.score = 0

    def __call__(self, eval_score: float) -> t.Tuple[bool, bool]:
        """
        Returns (should_save, should_stop) tuple of booleans
        """
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


def load_dataset(df: t.Union[Path, pd.DataFrame], tokenizer) -> TensorDataset:
    if isinstance(df, Path):
        df = pd.read_csv(df, sep='\t')
    encoded = [tokenizer.encode_plus(s, add_special_tokens=True) for s in df['Seq']]
    inp_ids, att_masks = map(
        lambda name: torch.tensor([e[name] for e in encoded], dtype=torch.long),
        ['input_ids', 'attention_mask'])

    labels = df['IsPositive'].values
    weights = torch.tensor(
        compute_sample_weight("balanced", labels), dtype=torch.float)[:, None]

    # This is float to work with 1-sized output loss
    labels = torch.tensor(labels, dtype=torch.float)[:, None]
    return TensorDataset(inp_ids, att_masks, labels, weights)


def train(
        model, tokenizer, train_ds: pd.DataFrame, eval_ds: pd.DataFrame,
        run_setup: RunSetup, opt_setup: OptSetup, stop_setup: StopSetup,
        checkpoint_dir=None, score_fn=calc_scores, device='cuda'
):
    train_y = train_ds.IsPositive.values
    train_weights = compute_sample_weight('balanced', train_y)
    tds = load_dataset(train_ds, tokenizer)
    train_loader = DataLoader(tds, sampler=RandomSampler(tds), batch_size=run_setup.BatchSize)

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

    eval_y = eval_ds.IsPositive.values
    eval_weights = compute_sample_weight('balanced', eval_y)
    eval_tds = load_dataset(eval_ds, tokenizer)
    eval_loader = DataLoader(
        eval_tds, sampler=SequentialSampler(eval_tds),
        batch_size=run_setup.BatchSize)

    if stop_setup:
        stopper = EarlyStopping(stop_setup.Rounds, stop_setup.Tolerance)
    else:
        stopper = None

    train_scores_hist, eval_scores_hist = [], []

    model.zero_grad()
    with tqdm(desc='Running epochs', total=run_setup.Epochs) as bar:
        for i_epoch in range(1, run_setup.Epochs + 1):

            train_loss, train_prob = run_epoch(
                model, train_loader, optimizer, scheduler, device, 'train')
            train_scores = score_fn(train_y, train_prob)
            train_scores_hist.append(train_scores)
            # TODO: find a way to match weights with samples after shuffling
            LOGGER.info(f'Finished epoch {i_epoch} with loss {train_loss} and (unweighted) scores {train_scores}')

            eval_loss, eval_prob = run_epoch(
                model, eval_loader, optimizer, scheduler, device, 'predict')
            eval_scores = score_fn(eval_y, eval_prob, eval_weights)
            eval_scores_hist.append(eval_scores)
            LOGGER.info(f'Fininshed evaluation with loss {eval_loss} and scores {eval_scores}')

            bar.set_postfix({'Train loss': train_loss, 'Eval loss': eval_loss})
            bar.update(1)

            if stopper is not None:
                should_save, should_stop = stopper(eval_scores.roc_auc)
                if should_stop:
                    LOGGER.info(f'Early-stopping the training at epoch {i_epoch}')
                    break
                if should_save and checkpoint_dir:
                    checkpoint_dir.mkdir(exist_ok=True, parents=True)
                    dump_state(
                        model, tokenizer, optimizer, scheduler, checkpoint_dir)
                    LOGGER.info(f'Dumped the current state to {checkpoint_dir}')

    return train_scores_hist, eval_scores_hist


def predict(model, loader, device):
    batch_iter = tqdm(loader, desc='Predicting batches')
    total_loss = 0
    predictions = []
    for batch in batch_iter:

        with torch.no_grad():
            inp_ids, att_mask, labels, weights = map(
                lambda tens: tens.to(device), batch)
            outputs = model(
                input_ids=inp_ids, attention_mask=att_mask,
                labels=labels, sample_weight=weights)
            loss, logits = outputs[:2]
            total_loss += loss.mean().item()
            predictions.append(logits.detach().cpu().numpy())

    predictions = np.concatenate(predictions)
    return total_loss, predictions


def run_epoch(model, loader, optimizer, scheduler, device, mode: str = 'train'):
    batch_iter = tqdm(loader, desc=f'Running {mode} batches')
    total_loss = 0
    predictions = []
    for batch in batch_iter:

        inp_ids, att_mask, labels, weights = map(
            lambda tens: tens.to(device), batch)
        if mode == 'train':
            model.train()
            outputs = model(
                input_ids=inp_ids, attention_mask=att_mask,
                labels=labels, sample_weight=weights)
        else:
            with torch.no_grad():
                model.eval()
                outputs = model(
                    input_ids=inp_ids, attention_mask=att_mask,
                    labels=labels, sample_weight=weights)
        loss, logits = outputs[:2]
        total_loss += loss.mean().item()
        predictions.append(logits.detach().cpu().numpy())

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
