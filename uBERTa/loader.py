import logging
import operator as op
import typing as t
from itertools import starmap, chain
from pathlib import Path

import numpy as np
import pandas as pd
import pytorch_lightning as pl
import torch
from more_itertools import windowed, sliding_window
from toolz import curry
from torch.utils.data import TensorDataset, RandomSampler, SequentialSampler, DataLoader

from uBERTa.base import ColNames, DefaultColNames, VALID_START
from uBERTa.tokenizer import DNATokenizer
from uBERTa.utils import fill_row_around_ones, train_test_split_on_idx, train_test_split_on_values

LOGGER = logging.getLogger(__name__)


class uBERTaLoader(pl.LightningDataModule):

    def __init__(
            self,
            df: t.Optional[pd.DataFrame] = None,
            window_size: t.Optional[int] = None,
            window_step: t.Optional[int] = None,
            tokenizer: t.Optional[DNATokenizer] = None,
            is_mlm_task: bool = False,
            mlm_mask_flank_size: int = 1,
            mlm_fraction_masked: float = 0.5,
            non_start_signal_to_zero: bool = False,
            scale_signal_bounds: t.Optional[t.Tuple[float, float]] = (0.01, 1.0),
            signal_bounds: t.Optional[t.Tuple[float, float]] = (0.1, 5000.0),
            token_level: bool = True,
            n_upsample_positives: t.Optional[int] = None,
            train_ds: t.Optional[pd.DataFrame] = None,
            val_ds: t.Optional[pd.DataFrame] = None,
            test_ds: t.Optional[pd.DataFrame] = None,
            train_tds: t.Optional[TensorDataset] = None,
            val_tds: t.Optional[TensorDataset] = None,
            test_tds: t.Optional[TensorDataset] = None,
            att_mask_tokens: t.Sequence[str] = ('[PAD]', '[SEP]'),
            val_fraction: float = 0.1, test_fraction: float = 0.1,
            split_col: str = DefaultColNames.gene_id,
            batch_size: int = 2 ** 5, num_proc: int = 8,
            col_names: ColNames = DefaultColNames,
            kmerize_cols: t.Sequence[str] = (
                    DefaultColNames.seq, DefaultColNames.seq_enum,
                    DefaultColNames.signal, DefaultColNames.classes),
            dataset_names: t.Sequence[str] = (
                    'train_ds.h5', 'val_ds.h5', 'test_ds.h5',
                    'train_tds.bin', 'val_tds.bin', 'test_tds.bin'),
            valid_start_codons: t.Sequence[str] = VALID_START
    ):
        """
        :param df: initial dataframe to work with; needed unless the prepared
            `train_ds`, `val_ds`, `test_ds` are provided.
        :param window_size: size of a window (aka maximum seq length), in tokens
        :param window_step: step for the sliding window to roll over seqs exceeding `window_size`
        :param tokenizer: a `DNATokenizer` instance
        :param is_mlm_task: flag indicating that datasets should be prepared for the masked language modeling objective.
            If true, some percent of the start codons and their surrounding tokens will be masked
        :param mlm_mask_flank_size: a number of tokens masked at each side of the start token
        :param mlm_fraction_masked: a fraction of potential start codons to mask among all the start codons
            in the dataset
        :param non_start_signal_to_zero: nullify experimental signal for tokens other than start codons
        :param scale_signal_bounds: scale experimental signal between these boundaries
        :param signal_bounds: cap experimental signal between these boundaries (applies before scaling)
        :param token_level: use token-level classification objective; if false, will prepare "centered" dataset,
            with start tokens at the center of sequences, for the sentence classification objective
        :param n_upsample_positives: a number of jittered copies created for each positive example. A jittered copy
            is a copy of a sequence centered at the proximity of a positive example, with length equal
            to the `window_size`
        :param train_ds: a prepared train dataset
        :param val_ds: a prepared validation dataset
        :param test_ds: a prepared test dataset
        :param train_tds: a prepared `TensorDataset`
        :param val_tds: a prepared `TensorDataset`
        :param test_tds: a prepared `TensorDataset`
        :param att_mask_tokens: a sequence of token names to be masked for attention
        :param val_fraction: a fraction of samples for validation
        :param test_fraction: a fraction of sampled set aside for testing
        :param split_col: a column name containing values the validation and testing fractions should be calculated on.
            To avoid overlapping, it's either a gene ID (then the fractions will be utilized), or the "dataset".
            In the latter case, it should designate samples by "Train", "Val", and "Test" values.
        :param batch_size: self-explanatory
        :param num_proc: a number of processors for data loaders
        :param col_names: a namedtuple with column names
        :param kmerize_cols: which columns should we kmerize
        :param dataset_names: names of all the datasets for saving
        :param valid_start_codons: nucleotide triplets treated as start codons (valid objects for  classification)
        """
        super().__init__()
        self.df = df
        self.window_size = window_size
        self.window_step = window_step
        self.is_mlm_task = is_mlm_task
        self.mlm_fraction_masked = mlm_fraction_masked
        self.mlm_mask_flank_size = mlm_mask_flank_size
        self.non_start_signal_to_zero = non_start_signal_to_zero
        self.scale_signal_bounds = scale_signal_bounds
        self.signal_bounds = signal_bounds
        self.token_level = token_level
        self.tokenizer = tokenizer
        self.val_fraction = val_fraction
        self.test_fraction = test_fraction
        self.n_upsample_positives = n_upsample_positives
        self.split_col = split_col
        self.batch_size = batch_size
        self.num_proc = num_proc
        self.col_names = col_names
        self.kmerize_cols = kmerize_cols
        self.dataset_names = dataset_names
        self.att_mask_tokens = att_mask_tokens
        self.valid_start_codons = valid_start_codons

        self.train_ds = train_ds
        self.val_ds = val_ds
        self.test_ds = test_ds

        self.train_tds = train_tds
        self.val_tds = val_tds
        self.test_tds = test_tds

        self.kmer = None if tokenizer is None else tokenizer.kmer

    def setup(self, stage: t.Optional[str] = None) -> None:
        """

        :param stage:
        :return:
        """
        self._setup()

    def _setup(self) -> None:

        if any(x is None for x in [self.train_tds, self.val_tds, self.test_tds]):
            if any(x is None for x in [self.train_ds, self.val_ds, self.test_ds]):
                LOGGER.info(f'Total initial sequences: {len(self.df)}')
                df = self.df.copy()

                if self.split_col == self.col_names.gene_id:
                    val, tmp = train_test_split_on_idx(df, self.val_fraction, self.split_col)
                    test, train = train_test_split_on_idx(
                        tmp, self.test_fraction / (1 - self.val_fraction), self.split_col)
                elif self.split_col == self.col_names.dataset:
                    datasets = train_test_split_on_values(df, self.col_names.dataset)
                    train, val, test = datasets['Train'], datasets['Val'], datasets['Test']
                else:
                    raise ValueError(f'Invalid col {self.split_col} to split datasets')

                LOGGER.info(f'Split datasets. '
                            f'Train: {len(train)}, '
                            f'Val: {len(val)}, '
                            f'Test: {len(test)}.')

                if self.token_level:
                    prep_ds = self._prep_token_level
                else:
                    prep_ds = self._prep_centered
                self.train_ds, self.val_ds, self.test_ds = starmap(
                    prep_ds, [(train, 'Train'), (val, 'Val'), (test, 'Test')])

            LOGGER.info(f'Finalized datasets. '
                        f'Train: {len(self.train_ds)}, '
                        f'Val: {len(self.val_ds)}, '
                        f'Test: {len(self.test_ds)}.')

            prep_tds = self._prep_tds_mlm if self.is_mlm_task else self._prep_tds_cls
            self.train_tds, self.val_tds, self.test_tds = map(
                prep_tds, [self.train_ds, self.val_ds, self.test_ds])
            LOGGER.debug(f'Prepared tensor datasets')

        return

    def save_all(self, base: Path, overwrite: bool = False):
        def save(name: str, ds: t.Union[TensorDataset, pd.DataFrame, None]):
            p = base / name
            if ds is None or (not overwrite and p.exists()):
                LOGGER.debug(f'Skipping existing {name}')
                return
            if isinstance(ds, TensorDataset):
                torch.save(ds, p)
            else:
                ds.to_hdf(str(p), name)

        dss = [self.train_ds, self.val_ds, self.test_ds,
               self.train_tds, self.val_tds, self.test_tds]

        if not base.exists():
            base.mkdir(exist_ok=True, parents=True)

        for _name, _ds in zip(self.dataset_names, dss):
            save(_name, _ds)

    def _prep_token_level(self, df: pd.DataFrame, df_name: str):
        df = df.copy()
        LOGGER.info(f'Preparing {df_name} with {len(df)} records for token-level task')

        size, step = self.window_size - 2, self.window_step
        if self.n_upsample_positives and df_name == 'Train':
            df_upsampled = self._upsample_positives(df, size // 2 - 1)
            LOGGER.info(f'Obtained {len(df_upsampled)} seqs with upsampled positives')
            seq_sizes = map(len, df_upsampled[self.col_names.seq])
            if any(x > size for x in seq_sizes):
                LOGGER.warning(f'Among the upsampled sequences, there is at least one with size '
                               f'less than a window size which would force to slide window over '
                               f'such examples, thus increasing the number of positives once more.')
            df = pd.concat([df, df_upsampled])

        df = self.kmerize(df)
        df = self._reduce_kmers(df, df_name)
        df = self._mask_invalid_starts(df, df_name)
        df = self._scale_signal(df, df_name)
        df = self.roll_window(df, size, step)

        idx_empty = df[self.col_names.classes].apply(lambda x: np.all(x == -100))

        if np.any(idx_empty):
            LOGGER.warning(f'Removing {idx_empty.sum()} out of {len(df)} '
                           f'windows without classes from {df_name}. '
                           f'Consider calibrating window parameters.')
            df = df[~idx_empty]

        return df

    def _prep_centered(self, df: pd.DataFrame, df_name: str):
        def slice_and_pad(xs: t.List[t.Any], idx: int, val: t.Any) -> t.List[t.Any]:
            xs_l = xs[np.max([0, idx - side]): idx]
            xs_r = xs[idx: idx + side]
            num_pad_l = (side - len(xs_l))
            num_pad_r = (side - len(xs_r))
            res = [val] * num_pad_l + xs_l + xs_r + [val] * num_pad_r
            return res

        def unravel(row):
            mask = row.Classes != -100
            seq = row[self.col_names.seq].split()
            sig = list(row[self.col_names.signal])
            prepend_values = [row[c] for c in cols_prepend]
            for idx in np.where(mask)[0]:
                pos = row[self.col_names.seq_enum][idx]
                cls = row[self.col_names.classes][idx]
                seq_c = slice_and_pad(seq, idx, '[PAD]')
                sig_c = slice_and_pad(sig, idx, 0.0)
                yield *prepend_values, seq_c, cls, pos, sig_c

        LOGGER.info(f'Preparing {df_name} with {len(df)} records for centered task')

        df = self.kmerize(df)
        df = self._reduce_kmers(df, df_name)
        df = self._mask_invalid_starts(df, df_name)
        df = self._scale_signal(df, df_name)

        LOGGER.info(f'Unraveling {df_name} rows')
        side = self.window_size // 2
        cols_roll = [self.col_names.seq, self.col_names.classes,
                     self.col_names.seq_enum, self.col_names.signal]
        cols_prepend = [c for c in df.columns if c not in cols_roll]
        length = len(df)
        df = pd.DataFrame(
            chain.from_iterable(map(unravel, map(op.itemgetter(1), df.iterrows()))),
            columns=cols_prepend + cols_roll)
        LOGGER.info(f'Unraveled {length} records into {len(df)}')

        return df

    def _upsample_positives(self, df: pd.DataFrame, jitter_ws: int):
        def spawn_positives(row):
            for pos in row[self.col_names.seq_enum_pos]:
                for _ in range(self.n_upsample_positives):
                    row_cp = row.copy()
                    idx = np.where(row[self.col_names.seq_enum] == pos)[0]
                    if not len(idx) == 1:
                        raise ValueError(f'Multiple indices {idx} for position {pos}')

                    idx = idx[0]
                    center = np.random.randint(idx - jitter_ws + 1, idx + jitter_ws - 1)
                    idx_l = max(0, center - jitter_ws)
                    idx_r = min(len(row[self.col_names.seq_enum]) - 1, center + jitter_ws)

                    row_cp[self.col_names.seq] = row_cp[self.col_names.seq][idx_l:idx_r]
                    row_cp[self.col_names.seq_enum] = row_cp[self.col_names.seq_enum][idx_l:idx_r]
                    row_cp[self.col_names.signal] = row_cp[self.col_names.signal][idx_l:idx_r]
                    row_cp[self.col_names.classes] = row_cp[self.col_names.classes][idx_l:idx_r]
                    row_cp[self.col_names.seq_enum_pos] = np.intersect1d(
                        row_cp[self.col_names.seq_enum], row_cp[self.col_names.seq_enum_pos])

                    if pos not in row_cp[self.col_names.seq_enum_pos]:
                        raise ValueError('Window missed positive value')
                    yield row_cp

        _df = df[~df[self.col_names.seq_enum_pos].isna()]
        LOGGER.info(f'Upsampling positives from {len(_df)} sequences and jitter window of {jitter_ws}')
        return pd.DataFrame(chain.from_iterable(map(spawn_positives, map(op.itemgetter(1), _df.iterrows()))))

    def kmerize(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()
        LOGGER.info(f'Using kmer {self.tokenizer.kmer} on {self.kmerize_cols}')

        if self.tokenizer is None:
            raise ValueError('Need tokenizer to kmerize and encode values')
        kmer = self.tokenizer.kmer

        def kmerize_seq(seq):
            return " ".join(map(lambda s: "".join(s), sliding_window(seq, kmer)))

        def kmerize_arr(arr):
            return np.array(list(sliding_window(arr, kmer)))

        for col in self.kmerize_cols:
            if col not in df.columns:
                raise ValueError(f'{col} is absent to kmerize')
            if isinstance(df[col].iloc[0], str):
                df[col] = df[col].apply(kmerize_seq)
            else:
                df[col] = df[col].apply(kmerize_arr)

        return df

    def _reduce_kmers(self, df: pd.DataFrame, df_name: str):
        LOGGER.debug(f'Reducing kmers for {df_name}')
        df = df.copy()
        df[[self.col_names.classes, self.col_names.seq_enum]] = list(starmap(
            self._reduce_by_positive,
            zip(df[self.col_names.classes], df[self.col_names.seq_enum])))
        df[self.col_names.signal] = list(starmap(
            self._reduce_signal,
            zip(df[self.col_names.classes], df[self.col_names.signal])))
        return df

    @staticmethod
    def _reduce_by_positive(classes, positions):
        def _reduce(cs, ps):
            idx_1 = cs == 1
            idx_0 = cs == 0
            if np.any(idx_1):
                return cs[idx_1][0], ps[idx_1][0]
            if np.any(idx_0):
                return cs[idx_0][0], ps[idx_0][0]
            return -100, 0

        assert len(classes.shape) == len(positions.shape) == 2
        assert len(classes) == len(positions)
        if classes.shape[1] == 1:
            return np.squeeze(classes), np.squeeze(positions)

        classes = classes[:, :-2]
        positions = positions[:, :-2]

        reduced = np.array(list(starmap(_reduce, zip(classes, positions))))
        return np.squeeze(reduced[:, 0]), np.squeeze(reduced[:, 1])

    def _reduce_signal(self, classes, signal):
        """
        Given reduced classes and signal, sum-reduce signal.
        Optionally, put zeros at the positions of -100-masked classes.
        """
        assert len(signal.shape) == 2
        assert len(classes.shape) == 1
        signal = np.sum(signal, axis=1)
        if self.non_start_signal_to_zero:
            signal[classes == -100] = 0
        return signal

    def _scale_signal(self, df: pd.DataFrame, df_name: str):
        LOGGER.debug(f'Capping and scaling signal for {df_name}')

        if self.signal_bounds is not None:
            df[self.col_names.signal] = df[self.col_names.signal].apply(
                lambda x: np.clip(x, self.signal_bounds[0], self.signal_bounds[1]))
            LOGGER.debug(f'Capped signal in {self.signal_bounds}')
        if self.scale_signal_bounds is not None:
            a, b = self.scale_signal_bounds
            sig_values = list(chain.from_iterable(df[self.col_names.signal]))
            min_sig, max_sig = np.min(sig_values), np.max(sig_values)
            if max_sig - min_sig != 0:
                df[self.col_names.signal] = df[self.col_names.signal].apply(
                    lambda x: a + (b - a) * (x - min_sig) / (max_sig - min_sig))
            LOGGER.debug(f'Scaled signal between {a} and {b}')
        return df

    def _mask_invalid_starts(self, df: pd.DataFrame, df_name: str) -> pd.DataFrame:
        def mask(row):
            seq = row[self.col_names.seq].split()
            m = np.array([x not in self.valid_start_codons for x in seq])
            row[self.col_names.classes][m] = -100
            row[self.col_names.seq_enum][m] = 0
            return row

        LOGGER.debug(f'Filtering to {self.valid_start_codons} for {df_name}')
        return pd.DataFrame(map(mask, map(op.itemgetter(1), df.iterrows())))

    def roll_window(self, df: pd.DataFrame, window_size: int, window_step: int) -> pd.DataFrame:
        LOGGER.info(f'Rolling window with size {window_size}, step {window_step}')
        _windowed = curry(windowed)(n=window_size, step=window_step)
        cols_roll = [self.col_names.seq, self.col_names.classes,
                     self.col_names.seq_enum, self.col_names.signal]
        cols_prepend = [c for c in df.columns if c not in cols_roll]

        def roll(row):
            seq, cls, seq_enum, signal = (
                row[self.col_names.seq], row[self.col_names.classes],
                row[self.col_names.seq_enum], row[self.col_names.signal])
            seq = seq.split()
            assert len(seq) == len(cls) == len(seq_enum) == len(signal)
            seq_chunks = _windowed(seq, fillvalue='[PAD]')
            cls_chunks = _windowed(cls, fillvalue=-100)
            seq_enum_chunks = _windowed(seq_enum, fillvalue=0)
            signal_chunks = _windowed(signal, fillvalue=0.0)

            prepend_values = [row[c] for c in cols_prepend]
            for seq_chunk, cls_chunk, seq_enum_chunk, signal_chunk in zip(
                    seq_chunks, cls_chunks, seq_enum_chunks, signal_chunks):
                yield (*prepend_values,
                       " ".join(seq_chunk), np.array(cls_chunk),
                       np.array(seq_enum_chunk), np.array(signal_chunk))

        rolled = chain.from_iterable(
            map(roll, map(op.itemgetter(1), df.iterrows())))

        return pd.DataFrame(rolled, columns=cols_prepend + cols_roll)

    def _prep_tds_mlm(self, df: pd.DataFrame) -> TensorDataset:
        """
        """
        inp_ids, att_msk, classes = self._prep_tds_cls(df).tensors[:3]

        cls_msk = classes != -100
        classes[cls_msk] = inp_ids[cls_msk]
        pos_idx = np.where(cls_msk)
        l = len(pos_idx[0])
        sel_idx_pos = np.random.choice(
            np.arange(0, l),
            int(self.mlm_fraction_masked * l),
            replace=False)
        sel_idx = (pos_idx[0][sel_idx_pos], pos_idx[1][sel_idx_pos])
        cls_msk[sel_idx] = False
        for _ in range(self.mlm_mask_flank_size):
            cls_msk = fill_row_around_ones(cls_msk)
        cls_msk[:, 0] = False
        cls_msk[:, -1] = False
        inp_ids[cls_msk] = 4
        return TensorDataset(inp_ids, att_msk, classes)

    def _prep_tds_cls(self, df: pd.DataFrame) -> TensorDataset:
        seqs = [s.split() if isinstance(s, str) else s for s in df[self.col_names.seq]]
        encoded = list(map(self.tokenizer.encode, seqs))
        inp_ids = torch.tensor(encoded, dtype=torch.long)
        att_msk = torch.ones(inp_ids.shape, dtype=torch.int)
        for tk in self.att_mask_tokens:
            att_msk[inp_ids == self.tokenizer.vocab[tk]] = 0

        signal = np.pad(
            np.vstack(df[self.col_names.signal]),
            ((0, 0), (1, 1)), constant_values=0)
        signal = torch.tensor(signal, dtype=torch.float)
        if self.token_level:
            classes = np.pad(
                np.vstack(df[self.col_names.classes]),
                ((0, 0), (1, 1)), constant_values=-100)
            classes = torch.tensor(classes, dtype=torch.long)
        else:
            classes = torch.tensor(df[self.col_names.classes].values, dtype=torch.long)
        return TensorDataset(inp_ids, att_msk, classes, signal)

    def train_dataloader(self) -> DataLoader:
        return DataLoader(
            self.train_tds, sampler=RandomSampler(self.train_tds),
            batch_size=self.batch_size, num_workers=self.num_proc)

    def val_dataloader(self) -> DataLoader:
        return DataLoader(
            self.val_tds, sampler=SequentialSampler(self.val_tds),
            batch_size=self.batch_size, num_workers=self.num_proc)

    def test_dataloader(self) -> DataLoader:
        return DataLoader(
            self.test_tds, sampler=SequentialSampler(self.test_tds),
            batch_size=self.batch_size, num_workers=self.num_proc)

    def predict_dataloader(self) -> DataLoader:
        return self.val_dataloader()


if __name__ == '__main__':
    raise RuntimeError
