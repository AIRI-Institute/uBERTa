import logging
import operator as op
import typing as t
from itertools import count, starmap, chain
from pathlib import Path
from random import choice, randint

import networkx as nx
import numpy as np
import pandas as pd
import pysam
import pytorch_lightning as pl
import torch
from Bio.Seq import Seq
from more_itertools import sliding_window, windowed
from ncls import NCLS
from toolz import curry, identity
from torch.utils.data import TensorDataset, RandomSampler, SequentialSampler, DataLoader

from uBERTa.base import VALID_CHROM, VALID_CHROM_FLANKS, ColNames
from uBERTa.tokenizer import DNATokenizer

LOGGER = logging.getLogger(__name__)


class DatasetGenerator:
    """
    An interface class to generate (u)ORF datasets.

    # TODO: update docs
    Class is a container for the generation settings.
    Use case: generate datasets with different examples, but preserving the composition,
    i.e., the balance between different kinds of positive and negative examples.

        1. First, it samples different groups of positive uORFs (u, ma, m).
        2. Then, it samples ~same number of negative examples, multiplied by a constant
            (e.g., twice more negative than positive).
        3. The total number of sampled positive examples is distributed between groups
            of negative example types.
        4. Within each negative examples' group, it preserves the balance of start codons
            found in the sampled positive examples.

    For example, we want to take all positive examples (denote it as N), and sample 2
    times more negative ones (2*N), such that there are 0.8 random negative examples
    centered on start codons of the positive ones (2*N*0.8 in total) and 0.2 negative
    examples taken from putative uORFs with no experimental signal (2*N*0.2 in total).
    For all examples we want to take 50 nucleotides around the middle of the start codon.

    """

    def __init__(self, ds_path: Path, ref_path: Path, neg_multiplier: int = 1,
                 neg_fractions: t.Tuple[float, float, float, float] = (0.0, 0.5, 0.5, 0.0),
                 pos_fractions: t.Tuple[float, float, float] = (1.0, 1.0, 1.0),
                 level_ts: float = 0,
                 flank_size: int = 200,
                 kmer_size: t.Optional[int] = None,
                 use_analyzed: bool = True,
                 genes_of_pos: bool = False,
                 col_names: ColNames = ColNames()
                 ):
        """
        :param ds_path: A path to a dataset with columns with both positive and negative examples
            (what we call the "base" dataset).
        :param ref_path: A path to a Fasta reference file of the hg38 assembly of the human genome.
        :param neg_multiplier: Having N positive examples, generate N times this value negative ones.
        :param neg_fractions: Fractions of the different types of the negative data examples to
            generate, in the following order: (random, random_centered, valid uORF below level,
            valid uORF above level). These are explained below:
            - "random" are completely random element sampled from the genome.
            - "random centered" -- random elements of the human genome centered at the start codons
                of all the positive examples in the base dataset
            - "valid below level" -- valid potential uORF with the level <= the provided `level_ts`
            - "valid above level" -- valid potential uORF with the level > the provided `level_ts`
        :param pos_fractions: Fractions of the positive examples, absolute for each group in the
            following order: ("uORF", "maORF", "mORF").
        :param level_ts: A P-site initiation level threshold to separate "below" and "above level"
            negative examples.
        :param flank_size: A number of nucleotides to take around the middle
            position of the start codon.
        :param kmer_size: Tokenize sequences into kmers with this size.
        :param col_names: Column names of the dataset parsed from `ds_path`.
        """
        self.base_ds = pd.read_csv(ds_path, sep='\t')
        self.ref = Ref(str(ref_path))
        self.neg_multiplier = neg_multiplier
        self.neg_fractions = neg_fractions
        self.pos_fractions = pos_fractions
        self.level_ts = level_ts
        self.flank_size = flank_size
        self.kmer_size = kmer_size
        self.use_analyzed = use_analyzed
        self.genes_of_pos = genes_of_pos
        self.col_names = col_names
        self.last_generated: t.Optional[pd.DataFrame] = None

    def prefilter(self):
        df = self.base_ds.copy()
        LOGGER.debug(f'Pre-filtering initial dataset with {len(df)} records')
        if self.use_analyzed:
            df = df[df[self.col_names.analyzed]]
            LOGGER.debug(f'Filtered to {len(df)} records of analyzed genes')
        if self.genes_of_pos:
            idx = (df[self.col_names.analyzed] &
                   df[self.col_names.group].isin(['m', 'ma', 'u']))
            ids = set(df.loc[idx, self.col_names.gene_id])
            df = df[df[self.col_names.gene_id].isin(ids)]
            LOGGER.debug(f'Filtered to {len(df)} records of analyzed genes '
                         f'with at least one positive example')
        self.last_generated = df
        return df

    def assign_class_by_group(self, df: t.Optional[pd.DataFrame] = None,
                              g_pos: t.Tuple[str, ...] = ('m', 'ma', 'u'),
                              g_neg: t.Tuple[str, ...] = ('s',)):
        if df is None:
            df = self.last_generated
        else:
            df = df.copy()
        cls_col, grp_col = self.col_names.cls, self.col_names.group
        df[cls_col] = np.nan
        df.loc[df[grp_col].isin(g_pos), cls_col] = 1
        df.loc[df[grp_col].isin(g_neg), cls_col] = 0
        num_na, num_pos, num_neg, num_tot = map(
            len, [df[df[cls_col].isna()], df[df[cls_col] == 1], df[df[cls_col] == 0], df])
        LOGGER.debug(f'Assigned {num_pos} positive, {num_neg} negative; unassigned {num_na}; total {num_tot}')
        self.last_generated = df
        return df

    def sample(self, df: t.Optional[pd.DataFrame] = None):
        if df is None:
            df = self.last_generated
        else:
            df = df.copy()
        samples = sample_dataset(
            self.neg_multiplier, self.ref, df,
            self.neg_fractions, self.pos_fractions,
            self.level_ts, self.col_names
        )
        LOGGER.debug(f'Generated {len(samples)} samples')
        self.last_generated = samples
        return df

    def assign_cc(self, df: t.Optional[pd.DataFrame] = None):
        if df is None:
            df = self.last_generated
        else:
            df = df.copy()

        def sep_group(gg):
            centers = gg[self.col_names.start].values + 1
            starts = centers - self.flank_size
            ends = centers + self.flank_size
            ccs = group_overlapping(starts, ends, gg.index.values)

            gg[self.col_names.cc] = 0
            for i, cc in enumerate(ccs, start=1):
                gg.loc[list(cc), self.col_names.cc] = i
            return gg

        df = df.groupby(
            [self.col_names.chrom, self.col_names.strand],
            as_index=False
        ).apply(sep_group)
        self.last_generated = df
        return df

    def prepare_seqs(self, df: t.Optional[pd.DataFrame] = None,
                     merge_overlapping: bool = False):
        if df is None:
            df = self.last_generated
        else:
            df = df.copy()
        prep = prepare_overlapping_seqs if merge_overlapping else prepare_seqs_around
        seqs = prep(df, self.ref, self.flank_size, self.kmer_size, self.col_names)
        self.last_generated = seqs
        return seqs


class uBERTaLoader(pl.LightningDataModule):

    def __init__(self,
                 df: t.Optional[pd.DataFrame] = None,
                 window_size: t.Optional[int] = None,
                 window_step: t.Optional[int] = None,
                 tokenizer: t.Optional[DNATokenizer] = None,
                 is_mlm_task: bool = True, token_level: bool = True,
                 train_ds: t.Optional[pd.DataFrame] = None,
                 val_ds: t.Optional[pd.DataFrame] = None,
                 test_ds: t.Optional[pd.DataFrame] = None,
                 train_tds: t.Optional[TensorDataset] = None,
                 val_tds: t.Optional[TensorDataset] = None,
                 test_tds: t.Optional[TensorDataset] = None,
                 att_mask_tokens: t.Sequence[str] = ('[PAD]', '[SEP]'),
                 val_fraction: float = 0.1, test_fraction: float = 0.1,
                 batch_size: int = 2 ** 5, num_proc: int = 8,
                 col_names: ColNames = ColNames(),
                 dataset_names: t.Sequence[str] = (
                         'train_ds.h5', 'val_ds.h5', 'test_ds.h5',
                         'train_tds.bin', 'val_tds.bin', 'test_tds.bin')
                 ):
        super().__init__()
        self.df = df
        self.window_size = window_size
        self.window_step = window_step
        self.is_mlm_task = is_mlm_task
        self.token_level = token_level
        self.tokenizer = tokenizer
        self.val_fraction = val_fraction
        self.test_fraction = test_fraction
        self.batch_size = batch_size
        self.num_proc = num_proc
        self.col_names = col_names
        self.dataset_names = dataset_names
        self.att_mask_tokens = att_mask_tokens

        self.train_ds = train_ds
        self.val_ds = val_ds
        self.test_ds = test_ds

        self.train_tds = train_tds
        self.val_tds = val_tds
        self.test_tds = test_tds

        self.kmer = None if tokenizer is None else tokenizer.kmer

    def setup(self, stage: t.Optional[str] = None) -> None:
        self._setup()

    def _setup(self) -> None:

        if any(x is None for x in [self.train_tds, self.val_tds, self.test_tds]):
            # Prepare dataframes
            if any(x is None for x in [self.train_ds, self.val_ds, self.test_ds]):
                LOGGER.info(f'Total initial sequences: {len(self.df)}')
                val, tmp = train_test_split(self.df, self.val_fraction)
                test, train = train_test_split(
                    tmp, self.test_fraction / (1 - self.val_fraction))

                # Roll window only in case of token-level classification,
                # otherwise, use the unchanged dataset
                prep_ds = self._roll_ds if self.token_level else lambda x, y: x
                self.train_ds, self.val_ds, self.test_ds = starmap(
                    prep_ds, [(train, 'Train'), (val, 'Val'), (test, 'Test')])
            LOGGER.info(f'Train: {len(self.train_ds)}, Val: {len(self.val_ds)}, Test: {len(self.test_ds)}')

            # Prep tensor datasets
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
                ds.to_hdf(p, name)

        dss = [self.train_ds, self.val_ds, self.test_ds,
               self.train_tds, self.val_tds, self.test_tds]

        if not base.exists():
            base.mkdir(exist_ok=True, parents=True)

        for _name, _ds in zip(self.dataset_names, dss):
            save(_name, _ds)

    def _roll_ds(self, df: pd.DataFrame, df_name: str):
        LOGGER.debug(f'Processing {df_name} with {len(df)} records')

        # Window size is given in kmers -> only account for [CLS] and [SEP]
        size, step = self.window_size - 2, self.window_step
        rolled = self._roll_window(df, size, step)
        LOGGER.debug(f'Rolled window with size {size}, step {step}; records: {len(rolled)}')

        rolled[self.col_names.starts] = rolled[self.col_names.starts].apply(self._reduce_by_middle)
        rolled[self.col_names.classes] = rolled[self.col_names.classes].apply(self._reduce_by_middle)
        LOGGER.debug(f'Reduced labels to token level')
        return rolled

    def _prep_tds_mlm(self, df: pd.DataFrame) -> TensorDataset:
        """
        [  6    7  9   5    7    0 ]  input ids
        [-100 -100 0 -100 -100 -100]  classes
        ...
        Mask token ID is 4
        ->
        [  6    4  4   4    7 ]  input ids
        [  1    1  1   1    0 ]  attention mask (mask padding)
        [-100 -100 9 -100 -100]  labels
        """
        inp_ids, att_msk, classes = self._prep_tds_cls(df).tensors
        cls_msk = classes != -100
        classes[cls_msk] = inp_ids[cls_msk]
        cls_msk = fill_row_around_ones(cls_msk)
        cls_msk[:, 0] = False
        cls_msk[:, -1] = False
        inp_ids[cls_msk] = 4
        return TensorDataset(inp_ids, att_msk, classes)

    def _prep_tds_cls(self, df: pd.DataFrame) -> TensorDataset:
        # For some reason, I have to split tokens manually for encode
        seqs = (s.split() for s in df[self.col_names.seq])
        encoded = list(map(self.tokenizer.encode, seqs))
        inp_ids = torch.tensor(encoded, dtype=torch.long)
        att_msk = torch.ones(inp_ids.shape, dtype=torch.int)
        for tk in self.att_mask_tokens:
            att_msk[inp_ids == self.tokenizer.vocab[tk]] = 0
        if self.token_level:
            classes = np.pad(
                np.vstack(df[self.col_names.classes]),
                ((0, 0), (1, 1)), constant_values=-100)
            classes = torch.tensor(classes, dtype=torch.long)
        else:
            classes = torch.tensor(df[self.col_names.cls].values, dtype=torch.long)
        return TensorDataset(inp_ids, att_msk, classes)

    def _roll_window(self, df: pd.DataFrame, window_size: int, window_step: int) -> pd.DataFrame:
        _windowed = curry(windowed)(n=window_size, step=window_step)

        def roll(row):
            seq, cls, starts = (
                row[self.col_names.seq], row[self.col_names.classes], row[self.col_names.starts])
            seq = seq.split()
            assert len(seq) == len(cls) == len(starts)
            seq_chunks = _windowed(seq, fillvalue='[PAD]')
            cls_chunks = _windowed(cls, fillvalue=np.full(self.kmer, -100))
            starts_chunks = _windowed(starts, fillvalue=np.zeros(self.kmer, dtype=int))
            for seq_chunk, cls_chunk, starts_chunk in zip(seq_chunks, cls_chunks, starts_chunks):
                yield (row[self.col_names.chrom], row[self.col_names.strand],
                       " ".join(seq_chunk), np.array(cls_chunk), np.array(starts_chunk))

        rolled = chain.from_iterable(
            map(roll, map(op.itemgetter(1), df.iterrows())))
        cols = [self.col_names.chrom, self.col_names.strand, self.col_names.seq,
                self.col_names.classes, self.col_names.starts]

        return pd.DataFrame(rolled, columns=cols)

    @staticmethod
    def _reduce_by_middle(a):
        assert len(a.shape) >= 2
        assert a.shape[1] % 2 == 1
        return np.max(a[:, 1:-1], axis=1)

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


class Ref(pysam.FastaFile):
    """
    A wrapper around `pysam.FastaFile` for easier sampling.
    """

    def get_names(self, valid=True):
        return [n for n in self.references
                if not valid or n in VALID_CHROM]

    def get_lengths(self, valid=True):
        return {n: l for n, l in zip(self.references, self.lengths)
                if not valid or n in VALID_CHROM}

    def fetch_around(self, chrom, pos, strand, flank_size):
        length = self.get_lengths()[chrom]
        start = max([pos - flank_size, 0])
        end = min([pos + flank_size + 1, length])
        seq = self.fetch(chrom, start, end)
        if strand == '-':
            seq = reverse_complement(seq)
        return seq

    def random_start(self):
        chrom, length = choice(list(self.get_lengths().items()))
        f1, f2 = VALID_CHROM_FLANKS[chrom]
        idx = randint(f1, length - f2)
        strand = choice(['+', '-'])
        return chrom, idx, strand, length

    def scan_codon_from(
            self, codon, chrom, start, strand,
            length: t.Optional[int] = None,
            flank_size: int = 100,
            max_iter: t.Optional[int] = None
    ):
        border = flank_size or 3
        if length is None:
            length = self.get_lengths()[chrom]
        if strand == '-':
            _range = range(start, border, -1)
            get_idx = lambda pos: (pos - 3, pos)
            get_seq = lambda seq: reverse_complement(seq)
        else:
            _range = range(start, length - border)
            get_idx = lambda pos: (pos, pos + 3)
            get_seq = identity

        for it, i in enumerate(_range):
            start, end = get_idx(i)
            _seq = get_seq(self.fetch(chrom, start, end))
            if _seq == codon:
                return i
            if max_iter and it >= max_iter:
                break
        return None

    def find_codon(
            self, codon: str,
            max_scans: t.Optional[int] = None,
            max_iter_per_scan: t.Optional[int] = None,
            fetch: bool = False,
            flank_size: t.Optional[int] = 100,
    ):
        assert max_scans is None or max_scans >= 1
        idx_codon = None
        iters = range(max_scans) if max_scans else count()
        for _ in iters:
            pos = self.random_start()
            idx_codon = self.scan_codon_from(
                codon, *pos, flank_size,
                max_iter_per_scan)
            if idx_codon:
                break
        if idx_codon is None:
            return None

        chrom, _, strand, length = pos

        seq = None
        if fetch and flank_size:
            i_start = idx_codon + 1
            seq = self.fetch_around(chrom, i_start, strand, flank_size)

        return chrom, idx_codon, strand, length, seq


def reverse_complement(s: str) -> str:
    return str(Seq(s).reverse_complement())


def kmerize(seq: str, kmer_size: int) -> str:
    return " ".join(map(lambda s: "".join(s), sliding_window(seq, kmer_size)))


def fill_row_around_ones(a):
    """
    adapted from https://stackoverflow.com/questions/40223114/numpy-fill-fields-surrounding-a-1-in-an-array
    """
    rows, cols = a.shape
    padded = np.pad(a, 1, 'constant', constant_values=0)
    result = np.copy(a)
    for r, c in ((1, 0), (1, 2)):
        result |= padded[r:r + rows, c:c + cols]
    if isinstance(a, np.ndarray):
        return result
    return torch.tensor(result, dtype=a.dtype)


def sample_dataset(
        neg_multiplier, ref: Ref, ds: pd.DataFrame,
        neg_fractions: t.Tuple[float, float, float, float],
        pos_fractions: t.Tuple[float, float, float] = (1.0, 1.0, 1.0),
        level_ts: float = 0,
        col_names: ColNames = ColNames(),
        random_state: t.Optional[int] = None
) -> pd.DataFrame:
    LOGGER.debug(f'Obtained a dataset with {len(ds)} records')

    idx_pos = ds[col_names.group].isin(['m', 'u', 'ma'])
    ds_pos, ds_neg = ds[idx_pos], ds[~idx_pos]
    LOGGER.debug(f'Found {len(ds_pos)} positive and {len(ds_neg)} (potential) negative records')

    samples_pos = sample_pos(pos_fractions, ds_pos)
    LOGGER.debug(f'Sampled {len(samples_pos)} positive examples')

    samples_neg = sample_neg(neg_multiplier, neg_fractions, samples_pos, ds_neg,
                             ref, level_ts, col_names, random_state)
    LOGGER.debug(f'Sampled {len(samples_neg)} negative examples')

    samples_pos = samples_pos[[col_names.chrom, col_names.start, col_names.strand,
                               col_names.codon, col_names.group]]
    samples_pos[col_names.cls] = True
    samples_neg[col_names.cls] = False

    samples = pd.concat([samples_pos, samples_neg])
    LOGGER.debug(f'Concatenated into {len(samples)} total samples')

    return samples


def sample_pos(
        pos_fractions,
        df_pos: pd.DataFrame,
        col_names: ColNames = ColNames(),
        random_state: t.Optional[int] = None
) -> pd.DataFrame:
    def sample_group(g_name: str, frac: float) -> pd.DataFrame:
        sub = df_pos[df_pos[col_names.group] == g_name]
        num = int(len(sub) * frac)
        expr = f'{num}={len(sub)} * {frac}'
        LOGGER.debug(
            f'Will sample {expr} positive examples of group {g_name}')
        if num == len(sub):
            return sub
        return sub.sample(num, random_state=random_state)

    return pd.concat(starmap(
        sample_group, zip(['u', 'ma', 'm'], pos_fractions)))


def sample_neg(
        neg_multiplier,
        neg_fractions,
        df_pos: pd.DataFrame,
        df_neg: pd.DataFrame,
        ref: Ref,
        level_ts: float = 0.0,
        col_names: ColNames = ColNames(),
        random_state: int = None
) -> pd.DataFrame:
    def sample_random(codon, num):
        if num <= 0:
            return pd.DataFrame(columns=sel_columns[:3])
        samples = []
        if codon:
            sampler = lambda: ref.find_codon(
                codon, max_scans=100,
                max_iter_per_scan=1000,
                fetch=False)[:-1]
        else:
            sampler = lambda: ref.random_start()
        while len(samples) < num:
            samples.append(sampler())
        LOGGER.debug(f'Found {len(samples)} codons {codon}')
        samples = pd.DataFrame(
            (x[:3] for x in samples), columns=sel_columns[:3])
        samples[col_names.codon] = codon
        return samples

    @curry
    def sample_valid(
            codon: str, num: int,
            min_level: t.Optional[float] = None,
            max_level: t.Optional[float] = None
    ):
        if num == 0:
            return pd.DataFrame(columns=sel_columns)
        idx = df_neg[col_names.codon] == codon
        if max_level is not None:
            idx &= df_neg[col_names.level] <= max_level
        if min_level is not None:
            idx &= df_neg[col_names.level] >= min_level
        sub = df_neg[idx]
        LOGGER.debug(f'Found {len(sub)} records with codon {codon} '
                     f'and max level {max_level}')
        if num > len(sub):
            LOGGER.warning(f'The number of desired samples {num} '
                           f'exceeds the number of existing ones with '
                           f'codon {codon} and max level {max_level}')
        elif num == len(sub):
            pass
        else:
            sub = sub.sample(min([num, len(sub)]), random_state=random_state)
            LOGGER.debug(f'Sampled {len(sub)} samples with codon {codon} '
                         f'and max level {max_level}')
        return sub[sel_columns].copy()

    sel_columns = [col_names.chrom, col_names.start,
                   col_names.strand, col_names.codon]

    sum_neg = sum(neg_fractions)
    if sum_neg != 1:
        raise ValueError(f'The `neg_fractions` {neg_fractions} '
                         f'must sum to 1.0; got {sum_neg}')
    neg_total = len(df_pos) * neg_multiplier
    LOGGER.debug(f'Expecting around {neg_total} total negative examples')

    # Derive per codon counts of samples within each group in the provided fractions
    codon_counts = df_pos[col_names.codon].value_counts().to_dict()
    codon_fractions = {k: v / len(df_pos) for k, v in codon_counts.items()}

    groups = ['random', 'below_level', 'above_level']
    # Total number of examples to sample for each group except the totally random one
    group_counts = {
        name: int(neg_total * frac) for name, frac in zip(groups, neg_fractions[1:])}
    # Counts of samples per codon for each group except the totally random one
    # as the latter does not require any specific start codon
    count_group_codons = lambda total: {
        k: int(total * v) for k, v in codon_fractions.items()}
    samples_per_codon = {
        name: count_group_codons(total) for name, total in zip(
            groups,
            map(group_counts.get, groups))
    }
    # Incorporate the counts for totally random group
    samples_per_codon[groups[0]][None] = int(neg_fractions[0] * neg_total)
    # Output the progress
    expected_samples = sum(sum(d.values()) for d in samples_per_codon.values())
    LOGGER.debug(f'Samples per codon counts: {samples_per_codon}')
    LOGGER.debug(f'Num samples after the per-codon correction: {expected_samples}')

    random_samples = pd.concat(starmap(
        sample_random,
        samples_per_codon['random'].items()))
    random_samples[col_names.group] = 'random'
    LOGGER.debug(f'Sampled {len(random_samples)} random start codons')

    valid_samples_below = pd.concat(starmap(
        sample_valid(max_level=level_ts),
        samples_per_codon['below_level'].items()))
    valid_samples_below[col_names.group] = 'below_level'
    LOGGER.debug(f'Sampled {len(valid_samples_below)} with level <= {level_ts}')

    valid_samples_above = pd.concat(starmap(
        sample_valid(min_level=level_ts + 0.01),
        samples_per_codon['above_level'].items()))
    valid_samples_above[col_names.group] = 'above_level'
    LOGGER.debug(f'Sampled {len(valid_samples_above)} with level > {level_ts + 0.01}')

    return pd.concat(
        [random_samples, valid_samples_below, valid_samples_above],
        ignore_index=True)


def prepare_seqs_around(
        ds: pd.DataFrame, ref: Ref, flank_size: int,
        kmer_size: t.Optional[int] = None,
        col_names: ColNames = ColNames()
) -> pd.DataFrame:
    def prepare_seq(chrom, pos, strand):
        seq = ref.fetch_around(chrom, pos + 1, strand, flank_size)
        if kmer_size:
            seq = kmerize(seq, kmer_size)
        return seq.upper()

    names = [col_names.chrom, col_names.start, col_names.strand, col_names.cls]

    return pd.DataFrame(
        ((chrom, start, strand, int(pos), prepare_seq(chrom, start, strand))
         for _, chrom, start, strand, pos in ds[names].itertuples()),
        columns=names + [col_names.seq]
    )


def group_overlapping(starts, ends, ids):
    """
    Find all overlapping intervals, turn them into a graph.
    Return graph's conneted components.
    """
    tree = NCLS(starts, ends, ids)
    idx_q, idx_s = tree.all_overlaps_both(starts, ends, ids)
    idx_self = idx_q == idx_s
    idx_q, idx_s = idx_q[~idx_self], idx_s[~idx_self]
    g = nx.Graph(zip(idx_q, idx_s))
    return nx.connected.connected_components(g)


def prepare_overlapping_seqs(
        ds: pd.DataFrame,
        ref: Ref,
        flank_size: int,
        kmer_size: t.Optional[int] = None,
        col_names: ColNames = ColNames(),
):
    def process_group(gg):
        gg = gg.sort_values(col_names.start)
        chrom = gg[col_names.chrom].iloc[0]
        strand = gg[col_names.strand].iloc[0]

        starts = gg[col_names.start].values
        centers = starts + 1
        start = centers[0] - flank_size
        end = centers[-1] + flank_size + 1

        centers = centers - start
        ann_classes = np.full(end - start, -100)
        ann_classes[centers] = gg[col_names.cls].values

        _starts = np.zeros(end - start, dtype=int)
        _starts[ann_classes != -100] = starts

        seq = ref.fetch(chrom, start, end)
        if strand == '-':
            seq = reverse_complement(seq)
            ann_classes = np.flip(ann_classes)
            _starts = np.flip(_starts)

        if kmer_size:
            _seq = kmerize(seq, kmer_size)
            _ann, _st = map(
                lambda x: np.array(list(sliding_window(x, kmer_size))),
                [ann_classes, _starts])
            return chrom, strand, _seq.upper(), _ann, _st

        return chrom, strand, seq.upper(), ann_classes, _starts

    idx = ds[col_names.cc] == 0
    group_vars = [col_names.chrom, col_names.strand, col_names.cc, col_names.start]
    getter = op.itemgetter(1)
    groups_solitary = map(getter, ds[idx].groupby(group_vars))
    groups_overlapping = map(getter, ds[~idx].groupby(group_vars[:-1]))
    groups = chain(groups_solitary, groups_overlapping)
    return pd.DataFrame(
        map(process_group, groups),
        columns=[col_names.chrom, col_names.strand, col_names.seq,
                 col_names.classes, col_names.starts])


def train_test_split(
        df: pd.DataFrame, test_fraction: float
) -> t.Tuple[pd.DataFrame, pd.DataFrame]:
    idx = np.zeros(len(df)).astype(bool)
    idx[np.random.randint(0, len(df), int(len(df) * test_fraction))] = True
    return df[idx], df[~idx]


if __name__ == '__main__':
    raise RuntimeError
