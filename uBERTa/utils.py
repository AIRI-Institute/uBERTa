import logging
import sys
import typing as t
from itertools import count
from pathlib import Path
from random import choice, randint

import numpy as np
import pandas as pd
import pyBigWig as pBW
import pysam
import torch
from Bio.Seq import Seq
from more_itertools import sliding_window
from toolz import curry, identity

from uBERTa.base import VALID_CHROM, VALID_CHROM_FLANKS

LOGGER = logging.getLogger(__name__)


class pBWs:
    """
    A class to simultaneously query multiple pybigwig files
    """

    def __init__(self, files: t.Iterable[t.Union[Path, str]]):
        self.handles = [pBW.open(str(p)) for p in files]

    def query(self, chrom: str, start: int, end: int, agg_fn=curry(np.sum)(axis=0)):
        values = np.array([bw.values(chrom, start, end) for bw in self.handles])
        if agg_fn:
            values = agg_fn(np.nan_to_num(values))
        return values


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


def train_test_split(
        df: pd.DataFrame, test_fraction: float
) -> t.Tuple[pd.DataFrame, pd.DataFrame]:
    idx = np.zeros(len(df)).astype(bool)
    idx[np.random.randint(0, len(df), int(len(df) * test_fraction))] = True
    return df[idx], df[~idx]


def train_test_split_on_idx(df: pd.DataFrame, test_fraction: float, col: str):
    vs = df[col].unique()
    idx = np.zeros(len(vs)).astype(bool)
    idx[np.random.randint(0, len(vs), int(len(vs) * test_fraction))] = True
    vs_train, vs_test = vs[idx], vs[~idx]
    return df[df[col].isin(vs_train)], df[df[col].isin(vs_test)]


def train_test_split_on_values(df: pd.DataFrame, col: str):
    return {v: df[df[col] == v] for v in df[col].unique()}


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


def split_values(
        df: pd.DataFrame, col: str, to_array: bool = True,
        dtype=np.int, sep=',', conv_to=None) -> pd.DataFrame:
    def split(vs):
        if not isinstance(vs, str):
            return vs
        _vs = vs.split(sep)
        if conv_to:
            _vs = list(map(conv_to, _vs))
        if to_array:
            _vs = np.array(_vs, dtype=dtype)
        return _vs

    df[col] = df[col].apply(split)
    return df


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


if __name__ == '__main__':
    raise RuntimeError
