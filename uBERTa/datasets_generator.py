import logging
import typing as t
from itertools import count, starmap
from pathlib import Path
from random import choice, randint

import pandas as pd
import pysam
from Bio.Seq import Seq
from more_itertools import sliding_window
from toolz import curry, identity
from tqdm.auto import tqdm

from uBERTa.base import VALID_CHROM, VALID_CHROM_FLANKS, ColNames

LOGGER = logging.getLogger(__name__)


class DatasetGenerator:
    """
    An interface class to generate (u)ORF datasets.

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

    >>> from pathlib import Path
    >>> ds_path, ref_path = Path('/path/to/dataset'), Path('path/to/ref')
    >>> datagen = DatasetGenerator(ds_path, ref_path, 2, (0, 0.8, 0.2, 0), flank_size=50)
    >>> dataset = datagen()
    """
    def __init__(self, ds_path: Path, ref_path: Path, neg_multiplier: int,
                 neg_fractions: t.Tuple[float, float, float, float],
                 pos_fractions: t.Tuple[float, float, float] = (1.0, 1.0, 1.0),
                 level_ts: float = 0,
                 flank_size: int = 100,
                 kmer_size: t.Optional[int] = None,
                 use_analyzed: bool = True,
                 genes_of_pos: bool = False,
                 col_names: ColNames = ColNames(),
                 drop_meta: bool = False):
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
        :param drop_meta: Leave only two columns: "Seq" and "Class", drop the rest.
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
        self.drop_meta = drop_meta
        self.last_generated: t.Optional[pd.DataFrame] = None

    def __call__(self) -> pd.DataFrame:
        df = self._prefilter()
        samples = sample_dataset(
            self.neg_multiplier, self.ref, df,
            self.neg_fractions, self.pos_fractions,
            self.level_ts, self.col_names
        )
        seqs = sampled2train(
            samples, self.ref, self.flank_size,
            self.col_names, self.kmer_size)
        if self.drop_meta:
            seqs = seqs[['Seq', 'IsPositive']]
        self.last_generated = seqs
        return seqs

    def _prefilter(self):
        df = self.base_ds.copy()
        LOGGER.info(f'Pre-filtering initial dataset with {len(df)} records')
        if self.use_analyzed:
            df = df[df[self.col_names.analyzed]]
            LOGGER.info(f'Filtered to {len(df)} records of analyzed genes')
        if self.genes_of_pos:
            idx = (df[self.col_names.analyzed] &
                   df[self.col_names.group.isin(['m', 'ma', 'u'])])
            ids = set(df.loc[df[idx, self.col_names.gene_id]])
            df = df[df[self.col_names.gene_id].isin(ids)]
            LOGGER.info(f'Filtered to {len(df)} records of analyzed genes '
                        f'with at least one positive example')
        return df


def reverse_complement(s: str):
    return Seq(s).reverse_complement()


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
            seq = str(reverse_complement(seq))
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
            offset = -2 if strand == '-' else 1
            i_start = idx_codon + offset
            seq = self.fetch_around(chrom, i_start, strand, flank_size)

        return chrom, idx_codon, strand, length, seq


def sample_dataset(
        neg_multiplier, ref: Ref, ds: pd.DataFrame,
        neg_fractions: t.Tuple[float, float, float, float],
        pos_fractions: t.Tuple[float, float, float] = (1.0, 1.0, 1.0),
        level_ts: float = 0,
        col_names: ColNames = ColNames(),
        random_state: t.Optional[int] = None
) -> pd.DataFrame:
    LOGGER.info(f'Obtained a dataset with {len(ds)} records')

    idx_pos = ds[col_names.group].isin(['m', 'u', 'ma'])
    ds_pos, ds_neg = ds[idx_pos], ds[~idx_pos]
    LOGGER.info(f'Found {len(ds_pos)} positive and {len(ds_neg)} (potential) negative records')

    samples_pos = sample_pos(pos_fractions, ds_pos)
    LOGGER.info(f'Sampled {len(samples_pos)} positive examples')

    samples_neg = sample_neg(neg_multiplier, neg_fractions, samples_pos, ds_neg,
                             ref, level_ts, col_names, random_state)
    LOGGER.info(f'Sampled {len(samples_neg)} negative examples')

    samples_pos = samples_pos[[col_names.chrom, col_names.start, col_names.strand,
                               col_names.codon, col_names.group]]
    samples_pos[col_names.positive] = True
    samples_neg[col_names.positive] = False

    samples = pd.concat([samples_pos, samples_neg])
    LOGGER.info(f'Concatenated into {len(samples)} total samples')

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
        LOGGER.info(
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
        LOGGER.info(f'Found {len(samples)} codons {codon}')
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
        LOGGER.info(f'Found {len(sub)} records with codon {codon} '
                    f'and max level {max_level}')
        if num > len(sub):
            LOGGER.warning(f'The number of desired samples {num} '
                           f'exceeds the number of existing ones with '
                           f'codon {codon} and max level {max_level}')
        elif num == len(sub):
            pass
        else:
            sub = sub.sample(min([num, len(sub)]), random_state=random_state)
            LOGGER.info(f'Sampled {len(sub)} samples with codon {codon} '
                        f'and max level {max_level}')
        return sub[sel_columns].copy()

    sel_columns = [col_names.chrom, col_names.start,
                   col_names.strand, col_names.codon]

    sum_neg = sum(neg_fractions)
    if sum_neg != 1:
        raise ValueError(f'The `neg_fractions` {neg_fractions} '
                         f'must sum to 1.0; got {sum_neg}')
    neg_total = len(df_pos) * neg_multiplier
    LOGGER.info(f'Expecting around {neg_total} total negative examples')

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
    LOGGER.info(f'Num samples after the per-codon correction: {expected_samples}')

    random_samples = pd.concat(starmap(
        sample_random,
        samples_per_codon['random'].items()))
    random_samples[col_names.group] = 'random'
    LOGGER.info(f'Sampled {len(random_samples)} random start codons')

    valid_samples_below = pd.concat(starmap(
        sample_valid(max_level=level_ts),
        samples_per_codon['below_level'].items()))
    valid_samples_below[col_names.group] = 'below_level'
    LOGGER.info(f'Sampled {len(valid_samples_below)} with level <= {level_ts}')

    valid_samples_above = pd.concat(starmap(
        sample_valid(min_level=level_ts + 0.01),
        samples_per_codon['above_level'].items()))
    valid_samples_above[col_names.group] = 'above_level'
    LOGGER.info(f'Sampled {len(valid_samples_above)} with level > {level_ts + 0.01}')

    return pd.concat(
        [random_samples, valid_samples_below, valid_samples_above],
        ignore_index=True)


def sampled2train(
        ds: pd.DataFrame, ref: Ref, flank_size: int,
        col_names: ColNames = ColNames(),
        kmer_size: t.Optional[int] = None, kmer_sep: str = ' '
) -> pd.DataFrame:
    def prepare_seq(chrom, pos, strand):
        offset = -2 if strand == '-' else 1
        pos = pos + offset
        seq = ref.fetch_around(chrom, pos, strand, flank_size)
        if kmer_size:
            seq = kmer_sep.join(map(
                lambda s: "".join(s), sliding_window(seq, kmer_size)))
        return seq.upper()

    names = [col_names.chrom, col_names.start, col_names.strand, col_names.positive]
    # it = tqdm(ds[names].itertuples(), total=len(ds), desc='Fetching seqs')
    it = ds[names].itertuples()

    return pd.DataFrame(
        ((chrom, start, strand, int(pos), prepare_seq(chrom, start, strand))
         for _, chrom, start, strand, pos in it),
        columns=names + ['Seq']
    )


if __name__ == '__main__':
    raise RuntimeError
