import sys
from collections import namedtuple

VALID_CHROM = (*(f'chr{i}' for i in range(23)), 'chrX', 'chrY')
VALID_CHROM_FLANKS = {
    'chr1': (10000, 10000),
    'chr2': (10000, 10000),
    'chr3': (10000, 60000),
    'chr4': (10000, 10000),
    'chr5': (10000, 60000),
    'chr6': (60000, 60000),
    'chr7': (10000, 10000),
    'chr8': (60000, 60000),
    'chr9': (10000, 60000),
    'chr10': (10000, 10000),
    'chr11': (60000, 10000),
    'chr12': (10000, 10000),
    'chr13': (16000000, 10000),
    'chr14': (16000000, 160000),
    'chr15': (17000000, 10000),
    'chr16': (10000, 110000),
    'chr17': (60000, 10000),
    'chr18': (10000, 110000),
    'chr19': (60000, 10000),
    'chr20': (60000, 110000),
    'chr21': (5010000, 10000),
    'chr22': (10510000, 10000),
    'chrX': (10000, 10000),
    'chrY': (10000, 10000)
}

_ColNames = namedtuple(
    'ColNames',
    ['chrom', 'start', 'end', 'codon', 'strand', 'group', 'level', 'positive'])
_defaults = [
    'Chrom', 'StartCodonStart', 'StartCodonEnd', 'StartCodonFetched',
    'Strand', 'Group', 'LevelStartCodonStartFetchedAround2', 'IsPositive']
if sys.version < "3.7":
    ColNames = lambda: _ColNames(*_defaults)
else:
    ColNames = namedtuple(
        'ColNames',
        ['chrom', 'start', 'end', 'codon', 'strand', 'group', 'level', 'positive'],
        defaults=_defaults
    )

if __name__ == '__main__':
    raise RuntimeError()
