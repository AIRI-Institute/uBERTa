import typing as t
from itertools import product
from pathlib import Path
from typing import Dict

from transformers import PreTrainedTokenizer

SPECIAL = ('[PAD]', '[UNK]', '[CLS]', '[SEP]', '[MASK]')
ALPHABET = ('A', 'T', 'C', 'G')


def construct_vocab(kmer: int):
    vocab = [*SPECIAL, *map(lambda x: "".join(x), product(ALPHABET, repeat=kmer))]
    return {c: i for i, c in enumerate(vocab)}


class DNATokenizer(PreTrainedTokenizer):
    def __init__(self, kmer: int, vocab: t.Optional[t.Mapping[str, int]] = None, **kwargs):

        pad, unk, cls, sep, msk = SPECIAL

        super().__init__(
            unk_token=unk,
            sep_token=sep,
            pad_token=pad,
            cls_token=cls,
            mask_token=msk,
            **kwargs,
        )
        self.max_len_single_sentence = self.max_len - 2  # take into account special tokens
        self.do_lower_case = False
        self.vocab = vocab or construct_vocab(kmer)
        self.kmer = kmer

    def get_vocab(self) -> Dict[str, int]:
        return self.vocab

    @property
    def vocab_size(self):
        return len(self.vocab)

    def _tokenize(self, text, **kwargs):
        if isinstance(text, str):
            text = text.split()
        return [self.vocab[tok] for tok in text]

    def _convert_token_to_id(self, token):
        """ Converts a token (str) in an id using the vocab. """
        return self.vocab.get(token, self.vocab.get(self.unk_token))

    def _convert_id_to_token(self, index):
        """Converts an index (integer) in a token (str) using the vocab."""
        return self.ids_to_tokens.get(index, self.unk_token)

    def convert_tokens_to_string(self, tokens):
        """ Converts a sequence of tokens (string) in a single string. """
        out_string = " ".join(tokens).replace(" ##", "").strip()
        return out_string

    def build_inputs_with_special_tokens(self, token_ids_0, token_ids_1=None) -> t.List[int]:
        assert token_ids_1 is None and len(token_ids_0) < 510
        return [self.vocab['[CLS]'], *token_ids_0, self.vocab['[SEP]']]

    def get_special_tokens_mask(self, token_ids_0, token_ids_1=None, already_has_special_tokens=False) -> t.List[int]:
        assert token_ids_1 is None and len(token_ids_0) < 510
        if already_has_special_tokens:
            return [1 if x in SPECIAL else 0 for x in token_ids_0]
        return [1] + ([0] * len(token_ids_0)) + [1]

    def create_token_type_ids_from_sequences(self, token_ids_0, token_ids_1=None) -> t.List[int]:
        return (len(token_ids_0) + 2) * [0]

    def save_vocabulary(self, vocab_path: t.Union[Path, str], **kwargs) -> Path:
        if not isinstance(vocab_path, Path):
            vocab_path = Path(vocab_path)
        vocab_file = vocab_path / 'vocab.txt' if vocab_path.is_dir() else vocab_path
        with vocab_file.open('w') as f:
            print(*self.vocab.keys(), sep='\n', file=f)

        return vocab_file


if __name__ == '__main__':
    raise RuntimeError
