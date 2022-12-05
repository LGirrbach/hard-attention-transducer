from __future__ import annotations

from typing import List
from collections import Counter
from actions import Copy, CopyShift, Deletion, Substitution, Insertion, Action, Noop

COPY = Copy()
COPY_SHIFT = CopyShift()
DELETION = Deletion()
NOOP = Noop()
FIXED_ACTIONS = [COPY, COPY_SHIFT, DELETION, NOOP]


class SourceVocabulary:
    PAD_TOKEN = "<PAD>"
    UNK_TOKEN = "<UNK>"
    SOS_TOKEN = "<SOS>"
    EOS_TOKEN = "<EOS>"

    def __init__(self, symbols: List[str]):
        self.specials = self.get_specials()
        self.tokens = self.specials + list(sorted(set(symbols)))

        self.token2idx = {token: idx for idx, token in enumerate(self.tokens)}
        self.idx2token = {idx: token for idx, token in enumerate(self.tokens)}

        self.unk_idx = self.token2idx[self.UNK_TOKEN]

    def get_specials(self) -> List[str]:
        return [self.PAD_TOKEN, self.UNK_TOKEN, self.SOS_TOKEN, self.EOS_TOKEN]

    def __len__(self):
        return len(self.tokens)

    def __getitem__(self, idx: int) -> str:
        return self.tokens[idx]

    def is_special(self, token: str):
        return token in self.specials

    def index_sequence(self, tokens: List[str]) -> List[int]:
        return [self.token2idx.get(token, self.unk_idx) for token in tokens]

    def convert_idx(self, idx: List[int]) -> List[str]:
        return [self.idx2token.get(index, self.UNK_TOKEN) for index in idx]

    @classmethod
    def build_vocabulary(cls, sequences: List[List[str]], min_frequency: int = 1) -> SourceVocabulary:
        all_symbols = []
        for sequence in sequences:
            all_symbols.extend(sequence)

        symbol_counts = Counter(all_symbols)
        filtered_symbols = list(sorted([symbol for symbol, count in symbol_counts.items() if count >= min_frequency]))
        return cls(symbols=filtered_symbols)


class TransducerVocabulary:
    PAD_TOKEN = "<PAD>"
    UNK_TOKEN = "<UNK>"
    SOS_TOKEN = "<SOS>"
    EOS_TOKEN = "<EOS>"

    def __init__(self, symbols: List[str]):
        self.symbols = self.get_special_symbols() + list(sorted(set(symbols)))
        self.symbol2idx = {symbol: idx for idx, symbol in enumerate(self.symbols)}
        self.idx2symbol = {idx: symbol for idx, symbol in enumerate(self.symbols)}

        self.insertions: List[Insertion] = [Insertion(token) for token in self.symbols]
        self.substitutions: List[Substitution] = [Substitution(token) for token in self.symbols]
        self.actions: List[Action] = self.insertions + self.substitutions + self.get_special_actions()

        self.insertion2idx = {action.token: idx for idx, action in enumerate(self.insertions)}
        self.substitution2idx = {
            action.token: len(self.insertions) + idx for idx, action in enumerate(self.substitutions)
        }

    def get_special_symbols(self) -> List[str]:
        return [self.PAD_TOKEN, self.UNK_TOKEN, self.SOS_TOKEN, self.EOS_TOKEN]

    @staticmethod
    def get_special_actions() -> List[Action]:
        return FIXED_ACTIONS

    def __len__(self) -> int:
        return len(self.actions)

    def __getitem__(self, idx: int) -> Action:
        return self.actions[idx]

    def get_insertion_index(self, symbol: str) -> int:
        return self.insertion2idx.get(symbol, 1)

    def get_substitution_index(self, symbol: str) -> int:
        return self.substitution2idx.get(symbol, len(self.insertions) + 1)

    def get_deletion_index(self) -> int:
        return len(self.actions) - len(FIXED_ACTIONS) + FIXED_ACTIONS.index(DELETION)

    def get_copy_index(self) -> int:
        return len(self.actions) - len(FIXED_ACTIONS) + FIXED_ACTIONS.index(COPY)

    def get_copy_shift_index(self) -> int:
        return len(self.actions) - len(FIXED_ACTIONS) + FIXED_ACTIONS.index(COPY_SHIFT)

    def get_noop_index(self) -> int:
        return len(self.actions) - len(FIXED_ACTIONS) + FIXED_ACTIONS.index(NOOP)

    def get_symbol_index(self, symbol: str) -> int:
        return self.symbol2idx.get(symbol, self.symbol2idx[self.UNK_TOKEN])

    def index_sequence(self, symbols: List[str]) -> List[int]:
        return [self.get_symbol_index(symbol) for symbol in symbols]

    @classmethod
    def build_vocabulary(cls, sequences: List[List[str]], min_frequency: int = 1) -> TransducerVocabulary:
        all_symbols = []
        for sequence in sequences:
            all_symbols.extend(sequence)

        symbol_counts = Counter(all_symbols)
        filtered_symbols = list(sorted([symbol for symbol, count in symbol_counts.items() if count >= min_frequency]))
        return cls(symbols=filtered_symbols)


class Seq2SeqVocabulary:
    PAD_TOKEN = "<PAD>"
    UNK_TOKEN = "<UNK>"
    SOS_TOKEN = "<SOS>"
    EOS_TOKEN = "<EOS>"

    def __init__(self, symbols: List[str]):
        self.symbols = self.get_special_symbols() + list(sorted(set(symbols)))
        self.symbol2idx = {symbol: idx for idx, symbol in enumerate(self.symbols)}
        self.idx2symbol = {idx: symbol for idx, symbol in enumerate(self.symbols)}

    def get_special_symbols(self) -> List[str]:
        return [self.PAD_TOKEN, self.UNK_TOKEN, self.SOS_TOKEN, self.EOS_TOKEN]

    def __len__(self) -> int:
        return len(self.symbols)

    def __getitem__(self, idx: int) -> str:
        return self.symbols[idx]

    def get_symbol_index(self, symbol: str) -> int:
        return self.symbol2idx.get(symbol, self.symbol2idx[self.UNK_TOKEN])

    def index_sequence(self, symbols: List[str]) -> List[int]:
        return [self.get_symbol_index(symbol) for symbol in symbols]

    @classmethod
    def build_vocabulary(cls, sequences: List[List[str]]) -> Seq2SeqVocabulary:
        all_tokens = list(set.union(*(set(sequence) for sequence in sequences)))
        return cls(symbols=all_tokens)
