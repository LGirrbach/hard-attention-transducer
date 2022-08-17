import torch

from abc import ABC
from typing import List
from typing import Optional
from typing import Iterable
from collections import namedtuple
from torch.utils.data import Dataset
from vocabulary import SourceVocabulary
from torch.nn.utils.rnn import pad_sequence
from vocabulary import TransducerVocabulary

Batch = namedtuple(
    "Batch",
    [
        "sources", "targets", "insertion_labels", "substitution_labels", "copy_matrix", "source_lengths",
        "target_lengths", "copy_index", "deletion_index", "noop_index", "raw_sources", "raw_targets",
        "features", "feature_lengths", "raw_features"
    ]
)

RawDataset = namedtuple("RawDataset", ["sources", "targets", "features"])
RawBatchElement = namedtuple("RawBatch", ["source", "target", "features"])


class TransducerDatasetTrain(Dataset, ABC):
    def __init__(self, dataset: RawDataset, source_vocabulary: SourceVocabulary,
                 target_vocabulary: TransducerVocabulary, feature_vocabulary: Optional[SourceVocabulary] = None,
                 use_features: bool = False):
        super(TransducerDatasetTrain, self).__init__()

        sources: List[List[str]] = dataset.sources
        targets: List[List[str]] = dataset.targets
        features: Optional[List[List[str]]] = dataset.features

        self.use_features = use_features

        assert len(sources) == len(targets)
        if self.use_features:
            assert len(sources) == len(features)

        self.sources = sources
        self.targets = targets
        self.features = features
        self.source_vocabulary = source_vocabulary
        self.target_vocabulary = target_vocabulary
        self.feature_vocabulary = feature_vocabulary

        self.target_copy_index = self.target_vocabulary.get_copy_index()
        self.target_deletion_index = self.target_vocabulary.get_deletion_index()
        self.target_noop_index = self.target_vocabulary.get_noop_index()

    def __len__(self) -> int:
        return len(self.sources)

    def __getitem__(self, idx: int) -> RawBatchElement:
        if not self.use_features:
            return RawBatchElement(source=self.sources[idx], target=self.targets[idx], features=None)
        else:
            return RawBatchElement(source=self.sources[idx], target=self.targets[idx], features=self.features[idx])

    def _add_special_tokens_source(self, sequence: List[str]) -> List[str]:
        raise NotImplementedError

    def _add_special_tokens_target(self, sequence: List[str]) -> List[str]:
        raise NotImplementedError

    @staticmethod
    def _make_copy_matrix(source: List[str], target: List[str], batch_source_length: int, batch_target_length: int):
        raise NotImplementedError

    def collate_fn(self, batch: Iterable[RawBatchElement]) -> Batch:
        sources = [batch_element.source for batch_element in batch]
        targets = [batch_element.target for batch_element in batch]
        features = [batch_element.features for batch_element in batch]

        # Add SOS and EOS special tokens to source and target sequences
        sources = [self._add_special_tokens_source(source) for source in sources]
        targets = [self._add_special_tokens_target(target) for target in targets]

        # Calculate lengths of source and target sequences
        source_lengths = torch.tensor([len(sequence) for sequence in sources]).long()
        target_lengths = torch.tensor([len(sequence) for sequence in targets]).long()

        # Index source sequences
        indexed_sources = [
            torch.tensor(self.source_vocabulary.index_sequence(sequence)).long() for sequence in sources
        ]
        indexed_sources = pad_sequence(indexed_sources, padding_value=0, batch_first=True)

        # Index target sequences
        indexed_targets = [
            torch.tensor(self.target_vocabulary.index_sequence(sequence)).long() for sequence in targets
        ]
        indexed_targets = pad_sequence(indexed_targets, padding_value=0, batch_first=True)

        # Get substitution indices
        substitution_labels = [
            [self.target_vocabulary.get_substitution_index(symbol) for symbol in target] for target in targets
        ]
        substitution_labels = [torch.tensor(labels).long() for labels in substitution_labels]
        substitution_labels = pad_sequence(substitution_labels, padding_value=0, batch_first=True)

        # Get insertion labels
        insertion_labels = [
            [self.target_vocabulary.get_insertion_index(symbol) for symbol in target] for target in targets
        ]
        insertion_labels = [torch.tensor(labels).long() for labels in insertion_labels]
        insertion_labels = pad_sequence(insertion_labels, padding_value=0, batch_first=True)

        # Make copy matrix
        batch_source_length = indexed_sources.shape[1]
        batch_target_length = indexed_targets.shape[1]

        batch_copy_matrix = []
        for source, target in zip(sources, targets):
            copy_matrix = self._make_copy_matrix(source, target, batch_source_length, batch_target_length)
            batch_copy_matrix.append(copy_matrix)

        batch_copy_matrix = torch.stack(batch_copy_matrix)

        # Index features
        if not self.use_features:
            indexed_features = None
            feature_lengths = None
        else:
            features = [
                [self.feature_vocabulary.SOS_TOKEN] + feats + [self.feature_vocabulary.EOS_TOKEN]
                for feats in features
            ]
            indexed_features = [self.feature_vocabulary.index_sequence(feats) for feats in features]
            indexed_features = [torch.tensor(feats).long() for feats in indexed_features]
            indexed_features = pad_sequence(indexed_features, batch_first=True, padding_value=0)
            feature_lengths = torch.tensor([len(feats) for feats in features]).long()

        return Batch(
            sources=indexed_sources, targets=indexed_targets, insertion_labels=insertion_labels,
            substitution_labels=substitution_labels, copy_matrix=batch_copy_matrix, source_lengths=source_lengths,
            target_lengths=target_lengths, copy_index=self.target_copy_index,
            deletion_index=self.target_deletion_index, noop_index=self.target_noop_index,
            raw_sources=sources, raw_targets=targets, features=indexed_features, feature_lengths=feature_lengths,
            raw_features=features
        )


class AutoregressiveTransducerDatasetTrain(TransducerDatasetTrain):
    def _add_special_tokens_source(self, sequence: List[str]) -> List[str]:
        sequence = [self.source_vocabulary.SOS_TOKEN] + sequence
        sequence = sequence + [self.source_vocabulary.EOS_TOKEN, self.source_vocabulary.EOS_TOKEN]
        return sequence

    def _add_special_tokens_target(self, sequence: List[str]) -> List[str]:
        sequence = [self.target_vocabulary.SOS_TOKEN, self.target_vocabulary.SOS_TOKEN] + sequence
        sequence = sequence + [self.target_vocabulary.EOS_TOKEN]
        return sequence

    @staticmethod
    def _make_copy_matrix(source: List[str], target: List[str], batch_source_length: int, batch_target_length: int):
        copy_matrix = torch.zeros(batch_source_length, batch_target_length, dtype=torch.bool)
        for source_index, source_symbol in enumerate(source):
            for target_index, target_symbol in enumerate(target[1:]):
                if source_symbol == target_symbol:
                    copy_matrix[source_index, target_index] = True

        return copy_matrix


class NonAutoregressiveTransducerDatasetTrain(TransducerDatasetTrain):
    def _add_special_tokens_source(self, sequence: List[str]) -> List[str]:
        sequence = [self.source_vocabulary.SOS_TOKEN] + sequence + [self.source_vocabulary.EOS_TOKEN]
        return sequence

    def _add_special_tokens_target(self, sequence: List[str]) -> List[str]:
        sequence = [self.target_vocabulary.SOS_TOKEN] + sequence + [self.target_vocabulary.EOS_TOKEN]
        return sequence

    @staticmethod
    def _make_copy_matrix(source: List[str], target: List[str], batch_source_length: int, batch_target_length: int):
        copy_matrix = torch.zeros(batch_source_length, batch_target_length, dtype=torch.bool)
        for source_index, source_symbol in enumerate(source):
            for target_index, target_symbol in enumerate(target):
                if source_symbol == target_symbol:
                    copy_matrix[source_index, target_index] = True

        return copy_matrix
