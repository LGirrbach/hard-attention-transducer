import torch

from typing import List
from torch import Tensor
from dataset import Batch
from tqdm.auto import tqdm
from typing import Optional
from settings import Settings
from dataset import RawDataset
from inference import AlignmentPosition
from torch.utils.data import DataLoader
from models.base import TransducerModel
from vocabulary import SourceVocabulary
from vocabulary import TransducerVocabulary
from trainer import _autoregressive_get_scores
from loss import _autoregressive_slow_recursion
from dataset import AutoregressiveTransducerDatasetTrain
from actions import Deletion, Copy, CopyShift, Insertion, Substitution


def decode_backpointers(source: List[str], target: List[str], backpointers) -> List[AlignmentPosition]:
    def make_alignment(source_index: int):
        return {"source_symbol": source[source_index], "target_symbols": [], "actions": []}

    i, j = len(source), len(target) - 1

    aligned_positions = []
    current_alignment = None  # make_alignment(i-1)

    while i >= 0 or j >= 0:
        operation = backpointers[i][j]

        if operation == Insertion:
            current_alignment["target_symbols"].append(target[j])
            current_alignment["actions"].append(Insertion(target[j]))
            j -= 1

        elif operation == Copy:
            if current_alignment:
                current_alignment["target_symbols"].append(target[j])
            current_alignment["actions"].append(Copy())
            j -= 1

        elif operation == Deletion:
            if current_alignment:
                aligned_positions.append(current_alignment)
            current_alignment = make_alignment(i - 1)

            current_alignment["actions"].append(Deletion())
            i -= 1

        elif operation == Substitution:
            if current_alignment:
                aligned_positions.append(current_alignment)
            current_alignment = make_alignment(i - 1)

            current_alignment["target_symbols"].append(target[j])
            current_alignment["actions"].append(Substitution(target[j]))
            i, j = i-1, j-1

        elif operation == CopyShift:
            if current_alignment:
                aligned_positions.append(current_alignment)
            current_alignment = make_alignment(i - 1)

            current_alignment["target_symbols"].append(target[j])
            current_alignment["actions"].append(CopyShift())
            i, j = i - 1, j - 1

    aligned_positions = [
        AlignmentPosition(
            symbol=position["source_symbol"],
            actions=list(reversed(position["actions"])),
            predictions=list(reversed(position["target_symbols"]))
        )
        for position in reversed(aligned_positions)
    ]

    return aligned_positions


def autoregressive_batched_viterbi(scores: Tensor, batch: Batch, settings: Settings):
    # Define constants
    batch_size, source_length, target_length, num_labels = scores.shape
    neg_inf_value = -1e6

    batch_indices = torch.arange(0, batch_size)
    neg_inf_vector = torch.full(size=(batch_size,), fill_value=neg_inf_value, device=settings.device)
    dummy_vector_zeros = torch.zeros(batch_size, device=settings.device)

    copy_matrix = batch.copy_matrix.to(settings.device)
    insertion_labels = batch.insertion_labels.to(settings.device)
    substitution_labels = batch.substitution_labels.to(settings.device)

    # Initialise score matrix and backpointers
    probability_matrix = [[neg_inf_vector for _ in range(target_length)] for _ in range(source_length + 1)]
    backpointers = [
        [
            [CopyShift for _ in range(batch_size)]
            for _ in range(target_length)
        ]
        for _ in range(source_length + 1)
    ]

    # Fill in dynamic programming grid and backpointers
    for source_index in range(0, source_length + 1):
        for target_index in range(0, target_length):
            if source_index == 0 and target_index == 0:
                probability_matrix[source_index][target_index] = dummy_vector_zeros
                continue

            probabilities, operation_mapping = _autoregressive_slow_recursion(
                probability_matrix=probability_matrix, scores=scores, source_index=source_index,
                target_index=target_index, source_length=source_length, deletion_index=batch.deletion_index,
                copy_index=batch.copy_index, copy_shift_index=batch.copy_shift_index, insertion_labels=insertion_labels,
                substitution_labels=substitution_labels, copy_matrix=copy_matrix, allow_copy=settings.allow_copy,
                enforce_copy=settings.enforce_copy, batch_indices=batch_indices, skip_undefined=False,
                neg_inf_vector=neg_inf_vector, source_lengths=batch.source_lengths
            )

            best_operations = torch.max(probabilities, dim=0)
            best_operation_indices = best_operations.indices.cpu().flatten().tolist()
            best_operation_scores = best_operations.values

            probability_matrix[source_index][target_index] = best_operation_scores
            for batch_elem_index, operation_index in enumerate(best_operation_indices):
                backpointers[source_index][target_index][batch_elem_index] = operation_mapping[operation_index]

    unpacked_backpointers = []

    zipped_lengths = zip(batch.source_lengths.cpu().tolist(), batch.target_lengths.cpu().tolist())
    for batch_elem_index, (elem_source_length, elem_target_length) in enumerate(zipped_lengths):
        unpacked_backpointers.append(
            [
                [backpointers[i][j][batch_elem_index] for j in range(elem_target_length)]
                for i in range(elem_source_length + 1)
            ]
        )
    probability_matrix = torch.stack([torch.stack(row) for row in probability_matrix])
    probability_matrix = torch.permute(probability_matrix, dims=(2, 0, 1))
    scores = probability_matrix[batch_indices, batch.source_lengths, batch.target_lengths - 1]

    return scores, unpacked_backpointers


def autoregressive_align(settings: Settings, model: TransducerModel, source_vocabulary: SourceVocabulary,
                         target_vocabulary: TransducerVocabulary, sources: List[List[str]], targets: List[List[str]],
                         feature_vocabulary: Optional[SourceVocabulary] = None, features: Optional[List[str]] = None):
    assert not (settings.enforce_copy and not settings.allow_copy), "Cannot both enforce copy and disable copying"

    model = model.eval()
    model = model.to(model.device)

    original_is_grad_enabled = torch.is_grad_enabled()
    torch.set_grad_enabled(mode=False)

    # Prepare Data
    dataset = RawDataset(sources=sources, targets=targets, features=features)
    dataset = AutoregressiveTransducerDatasetTrain(
        dataset=dataset, source_vocabulary=source_vocabulary, target_vocabulary=target_vocabulary,
        feature_vocabulary=feature_vocabulary, use_features=settings.use_features
    )
    dataloader = DataLoader(
        dataset=dataset, batch_size=settings.batch, shuffle=False, num_workers=0, collate_fn=dataset.collate_fn
    )

    # Align dataset (batch by batch)
    alignments = []
    alignment_scores = []

    for batch in tqdm(dataloader, desc="Aligning Progress", total=len(dataloader)):
        # Get prediction scores
        scores = _autoregressive_get_scores(model=model, batch=batch)
        # Run Viterbi algorithm
        batch_alignment_scores, batch_unpacked_backpointers = autoregressive_batched_viterbi(
            scores=scores, batch=batch, settings=settings
        )

        for source, target, backpointers in zip(batch.raw_sources, batch.raw_targets, batch_unpacked_backpointers):
            alignment = decode_backpointers(source=source, target=target, backpointers=backpointers)
            alignments.append(alignment)

        alignment_scores.extend(batch_alignment_scores.cpu().tolist())

    # Reset grad mode
    torch.set_grad_enabled(mode=original_is_grad_enabled)

    return {"alignments": alignments, "scores": alignment_scores}
