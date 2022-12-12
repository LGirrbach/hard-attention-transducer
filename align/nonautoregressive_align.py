import torch

from tqdm.auto import tqdm
from settings import Settings
from typing import List
from vocabulary import SourceVocabulary
from vocabulary import TransducerVocabulary
from typing import Optional
from models.base import TransducerModel
from dataset import RawDataset
from dataset import NonAutoregressiveTransducerDatasetTrain
from torch.utils.data import DataLoader
from trainer import _non_autoregressive_get_scores
from loss import non_autoregressive_transduction_loss
from inference import AlignmentPosition
from actions import Deletion, Copy, CopyShift, Insertion, Substitution


def decode_backpointers(tau: int, source: List[str], target: List[str], backpointers) -> List[AlignmentPosition]:

    def make_alignment(source_index: int):
        return {"source_symbol": source[source_index], "target_symbols": [], "actions": []}

    def get_source_symbol_index(source_index: int):
        return max(0, (source_index + 1) // tau)

    i, j = tau * len(source) - 1, len(target)

    aligned_positions = []
    current_alignment = None  # make_alignment(i-1)

    while i >= 0:
        operation, offset = backpointers[i][j]
        i += offset

        if operation != Deletion:
            j -= 1

        if operation == Insertion:
            current_alignment["target_symbols"].append(target[j])
            current_alignment["actions"].append(Insertion(target[j]))

        elif operation == Copy:
            if current_alignment:
                current_alignment["target_symbols"].append(target[j])
            current_alignment["actions"].append(Copy())

        elif operation == Deletion:
            if current_alignment:
                aligned_positions.append(current_alignment)
            current_alignment = make_alignment(get_source_symbol_index(source_index=i))

            current_alignment["actions"].append(Deletion())

        elif operation == Substitution:
            if current_alignment:
                aligned_positions.append(current_alignment)
            current_alignment = make_alignment(get_source_symbol_index(source_index=i))

            current_alignment["target_symbols"].append(target[j])
            current_alignment["actions"].append(Substitution(target[j]))

        elif operation == CopyShift:
            if current_alignment:
                aligned_positions.append(current_alignment)
            current_alignment = make_alignment(get_source_symbol_index(source_index=i))

            current_alignment["target_symbols"].append(target[j])
            current_alignment["actions"].append(CopyShift())

    aligned_positions.append(current_alignment)
    aligned_positions = [
        AlignmentPosition(
            symbol=position["source_symbol"],
            actions=list(reversed(position["actions"])),
            predictions=list(reversed(position["target_symbols"]))
        )
        for position in reversed(aligned_positions)
    ]

    return aligned_positions


def non_autoregressive_align(settings: Settings, model: TransducerModel, source_vocabulary: SourceVocabulary,
                             target_vocabulary: TransducerVocabulary, sources: List[List[str]],
                             targets: List[List[str]], feature_vocabulary: Optional[SourceVocabulary] = None,
                             features: Optional[List[str]] = None):
    assert not (settings.enforce_copy and not settings.allow_copy), "Cannot both enforce copy and disable copying"

    model = model.eval()
    model = model.to(model.device)

    original_is_grad_enabled = torch.is_grad_enabled()
    torch.set_grad_enabled(mode=False)

    # Prepare Data
    dataset = RawDataset(sources=sources, targets=targets, features=features)
    dataset = NonAutoregressiveTransducerDatasetTrain(
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
        scores, tau = _non_autoregressive_get_scores(model=model, batch=batch)
        # Run Viterbi algorithm
        backpointers, batch_alignment_scores = non_autoregressive_transduction_loss(
            scores=scores, source_lengths=batch.source_lengths, target_lengths=batch.target_lengths,
            insertion_labels=batch.insertion_labels, substitution_labels=batch.substitution_labels,
            copy_index=batch.copy_index, copy_shift_index=batch.copy_shift_index,
            deletion_index=batch.deletion_index, noop_index=batch.noop_index,
            copy_matrix=batch.copy_matrix, device=settings.device, allow_copy=settings.allow_copy,
            enforce_copy=settings.enforce_copy, tau=tau, noop_discount=settings.noop_discount, reduction="none",
            return_backpointers=True
        )

        for batch_elem_index, (source, target) in enumerate(zip(batch.raw_sources, batch.raw_targets)):
            batch_elem_backpointers = [
                [(None, 0) for _ in range(len(target) + 1)] for _ in range(tau * len(source))
            ]
            for i in range(tau * len(source)):
                for j in range(len(target) + 1):
                    batch_elem_backpointers[i][j] = backpointers[i][batch_elem_index][j]

            alignment = decode_backpointers(
                tau=tau, source=source, target=target, backpointers=batch_elem_backpointers
            )
            alignments.append(alignment)

        alignment_scores.extend(batch_alignment_scores.cpu().tolist())

    # Reset grad mode
    torch.set_grad_enabled(mode=original_is_grad_enabled)

    return {"alignments": alignments, "scores": alignment_scores}