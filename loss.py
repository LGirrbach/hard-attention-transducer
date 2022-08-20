import torch

from torch import Tensor
from typing import Optional


def _loss_reduction(loss: Tensor, reduction: str) -> Tensor:
    if reduction == "mean":
        return loss.mean()
    elif reduction == "sum":
        return loss.sum()
    elif reduction == "none":
        return loss
    else:
        raise ValueError(f"Invalid reduction: {reduction} (must be 'mean', 'sum', or 'none')")


def autoregressive_transduction_loss(scores: Tensor, source_lengths: Tensor, target_lengths: Tensor,
                                     insertion_labels: Tensor, substitution_labels: Tensor, copy_index: int,
                                     copy_shift_index: int, deletion_index: int, copy_matrix: Tensor,
                                     device: torch.device, allow_copy: bool = True, enforce_copy: bool = False,
                                     reduction: str = "mean") -> Tensor:
    assert not (enforce_copy and not allow_copy), "Cannot both enforce copy and disable copying"
    # scores: batch x source length x target length x #labels

    probability_matrix = [
        [torch.zeros(scores.shape[0], device=device) for _ in range(scores.shape[2])] for _ in range(scores.shape[1])
    ]
    batch_indices = torch.arange(0, scores.shape[0])
    copy_matrix = copy_matrix.to(device)

    for source_index in range(0, scores.shape[1]):
        for target_index in range(0, scores.shape[2]):
            if source_index == 0 and target_index == 0:
                continue

            probabilities = []

            # Calculate deletion scores
            if source_index > 0:
                deletion_scores = probability_matrix[source_index-1][target_index]
                deletion_scores = deletion_scores + scores[:, source_index-1, target_index, deletion_index]
                probabilities.append(deletion_scores)

            # Calculate insertion scores
            if target_index > 0:
                insertion_scores = probability_matrix[source_index][target_index - 1]
                insertion_prediction_scores =\
                    scores[batch_indices, source_index, target_index - 1, insertion_labels[:, target_index]]
                insertion_scores = insertion_scores + insertion_prediction_scores
                probabilities.append(insertion_scores)

            # Calculate substitution scores
            if source_index > 0 and target_index > 0:
                substitution_scores = probability_matrix[source_index - 1][target_index - 1]
                substitution_prediction_scores =\
                    scores[batch_indices, source_index - 1, target_index - 1, substitution_labels[:, target_index]]
                substitution_scores = substitution_scores + substitution_prediction_scores

                if enforce_copy:
                    can_copy_mask = copy_matrix[:, source_index-1, target_index-1]
                    substitution_scores = torch.masked_fill(substitution_scores, mask=can_copy_mask, value=-100)

                probabilities.append(substitution_scores)

            # Calculate copy scores
            if allow_copy and target_index > 0:
                copy_scores = probability_matrix[source_index][target_index - 1]
                copy_scores = copy_scores + scores[:, source_index, target_index-1, copy_index]

                copy_mask = torch.logical_not(copy_matrix[:, source_index, target_index-1])
                copy_scores = torch.masked_fill(copy_scores, mask=copy_mask, value=-100)

                probabilities.append(copy_scores)

            # Calculate copy shift scores
            if allow_copy and source_index > 0 and target_index > 0:
                copy_scores = probability_matrix[source_index][target_index - 1]
                copy_scores = copy_scores + scores[:, source_index, target_index - 1, copy_shift_index]

                copy_mask = torch.logical_not(copy_matrix[:, source_index, target_index - 1])
                copy_scores = torch.masked_fill(copy_scores, mask=copy_mask, value=-100)

                probabilities.append(copy_scores)

            probabilities = torch.stack(probabilities)
            probabilities = torch.logsumexp(probabilities, dim=0)
            probability_matrix[source_index][target_index] = probabilities

    probability_matrix = torch.stack([torch.stack(row) for row in probability_matrix])
    probability_matrix = torch.permute(probability_matrix, dims=(2, 0, 1))

    nll = -probability_matrix[batch_indices, source_lengths - 1, :][batch_indices, target_lengths - 1]
    loss = _loss_reduction(nll, reduction=reduction)

    return loss


def non_autoregressive_transduction_loss(scores: Tensor, source_lengths: Tensor, target_lengths: Tensor,
                                         insertion_labels: Tensor, substitution_labels: Tensor, copy_index: int,
                                         copy_shift_index: int, deletion_index: int, noop_index: int,
                                         copy_matrix: Tensor, tau: Optional[int], device: torch.device,
                                         allow_copy: bool = True, enforce_copy: bool = False,
                                         noop_discount: float = 1., reduction: str = "mean") -> Tensor:
    # scores: shape batch x timesteps x tau x #labels
    assert not (enforce_copy and not allow_copy), "Cannot both enforce copy and disable copying"

    batch_size = scores.shape[0]
    num_labels = scores.shape[-1]
    source_max_timesteps = tau * source_lengths.max().detach().cpu().item()
    target_max_timesteps = target_lengths.max().detach().cpu().item()

    probability_matrix = [
        [torch.full((batch_size,), fill_value=-100000, device=device) for _ in range(target_max_timesteps + 1)]
        for _ in range(source_max_timesteps + 1)
    ]
    probability_matrix[0][0] = torch.zeros(batch_size, device=device)

    batch_indices = torch.arange(0, batch_size)
    copy_matrix = copy_matrix.to(device)
    scores = scores.reshape(batch_size, -1, num_labels)

    for source_index in range(0, source_max_timesteps + 1):
        for target_index in range(0, target_max_timesteps + 1):
            probabilities = []
            source_element_index = source_index // tau

            # Process first prediction of input symbol differently than subsequent predictions
            # First prediction for symbol: Can only reach by delete, copy, substitute of previous symbol
            # Subsequent predictions: Can only reach by insertion from same symbol
            if source_index % tau == 0:
                for backward_arc in range(0, tau):
                    prev_symbol_start_index = source_index - tau
                    prev_position_index = prev_symbol_start_index + backward_arc
                    noop_scores = scores[:, prev_position_index+1:source_index, noop_index]
                    noop_scores = noop_scores.sum(dim=-1) / noop_discount

                    # Calculate deletion scores
                    if source_index > 0:
                        deletion_scores = probability_matrix[prev_position_index][target_index]
                        deletion_scores = deletion_scores + scores[:, prev_position_index, deletion_index]
                        deletion_scores = deletion_scores + noop_scores
                        probabilities.append(deletion_scores)

                    # Calculate substitution scores
                    if source_index > 0 and target_index > 0:
                        substitution_scores = probability_matrix[prev_position_index][target_index - 1]
                        current_substitution_labels = substitution_labels[:, target_index - 1]
                        substitution_prediction_scores = \
                            scores[batch_indices, prev_position_index, current_substitution_labels]

                        substitution_scores = substitution_scores + substitution_prediction_scores
                        substitution_scores = substitution_scores + noop_scores

                        if enforce_copy:
                            can_copy_mask = copy_matrix[:, source_element_index - 1, target_index - 1]
                            substitution_scores = torch.masked_fill(
                                substitution_scores, mask=can_copy_mask, value=-100000
                            )

                        probabilities.append(substitution_scores)

                    # Calculate copy scores
                    if allow_copy and source_index > 0 and target_index > 0:
                        copy_scores = probability_matrix[prev_position_index][target_index - 1]
                        copy_scores = copy_scores + scores[:, prev_position_index, copy_shift_index]
                        copy_scores = copy_scores + noop_scores

                        copy_mask = torch.logical_not(copy_matrix[:, source_element_index - 1, target_index - 1])
                        copy_scores = torch.masked_fill(copy_scores, mask=copy_mask, value=-100000)

                        probabilities.append(copy_scores)

            elif source_index % tau != 0 and target_index > 0:
                # We can only predict multiple symbols from one input symbol by making insertions or copy
                previous_scores = probability_matrix[source_index - 1][target_index - 1]

                current_insertion_labels = insertion_labels[:, target_index - 1]
                insertion_prediction_scores = scores[batch_indices, source_index - 1, current_insertion_labels]

                copy_scores = scores[batch_indices, source_index - 1, copy_index]
                copy_mask = torch.logical_not(copy_matrix[:, source_element_index, target_index - 1])
                copy_scores = torch.masked_fill(copy_scores, mask=copy_mask, value=-100000)

                probabilities.append(previous_scores + insertion_prediction_scores)
                probabilities.append(previous_scores + copy_scores)

            if len(probabilities) == 1:
                probability_matrix[source_index][target_index] = probabilities[0].flatten()

            if len(probabilities) > 0:
                probabilities = torch.stack(probabilities)
                probabilities = torch.logsumexp(probabilities, dim=0)
                probability_matrix[source_index][target_index] = probabilities

            else:
                continue

    probability_matrix = torch.stack([torch.stack(row) for row in probability_matrix])
    probability_matrix = torch.permute(probability_matrix, dims=(2, 0, 1))

    nll = -probability_matrix[batch_indices, tau * source_lengths, :][batch_indices, target_lengths]
    loss = _loss_reduction(nll, reduction=reduction)

    return loss
