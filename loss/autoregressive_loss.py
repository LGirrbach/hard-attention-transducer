import torch

from typing import List
from torch import Tensor
from util import make_mask_3d
from loss.loss_utils import shoot_nans
from loss.loss_utils import neg_inf_value
from loss.loss_utils import _loss_reduction
from actions import Deletion, Copy, CopyShift, Insertion, Substitution


def fast_autoregressive_transduction_loss(scores: Tensor, source_lengths: Tensor, target_lengths: Tensor,
                                          insertion_labels: Tensor, substitution_labels: Tensor, copy_index: int,
                                          copy_shift_index: int, deletion_index: int, copy_matrix: Tensor,
                                          device: torch.device, allow_copy: bool = True, enforce_copy: bool = False,
                                          reduction: str = "mean") -> Tensor:
    assert not (enforce_copy and not allow_copy), "Cannot both enforce copy and disable copying"
    # scores: batch x source length x target length x #labels

    # Define constants
    batch_size, source_length, target_length, num_labels = scores.shape
    neg_inf_batch_vector = torch.full((batch_size, 1), fill_value=neg_inf_value, device=device)
    expanded_neg_inf_batch_vector_1 = neg_inf_batch_vector.transpose(0, 1).unsqueeze(2).repeat(1, 1, target_length-1)
    expanded_neg_inf_batch_vector_2 = neg_inf_batch_vector.unsqueeze(0).repeat(target_length-1, 1, 1)

    probability_matrix = []

    batch_indices = torch.arange(0, batch_size)

    # Modify copy matrix for fast loss (index shift)
    cannot_copy_column = torch.zeros(copy_matrix.size(0), copy_matrix.size(1), 1, dtype=torch.bool)
    copy_matrix = torch.cat([copy_matrix[:, :, 1:], cannot_copy_column], dim=-1)
    copy_matrix = copy_matrix.to(device)

    # Make mask for insertions
    insertion_mask = make_mask_3d(source_lengths=source_lengths, target_lengths=target_lengths)
    # shape: [batch x #source symbols x # target symbols]
    insertion_mask = torch.cat([insertion_mask, torch.ones(batch_size, 1, target_length, dtype=torch.bool)], dim=1)
    insertion_mask = insertion_mask.to(device)

    # Make lower triangular mask
    triangular_mask = torch.ones(batch_size, target_length - 1, target_length - 1, device=device)
    triangular_mask = torch.logical_not(torch.triu(triangular_mask).bool())
    flipped_triangular_mask = triangular_mask.flip(dims=(1,))

    # Make substitution and insertion labels
    substitution_labels = substitution_labels[:, 1:].unsqueeze(2).to(device)
    insertion_labels = insertion_labels[:, 1:].unsqueeze(2).to(device)

    for source_index in range(0, source_length + 1):
        probabilities = []

        # Calculate deletion scores
        if source_index > 0:
            deletion_scores = probability_matrix[source_index - 1]
            deletion_scores = deletion_scores + scores[:, source_index - 1, :, deletion_index]
            probabilities.append(deletion_scores)

        # Calculate substitution scores
        if source_index > 0:
            substitution_prediction_scores = scores[:, source_index - 1, :-1, :]
            substitution_prediction_scores = torch.gather(
                substitution_prediction_scores, index=substitution_labels, dim=2
            )
            substitution_prediction_scores = substitution_prediction_scores.squeeze(2)

            substitution_scores = probability_matrix[source_index - 1][:, :-1]
            substitution_scores = substitution_scores + substitution_prediction_scores
            substitution_scores = torch.cat([neg_inf_batch_vector, substitution_scores], dim=1)

            if enforce_copy:
                can_copy_mask = copy_matrix[:, source_index - 1, :]
                substitution_scores = torch.masked_fill(
                    substitution_scores, mask=can_copy_mask, value=neg_inf_value
                )

            probabilities.append(substitution_scores)

        # Calculate copy shift scores
        if allow_copy and source_index > 0:
            copy_shift_prediction_scores = scores[:, source_index-1, :-1, copy_shift_index]
            copy_shift_scores = probability_matrix[source_index - 1][:, :-1]
            copy_shift_scores = copy_shift_scores + copy_shift_prediction_scores
            copy_shift_scores = torch.cat([neg_inf_batch_vector, copy_shift_scores], dim=1)

            copy_mask = torch.logical_not(copy_matrix[:, source_index - 1, :])
            copy_scores = torch.masked_fill(copy_shift_scores, mask=copy_mask, value=neg_inf_value)

            probabilities.append(copy_scores)

        # Combine probabilities
        if source_index > 0:
            probabilities = torch.stack(probabilities)
        else:
            probabilities = torch.cat(
                [torch.zeros([1, batch_size, 1], device=device), expanded_neg_inf_batch_vector_1], dim=2
            )

        # Calculate insertion scores
        if source_index < source_length:
            # Expand the row internal recursion (insert ops, from left to right)
            # 1. Get scores for insert op
            insertion_prediction_scores = scores[:, source_index, :-1, :]
            insertion_prediction_scores = torch.gather(
                insertion_prediction_scores, index=insertion_labels, dim=2
            )

            # 2. Build matrix that describes the weights partial probabilities in the row
            #    are multiplied with when added to later partial probabilities
            #    We expand the recursion, so we have to calculate cumulative sums (in log space)
            #    and zero out invalid terms
            #
            # 2.1. Copy scores across all timesteps
            insertion_prediction_scores = insertion_prediction_scores.repeat(1, 1, target_length-1)
            # 2.2. Zero out invalid terms (that do not appear in the expanded recursion)
            insertion_prediction_scores = torch.flip(insertion_prediction_scores, dims=(1,))
            insertion_prediction_scores = torch.masked_fill(
                insertion_prediction_scores, mask=flipped_triangular_mask, value=0.
            )
            # 2.3. Sum log-probs for each combination of timesteps
            insertion_prediction_scores = torch.cumsum(insertion_prediction_scores, dim=1)
            insertion_prediction_scores = torch.flip(insertion_prediction_scores, dims=(1,))
            # 2.4. Second masking: This time for real, not only to ensure correct behaviour of cumsum
            insertion_prediction_scores = torch.masked_fill(
                insertion_prediction_scores, mask=triangular_mask, value=neg_inf_value
            )

            # 2.5. Calculate expanded recursion
            if source_index > 0:
                partial_probabilities = torch.logsumexp(probabilities, dim=0)
            else:
                partial_probabilities = probabilities.squeeze(0)

            partial_probabilities = partial_probabilities[:, :-1]
            partial_probabilities = partial_probabilities.unsqueeze(2)
            partial_probabilities = partial_probabilities.repeat(1, 1, target_length-1)

            insertion_scores = insertion_prediction_scores + partial_probabilities
            insertion_scores = insertion_scores.transpose(0, 1)

            # 2.6. We don't calculate the true insertion scores but just add them in the logsumexp
            #      of all scores in order to avoid chained logsumexp
            insertion_scores = torch.cat([expanded_neg_inf_batch_vector_2, insertion_scores], dim=2)

            # 2.7. Mask out invalid insertion positions
            current_insertion_mask = insertion_mask[:, source_index, :]
            current_insertion_mask = current_insertion_mask.unsqueeze(0)
            current_insertion_mask = current_insertion_mask.repeat(insertion_scores.size(0), 1, 1)
            insertion_scores = torch.masked_fill(insertion_scores, mask=current_insertion_mask, value=neg_inf_value)

            probabilities = torch.cat([probabilities, insertion_scores], dim=0)

        probabilities = torch.logsumexp(probabilities, dim=0)
        probability_matrix.append(probabilities)

    probability_matrix = torch.stack(probability_matrix)
    probability_matrix = probability_matrix.transpose(0, 1)

    nll = -probability_matrix[batch_indices, source_lengths, target_lengths - 1]
    nll = shoot_nans(nll)
    loss = _loss_reduction(nll, reduction=reduction)

    return loss


def _autoregressive_slow_recursion(probability_matrix: List[List[Tensor]], scores: Tensor, source_index: int,
                                   target_index: int, source_length: int, deletion_index: int, copy_index: int,
                                   copy_shift_index: int, insertion_labels: Tensor, substitution_labels: Tensor,
                                   copy_matrix: Tensor, allow_copy: bool, enforce_copy: bool, batch_indices: Tensor,
                                   skip_undefined: bool, neg_inf_vector: Tensor, source_lengths: Tensor):
    probabilities = []

    not_can_copy_insert = torch.lt(source_lengths-1, source_index).to(scores.device)

    operation_mapping = dict()
    operation_counter = 0

    # Calculate deletion scores
    if source_index > 0:
        deletion_scores = probability_matrix[source_index - 1][target_index]
        deletion_scores = deletion_scores + scores[:, source_index - 1, target_index, deletion_index]
        probabilities.append(deletion_scores)

        operation_mapping[operation_counter] = Deletion
        operation_counter += 1

    elif not skip_undefined:
        probabilities.append(neg_inf_vector)

        operation_mapping[operation_counter] = Deletion
        operation_counter += 1

    # Calculate insertion scores
    if target_index > 0 and source_index < source_length:
        insertion_scores = probability_matrix[source_index][target_index - 1]
        insertion_prediction_scores = \
            scores[batch_indices, source_index, target_index - 1, insertion_labels[:, target_index]]
        insertion_scores = insertion_scores + insertion_prediction_scores
        insertion_scores = torch.masked_fill(insertion_scores, mask=not_can_copy_insert, value=neg_inf_value)
        probabilities.append(insertion_scores)

        operation_mapping[operation_counter] = Insertion
        operation_counter += 1

    elif not skip_undefined:
        probabilities.append(neg_inf_vector)

        operation_mapping[operation_counter] = Insertion
        operation_counter += 1

    # Calculate substitution scores
    if source_index > 0 and target_index > 0:
        substitution_scores = probability_matrix[source_index - 1][target_index - 1]
        substitution_prediction_scores = \
            scores[batch_indices, source_index - 1, target_index - 1, substitution_labels[:, target_index]]
        substitution_scores = substitution_scores + substitution_prediction_scores

        if enforce_copy:
            can_copy_mask = copy_matrix[:, source_index - 1, target_index - 1]
            substitution_scores = torch.masked_fill(
                substitution_scores, mask=can_copy_mask, value=neg_inf_value
            )

        probabilities.append(substitution_scores)

        operation_mapping[operation_counter] = Substitution
        operation_counter += 1

    elif not skip_undefined:
        probabilities.append(neg_inf_vector)

        operation_mapping[operation_counter] = Substitution
        operation_counter += 1

    # Calculate copy scores
    if allow_copy and target_index > 0 and source_index < source_length:
        copy_scores = probability_matrix[source_index][target_index - 1]
        copy_scores = copy_scores + scores[:, source_index, target_index - 1, copy_index]

        copy_mask = torch.logical_not(copy_matrix[:, source_index, target_index - 1])
        copy_scores = torch.masked_fill(copy_scores, mask=copy_mask, value=neg_inf_value)
        copy_scores = torch.masked_fill(copy_scores, mask=not_can_copy_insert, value=neg_inf_value)

        probabilities.append(copy_scores)

        operation_mapping[operation_counter] = Copy
        operation_counter += 1

    elif not skip_undefined:
        probabilities.append(neg_inf_vector)

        operation_mapping[operation_counter] = Copy
        operation_counter += 1

    # Calculate copy shift scores
    if allow_copy and source_index > 0 and target_index > 0:
        copy_scores = probability_matrix[source_index - 1][target_index - 1]
        copy_scores = copy_scores + scores[:, source_index - 1, target_index - 1, copy_shift_index]

        copy_mask = torch.logical_not(copy_matrix[:, source_index - 1, target_index - 1])
        copy_scores = torch.masked_fill(copy_scores, mask=copy_mask, value=neg_inf_value)

        probabilities.append(copy_scores)

        operation_mapping[operation_counter] = CopyShift
        operation_counter += 1

    elif not skip_undefined:
        probabilities.append(neg_inf_vector)

        operation_mapping[operation_counter] = CopyShift
        operation_counter += 1

    probabilities = torch.stack(probabilities)
    return probabilities, operation_mapping


def autoregressive_transduction_loss(scores: Tensor, source_lengths: Tensor, target_lengths: Tensor,
                                     insertion_labels: Tensor, substitution_labels: Tensor, copy_index: int,
                                     copy_shift_index: int, deletion_index: int, copy_matrix: Tensor,
                                     device: torch.device, allow_copy: bool = True, enforce_copy: bool = False,
                                     reduction: str = "mean") -> Tensor:
    assert not (enforce_copy and not allow_copy), "Cannot both enforce copy and disable copying"
    # scores: batch x source length x target length x #labels

    # Define constants
    batch_size, source_length, target_length, num_labels = scores.shape
    neg_inf_vector = torch.full(size=(batch_size,), fill_value=neg_inf_value, device=device)

    probability_matrix = [
        [neg_inf_vector for _ in range(target_length)]
        for _ in range(source_length + 1)
    ]

    batch_indices = torch.arange(0, batch_size)
    copy_matrix = copy_matrix.to(device)

    for source_index in range(0, source_length + 1):
        for target_index in range(0, target_length):
            if source_index == 0 and target_index == 0:
                probability_matrix[source_index][target_index] = torch.zeros(batch_size, device=device)
                continue

            probabilities, _ = _autoregressive_slow_recursion(
                probability_matrix=probability_matrix, scores=scores, source_index=source_index,
                target_index=target_index, source_length=source_length, deletion_index=deletion_index,
                copy_index=copy_index, copy_shift_index=copy_shift_index, insertion_labels=insertion_labels,
                substitution_labels=substitution_labels, copy_matrix=copy_matrix, allow_copy=allow_copy,
                enforce_copy=enforce_copy, batch_indices=batch_indices, skip_undefined=True,
                neg_inf_vector=neg_inf_vector, source_lengths=source_lengths
            )

            probabilities = torch.logsumexp(probabilities, dim=0)
            probability_matrix[source_index][target_index] = probabilities

    probability_matrix = torch.stack([torch.stack(row) for row in probability_matrix])
    probability_matrix = torch.permute(probability_matrix, dims=(2, 0, 1))

    nll = -probability_matrix[batch_indices, source_lengths, target_lengths - 1]
    nll = shoot_nans(nll)
    loss = _loss_reduction(nll, reduction=reduction)

    return loss
