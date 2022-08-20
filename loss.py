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
    neg_inf_val = -1e6

    # Define shape constants
    batch_size = scores.shape[0]
    num_labels = scores.shape[-1]
    source_max_timesteps = tau * source_lengths.max().detach().cpu().item()
    target_max_timesteps = target_lengths.max().detach().cpu().item()

    # Initialise probability matrix for dynamic programming
    initial_probabilities = torch.full(size=(batch_size, target_max_timesteps+1), fill_value=neg_inf_val, device=device)
    initial_probabilities[:, 0] = 0.0
    probability_matrix = [initial_probabilities]

    # Define batch indexers
    batch_indices = torch.arange(0, batch_size)
    batch_target_indices = torch.arange(0, batch_size * target_max_timesteps)
    batch_target_tau_indices = torch.arange(0, batch_size * target_max_timesteps * tau)

    # Move copy matrix to device
    copy_matrix = copy_matrix.to(device)

    # Define proxies for scores
    grouped_scores = scores
    expanded_scores = scores.reshape(batch_size, -1, num_labels)

    # Make substitution labels
    # substitution_labels shape: batch x source timesteps x target timesteps
    substitution_labels = substitution_labels.unsqueeze(1)
    substitution_labels = substitution_labels.expand((batch_size, tau, target_max_timesteps))
    substitution_labels = substitution_labels.flatten()

    # Make insertion labels
    insertion_labels = insertion_labels.flatten()

    for source_index in range(1, source_max_timesteps + 1):
        source_element_index = source_index // tau

        if source_index % tau != 0:
            # Get previous scores
            previous_scores = probability_matrix[source_index - 1][:, :-1]  # shape batch x target_max_timesteps

            # Get insertion scores
            # insertion labels shape: batch x target_max_timesteps
            insertion_prediction_scores = expanded_scores[:, source_index - 1, :]
            insertion_prediction_scores = insertion_prediction_scores.unsqueeze(1)
            insertion_prediction_scores = insertion_prediction_scores.expand(
                (batch_size, target_max_timesteps, num_labels)
            )
            insertion_prediction_scores = insertion_prediction_scores.reshape(-1, num_labels)
            insertion_prediction_scores = insertion_prediction_scores[batch_target_indices, insertion_labels]
            insertion_prediction_scores = insertion_prediction_scores.reshape(batch_size, target_max_timesteps)

            # Get copy labels
            copy_scores = expanded_scores[batch_indices, source_index - 1, copy_index]
            copy_scores = copy_scores.unsqueeze(1)
            copy_scores = copy_scores.expand((batch_size, target_max_timesteps))

            copy_mask = copy_matrix[:, source_element_index, :]
            copy_scores = torch.masked_fill(copy_scores, mask=~copy_mask, value=neg_inf_val)

            # Add to probability matrix
            insertion_scores = previous_scores + insertion_prediction_scores
            copy_scores = previous_scores + copy_scores
            source_index_scores = torch.logsumexp(torch.stack([insertion_scores, copy_scores]), dim=0)

            padding = torch.full(size=(batch_size, 1), fill_value=neg_inf_val, device=source_index_scores.device)
            source_index_scores = torch.cat([padding, source_index_scores], dim=-1)
            probability_matrix.append(source_index_scores)

        else:
            probabilities = []

            previous_symbol_scores = grouped_scores[batch_indices, source_element_index - 1]
            # previous_symbol_scores shape: batch x tau x labels
            previous_probabilities = torch.stack(probability_matrix[source_index-tau:])
            # previous_probabilities shape: tau x batch x target timesteps + 1
            previous_probabilities = previous_probabilities.transpose(0, 1)
            # previous_probabilities shape: batch x tau x target timesteps + 1

            # Calculate noop scores
            noop_scores = torch.cumsum(previous_symbol_scores[batch_indices, 1:, noop_index], dim=-1)
            noop_scores = noop_scores / noop_discount
            padding = torch.zeros(batch_size, 1, device=noop_scores.device)
            noop_scores = torch.cat([padding, noop_scores], dim=-1)
            noop_scores = torch.flip(noop_scores, dims=[1])
            # noop_scores shape: batch x tau
            noop_scores_expanded = noop_scores.unsqueeze(2)
            noop_scores_expanded = noop_scores_expanded.expand(batch_size, tau, target_max_timesteps + 1)
            # noop_scores_expanded shape: batch x tau x target timesteps + 1

            # Calculate deletion scores
            deletion_scores = previous_symbol_scores[batch_indices, :, deletion_index]
            deletion_scores = deletion_scores.unsqueeze(2)
            deletion_scores = deletion_scores.expand(batch_size, tau, target_max_timesteps + 1)
            deletion_scores = deletion_scores + noop_scores_expanded + previous_probabilities
            probabilities.append(deletion_scores)

            # Make current copy mask
            current_copy_mask = copy_matrix[batch_indices, source_element_index-1, :]
            current_copy_mask = current_copy_mask.unsqueeze(1)
            current_copy_mask = current_copy_mask.expand((batch_size, tau, target_max_timesteps))

            # Calculate substitution scores
            substitution_scores = previous_symbol_scores.reshape(batch_size * tau, 1, num_labels)
            substitution_scores = substitution_scores.expand((batch_size * tau, target_max_timesteps, num_labels))
            substitution_scores = substitution_scores.reshape(-1, num_labels)

            substitution_scores = substitution_scores[batch_target_tau_indices, substitution_labels]
            substitution_scores = substitution_scores.reshape(batch_size, tau, target_max_timesteps)

            if enforce_copy:
                substitution_scores = torch.masked_fill(substitution_scores, mask=current_copy_mask, value=neg_inf_val)

            substitution_scores = substitution_scores + noop_scores_expanded[:, :, :-1]
            substitution_scores = substitution_scores + previous_probabilities[:, :, :-1]

            padding = torch.full(size=(batch_size, tau, 1), fill_value=neg_inf_val, device=substitution_scores.device)
            substitution_scores = torch.cat([padding, substitution_scores], dim=-1)
            probabilities.append(substitution_scores)

            # Calculate copy scores
            if allow_copy:
                copy_scores = previous_symbol_scores[batch_indices, :, copy_shift_index]
                copy_scores = copy_scores.unsqueeze(2)
                copy_scores = copy_scores.expand((batch_size, tau, target_max_timesteps))
                copy_scores = torch.masked_fill(copy_scores, mask=~current_copy_mask, value=neg_inf_val)
                copy_scores = copy_scores + noop_scores_expanded[:, :, :-1] + previous_probabilities[:, :, :-1]
                copy_scores = torch.cat([padding, copy_scores], dim=-1)
                probabilities.append(copy_scores)

            # Calculate row probabilities
            probabilities = torch.cat(probabilities, dim=1)
            probabilities = torch.logsumexp(probabilities, dim=1)
            probability_matrix.append(probabilities)

    probability_matrix = torch.stack(probability_matrix)  # shape source timesteps + 1 x batch x target timesteps + 1
    probability_matrix = torch.transpose(probability_matrix, 0, 1)

    nll = -probability_matrix[batch_indices, tau * source_lengths, :][batch_indices, target_lengths]
    loss = _loss_reduction(nll, reduction=reduction)

    return loss
