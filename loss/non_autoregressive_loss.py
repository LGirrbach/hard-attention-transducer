import torch

from torch import Tensor
from typing import Optional
from loss.loss_utils import _loss_reduction
from actions import Deletion, Copy, CopyShift, Insertion, Substitution


def _non_autoregressive_make_backpointers_start(deletion_scores: Tensor, substitution_scores: Tensor,
                                                copy_scores: Optional[Tensor]):
    backpointers = []
    tau = deletion_scores.size(1)

    with torch.no_grad():
        best_scores_by_operation = []

        best_deletion_scores = torch.max(deletion_scores, dim=1)
        best_scores_by_operation.append(best_deletion_scores.values)
        best_substitution_scores = torch.max(substitution_scores, dim=1)
        best_scores_by_operation.append(best_substitution_scores.values)

        if copy_scores is not None:
            best_copy_scores = torch.max(copy_scores, dim=1)
            best_scores_by_operation.append(best_copy_scores.values)

        best_operation_indices = torch.max(torch.stack(best_scores_by_operation), dim=0)
        best_operation_indices = best_operation_indices.indices.cpu().tolist()

        for batch_elem_index, batch_elem_ops in enumerate(best_operation_indices):
            batch_elem_backpointers = []

            for batch_elem_target_index, batch_elem_op in enumerate(batch_elem_ops):
                if batch_elem_op == 0:
                    operation = Deletion
                    offset = best_deletion_scores.indices[batch_elem_index, batch_elem_target_index]
                    offset = offset.item() - tau

                elif batch_elem_op == 1:
                    operation = Substitution
                    offset = best_substitution_scores.indices[batch_elem_index, batch_elem_target_index]
                    offset = offset.item() - tau

                else:
                    operation = CopyShift
                    offset = best_copy_scores.indices[batch_elem_index, batch_elem_target_index]
                    offset = offset.item() - tau

                batch_elem_backpointers.append((operation, offset))

            backpointers.append(batch_elem_backpointers)

    return backpointers


def _non_autoregressive_make_backpointers_non_start(probabilities: Tensor):
    backpointers = []

    with torch.no_grad():
        best_operations = torch.max(probabilities, dim=0)

        for batch_elem_ops in best_operations.indices.cpu().tolist():
            batch_elem_backpointers = []

            for operation_index in batch_elem_ops:
                if operation_index == 0:
                    batch_elem_backpointers.append((Insertion, -1))
                else:
                    batch_elem_backpointers.append((Copy, -1))

            backpointers.append(batch_elem_backpointers)

    return backpointers


def non_autoregressive_transduction_loss(scores: Tensor, source_lengths: Tensor, target_lengths: Tensor,
                                         insertion_labels: Tensor, substitution_labels: Tensor, copy_index: int,
                                         copy_shift_index: int, deletion_index: int, noop_index: int,
                                         copy_matrix: Tensor, tau: Optional[int], device: torch.device,
                                         allow_copy: bool = True, enforce_copy: bool = False,
                                         noop_discount: float = 1., reduction: str = "mean",
                                         return_backpointers: bool = False):
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

    # Make padding
    padding_1 = torch.full(size=(batch_size, 1), fill_value=neg_inf_val, device=device)
    padding_2 = torch.full(size=(batch_size, tau, 1), fill_value=neg_inf_val, device=device)

    # Initialise backpointers
    backpointers = []

    for source_index in range(1, source_max_timesteps + 1):
        source_element_index = source_index // tau
        probabilities = []

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

            insertion_scores = previous_scores + insertion_prediction_scores
            probabilities.append(insertion_scores)

            # Get copy labels
            if allow_copy:
                copy_mask = copy_matrix[:, source_element_index, :]
                copy_scores = expanded_scores[batch_indices, source_index - 1, copy_index]
                copy_scores = copy_scores.unsqueeze(1)
                copy_scores = copy_scores.expand((batch_size, target_max_timesteps))
                copy_scores = torch.masked_fill(copy_scores, mask=~copy_mask, value=neg_inf_val)
                copy_scores = previous_scores + copy_scores

                probabilities.append(copy_scores)

            # Add to probability matrix
            source_index_scores_stacked = torch.stack(probabilities)
            source_index_scores = torch.logsumexp(source_index_scores_stacked, dim=0)
            source_index_scores = torch.cat([padding_1, source_index_scores], dim=-1)
            probability_matrix.append(source_index_scores)

            if return_backpointers:
                backpointer_padding_size = (*source_index_scores_stacked.shape[:2], 1)
                backpointer_padding = torch.full(
                    size=backpointer_padding_size, fill_value=neg_inf_val, device=source_index_scores_stacked.device
                )
                source_index_backpointers = _non_autoregressive_make_backpointers_non_start(
                    torch.cat([backpointer_padding, source_index_scores_stacked], dim=2)
                )
                backpointers.append(source_index_backpointers)

        else:
            previous_symbol_scores = grouped_scores[batch_indices, source_element_index - 1]
            # previous_symbol_scores shape: batch x tau x labels
            previous_probabilities = torch.stack(probability_matrix[source_index-tau:])
            # previous_probabilities shape: tau x batch x target timesteps + 1
            previous_probabilities = previous_probabilities.transpose(0, 1)
            # previous_probabilities shape: batch x tau x target timesteps + 1

            # Calculate noop scores
            noop_scores = torch.cumsum(previous_symbol_scores[batch_indices, 1:, noop_index], dim=-1)
            noop_scores = noop_scores / noop_discount
            noop_scores = torch.cat([padding_1, noop_scores], dim=-1)
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
                substitution_scores = torch.masked_fill(
                    substitution_scores, mask=current_copy_mask, value=neg_inf_val
                )

            substitution_scores = substitution_scores + noop_scores_expanded[:, :, :-1]
            substitution_scores = substitution_scores + previous_probabilities[:, :, :-1]
            substitution_scores = torch.cat([padding_2, substitution_scores], dim=-1)
            probabilities.append(substitution_scores)

            # Calculate copy scores
            if allow_copy:
                copy_scores = previous_symbol_scores[batch_indices, :, copy_shift_index]
                copy_scores = copy_scores.unsqueeze(2)
                copy_scores = copy_scores.expand((batch_size, tau, target_max_timesteps))
                copy_scores = torch.masked_fill(copy_scores, mask=~current_copy_mask, value=neg_inf_val)
                copy_scores = copy_scores + noop_scores_expanded[:, :, :-1] + previous_probabilities[:, :, :-1]
                copy_scores = torch.cat([padding_2, copy_scores], dim=-1)
                probabilities.append(copy_scores)
            else:
                copy_scores = None

            # Calculate row probabilities
            probabilities = torch.cat(probabilities, dim=1)
            probabilities = torch.logsumexp(probabilities, dim=1)
            probability_matrix.append(probabilities)

            if return_backpointers:
                source_index_backpointers = _non_autoregressive_make_backpointers_start(
                    deletion_scores, substitution_scores, copy_scores
                )
                backpointers.append(source_index_backpointers)

    probability_matrix = torch.stack(probability_matrix)  # shape source timesteps + 1 x batch x target timesteps + 1
    probability_matrix = torch.transpose(probability_matrix, 0, 1)

    nll = -probability_matrix[batch_indices, tau * source_lengths, target_lengths]

    if return_backpointers:
        loss = _loss_reduction(nll, reduction="none")
        return backpointers, loss

    else:
        loss = _loss_reduction(nll, reduction=reduction)
        return loss
