import torch

from torch import Tensor
from loss.loss_utils import _loss_reduction

# Global variables
cross_entropy = torch.nn.CrossEntropyLoss(ignore_index=0, reduction="none")


def soft_attention_loss(scores: Tensor, target_labels: Tensor, reduction: str = "mean") -> Tensor:
    # scores: batch x target length x #labels
    # target_labels: batch x target length

    batch, target_length, num_labels = scores.shape
    scores = torch.reshape(scores, (-1, num_labels)).contiguous()
    target_labels = torch.flatten(target_labels).contiguous()
    target_labels = target_labels.to(scores.device)

    nll = cross_entropy(scores, target_labels)

    if reduction == "mean":
        return nll.sum() / (target_labels != 0).sum().float().clamp(1.0)
    else:
        return _loss_reduction(loss=nll, reduction=reduction)
