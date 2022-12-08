from torch import Tensor

# Constants
neg_inf_value = -1e6


def _loss_reduction(loss: Tensor, reduction: str) -> Tensor:
    if reduction == "mean":
        return loss.mean()
    elif reduction == "sum":
        return loss.sum()
    elif reduction == "none":
        return loss
    else:
        raise ValueError(f"Invalid reduction: {reduction} (must be 'mean', 'sum', or 'none')")
