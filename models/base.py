import torch
import torch.nn as nn

from torch import Tensor
from entmax import entmax15
from entmax import sparsemax


class TransducerModel(nn.Module):
    def normalise_scores(self, scores: Tensor) -> Tensor:
        if self.scorer == "softmax":
            return torch.log_softmax(scores, dim=-1)

        elif self.scorer == "entmax":
            scores = torch.log(entmax15(scores, dim=-1))
            scores = torch.masked_fill(scores, mask=torch.isneginf(scores), value=-100000)
            return scores

        elif self.scorer == "sparsemax":
            scores = torch.log(sparsemax(scores, dim=-1))
            scores = torch.masked_fill(scores, mask=torch.isneginf(scores), value=-100000)
            return scores

        else:
            raise ValueError(f"Unknown scoring function: {self.scorer}")
