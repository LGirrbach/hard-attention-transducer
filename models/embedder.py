import torch
import torch.nn as nn

from positional_encodings.torch_encodings import Summer
from positional_encodings.torch_encodings import PositionalEncoding1D


class Embedder(nn.Module):
    def __init__(self, vocab_size: int, embedding_dim: int = 128, dropout: float = 0.0,
                 positional_encodings: bool = False):
        super(Embedder, self).__init__()

        self.vocab_size = vocab_size
        self.embedding_dim = embedding_dim
        self.dropout = dropout
        self.positional_encodings = positional_encodings

        self.embedding = nn.Embedding(
            num_embeddings=vocab_size, embedding_dim=embedding_dim, padding_idx=0,
        )
        self.dropout = nn.Dropout(p=self.dropout)

        if self.positional_encodings:
            self.positional_encoder = Summer(PositionalEncoding1D(self.embedding_dim))
        else:
            self.positional_encoder = nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        assert len(x.shape) == 2
        embedded = self.embedding(x)
        embedded = self.dropout(embedded)
        embedded = self.positional_encoder(embedded)
        return embedded
