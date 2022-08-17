import torch
import torch.nn as nn

from torch import Tensor
from typing import Optional
from models.embedder import Embedder
from models.base import TransducerModel
from models.encoder import BiLSTMEncoder
from models.feature_encoder import FeatureEncoder


class NonAutoregressiveLSTM(TransducerModel):
    def __init__(self, source_vocab_size: int, target_vocab_size: int, embedding_dim: int = 128,
                 hidden_size: int = 128, num_layers: int = 2, dropout: float = 0.0, temperature: float = 1.0,
                 scorer: str = "softmax", tau: int = 5, device: torch.device = torch.device("cpu"),
                 use_features: bool = False, feature_vocab_size: int = 0, feature_encoder_hidden: int = 128,
                 feature_encoder_layers: int = 0, feature_encoder_pooling: str = "mean"):
        super(NonAutoregressiveLSTM, self).__init__()

        self.source_vocab_size = source_vocab_size
        self.target_vocab_size = target_vocab_size
        self.embedding_dim = embedding_dim
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.dropout = dropout
        self.temperature = temperature
        self.tau = tau
        self.scorer = scorer
        self.device = device
        self.use_features = use_features
        self.feature_vocab_size = feature_vocab_size
        self.feature_encoder_hidden = feature_encoder_hidden
        self.feature_encoder_layers = feature_encoder_layers
        self.feature_encoder_pooling = feature_encoder_pooling

        # Make Embedders
        self.embedder = Embedder(vocab_size=source_vocab_size, embedding_dim=embedding_dim, dropout=dropout)

        # Make Encoder
        self.encoder = BiLSTMEncoder(
            input_size=self.embedder.embedding_dim, hidden_size=self.hidden_size, num_layers=self.num_layers,
            dropout=self.dropout
        )

        # Make feature encoder
        if self.use_features:
            self.feature_encoder = FeatureEncoder(
                vocab_size=self.feature_vocab_size, embedding_dim=self.embedding_dim,
                hidden_size=self.feature_encoder_hidden, num_layers=self.feature_encoder_layers, dropout=self.dropout,
                pooling=self.feature_encoder_pooling, context_dim=self.hidden_size, device=self.device
            )

        if self.use_features:
            classifier_input_dim = 2 * self.hidden_size
        else:
            classifier_input_dim = self.hidden_size

        # Make classification head
        self.classifier = nn.Sequential(
            nn.Dropout(p=self.dropout),
            nn.Linear(classifier_input_dim, 2 * self.hidden_size),
            nn.GELU(),
            nn.Dropout(p=self.dropout),
            nn.Linear(2 * self.hidden_size, 2 * self.hidden_size),
            nn.GELU(),
            nn.Linear(2 * self.hidden_size, self.tau * self.target_vocab_size)
        )

    def get_params(self):
        return {
            'source_vocab_size': self.source_vocab_size,
            'target_vocab_size': self.target_vocab_size,
            'embedding_dim': self.embedding_dim,
            'hidden_size': self.hidden_size,
            'num_layers': self.num_layers,
            'dropout': self.dropout,
            'temperature': self.temperature,
            'tau': self.tau,
            'scorer': self.scorer,
            'device': self.device,
            'use_features': self.use_features,
            'feature_vocab_size': self.feature_vocab_size,
            'feature_encoder_hidden': self.feature_encoder_hidden,
            'feature_encoder_layers': self.feature_encoder_layers,
            'feature_encoder_pooling': self.feature_encoder_pooling
        }

    def forward(self, sources: Tensor, lengths: Tensor, features: Optional[Tensor] = None,
                feature_lengths: Optional[Tensor] = None) -> Tensor:
        embedded = self.embedder(sources.to(self.device))
        encoded = self.encoder(embedded, lengths)

        # Make feature encodings
        if self.use_features:
            feature_encodings = self.feature_encoder(
                features, feature_lengths, encoded
            )
            encoded = torch.cat([encoded, feature_encodings], dim=-1)

        scores = self.classifier(encoded)
        scores = scores.reshape(scores.shape[0], scores.shape[1], self.tau, self.target_vocab_size)
        scores = self.normalise_scores(scores / self.temperature)

        return scores
