import torch
import torch.nn as nn

from torch import Tensor
from typing import Optional
from models.embedder import Embedder
from models.base import TransducerModel
from models.encoder import BiLSTMEncoder
from models.feature_encoder import FeatureEncoder


class NonAutoregressiveLSTMDecoder(nn.Module):
    def __init__(self, input_size: int, hidden_size: int, num_layers: int, dropout: float, device: torch.device,
                 target_vocab_size: int):
        super(NonAutoregressiveLSTMDecoder, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.dropout = dropout
        self.device = device
        self.target_vocab_size = target_vocab_size

        self.decoder = nn.LSTM(
            input_size=input_size, hidden_size=hidden_size, num_layers=num_layers,
            bias=True, bidirectional=False, batch_first=True, dropout=dropout, proj_size=0
        )
        self.h_0 = nn.Parameter(torch.zeros(self.num_layers, 1, self.hidden_size))
        self.c_0 = nn.Parameter(torch.zeros(self.num_layers, 1, self.hidden_size))

        self.classifier = nn.Sequential(
            nn.Dropout(p=dropout),
            nn.Linear(hidden_size, hidden_size),
            nn.GELU(),
            nn.Dropout(),
            nn.Linear(hidden_size, target_vocab_size)
        )

    def forward(self, encoder_outputs: Tensor, tau: int) -> Tensor:
        # Make input sequence
        batch, timesteps, num_features = encoder_outputs.shape

        decoder_inputs = encoder_outputs.reshape(-1, num_features).unsqueeze(1)
        decoder_inputs = decoder_inputs.expand((decoder_inputs.shape[0], tau, num_features))

        hidden = (
            self.h_0.expand((self.num_layers, batch * timesteps, self.hidden_size)).contiguous(),
            self.c_0.expand((self.num_layers, batch * timesteps, self.hidden_size)).contiguous()
        )

        decoder_outputs, _ = self.decoder(decoder_inputs.contiguous(), hidden)
        decoder_outputs = decoder_outputs.reshape(batch, timesteps, tau, self.hidden_size)
        decoder_outputs = self.classifier(decoder_outputs)

        return decoder_outputs


class NonAutoregressivePositionalDecoder(nn.Module):
    def __init__(self, embedding_size: int, max_decoding_length: int, input_size: int, hidden_size: int, dropout: float,
                 device: torch.device, target_vocab_size: int):
        super(NonAutoregressivePositionalDecoder, self).__init__()
        self.embedding_size = embedding_size
        self.max_decoding_length = max_decoding_length
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.dropout = dropout
        self.device = device
        self.target_vocab_size = target_vocab_size

        self.positions = nn.Parameter(torch.zeros(1, max_decoding_length, embedding_size))
        self.classifier = nn.Sequential(
            nn.Dropout(p=dropout),
            nn.Linear(input_size + embedding_size, 2 * hidden_size),
            nn.GELU(),
            nn.Dropout(p=dropout),
            nn.Linear(2 * hidden_size, 2 * hidden_size),
            nn.GELU(),
            nn.Linear(2 * hidden_size, target_vocab_size)
        )

    def forward(self, encoder_outputs: Tensor, tau: int) -> Tensor:
        # Make input sequence
        batch, timesteps, num_features = encoder_outputs.shape

        positions = self.positions[:, :tau, :].expand((batch * timesteps, tau, self.embedding_size))
        positions = positions.contiguous()

        decoder_inputs = encoder_outputs.reshape(-1, num_features).unsqueeze(1)
        decoder_inputs = decoder_inputs.expand((decoder_inputs.shape[0], tau, num_features))
        decoder_inputs = decoder_inputs.contiguous()

        decoder_inputs = torch.cat([decoder_inputs, positions], dim=-1)
        decoder_outputs = self.classifier(decoder_inputs)
        decoder_outputs = decoder_outputs.reshape(batch, timesteps, tau, self.target_vocab_size)

        return decoder_outputs


class NonAutoregressiveFixedTauDecoder(nn.Module):
    def __init__(self, input_size: int, hidden_size: int, target_vocab_size: int, dropout: float, tau: int):
        super(NonAutoregressiveFixedTauDecoder, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.target_vocab_size = target_vocab_size
        self.dropout = dropout
        self.tau = tau

        self.classifier = nn.Sequential(
            nn.Dropout(p=dropout),
            nn.Linear(input_size, 2 * hidden_size),
            nn.GELU(),
            nn.Dropout(p=dropout),
            nn.Linear(2 * hidden_size, 2 * hidden_size),
            nn.GELU(),
            nn.Linear(2 * hidden_size, tau * target_vocab_size)
        )

    def forward(self, encoder_outputs: Tensor) -> Tensor:
        scores = self.classifier(encoder_outputs)
        scores = scores.reshape(scores.shape[0], scores.shape[1], self.tau, self.target_vocab_size)
        return scores


class NonAutoregressiveLSTM(TransducerModel):
    def __init__(self, source_vocab_size: int, target_vocab_size: int, embedding_dim: int = 128,
                 hidden_size: int = 128, num_layers: int = 2, dropout: float = 0.0, temperature: float = 1.0,
                 scorer: str = "softmax", device: torch.device = torch.device("cpu"), use_features: bool = False,
                 feature_vocab_size: int = 0, feature_encoder_hidden: int = 128, feature_encoder_layers: int = 0,
                 feature_encoder_pooling: str = "mean", tau: Optional[int] = None, decoder_type: str = "position",
                 max_targets_per_symbol: int = 50):
        super(NonAutoregressiveLSTM, self).__init__()

        self.source_vocab_size = source_vocab_size
        self.target_vocab_size = target_vocab_size
        self.embedding_dim = embedding_dim
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.dropout = dropout
        self.temperature = temperature
        self.scorer = scorer
        self.device = device
        self.use_features = use_features
        self.feature_vocab_size = feature_vocab_size
        self.feature_encoder_hidden = feature_encoder_hidden
        self.feature_encoder_layers = feature_encoder_layers
        self.feature_encoder_pooling = feature_encoder_pooling
        self.tau = tau
        self.decoder_type = decoder_type
        self.max_targets_per_symbol = max_targets_per_symbol

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
            classifier_input_dim = self.hidden_size + self.feature_encoder_hidden
        else:
            classifier_input_dim = self.hidden_size

        # Make symbol decoder
        if decoder_type == "lstm":
            # assert tau is None
            self.decoder = NonAutoregressiveLSTMDecoder(
                input_size=classifier_input_dim, hidden_size=hidden_size, dropout=dropout, num_layers=num_layers,
                target_vocab_size=target_vocab_size, device=device
            )

        elif decoder_type == "position":
            # assert tau is None
            self.decoder = NonAutoregressivePositionalDecoder(
                embedding_size=embedding_dim, max_decoding_length=max_targets_per_symbol,
                input_size=classifier_input_dim, hidden_size=hidden_size, dropout=dropout, device=device,
                target_vocab_size=target_vocab_size
            )

        elif decoder_type == "fixed":
            assert isinstance(tau, int) and tau > 0
            self.decoder = NonAutoregressiveFixedTauDecoder(
                input_size=classifier_input_dim, hidden_size=hidden_size, dropout=dropout,
                target_vocab_size=target_vocab_size, tau=tau
            )

        else:
            raise ValueError(f"Unknown non-autoregressive decoder: {decoder_type}")

    def get_params(self):
        return {
            'source_vocab_size': self.source_vocab_size,
            'target_vocab_size': self.target_vocab_size,
            'embedding_dim': self.embedding_dim,
            'hidden_size': self.hidden_size,
            'num_layers': self.num_layers,
            'dropout': self.dropout,
            'temperature': self.temperature,
            'scorer': self.scorer,
            'device': self.device,
            'use_features': self.use_features,
            'feature_vocab_size': self.feature_vocab_size,
            'feature_encoder_hidden': self.feature_encoder_hidden,
            'feature_encoder_layers': self.feature_encoder_layers,
            'feature_encoder_pooling': self.feature_encoder_pooling,
            'tau': self.tau,
            'decoder_type': self.decoder_type,
            'max_targets_per_symbol': self.max_targets_per_symbol
        }

    def forward(self, sources: Tensor, lengths: Tensor, features: Optional[Tensor] = None,
                feature_lengths: Optional[Tensor] = None, tau: Optional[int] = None) -> Tensor:
        embedded = self.embedder(sources.to(self.device))
        encoded = self.encoder(embedded, lengths)

        # Make feature encodings
        if self.use_features:
            feature_encodings = self.feature_encoder(
                features.to(self.device), feature_lengths, encoded
            )
            encoded = torch.cat([encoded, feature_encodings], dim=-1)

        if self.decoder_type == "fixed":
            scores = self.decoder(encoded)
        else:
            scores = self.decoder(encoded, tau=tau)

        scores = self.normalise_scores(scores / self.temperature)

        return scores
