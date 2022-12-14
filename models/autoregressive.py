import torch
import torch.nn as nn

from torch import Tensor
from typing import Tuple
from typing import Optional
from models.embedder import Embedder
from models.decoder import LSTMDecoder
from models.base import TransducerModel
from models.encoder import BiLSTMEncoder
from models.decoder import DecoderOutput
from models.feature_encoder import FeatureEncoder


class LSTMEncoderDecoderModel(TransducerModel):
    def __init__(self, source_vocab_size: int, target_vocab_size: int, embedding_dim: int = 128,
                 hidden_size: int = 128, num_layers: int = 2, dropout: float = 0.0, temperature: float = 1.0,
                 scorer: str = "softmax", device: torch.device = torch.device("cpu"), use_features: bool = False,
                 feature_vocab_size: int = 0, feature_encoder_hidden: int = 128, feature_encoder_layers: int = 0,
                 feature_encoder_pooling: str = "mean", encoder_bridge: bool = False):
        super(LSTMEncoderDecoderModel, self).__init__()

        self.source_vocab_size = source_vocab_size
        self.target_vocab_size = target_vocab_size
        self.device = device
        self.embedding_dim = embedding_dim
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.dropout = dropout
        self.temperature = temperature
        self.scorer = scorer
        self.use_features = use_features
        self.feature_vocab_size = feature_vocab_size
        self.feature_encoder_hidden = feature_encoder_hidden
        self.feature_encoder_layers = feature_encoder_layers
        self.feature_encoder_pooling = feature_encoder_pooling
        self.encoder_bridge = encoder_bridge

        # Make Embedders
        self.encoder_embedder = Embedder(vocab_size=source_vocab_size, embedding_dim=embedding_dim, dropout=dropout)
        self.decoder_embedder = Embedder(vocab_size=target_vocab_size, embedding_dim=embedding_dim, dropout=dropout)

        # Make Encoder and Decoder
        self.encoder = BiLSTMEncoder(
            input_size=self.encoder_embedder.embedding_dim, hidden_size=self.hidden_size, num_layers=self.num_layers,
            dropout=self.dropout
        )
        self.decoder = LSTMDecoder(
            input_size=self.decoder_embedder.embedding_dim, hidden_size=self.hidden_size, num_layers=self.num_layers,
            dropout=self.dropout, encoder_bridge=encoder_bridge
        )

        # Make feature encoder
        if self.use_features:
            if self.feature_encoder_pooling == "dot":
                self.feature_encoder_hidden = 2 * self.hidden_size
            else:
                self.feature_encoder_hidden = self.feature_encoder_hidden

            self.feature_encoder = FeatureEncoder(
                vocab_size=self.feature_vocab_size, embedding_dim=self.embedding_dim,
                hidden_size=self.feature_encoder_hidden, num_layers=self.feature_encoder_layers, dropout=self.dropout,
                pooling=self.feature_encoder_pooling, context_dim=2*self.hidden_size, device=self.device
            )

        # Make prediction Head
        if self.use_features:
            prediction_head_input_size = 2 * self.hidden_size + self.feature_encoder_hidden
        else:
            prediction_head_input_size = 2 * self.hidden_size

        self.classifier = nn.Sequential(
            nn.Dropout(p=self.dropout),
            nn.Linear(prediction_head_input_size, self.hidden_size),
            nn.GELU(),
            nn.Dropout(p=self.dropout),
            nn.Linear(self.hidden_size, self.hidden_size),
            nn.GELU(),
            nn.Linear(self.hidden_size, self.target_vocab_size)
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
            'scorer': self.scorer,
            'device': self.device,
            'use_features': self.use_features,
            'feature_vocab_size': self.feature_vocab_size,
            'feature_encoder_hidden': self.feature_encoder_hidden,
            'feature_encoder_layers': self.feature_encoder_layers,
            'feature_encoder_pooling': self.feature_encoder_pooling,
            'encoder_bridge': self.encoder_bridge,
        }

    def encode(self, encoder_inputs: Tensor, lengths: Tensor) -> Tuple[Tensor, Tensor]:
        encoder_inputs = encoder_inputs.to(self.device)
        embedded = self.encoder_embedder(encoder_inputs)
        encoded = self.encoder(embedded, lengths)

        return encoded

    def decode(self, decoder_inputs: Tensor, decoder_lengths: Tensor, encoder_outputs: Tensor, encoder_lengths: Tensor,
               hidden: Tuple[Tensor, Tensor] = None) -> DecoderOutput:
        decoder_inputs = decoder_inputs.to(self.device)
        decoder_embedded = self.decoder_embedder(decoder_inputs)
        decoder_encoded, (old_hidden, new_hidden) = self.decoder(
            decoder_embedded, decoder_lengths, encoder_outputs, encoder_lengths, hidden=hidden
        )

        return decoder_encoded, (old_hidden, new_hidden)

    def get_transduction_scores(self, source_encodings: Tensor, target_encodings: Tensor,
                                features: Optional[Tensor] = None, feature_lengths: Optional[Tensor] = None) -> Tensor:
        batch, hidden = target_encodings.shape[0], target_encodings.shape[2]
        timesteps_decoder = target_encodings.shape[1]
        timesteps_encoder = source_encodings.shape[1]

        decoder_outputs = target_encodings.unsqueeze(2).expand((batch, timesteps_decoder, timesteps_encoder, hidden))
        encoder_outputs = source_encodings.unsqueeze(2).expand((batch, timesteps_encoder, timesteps_decoder, hidden))
        encoder_outputs = encoder_outputs.transpose(1, 2)
        classifier_inputs = torch.cat([encoder_outputs, decoder_outputs], dim=-1)

        # Make feature representations (optional)
        if self.use_features:
            feature_encodings = self.feature_encoder(
                features, feature_lengths, classifier_inputs.reshape(batch, -1, 2 * hidden)
            )
            feature_encodings = feature_encodings.reshape(batch, timesteps_decoder, timesteps_encoder, -1)
            classifier_inputs = torch.cat([classifier_inputs, feature_encodings], dim=-1)

        scores = self.classifier(classifier_inputs)
        scores = scores.transpose(1, 2)
        scores = self.normalise_scores(scores / self.temperature)

        return scores
