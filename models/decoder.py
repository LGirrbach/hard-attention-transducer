import torch
import torch.nn as nn

from typing import Tuple
from torch import Tensor
from util import make_mask_2d
from torch.nn.utils.rnn import pad_packed_sequence
from torch.nn.utils.rnn import pack_padded_sequence


DecoderOutput = Tuple[Tensor, Tuple[Tuple[Tensor, Tensor], Tuple[Tensor, Tensor]]]


class LSTMDecoder(nn.Module):
    def __init__(self, input_size: int, hidden_size: int = 128, num_layers: int = 1, dropout: float = 0.0,
                 encoder_bridge: bool = False):
        super(LSTMDecoder, self).__init__()

        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.dropout = dropout
        self.encoder_bridge = encoder_bridge

        self.lstm = nn.LSTM(
            input_size=input_size, hidden_size=hidden_size, num_layers=num_layers, batch_first=True,
            dropout=(dropout if num_layers > 1 else 0.0)
        )

        if self.encoder_bridge:
            self._encoder_bridge = nn.Linear(self.hidden_size, 2 * self.num_layers * self.hidden_size)
        else:
            # Initialise trainable hidden state initialisations
            self.h_0 = nn.Parameter(torch.zeros(self.num_layers, 1, self.hidden_size))
            self.c_0 = nn.Parameter(torch.zeros(self.num_layers, 1, self.hidden_size))

    @staticmethod
    def _pool_encoder_outputs(encoder_outputs: Tensor, encoder_lengths: Tensor) -> Tensor:
        mask = make_mask_2d(encoder_lengths, expand_dim=encoder_outputs.shape[2])
        mask = mask.to(device=encoder_outputs.device)
        encoder_outputs = torch.masked_fill(input=encoder_outputs, mask=mask, value=-torch.inf)
        encoder_outputs, _ = torch.max(encoder_outputs, dim=1)
        return encoder_outputs

    def _get_hidden(self, encoder_outputs: Tensor, encoder_lengths: Tensor, hidden: Tuple[Tensor, Tensor]):
        if hidden is not None:
            return hidden

        elif self.encoder_bridge:
            pooled_encoder_outputs = self._pool_encoder_outputs(encoder_outputs, encoder_lengths)
            pooled_encoder_outputs = self._encoder_bridge(pooled_encoder_outputs)
            batch_size = pooled_encoder_outputs.shape[0]
            pooled_encoder_outputs = pooled_encoder_outputs.reshape(batch_size, 2, self.num_layers, self.hidden_size)
            h_0 = pooled_encoder_outputs[:, 0].contiguous().transpose(0, 1)
            c_0 = pooled_encoder_outputs[:, 1].contiguous().transpose(0, 1)

            return h_0, c_0

        else:
            # Prepare hidden states
            batch_size = encoder_outputs.shape[0]
            h_0 = self.h_0.tile((1, batch_size, 1))
            c_0 = self.c_0.tile((1, batch_size, 1))

            return h_0, c_0

    def forward(self, inputs: Tensor, lengths: Tensor, encoder_outputs: Tensor, encoder_lengths: Tensor,
                hidden: Tuple[Tensor, Tensor] = None) -> DecoderOutput:
        # Pack sequence
        lengths = torch.clamp(lengths, 1)  # Enforce all lengths are >= 1 (required by pytorch)
        inputs = pack_padded_sequence(inputs, lengths, batch_first=True, enforce_sorted=False)

        # Initialise hidden states
        h_0, c_0 = self._get_hidden(encoder_outputs, encoder_lengths, hidden)
        old_hidden = (h_0, c_0)

        # Apply LSTM
        encoded, new_hidden = self.lstm(inputs, old_hidden)
        encoded, _ = pad_packed_sequence(encoded, batch_first=True)

        return encoded, (old_hidden, new_hidden)
