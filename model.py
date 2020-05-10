import torch
from torch import nn
import torch.nn.functional as F


class EF_LSTM(nn.Module):
    def __init__(self, num_classes, input_sizes, hidden_size, num_layers, dropout, bidirectional):
        super(EF_LSTM, self).__init__()
        self.LSTMs = nn.ModuleList([
            nn.LSTM(
                input_size=input_size,
                hidden_size=hidden_size,
                num_layers=num_layers,
                dropout=dropout if num_layers > 1 else 0,
                batch_first=True,
                bidirectional=bidirectional
            )
            for input_size in input_sizes
        ])

        linear_in_size = hidden_size * 3
        if bidirectional:
            linear_in_size = linear_in_size * 2

        self.out = nn.Sequential(
            nn.Linear(linear_in_size, int(linear_in_size / 3)),
            nn.ReLU(),
            nn.Linear(int(linear_in_size / 3), num_classes)
        )

    def forward(self, X_text, X_audio, X_vision):
        # (batch, seq_len, num_directions * hidden_size)
        output_text, _ = self.LSTMs[0](X_text)
        output_audio, _ = self.LSTMs[1](X_audio)
        output_vision, _ = self.LSTMs[2](X_vision)

        # (batch, num_directions * hidden_size)
        output_text = output_text[:, -1, :]
        output_audio = output_audio[:, -1, :]
        output_vision = output_vision[:, -1, :]

        logits = self.out(torch.cat((output_text, output_audio, output_vision), dim=1))

        return logits
