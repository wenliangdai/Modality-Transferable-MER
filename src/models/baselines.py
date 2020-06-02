import torch
from torch import nn
# import torch.nn.functional as F


class EF_LSTM(nn.Module):
    def __init__(self, num_classes, input_sizes, hidden_size, hidden_sizes, num_layers, dropout, bidirectional):
        super(EF_LSTM, self).__init__()
        self.LSTM = nn.LSTM(
            input_size=sum(input_sizes),
            hidden_size=hidden_size,
            num_layers=num_layers,
            dropout=dropout if num_layers > 1 else 0,
            batch_first=True,
            bidirectional=bidirectional
        )

        linear_in_size = hidden_size
        if bidirectional:
            linear_in_size = linear_in_size * 2

        self.out = nn.Sequential(
            nn.Linear(linear_in_size, linear_in_size),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(linear_in_size, num_classes)
        )

    def forward(self, X_text, X_audio, X_vision):
        # (batch, seq_len, num_directions * hidden_size)
        fused_input = torch.cat((X_text, X_audio, X_vision), dim=-1)
        output_fused, _ = self.LSTM(fused_input)

        # (batch, num_directions * hidden_size)
        output_fused = output_fused[:, -1, :]

        logits = self.out(output_fused)

        return logits


class LF_LSTM(nn.Module):
    def __init__(self, num_classes, input_sizes, hidden_size, hidden_sizes, num_layers, dropout, bidirectional):
        super(LF_LSTM, self).__init__()
        self.LSTMs = nn.ModuleList([
            nn.LSTM(
                input_size=input_size,
                hidden_size=hidden_sizes[i],
                num_layers=num_layers,
                dropout=dropout if num_layers > 1 else 0,
                batch_first=True,
                bidirectional=bidirectional
            )
            for i, input_size in enumerate(input_sizes)
        ])

        linear_in_size = hidden_size * 3
        if bidirectional:
            linear_in_size = linear_in_size * 2

        self.out = nn.Sequential(
            nn.Linear(linear_in_size, int(linear_in_size / 3)),
            nn.ReLU(),
            nn.Dropout(0.1),
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


class EF_LF_LSTM(nn.Module):
    def __init__(self, num_classes, input_sizes, hidden_size, hidden_sizes, num_layers, dropout, bidirectional):
        super(EF_LF_LSTM, self).__init__()
        self.EF_LSTM = nn.LSTM(
            input_size=sum(input_sizes),
            hidden_size=hidden_size,
            num_layers=num_layers,
            dropout=dropout if num_layers > 1 else 0,
            batch_first=True,
            bidirectional=bidirectional
        )

        self.LF_LSTMs = nn.ModuleList([
            nn.LSTM(
                input_size=input_size,
                hidden_size=hidden_sizes[i],
                num_layers=num_layers,
                dropout=dropout if num_layers > 1 else 0,
                batch_first=True,
                bidirectional=bidirectional
            )
            for i, input_size in enumerate(input_sizes)
        ])

        # linear_in_size = hidden_size * 4
        linear_in_size = hidden_size + sum(hidden_sizes)
        if bidirectional:
            linear_in_size = linear_in_size * 2

        self.out = nn.Sequential(
            nn.Linear(linear_in_size, int(linear_in_size / 2)),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(int(linear_in_size / 2), num_classes)
        )

    def forward(self, X_text, X_audio, X_vision):
        # (batch, seq_len, num_directions * hidden_size)
        output_text, _ = self.LF_LSTMs[0](X_text)
        output_audio, _ = self.LF_LSTMs[1](X_audio)
        output_vision, _ = self.LF_LSTMs[2](X_vision)

        fused_input = torch.cat((X_text, X_audio, X_vision), dim=-1)
        output_fused, _ = self.EF_LSTM(fused_input)

        # (batch, num_directions * hidden_size)
        output_text = output_text[:, -1, :]
        output_audio = output_audio[:, -1, :]
        output_vision = output_vision[:, -1, :]
        output_fused = output_fused[:, -1, :]

        logits = self.out(torch.cat((output_text, output_audio, output_vision, output_fused), dim=1))

        return logits
