import torch
from torch import nn


class EF_RNN(nn.Module):
    def __init__(self, num_classes, input_sizes, hidden_size, hidden_sizes, num_layers, dropout,
                 bidirectional=False, gru=False):
        super(EF_RNN, self).__init__()

        self.num_classes = num_classes

        RnnModel = nn.GRU if gru else nn.LSTM

        self.RNN = RnnModel(
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

        self.out = nn.Linear(linear_in_size, num_classes)

    def forward(self, X_text, X_audio, X_vision):
        # (batch, seq_len, num_directions * hidden_size)
        fused_input = torch.cat((X_text, X_audio, X_vision), dim=-1)
        output_fused, _ = self.RNN(fused_input)

        # (batch, num_directions * hidden_size)
        output_fused = output_fused[:, -1, :]

        logits = self.out(output_fused)

        return logits


class LF_RNN(nn.Module):
    def __init__(self, num_classes, input_sizes, hidden_size, hidden_sizes, num_layers, dropout, bidirectional=False, gru=False):
        super(LF_RNN, self).__init__()

        self.num_classes = num_classes

        RnnModel = nn.GRU if gru else nn.LSTM

        self.RNNs = nn.ModuleList([
            RnnModel(
                input_size=input_size,
                hidden_size=hidden_sizes[i],
                num_layers=num_layers,
                dropout=dropout if num_layers > 1 else 0,
                batch_first=True,
                bidirectional=bidirectional
            )
            for i, input_size in enumerate(input_sizes)
        ])

        linear_in_size = sum(hidden_sizes)
        if bidirectional:
            linear_in_size = linear_in_size * 2

        self.out = nn.Linear(linear_in_size, num_classes)

    def forward(self, X_text, X_audio, X_vision):
        # (batch, seq_len, num_directions * hidden_size)
        output_text, _ = self.RNNs[0](X_text)
        output_audio, _ = self.RNNs[1](X_audio)
        output_vision, _ = self.RNNs[2](X_vision)

        # (batch, num_directions * hidden_size)
        output_text = output_text[:, -1, :]
        output_audio = output_audio[:, -1, :]
        output_vision = output_vision[:, -1, :]

        logits = self.out(torch.cat((output_text, output_audio, output_vision), dim=1))

        return logits


class EF_LF_RNN(nn.Module):
    def __init__(self, num_classes, input_sizes, hidden_size, hidden_sizes, num_layers, dropout, bidirectional=False, gru=False):
        super(EF_LF_RNN, self).__init__()

        RnnModel = nn.GRU if gru else nn.LSTM

        self.EF_RNN = RnnModel(
            input_size=sum(input_sizes),
            hidden_size=hidden_size,
            num_layers=num_layers,
            dropout=dropout if num_layers > 1 else 0,
            batch_first=True,
            bidirectional=bidirectional
        )

        self.LF_RNNs = nn.ModuleList([
            RnnModel(
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
        output_text, _ = self.LF_RNNs[0](X_text)
        output_audio, _ = self.LF_RNNs[1](X_audio)
        output_vision, _ = self.LF_RNNs[2](X_vision)

        fused_input = torch.cat((X_text, X_audio, X_vision), dim=-1)
        output_fused, _ = self.EF_RNN(fused_input)

        # (batch, num_directions * hidden_size)
        output_text = output_text[:, -1, :]
        output_audio = output_audio[:, -1, :]
        output_vision = output_vision[:, -1, :]
        output_fused = output_fused[:, -1, :]

        logits = self.out(torch.cat((output_text, output_audio, output_vision, output_fused), dim=1))

        return logits


class TextSelectiveRNN(nn.Module):
    def __init__(self, num_classes, input_sizes, hidden_size, hidden_sizes, num_layers, dropout, bidirectional=False, gru=False):
        super(TextSelectiveRNN, self).__init__()

        RnnModel = nn.GRU if gru else nn.LSTM

        self.RNNs = nn.ModuleList([
            RnnModel(
                input_size=input_size,
                hidden_size=hidden_sizes[i],
                num_layers=num_layers,
                dropout=dropout if num_layers > 1 else 0,
                batch_first=True,
                bidirectional=bidirectional
            )
            for i, input_size in enumerate(input_sizes)
        ])

        linear_in_size = sum(hidden_sizes)
        if bidirectional:
            linear_in_size = linear_in_size * 2

        self.out = nn.Sequential(
            nn.Linear(linear_in_size, int(linear_in_size / 3)),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(int(linear_in_size / 3), num_classes)
        )

        self.attention = Attention(hidden_sizes[0], hidden_sizes[0])

    def forward(self, X_text, X_audio, X_vision):
        # (batch, seq_len, num_directions * hidden_size)
        output_text, _ = self.RNNs[0](X_text)
        output_audio, _ = self.RNNs[1](X_audio)
        output_vision, _ = self.RNNs[2](X_vision)

        # (batch, num_directions * hidden_size)
        output_text = output_text[:, -1, :]

        attn_audio = self.attention(output_text, output_audio)
        attn_vision = self.attention(output_text, output_vision)

        logits = self.out(torch.cat((output_text, attn_audio, attn_vision), dim=1))

        return logits


class Attention(nn.Module):
    def __init__(self, enc_hid_dim, dec_hid_dim):
        super().__init__()
        self.attn = nn.Linear(enc_hid_dim + dec_hid_dim, dec_hid_dim)
        self.v = nn.Linear(dec_hid_dim, 1, bias = False)

    def forward(self, hidden, encoder_outputs):

        #hidden = [batch size, dec hid dim]
        #encoder_outputs = [batch size, src len, enc hid dim]

        batch_size = encoder_outputs.shape[0]
        src_len = encoder_outputs.shape[1]

        #repeat decoder hidden state src_len times
        hidden = hidden.unsqueeze(1).repeat(1, src_len, 1)

        # encoder_outputs = encoder_outputs.permute(1, 0, 2)

        #hidden = [batch size, src len, dec hid dim]
        #encoder_outputs = [batch size, src len, enc hid dim]

        energy = torch.tanh(self.attn(torch.cat((hidden, encoder_outputs), dim = 2)))

        #energy = [batch size, src len, dec hid dim]

        attention = self.v(energy).squeeze(2)

        #attention= [batch size, src len]

        attn_weights = F.softmax(attention, dim=1).unsqueeze(-1)

        return torch.sum(encoder_outputs * attn_weights, dim=1)
