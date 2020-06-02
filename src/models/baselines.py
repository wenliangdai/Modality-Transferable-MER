import torch
from torch import nn
import torch.nn.functional as F


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


class TextSelectiveLSTM(nn.Module):
    def __init__(self, num_classes, input_sizes, hidden_size, hidden_sizes, num_layers, dropout, bidirectional):
        super(TextSelectiveLSTM, self).__init__()
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

    # def attention(self, attender, attendee):
    #     # attender (batch, hid_dim)
    #     # attendee (batch, seq_len, hid_dim)
    #     attender = attender.unsqueeze(-1)
    #     attn_weights = torch.bmm(attendee, attender) # (batch, seq_len, 1)
    #     attn_weights = F.softmax(attn_weights, dim=1)
    #     res = torch.sum(attendee * attn_weights, dim=1) # (batch, hid_dim)
    #     return res

    def forward(self, X_text, X_audio, X_vision):
        # (batch, seq_len, num_directions * hidden_size)
        output_text, _ = self.LSTMs[0](X_text)
        output_audio, _ = self.LSTMs[1](X_audio)
        output_vision, _ = self.LSTMs[2](X_vision)

        # (batch, num_directions * hidden_size)
        output_text = output_text[:, -1, :]
        # output_audio = output_audio[:, -1, :]
        # output_vision = output_vision[:, -1, :]

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
