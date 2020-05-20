import math
import torch
from torch import nn
from transformers import AlbertModel

class EF_Transformer(nn.Module):
    def __init__(self):
        super(EF_Transformer, self).__init__()
        encoder_layer = nn.TransformerEncoderLayer(d_model=160, nhead=8)
        norm = nn.LayerNorm(160)
        self.linear = nn.Linear(in_features=409, out_features=160, bias=True)
        self.pos_encoder = PositionalEncoding(160, dropout=0)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=4, norm=norm)
        self.out = nn.Sequential(
            nn.Linear(160, 80),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(80, 1)
        )

    def forward(self, X_text, X_audio, X_vision):
        fused_input = torch.cat((X_text, X_audio, X_vision), dim=-1)
        output = self.linear(fused_input)
        output = self.pos_encoder(output)
        output = self.transformer_encoder(output)
        output = torch.mean(output, dim=1)
        output = self.out(output)
        return output


class EF_ALBERT(nn.Module):
    def __init__(self):
        super(EF_Transformer, self).__init__()
        self.linear = nn.Linear(in_features=409, out_features=128, bias=True)
        self.albert = AlbertModel.from_pretrained('albert-base-v2')

    def forward(self, X_text, X_audio, X_vision):
        fused_input = torch.cat((X_text, X_audio, X_vision), dim=-1)
        output = self.linear(fused_input)
        output = self.transformer_encoder(output)
        output = torch.mean(output, dim=1)
        output = self.out(output)
        return output


'''
The PositionalEncoding module copied from:
https://pytorch.org/tutorials/beginner/transformer_tutorial.html
'''
class PositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout=0.1, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + self.pe[:x.size(0), :]
        return self.dropout(x)
