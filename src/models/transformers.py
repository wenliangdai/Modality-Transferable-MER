import torch
from torch import nn
# from transformers import

class EF_Transformer(nn.Module):
    def __init__(self):
        super(EF_Transformer, self).__init__()
        encoder_layer = nn.TransformerEncoderLayer(d_model=320, nhead=8)
        norm = nn.LayerNorm(320)
        self.linear = nn.Linear(in_features=409, out_features=320, bias=True)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=4, norm=norm)
        self.out = nn.Sequential(
            nn.Linear(320, 160),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(160, 1)
        )

    def forward(self, X_text, X_audio, X_vision):
        fused_input = torch.cat((X_text, X_audio, X_vision), dim=-1)
        output = self.linear(fused_input)
        output = self.transformer_encoder(output)
        output = torch.mean(output, dim=1)
        output = self.out(output)
        return output
