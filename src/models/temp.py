import torch
from torch import nn
import torch.nn.functional as F

class EmotionEmbAttnModel(nn.Module):
    def __init__(self, num_classes, input_sizes, hidden_size, hidden_sizes, num_layers, dropout, bidirectional):
        super(EmotionEmbAttnModel, self).__init__()

        self.affine_visual = nn.Linear(hidden_size, hidden_size)
        self.affine_audio = nn.Linear(hidden_size, hidden_size)

        self.LSTMs = nn.ModuleList([
            nn.LSTM(
                input_size=input_size,
                hidden_size=hidden_size,
                num_layers=num_layers,
                dropout=dropout if num_layers > 1 else 0,
                batch_first=True,
                bidirectional=bidirectional
            ) for i, input_size in enumerate(input_sizes)
        ])

    def forward(self, X_text, X_audio, X_vision):
        pass