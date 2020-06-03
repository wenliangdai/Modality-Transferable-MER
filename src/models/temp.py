import torch
from torch import nn
import torch.nn.functional as F

class EmotionEmbAttnModel(nn.Module):
    def __init__(self, num_classes, input_sizes, hidden_size, hidden_sizes, num_layers, dropout, bidirectional, emo_weight):
        super(EmotionEmbAttnModel, self).__init__()

        self.num_classes = num_classes

        self.affineVisual = nn.Linear(hidden_size, hidden_size)
        self.affineAudio = nn.Linear(hidden_size, hidden_size)

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

        linearInSize = hidden_size * 3
        if bidirectional:
            linearInSize = linearInSize * 2

        self.out = nn.Sequential(
            nn.Linear(linearInSize, int(linearInSize / 3)),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(int(linearInSize / 3), num_classes)
        )

        emo_weight = torch.FloatTensor(emo_weight)
        self.textEmoEmbs = nn.Embedding.from_pretrained(emo_weight)
        # self.textEmoEmbs.weight = emb_weight

    def attention(self, attender, attendee):
        # attender: (batch, hid_dim)
        # attendee: (batch, seq_len, hid_dim)
        attender = attender.unsqueeze(-1)
        attn_weights = torch.bmm(attendee, attender) # (batch, seq_len, 1)
        attn_weights = F.softmax(attn_weights, dim=1)
        res = torch.sum(attendee * attn_weights, dim=1) # (batch, hid_dim)
        return res

    def forward(self, X_text, X_audio, X_visual):
        # (batch, seq_len, num_directions * hidden_size)
        output_text, _ = self.LSTMs[0](X_text)
        output_audio, _ = self.LSTMs[1](X_audio)
        output_visual, _ = self.LSTMs[2](X_visual)

        batch_size = output_text.size(0)

        # (batch, num_directions * hidden_size)
        output_text = output_text[:, -1, :]
        output_audio = output_audio[:, -1, :]
        output_visual = output_visual[:, -1, :]

        # (num_classes, 300)
        text_emo_vecs = self.textEmoEmbs(torch.LongTensor(list(range(self.num_classes))))
        visual_emo_vecs = self.affineVisual(text_emo_vecs)
        audio_emo_vecs = self.affineAudio(text_emo_vecs)

        text_emo_vecs = text_emo_vecs.unsqueeze(0).repeat(batch_size, 1, 1)
        visual_emo_vecs = visual_emo_vecs.unsqueeze(0).repeat(batch_size, 1, 1)
        audio_emo_vecs = audio_emo_vecs.unsqueeze(0).repeat(batch_size, 1, 1)

        text_attn_feature = self.attention(output_text, text_emo_vecs)
        visual_attn_feature = self.attention(output_visual, visual_emo_vecs)
        audio_attn_feature = self.attention(output_audio, audio_emo_vecs)

        # TODO: try residual connection

        logits = self.out(torch.cat((text_attn_feature, visual_attn_feature, audio_attn_feature), dim=1))
        return logits
