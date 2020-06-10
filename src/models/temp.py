import torch
from torch import nn
import torch.nn.functional as F

class EmotionEmbAttnModel(nn.Module):
    def __init__(self, num_classes, input_sizes, hidden_size, hidden_sizes, num_layers, dropout, emo_weight, device, bidirectional=False, modalities='tav', gru=False):
        super(EmotionEmbAttnModel, self).__init__()

        self.num_classes = num_classes
        self.device = device
        self.modalities = modalities

        emo_weight = torch.FloatTensor(emo_weight)
        self.textEmoEmbs = nn.Embedding.from_pretrained(emo_weight)
        # self.textEmoEmbs.weight.requires_grad = False

        # def hook(module, grad_input, grad_output):
        #     print(module, grad_input, grad_output)
        #     exit(1)
        # self.textEmoEmbs.register_backward_hook(hook)


        self.affineAudio = nn.Linear(hidden_sizes[0], hidden_sizes[1])
        self.affineVisual = nn.Linear(hidden_sizes[0], hidden_sizes[2])

        RnnModel = nn.GRU if gru else nn.LSTM

        self.RNNs = nn.ModuleList([
            RnnModel(
                input_size=input_size,
                hidden_size=hidden_sizes[i],
                num_layers=num_layers,
                dropout=dropout if num_layers > 1 else 0,
                batch_first=True,
                bidirectional=bidirectional
            ) for i, input_size in enumerate(input_sizes)
        ])

        linearInSize = hidden_size
        if bidirectional:
            linearInSize = linearInSize * 2

        self.out = nn.Sequential(
            nn.Linear(linearInSize, int(linearInSize / 3)),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(int(linearInSize / 3), num_classes)
            # nn.Sigmoid()
        )

        # self.out = nn.Sequential(
        #     nn.Linear(emo_weight.size(0), num_classes),
        #     nn.Dropout(dropout),
        #     nn.Sigmoid()
        # )

    def attention(self, attender, attendee):
        # attender: (batch, hid_dim)
        # attendee: (batch, seq_len, hid_dim)
        attender = attender.unsqueeze(-1)
        attn_weights = torch.bmm(attendee, attender) # (batch, seq_len, 1)
        # attn_weights = F.softmax(attn_weights, dim=1)
        # res = torch.sum(attendee * attn_weights, dim=1) # (batch, hid_dim)
        # return res
        return attn_weights.squeeze(-1)

    def forward(self, X_text, X_audio, X_visual):
        # TODO: try residual connection

        batch_size = X_text.size(0)
        logits = None
        if 't' in self.modalities:
            output_text, _ = self.RNNs[0](X_text)
            output_text = output_text[:, -1, :]
            text_emo_vecs_origin = self.textEmoEmbs(torch.LongTensor(list(range(self.num_classes))).to(self.device))
            text_emo_vecs = text_emo_vecs_origin.unsqueeze(0).repeat(batch_size, 1, 1)
            text_attn_weights = self.attention(output_text, text_emo_vecs)
            logits = text_attn_weights if logits is None else logits + text_attn_weights

        if 'a' in self.modalities:
            output_audio, _ = self.RNNs[1](X_audio)
            output_audio = output_audio[:, -1, :]
            audio_emo_vecs = self.affineAudio(text_emo_vecs_origin)
            audio_emo_vecs = audio_emo_vecs.unsqueeze(0).repeat(batch_size, 1, 1)
            audio_attn_weights = self.attention(output_audio, audio_emo_vecs)
            logits = audio_attn_weights if logits is None else logits + audio_attn_weights

        if 'v' in self.modalities:
            output_visual, _ = self.RNNs[2](X_visual)
            output_visual = output_visual[:, -1, :]
            visual_emo_vecs = self.affineVisual(text_emo_vecs_origin)
            visual_emo_vecs = visual_emo_vecs.unsqueeze(0).repeat(batch_size, 1, 1)
            visual_attn_weights = self.attention(output_visual, visual_emo_vecs)
            logits = visual_attn_weights if logits is None else logits + visual_attn_weights

        return logits
