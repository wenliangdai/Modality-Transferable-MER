import torch

NUM_CLASSES = {
    'mosi': 1,
    'mosei_senti': 1,
    'mosei_emo': 6,
    'iemocap': 4 # if full, then should be 9
}

EMOTIONS = {
    'mosei_emo': ['anger', 'disgust', 'fear', 'happy', 'sad', 'surprise'],
    'iemocap': ['neutral', 'happy', 'sad', 'angry'],
    'iemocap9': ['angry', 'exciting', 'fear', 'sad', 'surprising', 'frustrating', 'happy', 'neutral', 'disgust']
    # 'iemocap': ['happy', 'sad', 'angry', 'neutral']
}

CRITERIONS = {
    'mosei_senti': torch.nn.L1Loss,
    'mosi': torch.nn.L1Loss,
    'iemocap': torch.nn.CrossEntropyLoss
}

MULT_PARAMS = {
    'mosei_senti': {
        'vonly': True,
        'aonly': True,
        'lonly': True,
        'num_heads': 8,
        'layers': 4,
        'attn_dropout': 0.2,
        'res_dropout': 0.1,
        'relu_dropout': 0.1,
        'out_dropout': 0.1,
        'embed_dropout': 0.2,
        'attn_dropout_a': 0.0,
        'attn_dropout_v': 0.0,
        'attn_mask': True,
        'output_dim': 1
    },
    'mosi': {
        'vonly': True,
        'aonly': True,
        'lonly': True,
        'num_heads': 10,
        'layers': 4,
        'attn_dropout': 0.2,
        'res_dropout': 0.1,
        'relu_dropout': 0.1,
        'out_dropout': 0.1,
        'embed_dropout': 0.2,
        'attn_dropout_a': 0.0,
        'attn_dropout_v': 0.0,
        'attn_mask': True,
        'output_dim': 1
    },
    'iemocap': {
        'vonly': True,
        'aonly': True,
        'lonly': True,
        'num_heads': 10,
        'layers': 4,
        'attn_dropout': 0.25,
        'res_dropout': 0.1,
        'relu_dropout': 0.1,
        'out_dropout': 0.1,
        'embed_dropout': 0.3,
        'attn_dropout_a': 0.0,
        'attn_dropout_v': 0.0,
        'attn_mask': False,
        'output_dim': 4
    }
}
