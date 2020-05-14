import torch

NUM_CLASSES = {
    'mosi': 1,
    'mosei_senti': 1,
    'iemocap': 8
}

# input_sizes = {
#     'mosei_senti': [300, 74, 35],
# }

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
        'attn_mask': False,
        'output_dim': 1
    },
    'mosi': {
        'vonly': True,
        'aonly': True,
        'lonly': True,
        'num_heads': 8,
        'layers': 4,
        'attn_dropout': 0.1,
        'res_dropout': 0.1,
        'relu_dropout': 0.1,
        'out_dropout': 0.1,
        'embed_dropout': 0.3,
        'attn_dropout_a': 0.0,
        'attn_dropout_v': 0.0,
        'attn_mask': False,
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
        'output_dim': 1
    }
}