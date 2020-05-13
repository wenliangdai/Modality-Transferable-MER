import os
import torch
import numpy as np
from torch.utils.data import DataLoader
from src.cli import get_args
from src.utils import get_data
from src.trainer import Trainer
from src.models import baselines # EF_LSTM, LF_LSTM, EF_LF_LSTM
from src.models.mult import MULTModel
from src.config import NUM_CLASSES, CRITERIONS, MULT_PARAMS

if __name__ == "__main__":
    args = get_args()

    # Fix seed for reproducibility
    seed = args['seed']
    torch.manual_seed(seed)
    np.random.seed(seed)

    # Set device
    # os.environ["CUDA_VISIBLE_DEVICES"] = args['cuda']
    device = torch.device(f"cuda:{args['cuda']}" if torch.cuda.is_available() else 'cpu')

    print("Start loading the data....")

    train_data = get_data(args, 'train')
    valid_data = get_data(args, 'valid')
    test_data = get_data(args, 'test')

    train_loader = DataLoader(train_data, batch_size=args['batch_size'], shuffle=True)
    valid_loader = DataLoader(valid_data, batch_size=args['batch_size'], shuffle=False)
    test_loader = DataLoader(test_data, batch_size=args['batch_size'], shuffle=False)

    dataloaders = {
        'train': train_loader,
        'valid': valid_loader,
        'test': test_loader
    }

    modal_dims = list(train_data.get_dim())

    model_type = args['model'].lower()

    if model_type == 'mult':
        mult_params = MULT_PARAMS[args['dataset']]
        mult_params['orig_d_l'] = modal_dims[0]
        mult_params['orig_d_a'] = modal_dims[1]
        mult_params['orig_d_v'] = modal_dims[2]
        model = MULTModel(mult_params)
    else:
        if model_type == 'lf':
            MODEL = baselines.LF_LSTM
        elif model_type == 'ef':
            MODEL = baselines.EF_LSTM
        elif model_type == 'eflf':
            MODEL = baselines.EF_LF_LSTM
        else:
            raise ValueError('Wrong model!')

        model = MODEL(
            num_classes=NUM_CLASSES[args['dataset']],
            input_sizes=modal_dims,
            hidden_size=args['hidden_size'],
            num_layers=args['num_layers'],
            dropout=args['dropout'],
            bidirectional=args['bidirectional']
        )
    model = model.to(device=device)

    optimizer = torch.optim.Adam(model.parameters(), lr=args['learning_rate'], weight_decay=args['weight_decay'])

    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', patience=args['patience'], verbose=True)

    trainer = Trainer(args, model, CRITERIONS[args['dataset']](), optimizer, scheduler, device, dataloaders)
    trainer.train()
