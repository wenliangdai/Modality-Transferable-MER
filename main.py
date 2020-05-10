import os
import torch
import numpy as np
from torch.utils.data import DataLoader
from cli import get_args
from utils import get_data
from trainer import Trainer
from model import EF_LSTM

if __name__ == "__main__":
    args = get_args()

    # Fix seed for reproducibility
    seed = args['seed']
    torch.manual_seed(seed)
    np.random.seed(seed)

    # Set device
    os.environ["CUDA_VISIBLE_DEVICES"] = args['cuda']
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

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

    num_classes = {
        'mosei_senti': 1,
    }

    input_sizes = {
        'mosei_senti': [300, 74, 35],
    }

    criterions = {
        'mosei_senti': torch.nn.L1Loss
    }

    model = EF_LSTM(
        num_classes=num_classes[args['dataset']],
        input_sizes=input_sizes[args['dataset']],
        hidden_size=args['hidden_size'],
        num_layers=args['num_layers'],
        dropout=args['dropout'],
        bidirectional=args['bidirectional']
    )
    model = model.to(device=device)

    optimizer = torch.optim.Adam(model.parameters(), lr=args['learning_rate'], weight_decay=args['weight_decay'])

    trainer = Trainer(args, model, criterions['mosei_senti'](), optimizer, device, dataloaders)
    trainer.train()
