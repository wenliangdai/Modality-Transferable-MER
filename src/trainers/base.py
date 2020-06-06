import os
import copy
import torch
from tqdm import tqdm
from tabulate import tabulate
from src.evaluate import eval_mosei_senti, eval_iemocap
from src.utils import save

class TrainerBase():
    def __init__(self, args, model, criterion, optimizer, scheduler, device, dataloaders):
        self.args = args
        self.model = model
        self.best_model = copy.deepcopy(model.state_dict())
        self.device = device
        self.criterion = criterion
        self.optimizer = optimizer
        self.dataloaders = dataloaders
        self.scheduler = scheduler
        self.earlyStop = args['early_stop']

        self.saving_path = f"./savings/{args['dataset']}/"

    def make_stat(self, prev, curr):
        new_stats = []
        for i in range(len(prev)):
            if curr[i] > prev[i]:
                new_stats.append(f'{curr[i]:.4f} \u2191')
            elif curr[i] < prev[i]:
                new_stats.append(f'{curr[i]:.4f} \u2193')
            else:
                new_stats.append(f'{curr[i]:.4f} -')
        return new_stats

    def get_saving_file_name(self):
        name = f"{self.args['model']}_Acc2_{self.best_valid_stats[1]}_Acc7_{self.best_valid_stats[3]}_rand{self.args['seed']}.pt"
        if self.args['gru']:
            name = f'gru_{name}'
        return name

    def save_stats(self):
        stats = {
            'train_stats': self.all_train_stats,
            'valid_stats': self.all_valid_stats,
            'test_stats': self.all_test_stats,
            'best_valid_stats': self.best_valid_stats,
            'best_epoch': self.best_epoch
        }

        save(stats, os.path.join(self.saving_path, 'stats', self.get_saving_file_name()))

    def save_model(self):
        save(self.best_model, os.path.join(self.saving_path, 'models', self.get_saving_file_name()))