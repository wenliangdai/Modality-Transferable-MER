import os
import copy
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
        best_test_stats = self.all_test_stats[self.best_epoch - 1]
        name = f"{self.args['model']}_wacc_{best_test_stats[0][6]}_f1_{best_test_stats[1][6]}_auc_{best_test_stats[2][6]}_rand{self.args['seed']}.pt"
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

        csv_path = os.path.join(self.saving_path, 'csv', self.get_saving_file_name()).replace('.pt', '.csv')
        dirname = os.path.dirname(csv_path)
        if not os.path.exists(dirname):
            os.makedirs(dirname)
        with open(csv_path, 'w') as f:
            for stat in self.all_test_stats[self.best_epoch - 1]:
                for n in stat:
                    f.write(f'{n:.4f},')
                f.write('\n')

    def save_model(self):
        save(self.best_model, os.path.join(self.saving_path, 'models', self.get_saving_file_name()))