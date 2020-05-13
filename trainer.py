import os
import copy
import torch
from tqdm import tqdm
from tabulate import tabulate
from evaluate import eval_mosei_senti
from utils import save

class Trainer():
    def __init__(self, args, model, criterion, optimizer, device, dataloaders):
        self.args = args
        self.model = model
        self.device = device
        self.criterion = criterion
        self.optimizer = optimizer
        self.dataloaders = dataloaders
        self.earlyStop = args['early_stop']

        self.all_train_stats = []
        self.all_valid_stats = []
        self.all_test_stats = []

        self.prev_train_stats = [float('inf')] + [-float('inf')] * 5
        self.prev_valid_stats = [float('inf')] + [-float('inf')] * 5
        self.prev_test_stats = [float('inf')] + [-float('inf')] * 5

        self.best_valid_stats = [float('inf')] + [-float('inf')] * 5
        self.best_epoch = -1

        self.saving_path = './savings/stats/'

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

    def train(self):
        headers = ['Phase', 'MAE', 'Acc2', 'Acc5', 'Acc7', 'F1', 'Corr']
        for epoch in range(1, self.args['epochs'] + 1):
            print(f'=== Epoch {epoch} ===')
            # t_mae, t_acc2, t_acc5, t_acc7, t_f1, t_corr = self.train_one_epoch()
            # v_mae, v_acc2, v_acc5, v_acc7, v_f1, v_corr = self.eval_one_epoch()
            train_stats = self.train_one_epoch()
            valid_stats = self.eval_one_epoch()
            test_stats = self.eval_one_epoch('test')

            self.all_train_stats.append(train_stats)
            self.all_valid_stats.append(valid_stats)
            self.all_test_stats.append(test_stats)

            for i in range(len(valid_stats)):
                if i == 0: # MAE
                    if valid_stats[i] < self.best_valid_stats[i]:
                        self.best_valid_stats[i] = valid_stats[i]
                else:
                    if valid_stats[i] > self.best_valid_stats[i]:
                        self.best_valid_stats[i] = valid_stats[i]
                        if i == 3: # Acc7
                            self.earlyStop = self.args['early_stop']
                            self.best_epoch = epoch
                    elif i == 3: # Acc7
                        self.earlyStop -= 1

            # For printing
            train_stats_str = self.make_stat(self.prev_train_stats, train_stats)
            valid_stats_str = self.make_stat(self.prev_valid_stats, valid_stats)
            test_stats_str = self.make_stat(self.prev_test_stats, test_stats)

            self.prev_train_stats = train_stats
            self.prev_valid_stats = valid_stats
            self.prev_test_stats = test_stats

            print(tabulate([['Train', *train_stats_str], ['Valid', *valid_stats_str], ['Test', *test_stats_str]], headers=headers))
            print()
            # End printing

            if self.earlyStop == 0:
                print('Early stopping...\n')
                break

        print('Best performance:')
        print(tabulate([
            [f'BEST ({self.best_epoch})', *self.best_valid_stats],
            [f'Test ({self.best_epoch})', *self.all_test_stats[self.best_epoch - 1]]
        ], headers=headers))

    def train_one_epoch(self):
        self.model.train()
        dataloader = self.dataloaders['train']
        epoch_loss = 0.0
        total_logits = None
        total_Y = None
        for X, Y, META in tqdm(dataloader, desc='Train'):
            _, X_text, X_audio, X_vision = X
            X_text = X_text.to(device=self.device)
            X_audio = X_audio.to(device=self.device)
            X_vision = X_vision.to(device=self.device)
            Y = Y.squeeze(-1).to(device=self.device)

            self.optimizer.zero_grad()
            with torch.set_grad_enabled(True):
                logits = self.model(X_text, X_audio, X_vision)
                loss = self.criterion(logits, Y)
                loss.backward()
                epoch_loss += loss.item() * Y.size(0)
                self.optimizer.step()
            total_logits = torch.cat((total_logits, logits), dim=0) if total_logits is not None else logits
            total_Y = torch.cat((total_Y, Y), dim=0) if total_Y is not None else Y
        epoch_loss /= len(dataloader.dataset)
        print(f'train loss = {epoch_loss}')
        # mae, acc2, acc5, acc7, f1, corr = eval_mosei_senti(logits, Y)
        return eval_mosei_senti(total_logits, total_Y)

    def eval_one_epoch(self, phase='valid'):
        self.model.eval()
        dataloader = self.dataloaders[phase]
        epoch_loss = 0.0
        total_logits = None
        total_Y = None
        for X, Y, META in tqdm(dataloader, desc=phase):
            _, X_text, X_audio, X_vision = X
            X_text = X_text.to(device=self.device)
            X_audio = X_audio.to(device=self.device)
            X_vision = X_vision.to(device=self.device)
            Y = Y.squeeze(-1).to(device=self.device)

            logits = self.model(X_text, X_audio, X_vision)
            loss = self.criterion(logits, Y)
            epoch_loss += loss.item() * Y.size(0)

            total_logits = torch.cat((total_logits, logits), dim=0) if total_logits is not None else logits
            total_Y = torch.cat((total_Y, Y), dim=0) if total_Y is not None else Y

        epoch_loss /= len(dataloader.dataset)
        print(f'{phase} loss = {epoch_loss}')
        # mae, acc2, acc5, acc7, f1, corr = eval_mosei_senti(logits, Y)
        return eval_mosei_senti(total_logits, total_Y)

    def save_stats(self):
        stats = {
            'train_stats': self.all_train_stats,
            'valid_stats': self.all_valid_stats,
            'test_stats': self.all_test_stats,
            'best_valid_stats': self.best_valid_stats,
            'best_epoch': self.best_epoch
        }

        save(stats, os.path.join(
            self.saving_path,
            f'{self.args['model']}_Acc2_{self.best_valid_stats[1]}_Acc7_{self.best_valid_stats[3]}_rand{self.args['seed']}.pt'
        ))

    def save_model(self):
        save(copy.deepcopy(self.model.state_dict()), os.path.join(
            self.saving_path,
            f'{self.args['model']}_Acc2_{self.best_valid_stats[1]}_Acc7_{self.best_valid_stats[3]}_rand{self.args['seed']}.pt'
        ))

