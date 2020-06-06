import copy
import torch
from tqdm import tqdm
from tabulate import tabulate
from src.evaluate import eval_mosei_senti
from src.trainers.base import TrainerBase


class SentiTrainer(TrainerBase):
    def __init__(self, args, model, criterion, optimizer, scheduler, device, dataloaders):
        super(SentiTrainer, self).__init__()
        self.all_train_stats = []
        self.all_valid_stats = []
        self.all_test_stats = []

        self.prev_train_stats = [float('inf')] + [-float('inf')] * 5
        self.prev_valid_stats = [float('inf')] + [-float('inf')] * 5
        self.prev_test_stats = [float('inf')] + [-float('inf')] * 5

        self.best_valid_stats = [float('inf')] + [-float('inf')] * 5
        self.best_epoch = -1

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
                            self.best_model = copy.deepcopy(self.model.state_dict())
                    elif i == 3: # Acc7
                        self.earlyStop -= 1

            train_stats_str = self.make_stat(self.prev_train_stats, train_stats)
            valid_stats_str = self.make_stat(self.prev_valid_stats, valid_stats)
            test_stats_str = self.make_stat(self.prev_test_stats, test_stats)

            self.prev_train_stats = train_stats
            self.prev_valid_stats = valid_stats
            self.prev_test_stats = test_stats

            print(tabulate([['Train', *train_stats_str], ['Valid', *valid_stats_str], ['Test', *test_stats_str]], headers=headers))
            print()

            if self.earlyStop == 0:
                print('Early stopping...\n')
                break

        print('=== Best performance ===')
        print(tabulate([
            [f'BEST ({self.best_epoch})', *self.best_valid_stats],
            [f'Test ({self.best_epoch})', *self.all_test_stats[self.best_epoch - 1]]
        ], headers=headers))

        self.save_stats()
        self.save_model()
        print('Results and model are saved!')

    def train_one_epoch(self):
        self.model.train()
        dataloader = self.dataloaders['train']
        epoch_loss = 0.0
        total_logits = None
        total_Y = None
        for X, Y, META in tqdm(dataloader, desc='Train'):
            X_text, X_audio, X_vision = X
            X_text = X_text.to(device=self.device)
            X_audio = X_audio.to(device=self.device)
            X_vision = X_vision.to(device=self.device)
            Y = Y.squeeze().to(device=self.device)

            self.optimizer.zero_grad()
            with torch.set_grad_enabled(True):
                logits = self.model(X_text, X_audio, X_vision)
                logits = logits.squeeze()
                loss = self.criterion(logits, Y)
                loss.backward()
                epoch_loss += loss.item() * Y.size(0)
                if self.args['clip'] > 0:
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.args['clip'])
                self.optimizer.step()
            total_logits = torch.cat((total_logits, logits), dim=0) if total_logits is not None else logits
            total_Y = torch.cat((total_Y, Y), dim=0) if total_Y is not None else Y

        epoch_loss /= len(dataloader.dataset)
        print(f'train loss = {epoch_loss}')
        # mae, acc2, acc5, acc7, f1, corr = eval_mosei_senti(logits, Y, exclude_zero=self.args['exclude_zero'])
        return eval_mosei_senti(total_logits, total_Y, exclude_zero=self.args['exclude_zero'])

    def eval_one_epoch(self, phase='valid'):
        self.model.eval()
        dataloader = self.dataloaders[phase]
        epoch_loss = 0.0
        total_logits = None
        total_Y = None
        for X, Y, META in tqdm(dataloader, desc=phase):
            X_text, X_audio, X_vision = X
            X_text = X_text.to(device=self.device)
            X_audio = X_audio.to(device=self.device)
            X_vision = X_vision.to(device=self.device)
            Y = Y.squeeze().to(device=self.device)

            with torch.set_grad_enabled(False):
                logits = self.model(X_text, X_audio, X_vision)
                logits = logits.squeeze()
                loss = self.criterion(logits, Y)
                epoch_loss += loss.item() * Y.size(0)

            total_logits = torch.cat((total_logits, logits), dim=0) if total_logits is not None else logits
            total_Y = torch.cat((total_Y, Y), dim=0) if total_Y is not None else Y

        epoch_loss /= len(dataloader.dataset)

        if phase == 'valid':
            self.scheduler.step(epoch_loss)

        print(f'{phase} loss = {epoch_loss}')
        # mae, acc2, acc5, acc7, f1, corr = eval_mosei_senti(logits, Y, exclude_zero=self.args['exclude_zero'])
        return eval_mosei_senti(total_logits, total_Y, exclude_zero=self.args['exclude_zero'])
