import torch
from tqdm import tqdm
from evaluate import eval_mosei_senti
from tabulate import tabulate

class Trainer():
    def __init__(self, args, model, criterion, optimizer, device, dataloaders):
        self.args = args
        self.model = model
        self.device = device
        self.criterion = criterion
        self.optimizer = optimizer
        self.dataloaders = dataloaders

    def train(self):
        for epoch in range(1, self.args['epochs'] + 1):
            print(f'=== Epoch {epoch} ===')
            # t_mae, t_acc2, t_acc5, t_acc7, t_f1, t_corr = self.train_one_epoch()
            # v_mae, v_acc2, v_acc5, v_acc7, v_f1, v_corr = self.validate_one_epoch()
            train_stats = self.train_one_epoch()
            valid_stats = self.validate_one_epoch()

            headers = ['Phase', 'MAE', 'Acc2', 'Acc5', 'Acc7', 'F1', 'Corr']
            print(tabulate([['Train', *train_stats], ['Valid', *valid_stats]], headers=headers))
            print()

    def train_one_epoch(self):
        self.model.train()
        dataloader = self.dataloaders['train']
        epoch_loss = 0.0
        for X, Y, META in tqdm(dataloader, desc='Training'):
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

        epoch_loss /= len(dataloader.dataset)
        print(f'Train loss = {epoch_loss}')
        # mae, acc2, acc5, acc7, f1, corr = eval_mosei_senti(logits, Y)
        return eval_mosei_senti(logits, Y)

    def validate_one_epoch(self):
        self.model.eval()
        dataloader = self.dataloaders['valid']
        epoch_loss = 0.0
        for X, Y, META in tqdm(dataloader, desc='Validating'):
            _, X_text, X_audio, X_vision = X
            X_text = X_text.to(device=self.device)
            X_audio = X_audio.to(device=self.device)
            X_vision = X_vision.to(device=self.device)
            Y = Y.squeeze(-1).to(device=self.device)

            logits = self.model(X_text, X_audio, X_vision)
            loss = self.criterion(logits, Y)
            epoch_loss += loss.item() * Y.size(0)

        epoch_loss /= len(dataloader.dataset)
        print(f'Valid loss = {epoch_loss}')
        # mae, acc2, acc5, acc7, f1, corr = eval_mosei_senti(logits, Y)
        return eval_mosei_senti(logits, Y)

    def test(self):
        pass

