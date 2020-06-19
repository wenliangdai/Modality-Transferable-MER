import copy
import torch
from torch.nn.utils import clip_grad_norm_
from tqdm import tqdm
from tabulate import tabulate
from src.evaluate import eval_mosei_emo
from src.trainers.base import TrainerBase
from src.utils import save
from src.gem import store_grad, overwrite_grad, project2cone2

class MoseiEmoTrainer(TrainerBase):
    def __init__(self, args, model, criterion, optimizer, scheduler, device, dataloaders, criterion2=None):
        super(MoseiEmoTrainer, self).__init__(args, model, criterion, optimizer, scheduler, device, dataloaders)
        self.all_train_stats = []
        self.all_valid_stats = []
        self.all_test_stats = []

        self.headers = [
            ['phase', 'anger (wacc)', 'disgust (wacc)', 'fear (wacc)', 'happy (wacc)', 'sad (wacc)', 'surprise (wacc)', 'average'],
            ['phase', 'anger (f1)', 'disgust (f1)', 'fear (f1)', 'happy (f1)', 'sad (f1)', 'surprise (f1)', 'average'],
            ['phase', 'anger (auc)', 'disgust (auc)', 'fear (auc)', 'happy (auc)', 'sad (auc)', 'surprise (auc)', 'average'],
            ['phase', 'acc', 'acc_subset', 'acc_intersect', 'auc_score']
        ]

        zsl = args['zsl']
        if zsl != -1:
            for i in range(3):
                self.headers[i] = self.headers[i][:zsl + 1] + self.headers[i][zsl + 2:]

        self.prev_train_stats = [
            [-float('inf')] * (model.num_classes + 2 if args['gem'] else model.num_classes + 1), # single wacc
            [-float('inf')] * (model.num_classes + 2 if args['gem'] else model.num_classes + 1), # single f1
            [-float('inf')] * (model.num_classes + 2 if args['gem'] else model.num_classes + 1), # single auc
            [-float('inf')] * 3 # acc all
        ]

        self.prev_valid_stats = [
            [-float('inf')] * (model.num_classes + 2 if args['gem'] else model.num_classes + 1), # single wacc
            [-float('inf')] * (model.num_classes + 2 if args['gem'] else model.num_classes + 1), # single f1
            [-float('inf')] * (model.num_classes + 2 if args['gem'] else model.num_classes + 1), # single auc
            [-float('inf')] * 3 # acc all
        ]

        self.prev_test_stats = [
            [-float('inf')] * (model.num_classes + 2 if args['gem'] else model.num_classes + 1), # single wacc
            [-float('inf')] * (model.num_classes + 2 if args['gem'] else model.num_classes + 1), # single f1
            [-float('inf')] * (model.num_classes + 2 if args['gem'] else model.num_classes + 1), # single auc
            [-float('inf')] * 3 # acc all
        ]

        self.best_valid_stats = [
            [-float('inf')] * (model.num_classes + 2 if args['gem'] else model.num_classes + 1), # single wacc
            [-float('inf')] * (model.num_classes + 2 if args['gem'] else model.num_classes + 1), # single f1
            [-float('inf')] * (model.num_classes + 2 if args['gem'] else model.num_classes + 1), # single auc
            [-float('inf')] * 3 # acc all
        ]

        self.best_epoch = -1

        if args['gem']:
            n_tasks = 2
            self.grad_dims_model = []
            for p in model.parameters():
                self.grad_dims_model.append(p.data.numel())
            self.grads_model = torch.Tensor(sum(self.grad_dims_model), n_tasks)
            self.grads_model = self.grads_model.to(device=self.device)

            self.criterion2 = criterion2

    def train(self):
        for epoch in range(1, self.args['epochs'] + 1):
            print(f'=== Epoch {epoch} ===')
            if self.args['gem']:
                train_stats = self.ft_one_epoch()
            else:
                train_stats = self.train_one_epoch()
            valid_stats = self.eval_one_epoch()
            test_stats = self.eval_one_epoch('test')

            self.all_train_stats.append(train_stats)
            self.all_valid_stats.append(valid_stats)
            self.all_test_stats.append(test_stats)

            for i in range(len(self.headers)):
                for j in range(len(valid_stats[i])):
                    is_pivot = (i == 0 and j == (len(valid_stats[i]) - 1)) # auc average
                    if valid_stats[i][j] > self.best_valid_stats[i][j]:
                        self.best_valid_stats[i][j] = valid_stats[i][j]
                        if is_pivot:
                            self.earlyStop = self.args['early_stop']
                            self.best_epoch = epoch
                            self.best_model = copy.deepcopy(self.model.state_dict())
                    elif is_pivot:
                        self.earlyStop -= 1

                train_stats_str = self.make_stat(self.prev_train_stats[i], train_stats[i])
                valid_stats_str = self.make_stat(self.prev_valid_stats[i], valid_stats[i])
                test_stats_str = self.make_stat(self.prev_test_stats[i], test_stats[i])

                self.prev_train_stats[i] = train_stats[i]
                self.prev_valid_stats[i] = valid_stats[i]
                self.prev_test_stats[i] = test_stats[i]

                print(tabulate([['Train', *train_stats_str], ['Valid', *valid_stats_str], ['Test', *test_stats_str]], headers=self.headers[i]))
                print()

            if self.earlyStop == 0:
                print('Early stopping...\n')
                break

        print('=== Best performance ===')
        print(tabulate([[f'Test ({self.best_epoch})', *self.all_test_stats[self.best_epoch - 1][0]]], headers=self.headers[0]))
        print(tabulate([[f'Test ({self.best_epoch})', *self.all_test_stats[self.best_epoch - 1][1]]], headers=self.headers[1]))
        print(tabulate([[f'Test ({self.best_epoch})', *self.all_test_stats[self.best_epoch - 1][2]]], headers=self.headers[2]))

        self.save_stats()
        self.save_model()
        print('Results and model are saved!')

        print(self.model.modality_weights.weight)

    def valid(self):
        valid_stats = self.eval_one_epoch()
        for i in range(len(self.headers)):
            print(tabulate([['Valid', *valid_stats[i]]], headers=self.headers[i]))
            print()
        # for stat in valid_stats:
        #     for n in stat:
        #         print(f'{n:.4f},', end='')
        # print()

    def test(self):
        test_stats = self.eval_one_epoch('test')
        for i in range(len(self.headers)):
            print(tabulate([['Test', *test_stats[i]]], headers=self.headers[i]))
            print()
        for stat in test_stats:
            for n in stat:
                print(f'{n:.4f},', end='')
        print()

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
                logits = self.model(X_text, X_audio, X_vision) # (batch_size, num_emotions), already after sigmoid/softmax
                loss = self.criterion(logits, Y)
                loss.backward()
                epoch_loss += loss.item() * Y.size(0)
                if self.args['clip'] > 0:
                    clip_grad_norm_(self.model.parameters(), self.args['clip'])
                self.optimizer.step()
            total_logits = torch.cat((total_logits, logits), dim=0) if total_logits is not None else logits
            total_Y = torch.cat((total_Y, Y), dim=0) if total_Y is not None else Y

        epoch_loss /= len(dataloader.dataset)
        print(f'train loss = {epoch_loss}')
        return eval_mosei_emo(total_logits, total_Y, self.args['threshold'], self.args['verbose'])

    def ft_one_epoch(self):
        self.model.train()
        train_data_generator = self.dataloaders['train']
        fsl_train_loader = self.dataloaders['ft_train']
        epoch_loss = 0.0
        total_logits = None
        total_Y = None
        for X, Y, META in tqdm(fsl_train_loader, desc='Train'):
            X_text, X_audio, X_vision = X
            X_text = X_text.to(device=self.device)
            X_audio = X_audio.to(device=self.device)
            X_vision = X_vision.to(device=self.device)
            Y = Y.squeeze().to(device=self.device)
            
            with torch.set_grad_enabled(True):
                # original data samples
                self.optimizer.zero_grad()
                X_ori, Y_ori, META_ori = next(train_data_generator)
                X_text_ori, X_audio_ori, X_vision_ori = X_ori
                X_text_ori = X_text_ori.to(device=self.device)
                X_audio_ori = X_audio_ori.to(device=self.device)
                X_vision_ori = X_vision_ori.to(device=self.device)
                Y_ori = Y_ori.squeeze().to(device=self.device)
                logits_ori = self.model(X_text_ori, X_audio_ori, X_vision_ori, oneemotionless=True)
                loss_ori = self.criterion2(logits_ori, Y_ori)
                loss_ori.backward()
                store_grad(self.model.parameters, self.grads_model, self.grad_dims_model, task_id=0)
                
                # ft data samples
                self.optimizer.zero_grad()
                logits = self.model(X_text, X_audio, X_vision) # (batch_size, num_emotions), already after sigmoid/softmax
                loss = self.criterion(logits, Y)
                loss.backward()
                store_grad(self.model.parameters, self.grads_model, self.grad_dims_model, task_id=1)
                epoch_loss += loss.item() * Y.size(0)

                dotp = torch.mm(self.grads_model[:, 1].unsqueeze(0), self.grads_model[:, 0].unsqueeze(1))
                if (dotp < 0).sum() != 0:
                    project2cone2(self.grads_model[:, 1].unsqueeze(1), self.grads_model[:, 0].unsqueeze(1), margin=self.args['margin'])
                    overwrite_grad(self.model.parameters, self.grads_model[:, 1], self.grad_dims_model)
                else:
                    print('Constraints are inconsistent, no solution. Do not use QP in this iteration')
                self.optimize_gem()

            total_logits = torch.cat((total_logits, logits), dim=0) if total_logits is not None else logits
            total_Y = torch.cat((total_Y, Y), dim=0) if total_Y is not None else Y

        epoch_loss /= len(fsl_train_loader.dataset)
        print(f'train loss = {epoch_loss}')
        return eval_mosei_emo(total_logits, total_Y, self.args['threshold'], self.args['verbose'])

    def optimize_gem(self):
        if self.args['clip'] > 0:
            clip_grad_norm_(self.model.parameters(), self.args['clip'])
        self.optimizer.step()

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
                loss = self.criterion(logits, Y)
                epoch_loss += loss.item() * Y.size(0)

            total_logits = torch.cat((total_logits, logits), dim=0) if total_logits is not None else logits
            total_Y = torch.cat((total_Y, Y), dim=0) if total_Y is not None else Y

        epoch_loss /= len(dataloader.dataset)

        if phase == 'valid':
            self.scheduler.step(epoch_loss)

        print(f'{phase} loss = {epoch_loss}')
        # save({
        #     'total_logits': total_logits.cpu().detach().numpy(),
        #     'total_Y': total_Y.cpu().detach().numpy()
        # }, './for_choose_thre2_val.pt')
        return eval_mosei_emo(total_logits, total_Y, self.args['threshold'], self.args['verbose'])


class IemocapTrainer(TrainerBase):
    def __init__(self, args, model, criterion, optimizer, scheduler, device, dataloaders):
        super(IemocapTrainer, self).__init__(args, model, criterion, optimizer, scheduler, device, dataloaders)
        self.all_train_stats = []
        self.all_valid_stats = []
        self.all_test_stats = []

        self.headers = [
            ['phase', 'neutral (acc)', 'happy (acc)', 'sad (acc)', 'angry (acc)', 'average'],
            ['phase', 'neutral (f1)', 'happy (f1)', 'sad (f1)', 'angry (f1)', 'average']
        ]

        zsl = args['zsl']
        if zsl != -1:
            for i in range(3):
                self.headers[i] = self.headers[i][:zsl + 1] + self.headers[i][zsl + 2:]

        self.prev_train_stats = [
            [-float('inf')] * (model.num_classes + 1), # single wacc
            [-float('inf')] * (model.num_classes + 1) # single f1
        ]

        self.prev_valid_stats = [
            [-float('inf')] * (model.num_classes + 1), # single wacc
            [-float('inf')] * (model.num_classes + 1) # single f1
        ]

        self.prev_test_stats = [
            [-float('inf')] * (model.num_classes + 1), # single wacc
            [-float('inf')] * (model.num_classes + 1) # single f1
        ]

        self.best_valid_stats = [
            [-float('inf')] * (model.num_classes + 1), # single wacc
            [-float('inf')] * (model.num_classes + 1) # single f1
        ]

        self.best_epoch = -1

    def train(self):
        for epoch in range(1, self.args['epochs'] + 1):
            print(f'=== Epoch {epoch} ===')
            train_stats = self.train_one_epoch()
            valid_stats = self.eval_one_epoch()
            test_stats = self.eval_one_epoch('test')

            self.all_train_stats.append(train_stats)
            self.all_valid_stats.append(valid_stats)
            self.all_test_stats.append(test_stats)

            for i in range(len(self.headers)):
                for j in range(len(valid_stats[i])):
                    is_pivot = (i == 0 and j == (len(valid_stats[i]) - 1)) # acc average
                    if valid_stats[i][j] > self.best_valid_stats[i][j]:
                        self.best_valid_stats[i][j] = valid_stats[i][j]
                        if is_pivot:
                            self.earlyStop = self.args['early_stop']
                            self.best_epoch = epoch
                            self.best_model = copy.deepcopy(self.model.state_dict())
                    elif is_pivot:
                        self.earlyStop -= 1

                train_stats_str = self.make_stat(self.prev_train_stats[i], train_stats[i])
                valid_stats_str = self.make_stat(self.prev_valid_stats[i], valid_stats[i])
                test_stats_str = self.make_stat(self.prev_test_stats[i], test_stats[i])

                self.prev_train_stats[i] = train_stats[i]
                self.prev_valid_stats[i] = valid_stats[i]
                self.prev_test_stats[i] = test_stats[i]

                print(tabulate([['Train', *train_stats_str], ['Valid', *valid_stats_str], ['Test', *test_stats_str]], headers=self.headers[i]))
                print()

            if self.earlyStop == 0:
                print('Early stopping...\n')
                break

        print('=== Best performance ===')
        print(tabulate([[f'Test ({self.best_epoch})', *self.all_test_stats[self.best_epoch - 1][0]]], headers=self.headers[0]))
        print(tabulate([[f'Test ({self.best_epoch})', *self.all_test_stats[self.best_epoch - 1][1]]], headers=self.headers[1]))

        self.save_stats()
        self.save_model()
        print('Results and model are saved!')
        if self.args['verbose']:
            print(self.model.modality_weights.weight)

    def valid(self):
        valid_stats = self.eval_one_epoch()
        for i in range(len(self.headers)):
            print(tabulate([['Valid', *valid_stats[i]]], headers=self.headers[i]))
            print()
        # for stat in valid_stats:
        #     for n in stat:
        #         print(f'{n:.4f},', end='')
        # print()

    def test(self):
        test_stats = self.eval_one_epoch('test')
        for i in range(len(self.headers)):
            print(tabulate([['Test', *test_stats[i]]], headers=self.headers[i]))
            print()
        for stat in test_stats:
            for n in stat:
                print(f'{n:.4f},', end='')
        print()

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
                logits = self.model(X_text, X_audio, X_vision) # (batch_size, num_emotions), already after sigmoid/softmax
                loss = self.criterion(logits, torch.argmax(Y, dim=-1))
                loss.backward()
                epoch_loss += loss.item() * Y.size(0)
                if self.args['clip'] > 0:
                    clip_grad_norm_(self.model.parameters(), self.args['clip'])
                self.optimizer.step()
            total_logits = torch.cat((total_logits, logits), dim=0) if total_logits is not None else logits
            total_Y = torch.cat((total_Y, Y), dim=0) if total_Y is not None else Y

        epoch_loss /= len(dataloader.dataset)
        print(f'train loss = {epoch_loss}')
        return eval_mosei_emo(total_logits, total_Y, self.args['threshold'], self.args['verbose'])

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
                loss = self.criterion(logits, torch.argmax(Y, dim=-1))
                epoch_loss += loss.item() * Y.size(0)

            total_logits = torch.cat((total_logits, logits), dim=0) if total_logits is not None else logits
            total_Y = torch.cat((total_Y, Y), dim=0) if total_Y is not None else Y

        epoch_loss /= len(dataloader.dataset)

        if phase == 'valid':
            self.scheduler.step(epoch_loss)

        print(f'{phase} loss = {epoch_loss}')
        # save({
        #     'total_logits': total_logits.cpu().detach().numpy(),
        #     'total_Y': total_Y.cpu().detach().numpy()
        # }, './for_choose_thre2_val.pt')
        return eval_mosei_emo(total_logits, total_Y, self.args['threshold'], self.args['verbose'])
