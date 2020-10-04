import numpy as np
from torch.utils.data.dataset import Dataset
import torch


class MOSI(Dataset):
    def __init__(self, id, text, audio, vision, labels, is20=False):
        super(MOSI, self).__init__()
        self.vision = torch.tensor(vision, dtype=torch.float32)
        self.labels = torch.tensor(labels.squeeze(), dtype=torch.float32)
        self.text = torch.tensor(text, dtype=torch.float32)
        self.audio = torch.tensor(audio, dtype=torch.float32)
        self.audio[self.audio == -np.inf] = 0
        self.id = id
        self.is20 = is20

    def get_seq_len(self):
        return self.text.shape[1], self.audio.shape[1], self.vision.shape[1]

    def get_dim(self):
        return self.text.shape[2], self.audio.shape[2], self.vision.shape[2]

    def __len__(self):
        return len(self.id)

    def __getitem__(self, index):
        if self.is20:
            meta = self.id[index]
        else:
            meta = (self.id[index][0].decode('UTF-8'), self.id[index][1].decode('UTF-8'), self.id[index][2].decode('UTF-8'))
        return (self.text[index], self.audio[index], self.vision[index]), self.labels[index], meta


class MOSEI(Dataset):
    def __init__(self, id, text, audio, vision, labels, zsl=-1, fsl=-1):
        super(MOSEI, self).__init__()

        if zsl != -1:
            zsl_text = []
            zsl_audio = []
            zsl_vision = []
            zsl_labels = []
            zsl_id = []

            labels = labels.tolist()
            for i in range(len(labels)):
                if labels[i][zsl] != 1:
                    zsl_text.append(text[i])
                    zsl_audio.append(audio[i])
                    zsl_vision.append(vision[i])
                    zsl_labels.append(labels[i][:zsl] + labels[i][zsl + 1:])
                    zsl_id.append(id[i])

            text = zsl_text
            audio = zsl_audio
            vision = zsl_vision
            labels = zsl_labels
            id = zsl_id

        if fsl != -1:
            fsl_num = np.ceil(np.sum(labels, axis=0)[fsl] * 0.01)
            counter = 0
            fsl_text = []
            fsl_audio = []
            fsl_vision = []
            fsl_labels = []
            fsl_id = []
            for i in range(len(labels)):
                if labels[i][fsl] == 1:
                    if counter <= fsl_num:
                        counter += 1
                    else:
                        continue
                fsl_text.append(text[i])
                fsl_audio.append(audio[i])
                fsl_vision.append(vision[i])
                fsl_labels.append(labels[i])
                fsl_id.append(id[i])

            text = fsl_text
            audio = fsl_audio
            vision = fsl_vision
            labels = fsl_labels
            id = fsl_id

        self.vision = torch.tensor(vision, dtype=torch.float32)
        self.labels = torch.tensor(labels, dtype=torch.float32)

        self.text = torch.tensor(text, dtype=torch.float32)
        self.audio = torch.tensor(audio, dtype=torch.float32)
        self.audio[self.audio == -np.inf] = 0
        self.id = id

    def get_seq_len(self):
        return self.text.shape[1], self.audio.shape[1], self.vision.shape[1]

    def get_dim(self):
        return self.text.shape[2], self.audio.shape[2], self.vision.shape[2]

    def __len__(self):
        return len(self.id)

    def get_pos_weight(self):
        pos_nums = self.labels.sum(dim=0)
        neg_nums = self.__len__() - pos_nums
        pos_weight = neg_nums / pos_nums
        return pos_weight

    def __getitem__(self, index):
        return (self.text[index], self.audio[index], self.vision[index]), self.labels[index], self.id[index]


class IEMOCAP(Dataset):
    def __init__(self, id, text, audio, vision, labels, zsl=-1):
        super(IEMOCAP, self).__init__()
        self.vision = torch.tensor(vision, dtype=torch.float32)
        self.labels = torch.tensor(labels, dtype=torch.float32)
        self.text = torch.tensor(text, dtype=torch.float32)
        self.audio = torch.tensor(audio, dtype=torch.float32)
        self.audio[self.audio == -np.inf] = 0
        self.id = id

    def get_seq_len(self):
        return self.text.shape[1], self.audio.shape[1], self.vision.shape[1]

    def get_dim(self):
        return self.text.shape[2], self.audio.shape[2], self.vision.shape[2]

    def get_pos_weight(self):
        pos_nums = self.labels.sum(dim=0)
        neg_nums = self.__len__() - pos_nums
        pos_weight = neg_nums / pos_nums
        return pos_weight

    def __len__(self):
        return len(self.id)

    def __getitem__(self, index):
        return (self.text[index], self.audio[index], self.vision[index]), self.labels[index], self.id[index]
