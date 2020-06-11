import numpy as np
from torch.utils.data.dataset import Dataset
import pickle
import os
from scipy import signal
import torch

# if torch.cuda.is_available():
#     torch.set_default_tensor_type('torch.cuda.FloatTensor')
# else:
#     torch.set_default_tensor_type('torch.FloatTensor')

############################################################################################
# This file provides basic processing script for the multimodal datasets we use. For other
# datasets, small modifications may be needed (depending on the type of the data, etc.)
############################################################################################

# class Multimodal_Datasets(Dataset):
#     def __init__(self, dataset_path, data='mosei_senti', split_type='train', if_align=False):
#         super(Multimodal_Datasets, self).__init__()
#         dataset_path = os.path.join(dataset_path, data + '_data.pkl' if if_align else data + '_data_noalign.pkl')
#         dataset = pickle.load(open(dataset_path, 'rb'))

#         # These are torch tensors
#         self.vision = torch.tensor(dataset[split_type]['vision'].astype(np.float32)).cpu().detach()
#         self.text = torch.tensor(dataset[split_type]['text'].astype(np.float32)).cpu().detach()
#         self.audio = dataset[split_type]['audio'].astype(np.float32)
#         self.audio[self.audio == -np.inf] = 0
#         self.audio = torch.tensor(self.audio).cpu().detach()
#         self.labels = torch.tensor(dataset[split_type]['labels'].astype(np.float32)).cpu().detach()

#         # Note: this is STILL an numpy array
#         self.meta = dataset[split_type]['id'] if 'id' in dataset[split_type].keys() else None

#         self.data = data

#         self.n_modalities = 3 # vision/ text/ audio

#     def get_n_modalities(self):
#         return self.n_modalities

#     def get_seq_len(self):
#         return self.text.shape[1], self.audio.shape[1], self.vision.shape[1]

#     def get_dim(self):
#         return self.text.shape[2], self.audio.shape[2], self.vision.shape[2]

#     def get_lbl_info(self):
#         # return number_of_labels, label_dim
#         return self.labels.shape[1], self.labels.shape[2]

#     def __len__(self):
#         return len(self.labels)

#     def __getitem__(self, index):
#         X = (index, self.text[index], self.audio[index], self.vision[index])
#         Y = self.labels[index]
#         META = (0, 0, 0) if self.meta is None else (self.meta[index][0], self.meta[index][1], self.meta[index][2])
#         if self.data == 'mosi':
#             META = (self.meta[index][0].decode('UTF-8'), self.meta[index][1].decode('UTF-8'), self.meta[index][2].decode('UTF-8'))
#         if self.data == 'iemocap':
#             Y = torch.argmax(Y, dim=-1)
#         return X, Y, META


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
    def __init__(self, id, text, audio, vision, labels, zsl=-1):
        super(MOSEI, self).__init__()

        zsl_text = []
        zsl_audio = []
        zsl_vision = []
        zsl_labels = []
        zsl_id = []
        if zsl != -1:
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
    def __init__(self, id, text, audio, vision, labels):
        super(IEMOCAP, self).__init__()
        self.vision = torch.tensor(vision, dtype=torch.float32)
        self.labels = torch.tensor(labels, dtype=torch.int32)
        self.text = torch.tensor(text, dtype=torch.float32)
        self.audio = torch.tensor(audio, dtype=torch.float32)
        self.audio[self.audio == -np.inf] = 0

        # "Neutral", "Happy", "Sad", "Angry"
        self.labels = torch.argmax(self.labels, dim=-1)
        self.id = id

    def get_seq_len(self):
        return self.text.shape[1], self.audio.shape[1], self.vision.shape[1]

    def get_dim(self):
        return self.text.shape[2], self.audio.shape[2], self.vision.shape[2]

    def __len__(self):
        return len(self.id)

    def __getitem__(self, index):
        return (self.text[index], self.audio[index], self.vision[index]), self.labels[index], self.id[index]
