import os
import pickle
import torch
import numpy as np

def save(toBeSaved, filename, mode='wb'):
    dirname = os.path.dirname(filename)
    if not os.path.exists(dirname):
        os.makedirs(dirname)
    file = open(filename, mode)
    pickle.dump(toBeSaved, file, protocol=4)
    file.close()

def load(filename, mode='rb'):
    file = open(filename, mode)
    loaded = pickle.load(file)
    file.close()
    return loaded

def pad_sents(sents, pad_token):
    sents_padded = []
    lens = get_lens(sents)
    max_len = max(lens)
    sents_padded = [sents[i] + [pad_token] * (max_len - l) for i, l in enumerate(lens)]
    return sents_padded

def sort_sents(sents, reverse=True):
    sents.sort(key=(lambda s: len(s)), reverse=reverse)
    return sents

def get_mask(sents, unmask_idx=1, mask_idx=0):
    lens = get_lens(sents)
    max_len = max(lens)
    mask = [([unmask_idx] * l + [mask_idx] * (max_len - l)) for l in lens]
    return mask

def get_lens(sents):
    return [len(sent) for sent in sents]

def get_max_len(sents):
    max_len = max([len(sent) for sent in sents])
    return max_len

def truncate_sents(sents, length):
    sents = [sent[:length] for sent in sents]
    return sents

def get_loss_weight(labels, label_order):
    nums = [np.sum(labels == lo) for lo in label_order]
    loss_weight = torch.tensor([n / len(labels) for n in nums])
    return loss_weight

def capitalize_first_letter(data):
    if type(data) == 'str':
        return data.capitalize()
    elif type(data) == 'list':
        return [word.capitalize() for word in data]
    elif type(data) == 'numpy.ndarray':
        return np.array([word.capitalize() for word in data])

def cmumosei_round(a):
    if a < -2:
        res = -3
    if -2 <= a and a < -1:
        res = -2
    if -1 <= a and a < 0:
        res = -1
    if 0 <= a and a <= 0:
        res = 0
    if 0 < a and a <= 1:
        res = 1
    if 1 < a and a <= 2:
        res = 2
    if a > 2:
        res = 3
    return res


# if __name__ == '__main__':
#     from tqdm import tqdm
#     f = open('/Users/wenliangdai/Documents/Codes/Datasets/glove.840B.300d.txt', 'r', encoding='utf-8')
#     lines = f.readlines()
#     word2emb = {}
#     for line in tqdm(lines):
#         line = line.replace('\xa0', '').split()
#         word2emb[line[0]] = [float(n) for n in line[1:]]
#     save(word2emb, '/Users/wenliangdai/Documents/glove.dict.840B.300d.pt')
