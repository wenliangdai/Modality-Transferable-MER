import torch
import os
import pickle
import h5py
import numpy as np
# from src.dataset import Multimodal_Datasets
from src.dataset import MOSI, MOSEI, IEMOCAP

# def get_data(args, split='train'):
#     dataset = args['dataset']
#     alignment = 'a' if args['aligned'] else 'na'
#     data_path = os.path.join(args['data_path'], dataset) + f'_{split}_{alignment}.dt'
#     print(data_path)
#     if not os.path.exists(data_path):
#         print(f"  - Creating new {split} data")
#         data = Multimodal_Datasets(args['data_path'], dataset, split, args['aligned'])
#         torch.save(data, data_path)
#     else:
#         print(f"  - Found cached {split} data")
#         data = torch.load(data_path)
#     return data

def get_data(dataset, seq_len, file_folder, aligned, phase):
    processed_path = f'./processed_datasets/{dataset}_{seq_len}_{phase}{"" if aligned else "_noalign"}.pt'
    if os.path.exists(processed_path):
        print(f'Load processed dataset! - {phase}')
        return load(processed_path)

    if dataset == 'mosi':
        if seq_len == 20:
            data_path = os.path.join(file_folder, f'X_{phase}.h5')
            label_path = os.path.join(file_folder, f'y_{phase}.h5')
            data = np.array(h5py.File(data_path, 'r')['data'])
            labels = np.array(h5py.File(label_path.replace('X', 'y'), 'r')['data'])
            text = data[:, :, :300]
            audio = data[:, :, 300:305]
            vision = data[:, :, 305:]
            this_dataset = MOSI(list(range(len(labels))), text, audio, vision, labels, is20=True)
        else:
            data_path = os.path.join(file_folder, f'mosi_data{"" if aligned else "_noalign"}.pkl')
            data = load(data_path)
            data = data[phase]
            this_dataset = MOSI(data['id'], data['text'], data['audio'], data['vision'], data['labels'])
    elif dataset == 'mosei_senti':
        if seq_len == 20:
            text_data = np.array(h5py.File(os.path.join(data_path, f'text_{phase}.h5'), 'r')['d1'])
            audio_data = np.array(h5py.File(os.path.join(data_path, f'audio_{phase}.h5'), 'r')['d1'])
            vision_data = np.array(h5py.File(os.path.join(data_path, f'vision_{phase}.h5'), 'r')['d1'])
            labels = np.array(h5py.File(os.path.join(data_path, f'y_{phase}.h5'), 'r')['d1'])
            this_dataset = MOSEI(list(range(len(labels))), text_data, audio_data, vision_data, labels)
        else:
            data_path = os.path.join(file_folder, f'mosei_senti_data{"" if aligned else "_noalign"}.pkl')
            data = load(data_path)
            data = data[phase]
            this_dataset = MOSEI(data['id'], data['text'], data['audio'], data['vision'], data['labels'])
    elif dataset == 'mosei_emo':
        text_data = np.array(h5py.File(os.path.join(data_path, f'text_{phase}.h5'), 'r')['d1'])
        audio_data = np.array(h5py.File(os.path.join(data_path, f'audio_{phase}.h5'), 'r')['d1'])
        vision_data = np.array(h5py.File(os.path.join(data_path, f'vision_{phase}.h5'), 'r')['d1'])
        labels = np.array(h5py.File(os.path.join(data_path, f'ey_{phase}.h5'), 'r')['d1'])
        this_dataset = MOSEI(list(range(len(labels))), text_data, audio_data, vision_data, labels)
    elif dataset == 'iemocap':
        data_path = os.path.join(file_folder, f'mosi_data{"" if aligned else "_noalign"}.pkl')
        data = load(data_path)
        data = data[phase]
        this_dataset = IEMOCAP(list(range(len(data['labels']))), data['text'], data['audio'], data['vision'], data['labels'])
    else:
        raise ValueError('Wrong dataset!')

    save(this_dataset, processed_path)

    return this_dataset

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

# def save_load_name(args, name=''):
#     if args.aligned:
#         name = name if len(name) > 0 else 'aligned_model'
#     elif not args.aligned:
#         name = name if len(name) > 0 else 'nonaligned_model'

#     return name + '_' + args.model


# def save_model(args, model, name=''):
#     name = save_load_name(args, name)
#     torch.save(model, f'pre_trained_models/{name}.pt')


# def load_model(args, name=''):
#     name = save_load_name(args, name)
#     model = torch.load(f'pre_trained_models/{name}.pt')
#     return model
