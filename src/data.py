import os
import pickle
import h5py
import numpy as np
# from src.dataset import Multimodal_Datasets
from src.dataset import MOSI, MOSEI, IEMOCAP
from src.utils import save, load, cmumosei_round

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

def get_data(args, phase):
    dataset = args['dataset']
    seq_len = args['data_seq_len']
    file_folder = args['data_folder']
    aligned = args['aligned']

    zsl = args['zsl']
    fsl = args['fsl'] if phase == 'train' else -1

    processed_path = f'./processed_datasets/{dataset}_{seq_len}_{phase}{"" if aligned else "_noalign"}.pt'
    if os.path.exists(processed_path) and zsl == -1 and fsl == -1:
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
            text_data = np.array(h5py.File(os.path.join(file_folder, f'text_{phase}.h5'), 'r')['d1'])
            audio_data = np.array(h5py.File(os.path.join(file_folder, f'audio_{phase}.h5'), 'r')['d1'])
            vision_data = np.array(h5py.File(os.path.join(file_folder, f'vision_{phase}.h5'), 'r')['d1'])
            labels = np.array(h5py.File(os.path.join(file_folder, f'y_{phase}.h5'), 'r')['d1'])
            this_dataset = MOSEI(list(range(len(labels))), text_data, audio_data, vision_data, labels)
        else:
            data_path = os.path.join(file_folder, f'mosei_senti_data{"" if aligned else "_noalign"}.pkl')
            data = load(data_path)
            data = data[phase]
            this_dataset = MOSEI(data['id'], data['text'], data['audio'], data['vision'], data['labels'])
    elif dataset == 'mosei_emo':
        text_data = np.array(h5py.File(os.path.join(file_folder, f'text_{phase}_emb.h5'), 'r')['d1'])
        audio_data = np.array(h5py.File(os.path.join(file_folder, f'audio_{phase}.h5'), 'r')['d1'])
        vision_data = np.array(h5py.File(os.path.join(file_folder, f'video_{phase}.h5'), 'r')['d1'])
        labels = np.array(h5py.File(os.path.join(file_folder, f'ey_{phase}.h5'), 'r')['d1']) # (N, 6)

        # Class order: Anger Disgust Fear Happy Sad Surprise
        if args['multi_level_classify']:
            # TODO: this should be length 24
            for i in range(labels.shape[0]):
                labels[i] = [cmumosei_round(l) for l in labels[i]]
        else:
            labels = np.array(labels > 0, np.int32)

        this_dataset = MOSEI(list(range(len(labels))), text_data, audio_data, vision_data, labels, zsl=zsl, fsl=fsl)
    elif dataset == 'iemocap':
        data_path = os.path.join(file_folder, f'mosi_data{"" if aligned else "_noalign"}.pkl')
        data = load(data_path)
        data = data[phase]
        this_dataset = IEMOCAP(list(range(len(data['labels']))), data['text'], data['audio'], data['vision'], data['labels'])
    else:
        raise ValueError('Wrong dataset!')

    if zsl == -1 and fsl == -1:
        save(this_dataset, processed_path)

    return this_dataset

def get_glove_emotion_embs(path):
    dataDict = load(path)
    return dataDict


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
