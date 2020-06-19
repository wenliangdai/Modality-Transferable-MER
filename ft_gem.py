import os
import torch
import numpy as np
from torch.utils.data import DataLoader
from src.cli import get_args
from src.utils import capitalize_first_letter, load, batch_generator
from src.data import get_data, get_glove_emotion_embs
from src.trainers.sentiment import SentiTrainer
from src.trainers.emotion import MoseiEmoTrainer, IemocapTrainer
from src.models import baselines # EF_LSTM, LF_LSTM, EF_LF_LSTM
from src.models.transformers import EF_Transformer
from src.models.mult import MULTModel
from src.models.temp import EmotionEmbAttnModel
from src.config import NUM_CLASSES, CRITERIONS, MULT_PARAMS, EMOTIONS
from src.sampler import ImbalancedDatasetSampler


if __name__ == "__main__":
    args = get_args()

    # Hijack Python's print function
    # if args['log_file'] != '':
    #     dirname = os.path.dirname(args['log_file'])
    #     if not os.path.exists(dirname):
    #         os.makedirs(dirname)
    #     __print = print
    #     __f = open(args['log_file'], 'a')
    #     def print(*args):
    #         __print(*args)
    #         __print(*args, file=__f)

    # Fix seed for reproducibility
    seed = args['seed']
    torch.manual_seed(seed)
    np.random.seed(seed)

    # Set device
    # os.environ["CUDA_VISIBLE_DEVICES"] = args['cuda']
    device = torch.device(f"cuda:{args['cuda']}" if torch.cuda.is_available() else 'cpu')

    print("Start loading the data....")

    train_data, fsl_train_data = get_data(args, 'train')
    
    args['zsl'] = -1    # to generate all emotions
    all_valid_data = get_data(args, 'valid')
    all_test_data = get_data(args, 'test')
    
    train_loader = DataLoader(train_data, batch_size=5120, shuffle=True)
    train_data_generator = batch_generator(train_loader)

    fsl_train_loader = DataLoader(fsl_train_data, batch_size=args['batch_size'], shuffle=True)
    fsl_valid_loader = DataLoader(all_valid_data, batch_size=args['batch_size'], shuffle=False)
    fsl_test_loader = DataLoader(all_test_data, batch_size=args['batch_size'], shuffle=False)
    print(f'Train samples = {len(train_loader.dataset)}')
    print(f'FT Train samples = {len(fsl_train_loader.dataset)}')
    print(f'Valid samples = {len(fsl_valid_loader.dataset)}')
    print(f'Test samples = {len(fsl_test_loader.dataset)}')

    
    dataloaders = {
        'train': train_data_generator,
        'ft_train': fsl_train_loader,
        'valid': fsl_valid_loader,
        'test': fsl_test_loader
    }
    
    modal_dims = list(train_data.get_dim())

    model_type = args['model'].lower()
    fusion_type = args['fusion'].lower()

    if model_type == 'mult':
        mult_params = MULT_PARAMS[args['dataset']]
        mult_params['orig_d_l'] = modal_dims[0]
        mult_params['orig_d_a'] = modal_dims[1]
        mult_params['orig_d_v'] = modal_dims[2]
        mult_params['hidden_dim'] = args['hidden_dim']
        model = MULTModel(mult_params)
    elif model_type == 'rnn':
        if fusion_type == 'lf':
            MODEL = baselines.LF_RNN
        elif fusion_type == 'ef':
            MODEL = baselines.EF_RNN
        elif fusion_type == 'eflf':
            MODEL = baselines.EF_LF_RNN
        elif fusion_type == 'ts':
            MODEL = baselines.TextSelectiveRNN
        else:
            raise ValueError('Wrong fusion!')

        model = MODEL(
            num_classes=NUM_CLASSES[args['dataset']],
            input_sizes=modal_dims,
            hidden_size=args['hidden_size'],
            hidden_sizes=args['hidden_sizes'],
            num_layers=args['num_layers'],
            dropout=args['dropout'],
            bidirectional=args['bidirectional'],
            gru=args['gru']
        )
    elif model_type == 'transformer':
        if fusion_type == 'lf':
            MODEL = EF_Transformer
        elif fusion_type == 'ef':
            MODEL = EF_Transformer
        elif fusion_type == 'eflf':
            MODEL = EF_Transformer
        else:
            raise ValueError('Wrong fusion!')

        model = MODEL()
    elif model_type == 'eea':
        ft_emo_list = EMOTIONS[args['dataset']]
        fsl2 = args['fsl2']
        emo_list = ft_emo_list[:fsl2] + ft_emo_list[fsl2 + 1:]
        if args['cap']:
            emo_list = capitalize_first_letter(emo_list)
            ft_emo_list = capitalize_first_letter(ft_emo_list)
        emo_weights = get_glove_emotion_embs(args['glove_emo_path'])
        emo_weight = []
        ft_emo_weight = []
        for emo in emo_list:
            emo_weight.append(emo_weights[emo])
        for emo in ft_emo_list:
            ft_emo_weight.append(emo_weights[emo])

        MODEL = EmotionEmbAttnModel
        model = MODEL(
            num_classes=len(emo_list),
            input_sizes=modal_dims,
            hidden_size=args['hidden_size'],
            hidden_sizes=args['hidden_sizes'],
            num_layers=args['num_layers'],
            dropout=args['dropout'],
            bidirectional=args['bidirectional'],
            modalities=args['modalities'],
            device=device,
            emo_weight=emo_weight,
            gru=args['gru'],
            ft_emo_weight=ft_emo_weight,
            ft_num_classes=len(ft_emo_list)
        )
    else:
        raise ValueError('Wrong model!')

    model = model.to(device=device)

    if args['ckpt'] != '':
        print("Loading models from %s" % args['ckpt'])
        state_dict = load(args['ckpt'])
        state_dict.pop('textEmoEmbs.weight')
        if state_dict['modality_weights.weight'].size(0) != len(args['modalities']):
            state_dict.pop('modality_weights.weight')
        model.load_state_dict(state_dict, strict=False)

    print(model)

    optimizer = torch.optim.Adam(model.parameters(), lr=args['learning_rate'], weight_decay=args['weight_decay'])

    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=args['patience'], verbose=True)

    if args['loss'] == 'l1':
        criterion = torch.nn.L1Loss()
    elif args['loss'] == 'mse':
        criterion = torch.nn.MSELoss()
    elif args['loss'] == 'ce':
        criterion = torch.nn.CrossEntropyLoss()
    elif args['loss'] == 'bce':
        fsl_pos_weight = fsl_train_data.get_pos_weight()
        fsl_pos_weight = fsl_pos_weight.to(device)
        criterion = torch.nn.BCEWithLogitsLoss(pos_weight=fsl_pos_weight)

        train_pos_weight = train_data.get_pos_weight()
        train_pos_weight = train_pos_weight.to(device)
        criterion2 = torch.nn.BCEWithLogitsLoss(pos_weight=train_pos_weight)

    if args['dataset'] == 'mosi' or args['dataset'] == 'mosei_senti':
        TRAINER = SentiTrainer
    elif args['dataset'] == 'mosei_emo':
        TRAINER = MoseiEmoTrainer
    elif args['dataset'] == 'iemocap':
        TRAINER = IemocapTrainer

    trainer = TRAINER(args, model, criterion, optimizer, scheduler, device, dataloaders, criterion2=criterion2)

    trainer.train()
