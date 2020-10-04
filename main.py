import copy
import torch
import numpy as np
from torch.utils.data import DataLoader
from src.cli import get_args
from src.utils import capitalize_first_letter, load
from src.data import get_data, get_glove_emotion_embs
from src.trainers.sentiment import SentiTrainer
from src.trainers.emotion import MoseiEmoTrainer, IemocapTrainer
from src.models import baselines # EF_LSTM, LF_LSTM, EF_LF_LSTM
from src.models.transformers import EF_Transformer
from src.models.mult import MULTModel
from src.models.eea import EmotionEmbAttnModel
from src.config import NUM_CLASSES, MULT_PARAMS, EMOTIONS


if __name__ == "__main__":
    args = get_args()

    # Fix seed for reproducibility
    seed = args['seed']
    torch.manual_seed(seed)
    np.random.seed(seed)

    # Set device
    # os.environ["CUDA_VISIBLE_DEVICES"] = args['cuda']
    device = torch.device(f"cuda:{args['cuda']}" if torch.cuda.is_available() else 'cpu')

    print("Start loading the data....")

    train_data = get_data(args, 'train')
    valid_data = get_data(args, 'valid')
    test_data = get_data(args, 'test')

    train_loader = DataLoader(train_data, batch_size=args['batch_size'], shuffle=True)
    valid_loader = DataLoader(valid_data, batch_size=args['batch_size'], shuffle=False)
    test_loader = DataLoader(test_data, batch_size=args['batch_size'], shuffle=False)

    print(f'Train samples = {len(train_loader.dataset)}')
    print(f'Valid samples = {len(valid_loader.dataset)}')
    print(f'Test samples = {len(test_loader.dataset)}')

    dataloaders = {
        'train': train_loader,
        'valid': valid_loader,
        'test': test_loader
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
        if args['zsl'] != -1:
            mult_params['output_dim'] = mult_params['output_dim'] + 1
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

        num_classes = NUM_CLASSES[args['dataset']]

        if args['zsl'] != -1:
            if args['dataset'] == 'iemocap':
                num_classes += 1
            else:
                num_classes -= 1

        model = MODEL(
            num_classes=num_classes,
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
        zsl = args['zsl']
        emo_list = EMOTIONS[args['dataset']]
        if zsl != -1:
            if args['dataset'] == 'iemocap':
                emo_list.append(EMOTIONS['iemocap9'][zsl])
            else:
                emo_list = emo_list[:zsl] + emo_list[zsl + 1:]

        if args['cap']:
            emo_list = capitalize_first_letter(emo_list)

        emo_weights = get_glove_emotion_embs(args['glove_emo_path'])
        emo_weight = []
        for emo in emo_list:
            emo_weight.append(emo_weights[emo])

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
            gru=args['gru']
        )
    else:
        raise ValueError('Wrong model!')

    model = model.to(device=device)

    # Load model checkpoint
    if args['ckpt'] != '':
        state_dict = load(args['ckpt'])
        if args['model'] == 'eea':
            state_dict.pop('textEmoEmbs.weight')
            if state_dict['modality_weights.weight'].size(0) != len(args['modalities']):
                state_dict.pop('modality_weights.weight')
        if args['model'] == 'rnn':
            if args['zsl_test'] != -1:
                out_weight = copy.deepcopy(model.out.weight)
                out_bias = copy.deepcopy(model.out.bias)
                pretrained_out_weight = state_dict['out.weight']
                pretrained_out_bias = state_dict['out.bias']
                indicator = 0
                for i in range(len(model.out.weight)):
                    if i == args['zsl_test']:
                        indicator = 1
                        continue
                    out_weight[i] = pretrained_out_weight[i - indicator]
                    out_bias[i] = pretrained_out_bias[i - indicator]
                model.out.weight = torch.nn.Parameter(out_weight)
                model.out.bias = torch.nn.Parameter(out_bias)
            state_dict.pop('out.weight')
            state_dict.pop('out.bias')
        if args['model'] == 'mult':
            if args['zsl_test'] != -1:
                out_weight = copy.deepcopy(model.out_layer.weight)
                out_bias = copy.deepcopy(model.out_layer.bias)
                pretrained_out_weight = state_dict['out_layer.weight']
                pretrained_out_bias = state_dict['out_layer.bias']
                indicator = 0
                for i in range(len(model.out_layer.weight)):
                    if i == args['zsl_test']:
                        indicator = 1
                        continue
                    out_weight[i] = pretrained_out_weight[i - indicator]
                    out_bias[i] = pretrained_out_bias[i - indicator]
                model.out_layer.weight = torch.nn.Parameter(out_weight)
                model.out_layer.bias = torch.nn.Parameter(out_bias)
            state_dict.pop('out_layer.weight')
            state_dict.pop('out_layer.bias')

        model.load_state_dict(state_dict, strict=False)

    if args['optim'] == 'adam':
        optimizer = torch.optim.Adam(model.parameters(), lr=args['learning_rate'], weight_decay=args['weight_decay'])
    elif args['optim'] == 'sgd':
        optimizer = torch.optim.SGD(model.parameters(), lr=args['learning_rate'], weight_decay=args['weight_decay'])

    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=args['patience'], verbose=True)

    if args['loss'] == 'l1':
        criterion = torch.nn.L1Loss()
    elif args['loss'] == 'mse':
        criterion = torch.nn.MSELoss()
    elif args['loss'] == 'ce':
        criterion = torch.nn.CrossEntropyLoss()
    elif args['loss'] == 'bce':
        pos_weight = train_data.get_pos_weight()
        pos_weight = pos_weight.to(device)
        criterion = torch.nn.BCEWithLogitsLoss(pos_weight=pos_weight)
        # criterion = torch.nn.BCEWithLogitsLoss()

    if args['dataset'] == 'mosi' or args['dataset'] == 'mosei_senti':
        TRAINER = SentiTrainer
    elif args['dataset'] == 'mosei_emo':
        TRAINER = MoseiEmoTrainer
    elif args['dataset'] == 'iemocap':
        TRAINER = IemocapTrainer

    trainer = TRAINER(args, model, criterion, optimizer, scheduler, device, dataloaders)

    if args['test']:
        trainer.test()
    elif args['valid']:
        trainer.valid()
    else:
        trainer.train()
