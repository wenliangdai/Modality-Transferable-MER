import argparse

def get_args():
    parser = argparse.ArgumentParser(description='Multi-modal emotion recognition')

    parser.add_argument('-lf', '--log-file', help='Log file', type=str, required=False, default='')

    # Training hyper-parameters
    parser.add_argument('-bs', '--batch-size', help='Batch size', type=int, required=True)
    parser.add_argument('-lr', '--learning-rate', help='Learning rate', type=float, required=True)
    parser.add_argument('-wd', '--weight-decay', help='Weight decay', type=float, required=False, default=0.0)
    parser.add_argument('-ep', '--epochs', help='Number of epochs', type=int, required=True)
    parser.add_argument('-es', '--early-stop', help='Early stop', type=int, required=False, default=4)
    parser.add_argument('-cu', '--cuda', help='Cude device number', type=str, required=False, default='0')
    parser.add_argument('-mo', '--model', help='model type', type=str, required=False, default='rnn')
    parser.add_argument('-fu', '--fusion', help='modality fusion type', type=str, required=False, default='ef')
    parser.add_argument('-cl', '--clip', help='Use clip to gradients', type=float, required=False, default=-1.0)
    parser.add_argument('-sc', '--scheduler', help='Use scheduler to optimizer', action='store_true')
    parser.add_argument('-se', '--seed', help='Random seed', type=int, required=False, default=0)
    parser.add_argument('-pa', '--patience', help='Patience of the scheduler', type=int, required=False, default=6)
    parser.add_argument('-ez', '--exclude-zero', help='Exclude zero in evaluation', action='store_true')
    parser.add_argument('--loss', help='loss function: l1/mse/ce', type=str, required=False, default='l1')

    parser.add_argument('--threshold', help='Threshold of for multi-label emotion recognition', type=float, required=False, default=0.5)
    # parser.add_argument('--threshold-wacc', help='Threshold of weighted accuracy for multi-label emotion recognition', type=float, required=False, default=0.2)
    # parser.add_argument('--threshold-f1', help='Threshold of f1 score for multi-label emotion recognition', type=float, required=False, default=0.4)

    parser.add_argument('--verbose', help='Verbose mode to print more logs', action='store_true')

    parser.add_argument('-mod', '--modalities', help='what modalities to use', type=str, required=False, default='tav')

    parser.add_argument('--valid', help='Valid mode', action='store_true')
    parser.add_argument('--test', help='Test mode', action='store_true')

    # Dataset
    parser.add_argument('--dataset', type=str, default='mosei_senti', help='Dataset to use')
    parser.add_argument('--aligned', action='store_true', help='Aligned experiment or not')
    parser.add_argument('--data-seq-len', help='Data sequence length', type=int, required=False, default=50)
    parser.add_argument('--data-folder', type=str, default='data', help='path for storing the dataset')
    parser.add_argument('--glove-emo-path', type=str, default='data/glove.emotions.840B.300d.pt')
    parser.add_argument('--emocap', action='store_true', help='Capitalize the first letter of emotion words')
    parser.add_argument('--multi-level-classify', help='MOSEI emotion multi level', action='store_true')
    parser.add_argument('--zsl', help='Do zero shot learning on which emotion (index)', type=int, required=False, default=-1)
    parser.add_argument('--fsl', help='Do few shot learning on which emotion (index)', type=int, required=False, default=-1)

    # Checkpoint
    parser.add_argument('--ckpt', type=str, required=False, default='')

    # LSTM
    parser.add_argument('-dr', '--dropout', help='dropout', type=float, required=False, default=0.1)
    parser.add_argument('-nl', '--num-layers', help='num of layers of LSTM', type=int, required=False, default=1)
    parser.add_argument('-hs', '--hidden-size', help='hidden vector size of LSTM', type=int, required=False, default=300)
    parser.add_argument('-hss', '--hidden-sizes', help='hidden vector size of LSTM', nargs='+', type=int, required=False, default=[256, 64, 32])
    parser.add_argument('-bi', '--bidirectional', help='Use Bi-LSTM', action='store_true')
    parser.add_argument('--gru', help='Use GRU rather than LSTM', action='store_true')


    # TRANSFORMER
    parser.add_argument('--hidden-dim', help='Transformers Hidden Unit Size', type=int, required=False, default=40)

    args = vars(parser.parse_args())
    return args
