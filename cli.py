import argparse

def get_args():
    parser = argparse.ArgumentParser(description='Test')

    # Training hyper-parameters
    parser.add_argument('-bs', '--batch-size', help='Batch size', type=int, required=True)
    parser.add_argument('-lr', '--learning-rate', help='Learning rate', type=float, required=True)
    parser.add_argument('-wd', '--weight-decay', help='Weight decay', type=float, required=False, default=0)
    parser.add_argument('-ep', '--epochs', help='Number of epochs', type=int, required=True)
    parser.add_argument('-es', '--early-stop', help='Early stop', type=int, required=False, default=4)
    parser.add_argument('-cu', '--cuda', help='Cude device number', type=str, required=False, default='0')
    parser.add_argument('-mo', '--model', help='lf/ef', type=str, required=False, default='lf')
    parser.add_argument('-cl', '--clip', help='Use clip to gradients', action='store_true')
    parser.add_argument('-sc', '--scheduler', help='Use scheduler to optimizer', action='store_true')
    parser.add_argument('-se', '--seed', help='Random seed', type=int, required=False, default=0)

    # Dataset
    parser.add_argument('--dataset', type=str, default='mosei_senti', help='Dataset to use')
    parser.add_argument('--aligned', action='store_true', help='Aligned experiment or not')
    parser.add_argument('--data-path', type=str, default='data', help='path for storing the dataset')

    parser.add_argument('--ckpt', type=str, required=False, default='')

    # LSTM
    parser.add_argument('-dr', '--dropout', help='dropout', type=float, required=False, default=0.1)
    parser.add_argument('-nl', '--num-layers', help='num of layers of LSTM', type=int, required=False, default=1)
    parser.add_argument('-hs', '--hidden-size', help='hidden vector size of LSTM', type=int, required=False, default=300)
    parser.add_argument('-bi', '--bidirectional', help='Use Bi-LSTM', action='store_true')

    args = vars(parser.parse_args())
    return args
