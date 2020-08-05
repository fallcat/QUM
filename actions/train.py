import torch
import argparse
import torch.nn as nn
from torch import optim
from torch.utils.data import DataLoader
from data.text import TextDataset
from models.classifier import TranslationClassifier
from tqdm import tqdm


def get_parser():
    parser = argparse.ArgumentParser()

    # model args
    # vocab_size, d_embed, output_size, num_heads, dropout_p=0.1, padding_idx=0
    parser.add_argument('--embedding-size', type=int, default=256, help='Embedding size of classifier')

    parser.add_argument('--output-size', type=int, default=2, help='Number of labels')

    parser.add_argument('--num-layers', type=int, default=1, help='Number of layers')

    parser.add_argument('--dropout-p', type=float, default=0.1, help='dropout percent')

    # data args
    # data dir has these files:
    parser.add_argument('--data-dir', type=str, default='./data/data1', help='Where data is stored')

    parser.add_argument('--split', type=str, default='train', help='Which split to use')

    # train args
    parser.add_argument('--optimizer', type=str, default='adam', choices=['adam', 'sgd'],
                        help='Which optimizer to use')

    parser.add_argument('--learning-rate', type=float, default=3e-4, help='Learning rate')

    parser.add_argument('--lr-scheduler', type=str, default='warmup', choices=['warmup', 'linear'],
                        help='Which learning rate scheduler to use. Choices: warmup, linear')

    parser.add_argument('--warmup-steps', type=int, default=4000, help='Warmup steps')

    parser.add_argument('--max-steps', type=int, default=10000, help='Max steps to train')

    parser.add_argument('--batch-size', type=int, default=100, help='Number of sentences per batch')

    parser.add_argument('--shuffle', action='store_true', default=False,
                        help='Use this flag when you want to shuffle the dataset')

    return parser


if __name__ == '__main__':
    parser = get_parser()
    args = parser.parse_args()

    use_cuda = torch.cuda.is_available()
    device = torch.device('cuda:0' if use_cuda else 'cpu')
    torch.backends.cudnn.benchmark = True

    training_set = TextDataset(args.data_dir, args.split)
    training_dataloader = DataLoader(training_set, batch_size=args.batch_size, shuffle=args.shuffle)

    model = TranslationClassifier(training_set.vocab_size, args.d_embed, training_set.label_size, args.num_heads,
                                  args.dropout_p, padding_idx=training_set.padding_idx)
    if args.optimizer == 'adam':
        optimizer = optim.Adam(model.parameters(), args.learning_rate, betas=(0.9, 0.98), eps=1e-9)
    elif args.optimizer == 'sgd':
        optimizer = optim.SGD(model.parameters(), lr=args.learning_rate, momentum=0.9)

    if 'cuda' in device.type:
        model = nn.DataParallel(model.cuda())

    step = 0
    while step < args.max_steps:
        for batch, label in training_dataloader:
            pass



