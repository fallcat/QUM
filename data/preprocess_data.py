"""
    Preprocess data for training and evaluating classifier
    Assume data follows this format: xxx xxx xxx .\t label
    Before processing, data_dir should include the following files: train.txt, valid.txt, test.txt.
    After processing, data_dir will include train.bin, valid.bin, test.bin, vocab.pkl.
    Each .bin file includes a tensor of the
"""

import os
import torch
import argparse
import pickle
import random
from tqdm import tqdm
from collections import defaultdict


SPLITS = ['train', 'valid', 'test']


def get_parser():
    parser = argparse.ArgumentParser()

    # data args
    # data dir has these files:
    parser.add_argument('--data-dir', type=str, default='./data/data1', help='Where data is stored')

    parser.add_argument('--raw-filename', type=str, default='data.txt', help='Not splitted data file')

    parser.add_argument('--split-ratio', type=str, default='8:1:1', help='Ratio to split train, valid, test')

    parser.add_argument('--seed', type=int, default=None, help='Seed when randomly splitting data')

    return parser


def split_data(data_dir, filename, ratio):
    """ Split data into train, valid and test if not already"""
    with open(os.path.join(data_dir, filename), 'rt') as input_file:
        data = list(input_file.readlines())

    random.shuffle(data)

    ratio = [float(r) for r in ratio.split(':')]
    split_points = []
    for r in ratio:
        split_points.append(int(r / sum(ratio) * len(data)))
    split_points[-1] = len(data)
    split_points = [0] + split_points

    for i, split in enumerate(SPLITS):
        with open(os.path.join(data_dir, f'{split}.txt'), 'wt') as output_file:
            output_file.writelines(data[split_points[i] : split_points[i+1]])


def build_vocab(data_dir, data):
    data_all = []
    for split in data:
        data_all.extend(data[split])

    print('building vocab and labels ...')
    vocab = defaultdict(int)
    labels = defaultdict(int)

    for line in tqdm(data_all):
        line = line.strip().split('\t')
        sentence = line[0].split()
        label = line[1]
        for tok in sentence:
            vocab[tok] += 1
        labels[label] += 1

    vocab = list(sorted(vocab.items(), key=lambda a: a[1], reverse=True))
    id2tok = ['<PAD>'] + [a[0] for a in vocab]
    tok2id = {a: i for i, a in enumerate(id2tok)}
    vocab_size = len(id2tok)

    labels = list(sorted(labels.items(), key=lambda a: a[1], reverse=True))
    id2label =  [a[0] for a in labels]
    label2id = {a: i for i, a in enumerate(id2label)}
    label_size = len(id2label)

    print('end building vocab and labels ...')
    print('vocab size', vocab_size)
    print('label size', label_size)
    vocab_dict = {'id2tok': id2tok, 'tok2id': tok2id, 'vocab_size': vocab_size,
                  'id2label': id2label, 'label2id': label2id, 'label_size': label_size}
    with open(os.path.join(data_dir, 'vocab.pkl'), 'wb') as output_file:
        pickle.dump(vocab_dict, output_file)
    return vocab_dict


def load_text(data_dir, filename):
    with open(os.path.join(data_dir, filename), 'rt') as text_file:
        return list(text_file.readlines())


def tensorize(line, symbol2id):
    return torch.tensor([symbol2id[tok] for tok in line])


def binarize(data_dir, split, data, vocab_dict):
    sentences = []
    labels = []

    print('binarizing data ...')
    for line in tqdm(data):
        line = line.strip().split('\t')
        sentence = line[0].split()
        label = line[1]
        sentences.append(tensorize(sentence, vocab_dict['tok2id']))
        labels.append(tensorize([label], vocab_dict['label2id']))

    sentences = torch.cat(sentences)
    labels = torch.cat(labels)

    print('end binarizing data ...')
    with open(os.path.join(data_dir, f'{split}.bin'), 'wb') as output_file:
        pickle.dump({'data': sentences, 'labels': labels}, protocol=pickle.HIGHEST_PROTOCOL)


if __name__ == '__main__':
    parser = get_parser()
    args = parser.parse_args()

    random.seed(args.seed)

    if not os.path.exists(args.data_dir):
        os.mkdir(args.data_dir)

    if not all(os.path.exists(os.path.join(args.data_dir, f'{split}.txt')) for split in SPLITS):
        if os.path.exists(os.path.join(args.data_dir, args.raw_filename)):
            split_data(args.data_dir, args.raw_filename, args.split_ratio)
        else:
            raise FileNotFoundError('Data is not found')

    data = {split: load_text(args.data_dir, f'{split}.txt') for split in SPLITS}
    vocab_path = os.path.join(args.data_dir, 'vocab.pkl')
    if not os.path.exists(vocab_path):
        vocab_dict = build_vocab(args.data_dir, data)
    else:
        print("vocab already built. loading vocab ...")
        with open(vocab_path, 'rb') as input_file:
            vocab_dict = pickle.load(input_file)

    for split in SPLITS:
        binarize(args.data_dir, split, data[split], vocab_dict)


