# Assume data follows this format: xxx xxx xxx .\t label
import torch
import torch.nn as nn
import os
import pickle
from torch.utils.data import Dataset, DataLoader

PROCESSED_FILES = ['vocab.pkl', 'train.bin', 'valid.bin', 'test.bin']


class TextDataset(Dataset):

    def __init__(self, data_dir, split):
        self.data_dir = data_dir
        self.split = split

        # load vocab
        vocab_dict = self.load_vocab()
        self.id2tok = vocab_dict['id2tok']
        self.tok2id = vocab_dict['tok2id']
        self.vocab_size = vocab_dict['vocab_size']
        self.id2label = vocab_dict['id2label']
        self.label2id = vocab_dict['label2id']
        self.label_size = vocab_dict['label_size']

        # load data
        self.data = self.load_data(split)

    def load_data(self, split):
        with open(os.path.join(self.data_dir, f'{split}.bin'), 'rb') as input_file:
            return pickle.load(input_file)

    def load_vocab(self):
        with open(os.path.join(self.data_dir, 'vocab.pkl'), 'rb') as input_file:
            return pickle.load(input_file)

    def __len__(self):
        return len(self.data['data'])

    def __getitem__(self, idx):
        return self.data['data'][idx], self.data['labels'][idx]

    @property
    def padding_idx(self):
        return self.tok2id['<PAD>']