import torch
import torch.nn as nn
from models.embedding import TokenEmbedding, PositionEmbedding


class TranslationClassifier(nn.Module):

    def __init__(self, vocab_size, d_embed,  output_size, num_heads, dropout_p=0.1, padding_idx=0):
        super(TranslationClassifier, self).__init__()

        self.token_embed = TokenEmbedding(vocab_size, d_embed, padding_idx=padding_idx)
        self.position_embed = PositionEmbedding(d_embed)
        self.dropout = nn.Dropout(dropout_p, inplace=True)
        self.mha = nn.modules.activation.MultiheadAttention(d_embed, num_heads, dropout=dropout_p)
        self.output_layer = nn.Linear(d_embed, output_size)

    def forward(self, inputs):
        state = self.embed(inputs)
        logits = self.output_layer(self.mha(state, state, state))
        return logits

    def embed(self, inputs):
        return self.dropout(self.token_embed(inputs) + self.position_embed(inputs))


class LSTMClassifier(nn.Module):
    def __init__(self, vocab_size, hidden_size, embedding_size, num_layers,
                 reduction_size, output_size):
        super(LSTMClassifier, self).__init__()
        self.hidden_size = hidden_size
        self.embedding_size = embedding_size
        self.num_layers = num_layers
        self.reduction_size = reduction_size
        self.embedding = nn.Embedding(vocab_size, embedding_size)
        self.lstm = nn.LSTM(embedding_size, hidden_size,
                            num_layers=num_layers,
                            bidirectional=True, batch_first=True)
        self.ff1 = nn.Linear(2 * hidden_size, reduction_size)
        self.ff2 = nn.Linear(reduction_size, output_size)
        self.sigmoid = nn.Sigmoid()

    def forward(self, inputs, input_lengths):
        embedded = self.embedding(inputs)
        packed_embedded = nn.utils.rnn.pack_padded_sequence(embedded, input_lengths, batch_first=True)
        o, _ = self.lstm(packed_embedded)
        reduction = self.sigmoid(self.ff1(o[-1]))
        output = self.sigmoid(self.ff2(reduction))
        return output
