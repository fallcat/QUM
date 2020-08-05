import torch
import torch.nn as nn
import threading


class TokenEmbedding(nn.Embedding):
    ''' An embedding layer used for the transformer '''
    def __init__(self, d_vocab, d_embed, padding_idx=0):
        super(TokenEmbedding, self).__init__(d_vocab, d_embed, padding_idx=padding_idx)

        self.scale = d_embed ** 0.5
        self.embed = nn.Embedding(d_vocab, d_embed)

    def forward(self, inputs):
        return self.scale * self.embed(inputs)


class PositionEmbedding(nn.Module):
    ''' Produce position embeddings '''
    def __init__(self, dim, freq=1e4):
        ''' Initialize the PositionEmbedding '''
        super(PositionEmbedding, self).__init__()

        self.dim = dim
        self.freq = freq

    _embeddings = threading.local()
    def forward(self, inputs): # pylint:disable=arguments-differ
        ''' Implement the forward pass of the embedding '''
        device = inputs.device
        max_length = inputs.shape[1]
        embedding_store = PositionEmbedding._embeddings.__dict__
        device_store = embedding_store.get(device, {})
        if (
                not device_store or
                self.dim not in device_store or
                device_store[self.dim].shape[0] < max_length
        ):
            positions = torch.arange(0., max_length, device=device).unsqueeze(1)

            # the tensor2tensor code is slightly different than described in the paper
            # dividing by (self.dim - 2) produces nearly identical results to their version
            # when comparing the tensorflow results to these torch results
            dims = torch.arange(0., self.dim, 2., device=device).unsqueeze(0) / (self.dim - 2)

            sin = torch.sin(positions / torch.pow(self.freq, dims))
            cos = torch.cos(positions / torch.pow(self.freq, dims))

            embeddings = torch.stack((sin, cos), 0)
            device_store[self.dim] = embeddings.transpose(0, 1).contiguous().view(-1, self.dim)

        embeddings = device_store[self.dim]
        embedding_store[device] = device_store
        return embeddings[:max_length].unsqueeze(0)