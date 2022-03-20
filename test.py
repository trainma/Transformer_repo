import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import math
import matplotlib.pyplot as plt
import numpy as np
import copy


# embedding=nn.Embedding(10,3)
# input1=torch.LongTensor([[1,2,4,5],[4,3,2,9]])
# print(embedding(input1),embedding(input1).size())
#
# embedding=nn.Embedding(10,3,padding_idx=0)
# input1=torch.LongTensor([[0,2,0,5]])
# print(embedding(input1),embedding(input1).size())

class Embeddings(nn.Module):
    def __init__(self, d_model, vocab):
        super().__init__()
        # d_model: the shape of embedding
        # vocab the size of vocab
        self.lut = nn.Embedding(vocab, d_model)
        self.d_model = d_model

    def forward(self, x):
        return self.lut(x) * math.sqrt(self.d_model)


d_model = 512
vocab = 1000
x = Variable(torch.LongTensor([[100, 2, 421, 508], [491, 998, 1, 221]]))
emb = Embeddings(d_model, vocab)
embr = emb(x)
print("embr:", embr)
print(embr.size())

# %%
m = nn.Dropout(p=.2)
input1 = torch.randn(4, 5)
output = m(input1)


class PositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout, max_len=5000):
        super().__init__()
        self.dropout = nn.Dropout(dropout)
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * -(math.log(10000.0) / d_model))

        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)

        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)

    def forward(self, x):
        # x represents the text series embedded
        x = x + Variable(self.pe[:, :x.size(1)], requires_grad=False)
        return self.dropout(x)


d_model = 512
dropout = .1
max_len = 60
x = embr
pe = PositionalEncoding(d_model, dropout, max_len)
pe_result = pe(x)
print(pe_result)

# %%
plt.figure(figsize=(15, 5))
pe = PositionalEncoding(20, 0)
y = pe(Variable(torch.zeros(1, 100, 20)))
plt.figure(figsize=(20, 16))
plt.plot(np.arange(100), y[0, :, 4:8].data.numpy())
plt.legend("dim %d" % p for p in [4, 5, 6, 7])


# %%
def subsequent_mask(size):
    attn_shape = (1, size, size)
    subsequent_mask = np.triu(np.ones(attn_shape), k=1).astype('uint8')
    return torch.from_numpy(1 - subsequent_mask)


size = 5
sm = subsequent_mask(size)
print("sm:", sm)
plt.figure(figsize=(10, 10))
plt.imshow(subsequent_mask(20)[0])

# %%
x = Variable(torch.randn(5, 5))
print(x)

mask = Variable(torch.zeros(5, 5))
print(mask)

y = x.masked_fill(mask == 0, -1e9)
print(y)


# %%
def attention(query, key, value, mask=None, dropout=None):
    d_k = query.size(-1)
    scores = torch.matmul(query, key.transpose(-2, -1)) / math.sqrt(d_k)

    if mask is not None:
        scores = scores.masked_fill(mask == 0, -1e9)

    p_attn = F.softmax(scores, dim=-1)

    if dropout is not None:
        p_attn = dropout(p_attn)

    return torch.matmul(p_attn, value), p_attn


query = key = value = pe_result
attn, p_attn = attention(query, key, value)
print(attn,attn.shape)
print(p_attn,p_attn.shape)
# %%
