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
print(attn, attn.shape)
print(p_attn, p_attn.shape)

# %%
# ?????????????????????copy?????????
import copy


# ??????????????????????????????, ??????????????????????????????????????????, ????????????????????????????????????.
# ???????????????clone???????????????????????????????????????????????????????????????. ???????????????????????????????????????.
def clones(module, N):
    """??????????????????????????????????????????, ????????????module?????????????????????????????????, N???????????????????????????"""
    # ????????????, ????????????for?????????module??????N???????????????, ????????????module??????????????????,
    # ??????????????????nn.ModuleList????????????????????????.
    return nn.ModuleList([copy.deepcopy(module) for _ in range(N)])


# ????????????????????????????????????????????????????????????
class MultiHeadedAttention(nn.Module):
    def __init__(self, head, embedding_dim, dropout=0.1):
        """?????????????????????, ????????????????????????head???????????????embedding_dim???????????????????????????
           dropout????????????dropout????????????0??????????????????0.1."""
        super(MultiHeadedAttention, self).__init__()

        # ??????????????????????????????????????????????????????assert???????????????h????????????d_model?????????
        # ???????????????????????????????????????????????????????????????.?????????embedding_dim/head???.
        assert embedding_dim % head == 0

        # ?????????????????????????????????????????????d_k
        self.d_k = embedding_dim // head

        # ????????????h
        self.head = head

        # ????????????????????????????????????nn???Linear???????????????????????????????????????embedding_dim x embedding_dim???????????????clones?????????????????????
        # ????????????????????????????????????????????????????????????Q???K???V??????????????????????????????????????????????????????????????????????????????.
        self.linears = clones(nn.Linear(embedding_dim, embedding_dim), 4)

        # self.attn???None???????????????????????????????????????????????????????????????????????????None.
        self.attn = None

        # ??????????????????self.dropout??????????????????nn??????Dropout?????????????????????0???????????????????????????dropout.
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, query, key, value, mask=None):
        """??????????????????, ?????????????????????????????????????????????????????????????????????Q, K, V???
           ????????????????????????????????????????????????mask????????????????????????None. """

        # ????????????????????????mask
        if mask is not None:
            # ??????unsqueeze????????????
            mask = mask.unsqueeze(0)

        # ???????????????????????????batch_size??????????????????query????????????1????????????????????????????????????.
        batch_size = query.size(0)

        # ?????????????????????????????????
        # ????????????zip?????????QKV?????????????????????????????????????????????for??????????????????QKV???????????????????????????
        # ?????????????????????????????????????????????????????????????????????view????????????????????????????????????????????????????????????????????????h??????????????????
        # ????????????????????????????????????????????????????????????????????????????????????-1????????????????????????
        # ??????????????????????????????????????????????????????.???????????????????????????????????????????????????
        # ????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????
        # ???attention???????????????????????????????????????????????????????????????????????????.??????????????????????????????????????????.
        query, key, value = \
            [model(x).view(batch_size, -1, self.head, self.d_k).transpose(1, 2)
             for model, x in zip(self.linears, (query, key, value))]

        # ???????????????????????????????????????????????????????????????attention??????
        # ???????????????????????????????????????attention??????.????????????mask???dropout????????????.
        x, self.attn = attention(query, key, value, mask=mask, dropout=self.dropout)

        # ?????????????????????????????????????????????????????????????????????????????????4?????????????????????????????????????????????????????????????????????????????????
        # ???????????????????????????????????????????????????????????????????????????????????????????????????????????????contiguous?????????
        # ????????????????????????????????????????????????????????????view???????????????????????????????????????
        # ??????????????????????????????view??????????????????????????????????????????.
        x = x.transpose(1, 2).contiguous().view(batch_size, -1, self.head * self.d_k)

        # ??????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????.
        return self.linears[-1](x)


head = 8

# ???????????????embedding_dim
embedding_dim = 512

# ????????????dropout
dropout = 0.2
query = value = key = pe_result

# ?????????????????????mask
mask = Variable(torch.zeros(8, 4, 4))

mha = MultiHeadedAttention(head, embedding_dim, dropout)
mha_result = mha.forward(query, key, value, mask)
print(mha_result)


# %%
class PositionwiseFeedForward(nn.Module):
    def __init__(self, d_model, d_ff, dropout=.1):
        super(PositionwiseFeedForward, self).__init__()
        self.w1 = nn.Linear(d_model, d_ff)
        self.w2 = nn.Linear(d_ff, d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        return self.w2(self.dropout(F.relu(self.w1(x))))


d_model = 512
d_ff = 64
dropout = .2
x = mha_result
ff = PositionwiseFeedForward(d_model, d_ff, dropout)
ff_result = ff(x)
print(ff_result, ff_result.size())


# %%
class LayerNorm(nn.Module):
    def __init__(self, features, eps=1e-6):
        super(LayerNorm, self).__init__()
        self.a2 = nn.Parameter(torch.ones(features))
        self.b2 = nn.Parameter(torch.zeros(features))

        self.eps = eps

    def forward(self, x):
        mean = x.mean(-1, keepdim=True)
        std = x.std(-1, keepdim=True)
        return self.a2 * (x - mean) / (std + self.eps) + self.b2


features = d_model = 512
eps = 1e-6
x = ff_result
print(x.shape)
ln = LayerNorm(features, eps)
ln_res = ln(x)
print(x)
print(ln_res)


# %%
class sublayerConnection(nn.Module):
    def __init__(self, size, dropout=.1):
        super(sublayerConnection, self).__init__()
        self.norm = LayerNorm(size)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, sublayer):
        return x + self.dropout(sublayer(self.norm(x)))


size = 512
dropout = .2
head = 8
d_model = 512
x = pe_result
mask = Variable(torch.zeros(8, 4, 4))
self_attn = MultiHeadedAttention(head, d_model)
sublayer = lambda x: self_attn(x, x, x, mask)
sc = sublayerConnection(size, dropout)
sc_result = sc.forward(x, sublayer)
print(sc_result)
print(sc_result.shape)


# %%
class EncoderLayer(nn.Module):
    def __init__(self, size, self_attn, feed_forward, dropout):
        super(EncoderLayer, self).__init__()
        self.self_attn = self_attn
        self.feed_forward = feed_forward
        self.sublayer = clones(sublayerConnection(size, dropout), 2)

        self.size = size

    def forward(self, x, mask):
        x = self.sublayer[0](x, lambda x: self.self_attn(x, x, x, mask))
        return self.sublayer[1](x, self.feed_forward)


size = 512
head = 8
d_model = 512
d_ff = 64
x = pe_result
dropout = 0.2
self_attn = MultiHeadedAttention(head, d_model)
ff = PositionwiseFeedForward(d_model, d_ff, dropout)
mask = Variable(torch.zeros(8, 4, 4))
el = EncoderLayer(size, self_attn, ff, dropout)
el_result = el(x, mask)
print(el_result)
print(el_result.shape)


# %%
class Encoder(nn.Module):
    def __init__(self, layer, N):
        super().__init__()
        self.layers = clones(layer, N)
        self.norm = LayerNorm(layer.size)

    def forward(self, x, mask):
        for layer in self.layers:
            x = layer(x, mask)
        return self.norm(x)

    # ????????????????????????layer, ??????????????????????????????????????????, ???????????????????????????????????????


# ????????????????????????????????????????????????, ??????????????????????????????????????????.
size = 512
head = 8
d_model = 512
d_ff = 64
c = copy.deepcopy
attn = MultiHeadedAttention(head, d_model)
ff = PositionwiseFeedForward(d_model, d_ff, dropout)
dropout = 0.2
layer = EncoderLayer(size, c(attn), c(ff), dropout)

# ?????????????????????????????????N
N = 8
mask = Variable(torch.zeros(8, 4, 4))

en = Encoder(layer, N)
en_result = en(x, mask)
print(en_result)
print(en_result.shape)


# %%
class DecoderLayer(nn.Module):
    def __init__(self, size, self_attn, src_attn, feed_forward, dropout):
        super(DecoderLayer, self).__init__()
        self.size = size
        self.self_attn = self_attn
        self.src_attn = src_attn
        self.feed_forward = feed_forward
        self.sublayer = clones(sublayerConnection(size, dropout), 3)

    def forward(self, x, memory, source_mask, target_mask):
        m = memory

        x = self.sublayer[0](x, lambda x: self.self_attn(x, x, x, target_mask))

        x = self.sublayer[1](x, lambda x: self.src_attn(x, m, m, source_mask))

        return self.sublayer[2](x, self.feed_forward)


head = 8
size = 512
d_model = 512
d_ff = 64
dropout = .2
self_attn = src_attn = MultiHeadedAttention(head, d_model, dropout)
ff = PositionwiseFeedForward(d_model, d_ff, dropout)

x = pe_result

# memory???????????????????????????
memory = en_result

# ?????????source_mask???target_mask????????????, ???????????????????????????????????????mask
mask = torch.zeros(8, 4, 4)
source_mask = target_mask = mask
dl = DecoderLayer(size, self_attn, src_attn, ff, dropout)
dl_result = dl(x, memory, source_mask, target_mask)
print(dl_result)
print(dl_result.shape)


# %%

class Decoder(nn.Module):
    def __init__(self, layer, N):
        super().__init__()
        self.layers = clones(layer, N)
        self.norm = LayerNorm(layer.size)

    def forward(self, x, memory, source_mask, target_mask):
        for layer in self.layers:
            x = layer(x, memory, source_mask, target_mask)
        return self.norm(x)


# ?????????????????????layer????????????????????????N
size = 512
d_model = 512
head = 8
d_ff = 64
dropout = 0.2
c = copy.deepcopy
attn = MultiHeadedAttention(head, d_model)
ff = PositionwiseFeedForward(d_model, d_ff, dropout)
layer = DecoderLayer(d_model, c(attn), c(attn), c(ff), dropout)
N = 8
# ????????????????????????????????????????????????
x = pe_result
memory = en_result
mask = torch.zeros(8, 4, 4)
source_mask = target_mask = mask
de = Decoder(layer, N)
de_result = de(x, memory, source_mask, target_mask)
print(de_result)
print(de_result.shape)

# %%
import torch.nn.functional as F


class Generator(nn.Module):
    def __init__(self, d_model, vocab_size):
        super().__init__()
        self.project = nn.Linear(d_model, vocab_size)

    def forward(self, x):
        return F.log_softmax(self.project(x), dim=-1)


d_model = 512

# ???????????????1000
vocab_size = 1000
x = de_result
gen = Generator(d_model, vocab_size)
gen_result = gen(x)
print(gen_result)
print(gen_result.shape)


# %%

class EncoderDecoder(nn.Module):
    def __init__(self, encoder, decoder, source_embed, target_embed, generator):
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.src_embed = source_embed
        self.tgt_embed = target_embed
        self.generator = generator

    def forward(self, source, target, source_mask, target_mask):
        return self.decode(self.encode(source, source_mask), source_mask, target, target_mask)

    def encode(self, source, source_mask):
        return self.encoder(self.src_embed(source), source_mask)

    def decode(self, memory, source_mask, target, target_mask):
        return self.decoder(self.tgt_embed(target), memory, source_mask, target_mask)


vocab_size = 1000
d_model = 512
encoder = en
decoder = de
source_embed = nn.Embedding(vocab_size, d_model)
target_embed = nn.Embedding(vocab_size, d_model)
generator = gen

source = target = torch.LongTensor([[100, 2, 421, 508], [491, 998, 1, 221]])
source_mask = target_mask = torch.zeros(8, 4, 4)

ed = EncoderDecoder(encoder, decoder, source_embed, target_embed, generator)
de_result = ed(source, target, source_mask, target_mask)
print(de_result.shape)


# %%

def make_model(source_vocab, target_vocab, N=6, d_model=512, d_ff=2048, head=8, dropout=.1):
    c = copy.deepcopy
    attn = MultiHeadedAttention(head, d_model)
    ff = PositionwiseFeedForward(d_model, d_ff, dropout)
    position = PositionalEncoding(d_model, dropout)

    model = EncoderDecoder(
        Encoder(EncoderLayer(d_model, c(attn), c(ff), dropout), N)
        , Decoder(DecoderLayer(d_model, c(attn), c(attn), c(ff), dropout), N),
        nn.Sequential(Embeddings(d_model, source_vocab), c(position)),
        nn.Sequential(Embeddings(d_model, target_vocab), c(position)),
        Generator(d_model, target_vocab)
    )

    for p in model.parameters():
        if p.dim() > 1:
            nn.init.xavier_uniform_(p)
    return model


# %%
source_vocab = 11
target_vocab = 11
N = 6
res = make_model(source_vocab, target_vocab, N)
print(res)

# %%
from pyitcast.transformer_utils import Batch


def data_generator(V, batch, num_batch):
    for i in range(num_batch):
        data = torch.from_numpy(np.random.randint(1, V, size=(batch, 10)))
        data[:, 0] = 1

        source = Variable(data, requires_grad=False)
        target = Variable(data, requires_grad=False)

        yield Batch(source, target)


V = 11
batch = 20
num_batch = 30
res = data_generator(V, batch, num_batch)
print(res)

res.__next__()

# %%
from pyitcast.transformer_utils import get_std_opt
from pyitcast.transformer_utils import LabelSmoothing
from pyitcast.transformer_utils import SimpleLossCompute

model = make_model(V, V, N=2)
model_optimizer = get_std_opt(model)
criterion = LabelSmoothing(size=V, padding_idx=0, smoothing=0.0)
loss = SimpleLossCompute(model.generator, criterion, model_optimizer)

# %%
from pyitcast.transformer_utils import run_epoch


def run(model, loss, epochs=10):
    for epoch in range(epochs):
        model.train()
        run_epoch(data_generator(V, 8, 20), model, loss)
        model.eval()
        run_epoch(data_generator(V, 8, 5), model, loss)

# ???????????????????????????greedy_decode, ??????????????????????????????????????????
# ??????????????????????????????????????????????????????????????????????????????,
# ????????????????????????????????????, ?????????????????????????????????.
from pyitcast.transformer_utils import greedy_decode


def run(model, loss, epochs=10):
    for epoch in range(epochs):
        model.train()

        run_epoch(data_generator(V, 8, 20), model, loss)

        model.eval()

        run_epoch(data_generator(V, 8, 5), model, loss)

    # ????????????????????????
    model.eval()

    # ?????????????????????
    source = Variable(torch.LongTensor([[1,3,2,5,4,6,7,8,9,10]]))

    # ???????????????????????????, ??????????????????1, ???????????????1???????????????
    # ?????????????????????????????????????????????.
    source_mask = Variable(torch.ones(1, 1, 10))

    # ?????????model, src, src_mask, ???????????????????????????max_len, ?????????10
    # ????????????????????????, ?????????1, ???????????????????????????1
    result = greedy_decode(model, source, source_mask, max_len=10, start_symbol=1)
    print(result)


if __name__ == '__main__':
    run(model, loss)