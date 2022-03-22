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
# 用于深度拷贝的copy工具包
import copy


# 首先需要定义克隆函数, 因为在多头注意力机制的实现中, 用到多个结构相同的线性层.
# 我们将使用clone函数将他们一同初始化在一个网络层列表对象中. 之后的结构中也会用到该函数.
def clones(module, N):
    """用于生成相同网络层的克隆函数, 它的参数module表示要克隆的目标网络层, N代表需要克隆的数量"""
    # 在函数中, 我们通过for循环对module进行N次深度拷贝, 使其每个module成为独立的层,
    # 然后将其放在nn.ModuleList类型的列表中存放.
    return nn.ModuleList([copy.deepcopy(module) for _ in range(N)])


# 我们使用一个类来实现多头注意力机制的处理
class MultiHeadedAttention(nn.Module):
    def __init__(self, head, embedding_dim, dropout=0.1):
        """在类的初始化时, 会传入三个参数，head代表头数，embedding_dim代表词嵌入的维度，
           dropout代表进行dropout操作时置0比率，默认是0.1."""
        super(MultiHeadedAttention, self).__init__()

        # 在函数中，首先使用了一个测试中常用的assert语句，判断h是否能被d_model整除，
        # 这是因为我们之后要给每个头分配等量的词特征.也就是embedding_dim/head个.
        assert embedding_dim % head == 0

        # 得到每个头获得的分割词向量维度d_k
        self.d_k = embedding_dim // head

        # 传入头数h
        self.head = head

        # 然后获得线性层对象，通过nn的Linear实例化，它的内部变换矩阵是embedding_dim x embedding_dim，然后使用clones函数克隆四个，
        # 为什么是四个呢，这是因为在多头注意力中，Q，K，V各需要一个，最后拼接的矩阵还需要一个，因此一共是四个.
        self.linears = clones(nn.Linear(embedding_dim, embedding_dim), 4)

        # self.attn为None，它代表最后得到的注意力张量，现在还没有结果所以为None.
        self.attn = None

        # 最后就是一个self.dropout对象，它通过nn中的Dropout实例化而来，置0比率为传进来的参数dropout.
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, query, key, value, mask=None):
        """前向逻辑函数, 它的输入参数有四个，前三个就是注意力机制需要的Q, K, V，
           最后一个是注意力机制中可能需要的mask掩码张量，默认是None. """

        # 如果存在掩码张量mask
        if mask is not None:
            # 使用unsqueeze拓展维度
            mask = mask.unsqueeze(0)

        # 接着，我们获得一个batch_size的变量，他是query尺寸的第1个数字，代表有多少条样本.
        batch_size = query.size(0)

        # 之后就进入多头处理环节
        # 首先利用zip将输入QKV与三个线性层组到一起，然后使用for循环，将输入QKV分别传到线性层中，
        # 做完线性变换后，开始为每个头分割输入，这里使用view方法对线性变换的结果进行维度重塑，多加了一个维度h，代表头数，
        # 这样就意味着每个头可以获得一部分词特征组成的句子，其中的-1代表自适应维度，
        # 计算机会根据这种变换自动计算这里的值.然后对第二维和第三维进行转置操作，
        # 为了让代表句子长度维度和词向量维度能够相邻，这样注意力机制才能找到词义与句子位置的关系，
        # 从attention函数中可以看到，利用的是原始输入的倒数第一和第二维.这样我们就得到了每个头的输入.
        query, key, value = \
            [model(x).view(batch_size, -1, self.head, self.d_k).transpose(1, 2)
             for model, x in zip(self.linears, (query, key, value))]

        # 得到每个头的输入后，接下来就是将他们传入到attention中，
        # 这里直接调用我们之前实现的attention函数.同时也将mask和dropout传入其中.
        x, self.attn = attention(query, key, value, mask=mask, dropout=self.dropout)

        # 通过多头注意力计算后，我们就得到了每个头计算结果组成的4维张量，我们需要将其转换为输入的形状以方便后续的计算，
        # 因此这里开始进行第一步处理环节的逆操作，先对第二和第三维进行转置，然后使用contiguous方法，
        # 这个方法的作用就是能够让转置后的张量应用view方法，否则将无法直接使用，
        # 所以，下一步就是使用view重塑形状，变成和输入形状相同.
        x = x.transpose(1, 2).contiguous().view(batch_size, -1, self.head * self.d_k)

        # 最后使用线性层列表中的最后一个线性层对输入进行线性变换得到最终的多头注意力结构的输出.
        return self.linears[-1](x)


head = 8

# 词嵌入维度embedding_dim
embedding_dim = 512

# 置零比率dropout
dropout = 0.2
query = value = key = pe_result

# 输入的掩码张量mask
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

    # 第一个实例化参数layer, 它是一个编码器层的实例化对象, 因此需要传入编码器层的参数


# 又因为编码器层中的子层是不共享的, 因此需要使用深度拷贝各个对象.
size = 512
head = 8
d_model = 512
d_ff = 64
c = copy.deepcopy
attn = MultiHeadedAttention(head, d_model)
ff = PositionwiseFeedForward(d_model, d_ff, dropout)
dropout = 0.2
layer = EncoderLayer(size, c(attn), c(ff), dropout)

# 编码器中编码器层的个数N
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

# memory是来自编码器的输出
memory = en_result

# 实际中source_mask和target_mask并不相同, 这里为了方便计算使他们都为mask
mask = torch.zeros(8, 4, 4)
source_mask = target_mask = mask

# %%
dl = DecoderLayer(size, self_attn, src_attn, ff, dropout)
dl_result = dl(x, memory, source_mask, target_mask)
print(dl_result)
print(dl_result.shape)
