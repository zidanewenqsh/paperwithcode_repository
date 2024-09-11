import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class PositionalEncoding(nn.Module):
    """
    位置编码模块，用于向输入序列中加入位置信息。
    """
    def __init__(self, d_model, dropout=0.1, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        # 根据模型维度和序列长度生成位置编码
        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * -(math.log(10000.0) / d_model))
        pe = torch.zeros(max_len, d_model)
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)  # 增加一个批次维度，便于后续的广播操作
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + self.pe[:, :x.size(1), :]  # 将位置编码加到输入序列上
        return self.dropout(x)  # 应用dropout防止过拟合

class MultiHeadAttention(nn.Module):
    """
    多头注意力机制模块。
    """
    def __init__(self, hidden_size, num_heads):
        super(MultiHeadAttention, self).__init__()
        self.num_heads = num_heads
        self.hidden_size = hidden_size
        self.head_dim = hidden_size // num_heads

        assert self.head_dim * num_heads == hidden_size, "hidden_size must be divisible by num_heads"

        # 定义转换输入的全连接层
        self.query = nn.Linear(hidden_size, hidden_size)
        self.key = nn.Linear(hidden_size, hidden_size)
        self.value = nn.Linear(hidden_size, hidden_size)
        self.out = nn.Linear(hidden_size, hidden_size)

    def forward(self, query, key, value, mask=None):
        batch_size = query.size(0)
        # 应用全连接层并调整形状以适应多头操作
        query = self.query(query).view(batch_size, -1, self.num_heads, self.head_dim).transpose(1, 2)
        key = self.key(key).view(batch_size, -1, self.num_heads, self.head_dim).transpose(1, 2)
        value = self.value(value).view(batch_size, -1, self.num_heads, self.head_dim).transpose(1, 2)

        # 计算注意力得分
        scores = torch.matmul(query, key.transpose(-2, -1)) / math.sqrt(self.head_dim)
        if mask is not None:
            scores = scores.masked_fill(mask == 0, float('-inf'))
        attention = torch.softmax(scores, dim=-1)
        context = torch.matmul(attention, value)
        context = context.transpose(1, 2).contiguous().view(batch_size, -1, self.hidden_size)
        return self.out(context)

class FeedForward(nn.Module):
    """
    前馈网络模块，用于在注意力机制后进一步处理信息。
    """
    def __init__(self, hidden_size, filter_size):
        super(FeedForward, self).__init__()
        self.linear1 = nn.Linear(hidden_size, filter_size)
        self.relu = nn.ReLU()
        self.linear2 = nn.Linear(filter_size, hidden_size)

    def forward(self, x):
        return self.linear2(self.relu(self.linear1(x)))

class EncoderLayer(nn.Module):
    """
    编码器层，包含一个多头注意力机制和一个前馈网络。
    """
    def __init__(self, hidden_size, num_heads, filter_size, dropout):
        super(EncoderLayer, self).__init__()
        self.self_attn = MultiHeadAttention(hidden_size, num_heads)
        self.feed_forward = FeedForward(hidden_size, filter_size)
        self.norm1 = nn.LayerNorm(hidden_size)
        self.norm2 = nn.LayerNorm(hidden_size)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, mask):
        x2 = self.norm1(x + self.self_attn(x, x, x, mask))  # 应用自注意力和层归一化
        x = x2 + self.dropout(self.feed_forward(self.norm2(x2)))  # 应用前馈网络和dropout
        return x

class DecoderLayer(nn.Module):
    """
    解码器层，包含两个多头注意力机制和一个前馈网络。
    """
    def __init__(self, hidden_size, num_heads, filter_size, dropout):
        super(DecoderLayer, self).__init__()
        self.self_attn = MultiHeadAttention(hidden_size, num_heads)
        self.enc_attn = MultiHeadAttention(hidden_size, num_heads)
        self.feed_forward = FeedForward(hidden_size, filter_size)
        self.norm1 = nn.LayerNorm(hidden_size)
        self.norm2 = nn.LayerNorm(hidden_size)
        self.norm3 = nn.LayerNorm(hidden_size)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, enc_x, self_mask, enc_mask):
        x2 = self.norm1(x + self.self_attn(x, x, x, self_mask))  # 自注意力
        x = x2 + self.dropout(self.enc_attn(x2, enc_x, enc_x, enc_mask))  # 编码器到解码器的注意力
        x = x + self.dropout(self.feed_forward(self.norm3(x)))  # 前馈网络
        return x

class Transformer(nn.Module):
    """
    Transformer模型，包含编码器和解码器。
    """
    def __init__(self, num_tokens, hidden_size, num_heads, filter_size, num_layers, dropout=0.5):
        super(Transformer, self).__init__()
        self.model_type = 'Transformer'
        self.pos_encoder = PositionalEncoding(hidden_size, dropout)
        self.encoder = nn.ModuleList([EncoderLayer(hidden_size, num_heads, filter_size, dropout) for _ in range(num_layers)])
        self.decoder = nn.ModuleList([DecoderLayer(hidden_size, num_heads, filter_size, dropout) for _ in range(num_layers)])
        self.src_mask = None
        self.trg_mask = None

    def forward(self, src, trg):
        src = self.pos_encoder(src)  # 应用位置编码
        trg = self.pos_encoder(trg)  # 应用位置编码
        enc_output = self.encode(src, self.src_mask)  # 编码过程
        output = self.decode(trg, enc_output, self.trg_mask, self.src_mask)  # 解码过程
        return output

    def encode(self, src, mask):
        for layer in self.encoder:
            src = layer(src, mask)  # 应用每一层编码器
        return src

    def decode(self, trg, enc_src, trg_mask, src_mask):
        for layer in self.decoder:
            trg = layer(trg, enc_src, trg_mask, src_mask)  # 应用每一层解码器
        return trg

def test_transformer():
    num_tokens = 100  # 示例词汇表大小
    d_model = 512  # 嵌入维度大小
    nhead = 8  # 多头注意力中的头数
    d_hid = 2048  # 前馈网络中间层大小
    nlayers = 6  # 层数
    dropout = 0.5  # Dropout比率

    src = torch.randn(10, 32, d_model)  # 源数据
    trg = torch.randn(10, 20, d_model)  # 目标数据

    model = Transformer(num_tokens, d_model, nhead, d_hid, nlayers, dropout)
    output = model(src, trg)
    print("Output shape:", output.shape)

# 运行测试函数
test_transformer()
