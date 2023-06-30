import torch.nn as nn
import torch
import numpy as np
import math
from lib import utils
device = 'cuda:0' if torch.cuda.is_available() else 'cpu'


# 节点聚类编码
class ClusteringEncode(nn.Module):

    def __init__(self, node_num):
        super().__init__()
        self.node_num = node_num
        self.hidden1 = nn.Sequential(
            nn.Linear(in_features=self.node_num, out_features=100, bias=True),
            nn.ReLU()
        )
        self.hidden2 = nn.Sequential(
            nn.Linear(in_features=100, out_features=100, bias=True),
            nn.ReLU()
            # nn.Sigmoid()
        )
        self.hidden3 = nn.Sequential(
            nn.Linear(in_features=100, out_features=self.node_num, bias=True),
            nn.Sigmoid()
        )

    def forward(self, x):
        fc1 = self.hidden1(x)
        fc2 = self.hidden2(fc1)
        output = self.hidden3(fc2)
        return output


# 位置编码
class PositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout=0.1, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + self.pe[:x.size(0), :]
        return self.dropout(x)


# 用来改变维度的，因为得到的数据是经过embedding的，需要原始维度
class DimChange(nn.Module):
    def __init__(self, n_feature, n_hidden, n_output):
        super(DimChange, self).__init__()
        # 定义每层用什么样的形式
        self.hidden = torch.nn.Linear(n_feature, n_hidden)
        self.predict = torch.nn.Linear(n_hidden, n_output)

    def forward(self, x):
        x = self.hidden(x)
        x = self.predict(x.view(-1))
        return x


# 标量点积注意力
class ScaledDotProductAttention(nn.Module):
    def __init__(self):
        super(ScaledDotProductAttention, self).__init__()
        self.sim_conv = nn.Conv2d(in_channels=1, out_channels=1, kernel_size=1,
                                  bias=True).to(device)  # 卷积核大小和相似度矩阵一样大小

    def forward(self, Q, K, V, attn_mask, normal_lps, d_k, n_heads):
        scores = torch.matmul(Q, K.transpose(-1, -2)) / np.sqrt(d_k)

        ############################################################################################
        # 边的相似度矩阵嵌入到scores里面
        # 此处注释，即可做消融实验
        batch = scores.shape[0]
        normal_lps = normal_lps.expand(batch, normal_lps.shape[0], normal_lps.shape[1])  # 将结构矩阵拓展batch次
        normal_lps = normal_lps.unsqueeze(1).float()  # unsqueeze(dim)在dim=1维度加一维
        node_top = self.sim_conv(normal_lps)
        # sim_link = sim_link.squeeze(1)
        node_top = node_top.repeat(1, n_heads, 1, 1)

        # sim_link = sim_link.unsqueeze(1).repeat(1, n_heads, 1, 1)
        scores = scores + node_top
        ############################################################################################

        # mask矩阵填充scores（用-1e9填充scores中与attn_mask中值为1位置相对应的元素）
        scores.masked_fill_(attn_mask, -1e9)  # Fills elements of self tensor with value where mask is True.

        attn = nn.Softmax(dim=-1)(scores)  # 对最后一个维度v做softmax
        context = torch.matmul(attn, V)  # context: [batch_size, n_heads, len_q, d_v]
        return context, attn


# 多头注意力模块
class MultiHeadAttention(nn.Module):
    """这个Attention类可以实现:
    Encoder的Self-Attention
    Decoder的Masked Self-Attention
    Encoder-Decoder的Attention
    """

    def __init__(self,d_model, d_k,d_v,n_heads):
        super(MultiHeadAttention, self).__init__()
        self.d_model = d_model
        self.d_k = d_k
        self.d_v = d_v
        self.n_heads = n_heads
        self.W_Q = nn.Linear(d_model, d_k * n_heads, bias=False)  # q,k必须维度相同，不然无法做点积
        self.W_K = nn.Linear(d_model, d_k * n_heads, bias=False)
        self.W_V = nn.Linear(d_model, d_v * n_heads, bias=False)
        self.fc = nn.Linear(n_heads * d_v, d_model, bias=False)

    def forward(self, input_Q, input_K, input_V, attn_mask, normal_lps):
        """
        input_Q: [batch_size, len_q, d_model]
        input_K: [batch_size, len_k, d_model]
        input_V: [batch_size, len_v(=len_k), d_model]
        attn_mask: [batch_size, seq_len, seq_len]
        """
        residual, batch_size = input_Q, input_Q.size(0)
        Q = self.W_Q(input_Q).view(batch_size, -1, self.n_heads, self.d_k).transpose(1, 2)
        # K: [batch_size, n_heads, len_k, d_k] # K和V的长度一定相同，维度可以不同
        K = self.W_K(input_K).view(batch_size, -1, self.n_heads, self.d_k).transpose(1, 2)
        # V: [batch_size, n_heads, len_v(=len_k), d_v]
        V = self.W_V(input_V).view(batch_size, -1, self.n_heads, self.d_v).transpose(1, 2)

        attn_mask = attn_mask.unsqueeze(1).repeat(1, self.n_heads, 1, 1)

        context, attn = ScaledDotProductAttention()(Q, K, V, attn_mask, normal_lps, self.d_k, self.n_heads)
        # 下面将不同头的输出向量拼接在一起
        context = context.transpose(1, 2).reshape(batch_size, -1, self.n_heads * self.d_v)
        output = self.fc(context)  # [batch_size, len_q, d_model]
        return nn.LayerNorm(self.d_model).to(device)(output + residual), attn


# 前馈网络
class PoswiseFeedForwardNet(nn.Module):
    def __init__(self,d_model, d_ff,):
        super(PoswiseFeedForwardNet, self).__init__()
        self.d_model =d_model
        self.d_ff =d_ff
        self.fc = nn.Sequential(
            nn.Linear(self.d_model, self.d_ff, bias=False),
            nn.ReLU(),
            nn.Linear(self.d_ff, self.d_model, bias=False)
        )

    def forward(self, inputs):
        """
        inputs: [batch_size, seq_len, d_model]
        """
        residual = inputs
        output = self.fc(inputs)
        return nn.LayerNorm(self.d_model).to(device)(output + residual)  # [batch_size, seq_len, d_model]


# node_feature
class NodeFeature(nn.Module):

    def __init__(self, node_num,d_node_feat):
        super().__init__()
        self.node_num = node_num
        self.d_node_feat = d_node_feat
        self.hidden1 = nn.Sequential(
            nn.Linear(in_features=self.node_num, out_features=100, bias=True),
            nn.ReLU()
        )
        self.hidden2 = nn.Sequential(
            nn.Linear(in_features=100, out_features=100, bias=True),
            nn.ReLU()
            # nn.Sigmoid()
        )
        self.hidden3 = nn.Sequential(
            nn.Linear(in_features=100, out_features=self.d_node_feat, bias=True),
            nn.Sigmoid()
        )

    def forward(self, x):
        h1 = self.hidden1(x)
        h2 = self.hidden2(h1)
        output1 = self.hidden3(h2)
        return output1


# 编码层
class EncoderLayer(nn.Module):
    def __init__(self,d_model, d_k,d_v,n_heads,d_ff):
        super(EncoderLayer, self).__init__()
        self.d_model = d_model
        self.d_k = d_k
        self.d_v = d_v
        self.d_ff = d_ff
        self.n_heads = n_heads
        self.enc_self_attn = MultiHeadAttention(self.d_model, self.d_k,self.d_v,self.n_heads)
        self.pos_ffn = PoswiseFeedForwardNet(self.d_model, self.d_ff)

    def forward(self, enc_inputs, enc_self_attn_mask, normal_lps):
        # 第一个enc_inputs * W_Q = Q
        # 第二个enc_inputs * W_K = K
        # 第三个enc_inputs * W_V = V
        enc_outputs, attn = self.enc_self_attn(enc_inputs, enc_inputs, enc_inputs,
                                               enc_self_attn_mask, normal_lps)
        enc_outputs = self.pos_ffn(enc_outputs)
        return enc_outputs, attn


# 解码层
class DecoderLayer(nn.Module):
    def __init__(self,d_model, d_k,d_v,n_heads,d_ff):
        self.d_model = d_model
        self.d_k = d_k
        self.d_v = d_v
        self.d_ff = d_ff
        self.n_heads = n_heads
        super(DecoderLayer, self).__init__()
        self.dec_self_attn = MultiHeadAttention(self.d_model, self.d_k,self.d_v,self.n_heads)
        self.dec_enc_attn = MultiHeadAttention(self.d_model, self.d_k,self.d_v,self.n_heads)
        self.pos_ffn = PoswiseFeedForwardNet(self.d_model, self.d_ff)

    def forward(self, dec_inputs, enc_outputs, dec_self_attn_mask, dec_enc_attn_mask, link_sim_mat):
        dec_outputs, dec_self_attn = self.dec_self_attn(dec_inputs, dec_inputs, dec_inputs,
                                                        dec_self_attn_mask, link_sim_mat)  # 这里的Q,K,V全是Decoder自己的输入
        dec_outputs, dec_enc_attn = self.dec_enc_attn(dec_outputs, enc_outputs, enc_outputs,
                                                      dec_enc_attn_mask,
                                                      link_sim_mat)  # Attention层的Q(来自decoder) 和 K,V(来自encoder)
        dec_outputs = self.pos_ffn(dec_outputs)
        return dec_outputs, dec_self_attn, dec_enc_attn  # dec_self_attn, dec_enc_attn这两个是为了可视化的


# 编码模块
class Encoder(nn.Module):
    def __init__(self,node_num,d_model,n_layers,d_k,d_v,n_heads,d_ff,r,dimc):
        super(Encoder, self).__init__()
        # self.src_emb = nn.Embedding(src_vocab_size, d_model)  # token Embedding
        self.node_num = node_num
        self.d_model = d_model
        self.n_layers = n_layers
        self.d_k = d_k
        self.d_v = d_v
        self.d_ff = d_ff
        self.n_heads = n_heads
        self.r = r
        self.dimc = dimc

        self.pos_emb = PositionalEncoding(self.d_model)  # Transformer中位置编码时固定的，不需要学习
        self.layers = nn.ModuleList([EncoderLayer(self.d_model, self.d_k,self.d_v,self.n_heads,self.d_ff)
                                     for _ in range(self.n_layers)])

    def forward(self, enc_inputs, normal_lps):
        """
        enc_inputs: 1, 2 X data.nums
        link_feature_input_v1: 1, 2*data.nums
        """
        #########################################################################################
        enc_outputs = enc_inputs.transpose(0, 1)
        enc_outputs = self.pos_emb(enc_outputs)
        enc_outputs = enc_outputs.transpose(0, 1)
        # Encoder输入序列的pad mask矩阵
        enc_self_attn_mask = utils.get_attn_pad_mask(enc_inputs, enc_inputs,self.node_num,self.dimc)
        enc_self_attns = []
        for layer in self.layers:  # for循环访问nn.ModuleList对象
            # 上一个block的输出enc_outputs作为当前block的输入
            # enc_outputs: [batch_size, src_len, d_model], enc_self_attn: [batch_size, n_heads, src_len, src_len]
            enc_outputs, enc_self_attn = layer(enc_outputs,
                                               enc_self_attn_mask,
                                               normal_lps)  # 传入的enc_outputs其实是input，传入mask矩阵是因为你要做self attention
            enc_self_attns.append(enc_self_attn)  # 这个只是为了可视化
        return enc_outputs, enc_self_attns, normal_lps


# 解码模块
class Decoder(nn.Module):
    def __init__(self,node_num,d_model,n_layers,d_k,d_v,n_heads,d_ff,r,dimc):
        super(Decoder, self).__init__()
        # self.tgt_emb = nn.Embedding(tgt_vocab_size, d_model)  # Decoder输入的embed词表
        self.node_num = node_num
        self.d_model = d_model
        self.n_layers = n_layers
        self.d_k = d_k
        self.d_v = d_v
        self.d_ff = d_ff
        self.n_heads = n_heads
        self.r = r
        self.dimc = dimc
        self.pos_emb = PositionalEncoding(self.d_model)
        self.layers = nn.ModuleList([DecoderLayer
                                     (self.d_model, self.d_k,self.d_v,self.n_heads,self.d_ff)
                                     for _ in range(self.n_layers)])  # Decoder的blocks

    def forward(self, dec_inputs, enc_inputs, enc_outputs, link_sim_mat):
        # dec_outputs = self.tgt_emb(dec_inputs)  # [batch_size, tgt_len, d_model]
        dec_outputs = dec_inputs
        dec_outputs = self.pos_emb(dec_outputs.transpose(0, 1)).transpose(0, 1).to(device)
        # Decoder输入序列的pad mask矩阵（这个例子中decoder是没有加pad的，实际应用中都是有pad填充的）
        dec_self_attn_pad_mask = utils.get_attn_pad_mask(dec_inputs, dec_inputs,self.node_num,self.dimc).to(device)
        # Masked Self_Attention：当前时刻是看不到未来的信息的
        dec_self_attn_subsequence_mask = utils.get_attn_subsequence_mask(dec_inputs).to(device)

        # Decoder中把两种mask矩阵相加（既屏蔽了pad的信息，也屏蔽了未来时刻的信息）
        dec_self_attn_mask = torch.gt((dec_self_attn_pad_mask + dec_self_attn_subsequence_mask), 0).to(device)

        # 这个mask主要用于encoder-decoder attention层
        # get_attn_pad_mask主要是enc_inputs的pad mask矩阵
        # (因为enc是处理K,V的，求Attention时是用v1,v2,..vm去加权的，
        # 要把pad对应的v_i的相关系数设为0，这样注意力就不会关注pad向量)
        #                       dec_inputs只是提供expand的size的
        dec_enc_attn_mask = utils.get_attn_pad_mask(dec_inputs, enc_inputs,self.node_num,self.dimc)

        dec_self_attns, dec_enc_attns = [], []
        for layer in self.layers:
            # Decoder的Block是上一个Block的输出dec_outputs(变化)和Encoder网络的输出enc_outputs(固定)
            dec_outputs, dec_self_attn, dec_enc_attn = layer(dec_outputs, enc_outputs, dec_self_attn_mask,
                                                             dec_enc_attn_mask, link_sim_mat)
            dec_self_attns.append(dec_self_attn)
            dec_enc_attns.append(dec_enc_attn)
        return dec_outputs, dec_self_attns, dec_enc_attns


# 这是TPformer的模型
class TPformer(nn.Module):
    def __init__(self,node_num,d_model,n_layers,d_k,d_v,n_heads,d_ff,r,d_node_feat):
        super(TPformer, self).__init__()
        self.node_num = node_num
        self.d_model = d_model
        self.n_layers = n_layers
        self.d_k = d_k
        self.d_v = d_v
        self.d_ff = d_ff
        self.n_heads = n_heads
        self.r = r
        self.d_node_feat = d_node_feat
        self.dimc = DimChange(n_feature=self.r+self.d_node_feat, n_hidden=self.node_num, n_output=1).to(device)
        self.node_embeding1 = NodeFeature(self.node_num, self.d_node_feat)
        self.node_embeding2 = NodeFeature(self.node_num, self.d_node_feat)
        self.encoder = Encoder(self.node_num,self.d_model,self.n_layers,self.d_k,self.d_v,
                               self.n_heads,self.d_ff,self.r,self.dimc).to(device)
        self.decoder = Decoder(self.node_num,self.d_model,self.n_layers,self.d_k,self.d_v,
                               self.n_heads,self.d_ff,self.r,self.dimc).to(device)
        self.projection = nn.Linear(d_model, d_model, bias=False).to(device)
        self.tanh = nn.Tanh()

    def forward(self, enc_inputs, dec_inputs, normal_lps, node_triplet1, node_triplet2):
        #########################################################################################
        # 节点结构嵌入
        batch = enc_inputs.shape[0]
        node_feature = self.node_embeding1(node_triplet1) + self.node_embeding2(node_triplet2)
        node_feature = node_feature.expand(batch, node_feature.shape[0], node_feature.shape[1])
        #########################################################################################
        # 补0保证output维度与input一致
        dim_diff = enc_inputs.shape[2]- dec_inputs.shape[2]
        if dim_diff > 0:
            dec_inputs = torch.cat([dec_inputs,
                                    torch.zeros(dec_inputs.shape[0],dec_inputs.shape[1],dim_diff).to(device)], dim=2)

        enc_inputs = torch.cat([enc_inputs, node_feature], dim=2)
        dec_inputs = torch.cat([dec_inputs, node_feature], dim=2)
        #########################################################################################
        enc_outputs, enc_self_attns, normal_lps = self.encoder(enc_inputs, normal_lps)
        dec_outputs, dec_self_attns, dec_enc_attns = self.decoder(dec_inputs, enc_inputs, enc_outputs, normal_lps)
        dec_logits = self.projection(dec_outputs)
        dec_logits = self.tanh(dec_logits)
        return dec_logits, enc_self_attns, dec_self_attns, dec_enc_attns