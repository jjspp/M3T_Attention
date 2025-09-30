import torch
from torch import nn
from torchinfo import summary
import math
import os
from torch import Tensor
from torch.nn import init
from torchviz import make_dot
from pathlib import Path
from fvcore.nn import FlopCountAnalysis

class GELU(nn.Module):
    def forward(self, input: Tensor) -> Tensor:
        return input * 0.5 * (1.0 + torch.erf(input / math.sqrt(2.0)))
class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len, device):
        super(PositionalEncoding, self).__init__()
        self.encoding = torch.zeros(max_len, d_model, device=device)
        self.encoding.requires_grad = False

        pos = torch.arange(0, max_len, device=device)
        pos = pos.float().unsqueeze(dim=1)
        _2i = torch.arange(0, d_model, step=2, device=device).float()
        self.encoding[:, 0::2] = torch.sin(pos / (10000  ** (_2i / d_model)))
        self.encoding[:, 1::2] = torch.cos(pos / (10000  ** (_2i / d_model)))

    def forward(self, x):
        # 获取序列长度
        seq_len = x.size(1)
        return self.encoding[:seq_len, :]
class TransformerEmbedding(nn.Module):
    """
    修改后的Transformer嵌入层，适用于连续数据
    """
    def __init__(self, input_dim, d_model, max_len, drop_prob, device):
        super().__init__()
        self.linear = nn.Linear(input_dim, d_model)  # 将输入投影到d_model维度
        self.pos_emb = PositionalEncoding(d_model, max_len, device)
        self.drop_out = nn.Dropout(p=drop_prob)

    def forward(self, x):
        # 线性投影
        x = self.linear(x)
        # 添加位置编码
        x = x + self.pos_emb(x)
        return self.drop_out(x)
class TokenEmbedding(nn.Embedding):
    def __init__(self, vocab_size, d_model):
        super(TokenEmbedding, self).__init__(vocab_size, d_model, padding_idx=1)


class LayerNorm(nn.Module):
    def __init__(self, d_model, eps=1e-12):
        super(LayerNorm, self).__init__()
        self.gamma = nn.Parameter(torch.ones(d_model))
        self.beta = nn.Parameter(torch.zeros(d_model))
        self.eps = eps

    def forward(self, x):
        mean = x.mean(-1, keepdim=True)
        var = x.var(-1, unbiased=False, keepdim=True)
        # '-1' means last dimension.

        out = (x - mean) / torch.sqrt(var + self.eps)
        out = self.gamma * out + self.beta
        return out

class MultiHeadAttention(nn.Module):

    def __init__(self, d_model, n_head):
        super(MultiHeadAttention, self).__init__()
        self.n_head = n_head
        self.attention = ScaleDotProductAttention()
        self.w_q = nn.Linear(d_model, d_model)
        self.w_k = nn.Linear(d_model, d_model)
        self.w_v = nn.Linear(d_model, d_model)
        self.w_concat = nn.Linear(d_model, d_model)

    def forward(self, q, k, v, mask=None):
        q, k, v = self.w_q(q), self.w_k(k), self.w_v(v)
        q, k, v = self.split(q), self.split(k), self.split(v)
        out, attention = self.attention(q, k, v, mask=mask)
        out = self.concat(out)
        out = self.w_concat(out)
        return out

    def split(self, tensor):

        batch_size, length, d_model = tensor.size()

        d_tensor = d_model // self.n_head
        tensor = tensor.view(batch_size, length, self.n_head, d_tensor).transpose(1, 2)

        return tensor

    def concat(self, tensor):
        batch_size, head, length, d_tensor = tensor.size()
        d_model = head * d_tensor

        tensor = tensor.transpose(1, 2).contiguous().view(batch_size, length, d_model)
        return tensor

class PositionwiseFeedForward(nn.Module):

    def __init__(self, d_model, hidden, drop_prob=0.1):
        super(PositionwiseFeedForward, self).__init__()
        self.linear1 = nn.Linear(d_model, hidden)
        self.linear2 = nn.Linear(hidden, d_model)
        self.relu = nn.GELU()
        self.dropout = nn.Dropout(p=drop_prob)

    def forward(self, x):
        x = self.linear1(x)
        x = self.relu(x)
        x = self.dropout(x)
        x = self.linear2(x)
        return x
class ScaleDotProductAttention(nn.Module):


    def __init__(self):
        super(ScaleDotProductAttention, self).__init__()
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, q, k, v, mask=None, e=1e-12):

        batch_size, head, length, d_tensor = k.size()

        k_t = k.transpose(2, 3)  # transpose
        score = (q @ k_t) / math.sqrt(d_tensor)  # scaled dot product

        if mask is not None:
            score = score.masked_fill(mask == 0, -10000)

        score = self.softmax(score)

        v = score @ v

        return v, score



class EncoderLayer(nn.Module):

    def __init__(self, d_model, ffn_hidden, n_head, drop_prob):
        super(EncoderLayer, self).__init__()
        self.attention = MultiHeadAttention(d_model=d_model, n_head=n_head)
        self.norm1 = LayerNorm(d_model=d_model)
        self.dropout1 = nn.Dropout(p=drop_prob)

        self.ffn = PositionwiseFeedForward(d_model=d_model, hidden=ffn_hidden, drop_prob=drop_prob)
        self.norm2 = LayerNorm(d_model=d_model)
        self.dropout2 = nn.Dropout(p=drop_prob)

    def forward(self, x, src_mask):
        _x = x
        x = self.attention(q=x, k=x, v=x, mask=src_mask)

        x = self.dropout1(x)
        x = self.norm1(x + _x)

        _x = x
        x = self.ffn(x)

        x = self.dropout2(x)
        x = self.norm2(x + _x)
        return x

class TemporalAttention(nn.Module):
    def __init__(self, hidden_dim, num_heads=6, dropout=0.25, is_causal=False):
        super().__init__()
        self.is_causal = is_causal  # 是否因果注意力（防止未来信息泄露）
        self.multihead_attn = nn.MultiheadAttention(
            embed_dim=hidden_dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True
        )
        self.norm = nn.LayerNorm(hidden_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        # 生成因果掩码（可选）
        if self.is_causal:
            seq_len = x.size(1)
            mask = torch.triu(torch.ones(seq_len, seq_len), diagonal=1).bool().to(x.device)
        else:
            mask = None

        # 注意力计算
        attn_output, _ = self.multihead_attn(
            query=x,
            key=x,
            value=x,
            attn_mask=mask
        )
        attn_output = self.dropout(attn_output)
        return self.norm(x + attn_output)


class DualScaleTemporalAttention(nn.Module):
    def __init__(self, dim, scale_factor=2, num_heads=6, dropout=0.25):
        super().__init__()
        self.scale_factor = scale_factor

        # 大尺度分支
        self.attn_large = nn.MultiheadAttention(
            embed_dim=dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True
        )

        # 小尺度分支
        self.avg_pool = nn.AvgPool1d(kernel_size=scale_factor)
        self.attn_small = nn.MultiheadAttention(
            embed_dim=dim,
            num_heads=num_heads // 2,   #减少小尺度的头数
            dropout=dropout,
            batch_first=True
        )

        # 第三尺度分支（头数为1）
        self.avg_pool_third = nn.AvgPool1d(kernel_size=scale_factor * 2, stride=scale_factor*2 )
        self.attn_third = nn.MultiheadAttention(
            embed_dim=dim,
            num_heads=1,
            dropout=dropout,
            batch_first=True
        )
        # 特征融合
        self.cross_attn1 = CrossAttention(dim, )
        self.cross_attn2 = CrossAttention(dim, )
        self.norm = nn.LayerNorm(dim)
        self.dropout = nn.Dropout(dropout)
        # 特征融合
        self.cross_attn = CrossAttention(dim)
        self.norm = nn.LayerNorm(dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        # 大尺度分支 [B, T, C]
        large_out, _ = self.attn_large(x, x, x)
        # 小尺度分支
        x_pool = self.avg_pool(x.permute(0, 2, 1)).permute(0, 2, 1)  # [B, T//2, C]
        small_out, _ = self.attn_small(x_pool, x_pool, x_pool)

        # 第三尺度分支
        x_pool_third = self.avg_pool_third(x.permute(0, 2, 1)).permute(0, 2, 1)  # [B, T//4, C]
        third_out, _ = self.attn_third(x_pool_third, x_pool_third, x_pool_third)

        # 级联融合
        fused1 = self.cross_attn1(large_out, small_out)
        fused2 = self.cross_attn2(fused1, third_out)
        return self.norm(x + self.dropout(fused2))


class CrossAttention(nn.Module):
    def __init__(self, dim, num_heads=6, dropout=0.25):
        super().__init__()
        self.attn = nn.MultiheadAttention(
            embed_dim=dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True
        )
        self.upsample = nn.Upsample(scale_factor=2, mode='linear')

    def forward(self, large_feat, small_feat):
        # 上采样小尺度特征
        small_up = self.upsample(small_feat.permute(0, 2, 1)).permute(0, 2, 1)

        # 交叉注意力
        attn_out, _ = self.attn(
            query=large_feat,
            key=small_up,
            value=small_up
        )
        return attn_out + large_feat

class Encoder(nn.Module):
    def __init__(self, input_dim, max_len, d_model, ffn_hidden, n_head, n_layers, drop_prob, device):
        super().__init__()
        # 时空卷积模块
        self.conv_block = nn.Sequential(
            nn.Conv1d(18, 72, kernel_size=3, padding=1, groups=18),  # 深度卷积处理时间维度  卷积核＞1
            nn.Conv1d(72, 72, kernel_size=1),  # 点卷积调整通道
            nn.BatchNorm1d(72),
            nn.GELU(),
            nn.MaxPool1d(kernel_size=2),
            nn.Dropout(drop_prob),
            nn.Conv1d(72, 36, kernel_size=1, groups=6),  # 分组卷积处理空间维度  卷积核＝1
            nn.Conv1d(36, 36, kernel_size=1),  # 点卷积恢复通道
            nn.BatchNorm1d(36),
            nn.GELU(),
            nn.MaxPool1d(kernel_size=2),
            nn.Conv1d(36, 36, kernel_size=3, padding=1),
            nn.Dropout(drop_prob),# 标准卷积融合特征
            # ECAAttention(),
            # ChannelAttention(36)  # 通道注意力
        )
        # 双向LSTM（输入输出特征数一致）
        self.bilstm1 = nn.LSTM(input_size=36, hidden_size=18, bidirectional=True, batch_first=True)
        #时间注意力
        self.temporal_attentions = nn.ModuleList([
            DualScaleTemporalAttention(36) for _ in range(6)
        ])

        # 位置编码前的线性投影
        self.projection = nn.Linear(36, 36)
        self.emb = TransformerEmbedding(input_dim=36,  # 修改输入维度
                                        d_model=d_model,
                                        max_len=max_len,
                                        drop_prob=drop_prob,
                                        device=device)
        self.layers = nn.ModuleList([EncoderLayer(d_model=d_model,
                                                  ffn_hidden=ffn_hidden,
                                                  n_head=n_head,
                                                  drop_prob=drop_prob)
                                     for _ in range(n_layers)])

    def forward(self, x, src_mask):
        # 时空特征提取
        residual = x.permute(0, 2, 1)
        x = self.conv_block(residual)   # 残差连接
        # 维度变换和投影
        x = x.permute(0, 2, 1)
        x, _ = self.bilstm1(x)  # [batch, seq_len, 18]
        for attn in self.temporal_attentions:
            x = attn(x)

        x = self.projection(x)  # [batch_size, seq_len, d_model]
        x = self.emb(x)

        for layer in self.layers:
            x = layer(x, src_mask)
        return x


# 新增通道注意力模块
class ChannelAttention(nn.Module):
    def __init__(self, channel, reduction=4):
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool1d(1)
        self.max_pool = nn.AdaptiveMaxPool1d(1)

        self.fc = nn.Sequential(
            nn.Linear(channel, channel // reduction),
            nn.GELU(),
            nn.Linear(channel // reduction, channel),
            nn.Sigmoid()
        )

    def forward(self, x):
        b, c, _ = x.size()
        avg_out = self.fc(self.avg_pool(x).view(b, c))
        max_out = self.fc(self.max_pool(x).view(b, c))
        out = avg_out + max_out
        return x * out.unsqueeze(2)


class ECAAttention(nn.Module):
    def __init__(self, kernel_size=3):
        super().__init__()
        self.gap = nn.AdaptiveAvgPool1d(1)  # 修改为1D平均池化
        self.conv = nn.Conv1d(1, 1, kernel_size=kernel_size, padding=(kernel_size - 1) // 2)
        self.sigmoid = nn.Sigmoid()
        self.init_weights()

    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv1d):  # 处理Conv1d初始化
                init.kaiming_normal_(m.weight, mode='fan_out')
                if m.bias is not None:
                    init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm1d):
                init.constant_(m.weight, 1)
                init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                init.normal_(m.weight, std=0.001)
                if m.bias is not None:
                    init.constant_(m.bias, 0)

    def forward(self, x):
        # 输入形状: [B, C, T]
        y = self.gap(x)          # [B, C, 1]
        y = y.permute(0, 2, 1)  # [B, 1, C]
        y = self.conv(y)         # [B, 1, C]
        y = self.sigmoid(y)      # [B, 1, C]
        y = y.permute(0, 2, 1)   # [B, C, 1]
        return x * y             # 广播乘法得到[B, C, T]


class EEGEncoder(nn.Module):
    def __init__(self, d_model, n_head, max_len, ffn_hidden, n_layers, drop_prob, device):
        super().__init__()
        self.device = device

        # 编码器部分（保持原有结构）
        self.encoder = Encoder(
            input_dim=18,
            d_model=d_model,
            max_len=max_len,
            ffn_hidden=ffn_hidden,
            n_head=n_head,
            n_layers=n_layers,
            drop_prob=drop_prob,
            device=device
        )

        # 上采样回归头
        self.regression_head = nn.Sequential(
            nn.Linear(d_model, 18),
            nn.GELU(),
            nn.Dropout(drop_prob),
            nn.Linear(18, 3)  # 输出 [batch, 25, 3]
        )

        # 添加转置卷积上采样层
        self.upsample = nn.ConvTranspose1d(
            in_channels=3,
            out_channels=3,
            kernel_size=4,  # 25 * 20=500
            stride=4
        )

    def forward(self, src):
        enc_out = self.encoder(src, src_mask=None)# 应为 [200,25,18]
        output = self.regression_head(enc_out)# 应为 [200,25,3]
        output = output.permute(0, 2, 1)# 应为 [200,3,25]
        output = self.upsample(output)# 应为 [200,3,500]
        put = output.permute(0, 2, 1) # 应为 [200,500,3]
        return put


if __name__ == '__main__':
    # 参数设置
    d_model = 36        #输入的特征维度
    n_head = 6          #多头注意力机制（Multi-Head Attention）中的头数
    max_len = 500       #模型能够处理的最大序列长度，在位置编码（Positional Encoding）中会用到这个值，用于生成固定长度的位置编码矩阵
    ffn_hidden = 64     #位置前馈网络中隐藏层的维度
    n_layers = 12       #编码器和解码器中各自包含的层数。
    drop_prob = 0.25     #dropout层参数
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    # 初始化模型
    model = EEGEncoder(d_model, n_head, max_len, ffn_hidden, n_layers, drop_prob, device)
    # 生成正确的示例输入（浮点类型，三维结构）
    batch_size = 64
    seq_len = 500
    eeg_data = torch.randn(batch_size, seq_len, 18).to(device)  # [batch, seq_len, features]
    # trajectory_data = torch.randn(batch_size, seq_len, 3).to(device)
    # 使用torchinfo查看模型结构
    summary(
        model,
        input_data=[eeg_data],
        device=device
    )
    # print(model)
    model.eval()
    # 提取卷积块
    conv_block = model.encoder.conv_block

    # 创建卷积块的输入（需要正确形状）
    # 原始输入: [batch_size, seq_len, 18] -> 转置为 [batch_size, 18, seq_len]
    conv_input = eeg_data.permute(0, 2, 1)  # [64, 18, 500]

    # 计算卷积块的FLOPs
    flops = FlopCountAnalysis(conv_block, conv_input)
    conv_flops = flops.total()
    conv_macs = conv_flops / 2  # 1 MAC = 2 FLOPs

    # 输出结果
    print("=" * 60)
    print("Convolution Block (conv_block) FLOPs/MACs Analysis")
    print("=" * 60)
    print(f"{'FLOPs':<20} {'MACs':<20}")
    print("-" * 60)
    print(f"{conv_flops:<20,} {conv_macs:<20,}")
    print(f"({conv_flops / 1e9:.4f} GFLOPs)  ({conv_macs / 1e9:.4f} GMACs)")
    print("=" * 60)

    # 详细分解每个卷积层的运算量
    print("\nDetailed Breakdown of Conv Block Layers:")
    print("-" * 60)

    # 遍历conv_block中的每个层
    for i, layer in enumerate(conv_block):
        if isinstance(layer, nn.Conv1d):
            # 计算该层的FLOPs
            layer_flops = FlopCountAnalysis(layer, conv_input).total()
            layer_macs = layer_flops / 2

            # 更新输入形状用于下一层
            with torch.no_grad():
                conv_input = layer(conv_input)

            # 输出层信息
            print(f"Layer {i + 1}: {layer.__class__.__name__}")
            print(f"  Kernel: {layer.kernel_size}, Stride: {layer.stride}, "
                  f"Padding: {layer.padding}, Groups: {layer.groups}")
            print(f"  Input shape: {conv_input.shape}")
            print(f"  FLOPs: {layer_flops:,} ({layer_flops / 1e6:.2f} MFLOPs)")
            print(f"  MACs: {layer_macs:,} ({layer_macs / 1e6:.2f} MMACs)")
            print("-" * 40)
        elif isinstance(layer, (nn.MaxPool1d, nn.AvgPool1d)):
            # 池化层通常不计入FLOPs，但我们会计算其输出形状
            with torch.no_grad():
                conv_input = layer(conv_input)
            print(f"Layer {i + 1}: {layer.__class__.__name__}")
            print(f"  Kernel: {layer.kernel_size}, Stride: {layer.stride}")
            print(f"  Output shape: {conv_input.shape}")
            print("-" * 40)
        else:
            # 其他层（BatchNorm, GELU, Dropout等）
            try:
                with torch.no_grad():
                    conv_input = layer(conv_input)
                print(f"Layer {i + 1}: {layer.__class__.__name__}")
                print(f"  Output shape: {conv_input.shape}")
                print("-" * 40)
            except:
                pass

