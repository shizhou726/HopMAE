import torch
import torch.nn as nn
import math

from utils import create_activation, cs_loss, get_loss_masked_feat, get_mask


class HopMAE(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.args = args
        self.encoder = Encoder(args)
        self.decoder = Decoder(args)
        self.apply(lambda module: init_params(module, n_layers=self.args.n_encoder_layers + self.args.n_decoder_layers))

    def forward(self, x, args):
        x_origin = x.clone().detach()
        
        x_mask = get_mask(x.shape, args.mask_rate, args.device)
        x_indices = x_mask > 0
        x = x * (1 - x_mask)
        
        z = self.encoder(x)
        z_mask = get_mask(z.shape, args.remask_rate, args.device)
        z = z * (1 - z_mask)

        x_hat = self.decoder(z)
        
        ma_loss = get_loss_masked_feat(x_origin, x_hat, x_indices)
        re_loss = cs_loss(x_origin, x_hat)
        
        return re_loss, ma_loss
    
    def embed(self, x):
        if self.args.pool_type == 'max':
            return self.encoder(x).max(dim=-2).values
        elif self.args.pool_type == 'mean':
            return self.encoder(x).mean(dim=-2)
        elif self.args.pool_type == 'self':
            return self.encoder(x)[:,0,:]
        else:
            raise NotImplementedError(f"{self.args.pool_type} is not implemented.")


class LogisticRegression(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.args = args
        self.layer = nn.Linear(self.args.hidden_dim, self.args.num_classes)

    def forward(self, x):
        logits = self.layer(x)
        return logits


def init_params(module, n_layers):
    if isinstance(module, nn.Linear):
        module.weight.data.normal_(mean=0.0, std=0.02 / math.sqrt(n_layers))
        if module.bias is not None:
            module.bias.data.zero_()
    if isinstance(module, nn.Embedding):
        module.weight.data.normal_(mean=0.0, std=0.02)


class FeedForwardNetwork(nn.Module):
    def __init__(self, args):
        super(FeedForwardNetwork, self).__init__()

        self.layer1 = nn.Linear(args.hidden_dim, args.ffn_dim)
        self.activation = create_activation(args.activation)
        self.layer2 = nn.Linear(args.ffn_dim, args.hidden_dim)

    def forward(self, x):
        x = self.layer1(x)
        x = self.activation(x)
        x = self.layer2(x)
        return x


class MultiHeadAttention(nn.Module):
    def __init__(self, args):
        super(MultiHeadAttention, self).__init__()

        self.num_heads = args.n_heads

        self.att_size = att_size = args.hidden_dim // args.n_heads
        self.scale = att_size ** -0.5

        self.linear_q = nn.Linear(args.hidden_dim, args.n_heads * att_size)
        self.linear_k = nn.Linear(args.hidden_dim, args.n_heads * att_size)
        self.linear_v = nn.Linear(args.hidden_dim, args.n_heads * att_size)
        self.att_dropout = nn.Dropout(args.attention_dropout)

        self.output_layer = nn.Linear(args.n_heads * att_size, args.hidden_dim)

    def forward(self, q, k, v, attn_bias=None):
        orig_q_size = q.size()

        d_k = self.att_size
        d_v = self.att_size
        batch_size = q.size(0)

        # head_i = Attention(Q(W^Q)_i, K(W^K)_i, V(W^V)_i)
        q = self.linear_q(q).view(batch_size, -1, self.num_heads, d_k)
        k = self.linear_k(k).view(batch_size, -1, self.num_heads, d_k)
        v = self.linear_v(v).view(batch_size, -1, self.num_heads, d_v)

        q = q.transpose(1, 2)                  # [b, h, q_len, d_k]
        v = v.transpose(1, 2)                  # [b, h, v_len, d_v]
        k = k.transpose(1, 2).transpose(2, 3)  # [b, h, d_k, k_len]

        # Scaled Dot-Product Attention.
        # Attention(Q, K, V) = softmax((QK^T)/sqrt(d_k))V
        q = q * self.scale
        x = torch.matmul(q, k)  # [b, h, q_len, k_len]
        if attn_bias is not None:
            x = x + attn_bias

        x = torch.softmax(x, dim=3)
        x = self.att_dropout(x)
        x = x.matmul(v)  # [b, h, q_len, attn]

        x = x.transpose(1, 2).contiguous()  # [b, q_len, h, attn]
        x = x.view(batch_size, -1, self.num_heads * d_v)

        x = self.output_layer(x)

        assert x.size() == orig_q_size
        return x


class EncoderLayer(nn.Module):
    def __init__(self, args):
        super(EncoderLayer, self).__init__()

        self.self_attention_norm = nn.LayerNorm(args.hidden_dim)
        self.self_attention = MultiHeadAttention(args)
        self.self_attention_dropout = nn.Dropout(args.dropout)

        self.ffn_norm = nn.LayerNorm(args.hidden_dim)
        self.ffn = FeedForwardNetwork(args)
        self.ffn_dropout = nn.Dropout(args.dropout)

    def forward(self, x, attn_bias=None):
        y = self.self_attention_norm(x)
        y = self.self_attention(y, y, y, attn_bias)
        y = self.self_attention_dropout(y)
        x = x + y

        y = self.ffn_norm(x)
        y = self.ffn(y)
        y = self.ffn_dropout(y)
        x = x + y
        return x


class Encoder(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.in_proj = nn.Linear(args.input_dim, args.hidden_dim)
        self.encoders = nn.ModuleList([EncoderLayer(args) for _ in range(args.n_encoder_layers)])

    def forward(self, x):
        x = self.in_proj(x)
        for encoder in self.encoders:
            x = encoder(x)
        return x


class Decoder(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.decoders = nn.ModuleList([EncoderLayer(args) for _ in range(args.n_decoder_layers)])
        self.out_proj = nn.Linear(args.hidden_dim, args.input_dim)

    def forward(self, x):
        for decoder in self.decoders:
            x = decoder(x)
        x = self.out_proj(x)
        return x