import torch
import torch.nn as nn
from torch.nn import functional as F
try:
    from torch.nn.utils.parametrizations import weight_norm
except:
    from torch.nn.utils import weight_norm


class Chomp1d(nn.Module):
    def __init__(self, chomp_size):
        super(Chomp1d, self).__init__()
        self.chomp_size = chomp_size

    def forward(self, x):
        return x[:, :, :-self.chomp_size].contiguous()


class FFN(nn.Module):
    def __init__(self, dim, dropout_rate=0.5):
        super(FFN, self).__init__()
        self.MLP = nn.Sequential(
            torch.nn.Conv1d(dim, dim, kernel_size=1),
            #             nn.Linear(in_features=dim, out_features=dim),
            nn.ReLU(),
            torch.nn.Conv1d(dim, dim, kernel_size=1),
            #             nn.Linear(in_features=dim, out_features=dim),
            nn.Dropout(p=dropout_rate)
        )

    def forward(self, x):
        return self.MLP(x.transpose(-1, -2)).transpose(-1, -2) + x


class CausalConv(nn.Module):
    def __init__(self, n_inputs, n_outputs, kernel_size, padding,
                 num_stack=2, dropout=0.2, stride=1, dilation=1):
        super(CausalConv, self).__init__()
        self.net = nn.Sequential()

        for i in range(num_stack):
            self.net.add_module('conv{}'.format(i + 1), weight_norm(nn.Conv1d(n_inputs, n_outputs, kernel_size,
                                                                              stride=stride, padding=padding,
                                                                              dilation=dilation)))
            self.net.add_module('chomp{}'.format(i + 1), Chomp1d(padding))
            self.net.add_module('relu{}'.format(i + 1), nn.SiLU())
            self.net.add_module('drop{}'.format(i + 1), nn.Dropout(dropout))
        self.downsample = nn.Conv1d(n_inputs, n_outputs, 1) if n_inputs != n_outputs else None

    def forward(self, x):
        out = self.net(x.permute(0, 2, 1)).permute(0, 2, 1)
        if self.downsample is None:
            return out + x
        else:
            return out + self.downsample(x)


class RMSNorm(nn.Module):
    def __init__(self, normalized_shape, device, eps=1e-5):
        super(RMSNorm, self).__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(normalized_shape, device=device))

    def forward(self, x):
        output = x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps) * self.weight
        return output


class AttentionPooling(nn.Module):
    def __init__(self, user_num, dim, dropout=0.5):
        super(AttentionPooling, self).__init__()
        self.dim = dim
        self.dropout = nn.Dropout(p=dropout)
        self.linear = nn.Linear(in_features=self.dim, out_features=self.dim)
        self.act = nn.Tanh()
        self.h = nn.Embedding(user_num, dim)
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x, user):
        # x: B, L, D, C
        x = self.dropout(self.act(self.linear(x.transpose(-2, -1))))  # B, L, C, D
        att = self.softmax(torch.einsum('blcd,bd->blc', x, self.h(user)))  # (B, L, C, D) * (B, 1, 1, D)
        return (x * att.unsqueeze(dim=3)).sum(dim=2), att


class Mamba4Rec(nn.Module):
    def __init__(self, user_num, item_num, args, device, dropout_rate=0.5):
        super(Mamba4Rec, self).__init__()
        self.user_num = user_num
        self.item_num = item_num
        self.args = args
        self.device = device

        self.item_emb = nn.Embedding(self.item_num + 1, args.dim, padding_idx=0)
#         self.pos_emb = nn.Embedding(args.max_len, args.dim)
        self.dropout = nn.Dropout(p=dropout_rate)

        self.conv_layers = nn.ModuleList()
        self.last_normal = RMSNorm(normalized_shape=args.dim, eps=1e-8, device=device)
        for i in range(args.num_blocks):
            self.conv_layers.append(CausalConv(n_inputs=args.dim, n_outputs=args.dim, kernel_size=args.kernal_size,
                                               padding=int(args.kernal_size - 1 + (args.dilation[i] - 1) * (
                                                           args.kernal_size - 1)),
#                                                padding=int(args.dilation[i]* (args.kernal_size - 1) / 2),  # w/o Causal
                                               num_stack=args.num_stack, dropout=dropout_rate, stride=1,
                                               dilation=args.dilation[i]))
        self.att_layer = AttentionPooling(user_num=user_num, dim=args.dim, dropout=dropout_rate)

    def seq_feats(self, seqs, users):
        seqs_emb = self.item_emb(seqs)
        seqs_emb *= self.item_emb.embedding_dim ** 0.5  # 消融
#         positions = torch.tile(torch.arange(seqs.shape[1]), [seqs.shape[0], 1]).to(self.device)
#         seqs_emb += self.pos_emb(positions)
        seqs_emb = self.dropout(seqs_emb)

        time_mask = (seqs == 0)
        seqs_emb *= ~time_mask.unsqueeze(-1)  # broadcast in last dim
        all_emb = [seqs_emb]
        for i in range(self.args.num_blocks):
            seqs_emb = self.conv_layers[i](seqs_emb)
            all_emb.append(seqs_emb)
        all_emb = torch.stack(all_emb, dim=-1)
#         all_emb = torch.mean(all_emb, dim=-1)  # w/o attn
        all_emb, att = self.att_layer(all_emb, users)
        all_emb *= ~time_mask.unsqueeze(dim=-1)
        return self.last_normal(all_emb), att

    def forward(self, users, seqs, pos_seqs, neg_seqs):
        seqs_emb, _ = self.seq_feats(seqs, users)
        pos_embs = self.item_emb(pos_seqs)
        neg_embs = self.item_emb(neg_seqs)
        return (seqs_emb * pos_embs).sum(dim=-1), (seqs_emb * neg_embs).sum(dim=-1)

    def predict(self, user, seq, items, attention_vision=False):
        if attention_vision:
            seq_emb, att = self.seq_feats(seq, user)
            seq_emb = seq_emb.squeeze(dim=0)[-1]
            items_emb = self.item_emb(items)
            return (seq_emb * items_emb).sum(dim=-1), att
        else:
            seq_emb = self.seq_feats(seq, user)[0].squeeze(dim=0)[-1]
            items_emb = self.item_emb(items)
            return (seq_emb * items_emb).sum(dim=-1)