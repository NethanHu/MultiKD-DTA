import torch
import torch.nn as nn
from torch.nn import LSTM, Linear, Dropout, ReLU, Conv1d, Softmax, LayerNorm
from torch_geometric.nn import GINConv, global_add_pool

import numpy as np

len_seq = 1000
hidden_dim = 256
n_heads = 8


def generate_masks(adj, adj_sizes, n_heads):
    out = torch.ones(adj.shape[0], adj.shape[1])
    max_size = adj.shape[1]
    for e_id, drug_len in enumerate(adj_sizes):
        out[e_id, drug_len: max_size] = 0
    out = out.unsqueeze(1).expand(-1, n_heads, -1)
    return out.cuda(device=adj.device)


def generate_out_masks(drug_sizes, adj, masks, source_lengths, n_heads):
    adj_size = adj.shape[2]
    sen_size = masks.shape[2]
    max_len = adj_size + sen_size
    out = torch.ones(adj.shape[0], max_len)
    for e_id in range(len(source_lengths)):
        src_len = source_lengths[e_id]
        drug_len = drug_sizes[e_id]
        out[e_id, drug_len: adj_size] = 0
        out[e_id, adj_size + src_len:] = 0
    out = out.unsqueeze(1).expand(-1, n_heads, -1)
    return out.cuda(device=adj.device)


class Affine(nn.Module):
    def __init__(self, dim):
        super(Affine, self).__init__()
        self.g = nn.Parameter(torch.ones(1, 1, dim))
        self.b = nn.Parameter(torch.zeros(1, 1, dim))

    def forward(self, x):
        return x * self.g + self.b


class SelfAttention(nn.Module):
    def __init__(self, channel):
        super(SelfAttention, self).__init__()
        self.Q = nn.Linear(channel, channel)
        self.K = nn.Linear(channel, channel)
        self.V = nn.Linear(channel, channel)
        self.softmax = nn.Softmax(dim=-1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, xs):
        Q = self.Q(xs)
        K = self.K(xs)
        scale = K.size(-1) ** (-0.5)
        attn = self.softmax(Q * scale)
        return attn * K


class WidthMLP(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(WidthMLP, self).__init__()
        self.affine_in = Affine(input_dim)
        self.linear_1 = Linear(input_dim, output_dim)
        self.linear_2 = Linear(output_dim, output_dim)
        self.relu = ReLU()
        self.affine_out = Affine(output_dim)
        self.self_attention = SelfAttention(len_seq)
        self.sampling_linear = Linear(output_dim, output_dim)
        self.dropout = Dropout(0.2)

    def forward(self, x):
        xd = self.affine_in(x).permute(0, 2, 1)
        xd = self.relu(self.linear_1(xd))
        xd = self.relu(self.linear_2(xd))
        xd = self.affine_out(xd).permute(0, 2, 1)
        xd = self.self_attention(xd)
        xd = self.dropout(self.relu(x + xd))
        return self.sampling_linear(xd.permute(0, 2, 1))


class DepthCNN(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(DepthCNN, self).__init__()
        self.affine_in = Affine(input_dim)
        # These three Conv-Kernel must be big and vast
        self.conv1 = Conv1d(input_dim, output_dim, kernel_size=15, padding=7)
        self.conv2 = Conv1d(output_dim, output_dim, kernel_size=31, padding=15)
        self.conv3 = Conv1d(output_dim, output_dim, kernel_size=7, padding=3)
        self.bn1 = nn.BatchNorm1d(output_dim)
        self.bn2 = nn.BatchNorm1d(output_dim)
        self.bn3 = nn.BatchNorm1d(output_dim)
        self.self_attention = SelfAttention(len_seq)
        self.sampling_linear = Linear(output_dim, output_dim)
        self.affine_out = Affine(output_dim)
        self.dropout = Dropout(0.2)
        self.relu = ReLU()

    def forward(self, x):
        xd = self.affine_in(x)  # Tensor(b, 1, 1000)
        xd = self.relu(self.bn1(self.conv1(xd)))  # Tensor(b, 256, 1000)
        xd = self.relu(self.bn2(self.conv2(xd)))  # Tensor(b, 256, 1000)
        xd = self.relu(self.bn3(self.conv3(xd)))  # Tensor(b, 256, 1000)
        xd = xd.permute(0, 2, 1)  # Tensor(b, 1000, 256)
        xd = self.affine_out(xd).permute(0, 2, 1)  # Tensor(b, 256, 1000)
        xd = self.self_attention(xd)  # Tensor(b, 256, 1000)
        xd = self.dropout(self.relu(x + xd))  # Tensor(b, 256, 1000)
        return self.sampling_linear(xd.permute(0, 2, 1))


class LinkAttention(nn.Module):
    def __init__(self, input_dim, n_heads):
        super(LinkAttention, self).__init__()
        self.query = Linear(input_dim, n_heads)
        self.input_dim = input_dim
        self.n_heads = n_heads
        self.softmax = Softmax(dim=-1)

    def forward(self, x, masks):
        query = self.query(x).transpose(1, 2)
        value = x

        minus_inf = -9e15 * torch.ones_like(query)
        e = torch.where(masks > 0.5, query, minus_inf)
        a = self.softmax(e)

        out = torch.matmul(a, value)
        out = torch.sum(out, dim=1).squeeze()
        return out, a


class GINConvLayer(nn.Module):
    def __init__(self, num_feat_in, hid_dim, num_feat_out):
        super(GINConvLayer, self).__init__()
        self.dropout = Dropout(0.2)
        self.relu = ReLU()
        # hid_dim = 32
        self.conv1 = GINConv(nn.Sequential(Linear(num_feat_in, hid_dim), ReLU()))
        self.bn1 = nn.BatchNorm1d(hid_dim)
        self.conv2 = GINConv(nn.Sequential(Linear(hid_dim, hid_dim), ReLU()))
        self.bn2 = nn.BatchNorm1d(hid_dim)
        self.conv3 = GINConv(nn.Sequential(Linear(hid_dim, hid_dim), ReLU()))
        self.bn3 = nn.BatchNorm1d(hid_dim)
        self.conv4 = GINConv(nn.Sequential(Linear(hid_dim, hid_dim), ReLU()))
        self.bn4 = nn.BatchNorm1d(hid_dim)
        self.conv5 = GINConv(nn.Sequential(Linear(hid_dim, hid_dim), ReLU()))
        self.bn5 = nn.BatchNorm1d(hid_dim)

        self.fc1_xd = Linear(hid_dim, num_feat_out)

    def forward(self, x, edge_index, batch):
        xd = self.bn1(self.relu(self.conv1(x, edge_index)))
        xd = self.bn2(self.relu(self.conv2(xd, edge_index)))
        xd = self.bn3(self.relu(self.conv3(xd, edge_index)))
        xd = self.bn4(self.relu(self.conv4(xd, edge_index)))
        xd = self.bn5(self.relu(self.conv5(xd, edge_index)))
        xd = global_add_pool(xd, batch)
        xd = self.dropout(self.relu(self.fc1_xd(xd)))
        return xd


class DrugGNN(nn.Module):
    def __init__(self, num_feat_in=78, num_feat_out=640, dr=0.2):
        super(DrugGNN, self).__init__()
        # Hyper-Parameters of Drug GNNs
        self.num_feat_in = num_feat_in
        self.num_feat_out = num_feat_out
        self.dropout = Dropout(dr)
        self.relu = ReLU()
        # Drug goes through GIN Layer
        self.ginconv = GINConvLayer(num_feat_in, 32, len_seq)
        # Drug goes through WiDeCNN Layer
        self.wiMLP = WidthMLP(1, 256)
        self.deCNN = DepthCNN(1, 256)
        # Drug goes FC Layer
        self.fc_xd_1 = Linear(num_feat_in * 10 * 2, num_feat_out)
        self.fusion_d = LinkAttention(hidden_dim, n_heads=n_heads)

    def forward(self, xd, edge_index, batch):
        xd_length = len(batch)
        xd = self.ginconv(xd, edge_index, batch)
        xd = xd.unsqueeze(-1).permute(0, 2, 1)  # xd = Tensor:(b, 1, 1000)

        # WidthMLP
        xd_mlp = self.wiMLP(xd)  # Tensor(b, 1, 1000) -> Tensor(b, 1000, 256)
        # DepthCNN
        xd_cnn = self.deCNN(xd)  # Tensor(b, 1, 1000) -> Tensor(b, 1000, 256)
        # 特征融合
        s_tensor = torch.stack((xd_mlp, xd_cnn), dim=3)
        sum_tensor = torch.sum(s_tensor, dim=3)
        mean = sum_tensor.mean(dim=2, keepdim=True)
        std = sum_tensor.std(dim=2, keepdim=True)
        xd_out = (sum_tensor - mean) / torch.add(std, 1e-5)

        xd_length = np.array([x.shape[0] for x in xd_out])
        xd_masks = generate_masks(xd_out, xd_length, n_heads)
        xd_cat, xt_attn = self.fusion_d(xd_out, xd_masks)
        return xd_length, xd_out, xd_masks, xd_cat, xt_attn


class ProteinLSTM(nn.Module):
    def __init__(self, num_feat_in=128, num_feat_out=640, rnn_layers=2, dr=0.2):
        super(ProteinLSTM, self).__init__()
        self.rnn_layers = rnn_layers
        self.is_bidirectional = True
        self.dropout = Dropout(dr)
        self.encode_rnn = LSTM(num_feat_in, num_feat_in, num_layers=self.rnn_layers, batch_first=True,
                               bidirectional=self.is_bidirectional, dropout=dr)
        self.fc_xt_1 = Linear(640, num_feat_out)
        self.fusion_t = LinkAttention(hidden_dim, n_heads)

    def forward(self, target):
        xt_length = np.array([x.shape[0] for x in target])
        xt = self.fc_xt_1(target)
        xt_out, _ = self.encode_rnn(xt)
        xt_masks = generate_masks(xt_out, xt_length, n_heads)
        xt_cat, xt_attn = self.fusion_t(xt_out, xt_masks)
        return xt_length, xt_out, xt_masks, xt_cat, xt_attn


class Predictor(nn.Module):
    def __init__(self, num_feat_xd=78, num_feat_xt=128, num_feat_out=128, dr=0.2):
        super(Predictor, self).__init__()
        self.drugGNN = DrugGNN(num_feat_xd, len_seq, dr=dr)
        self.proteinLSTM = ProteinLSTM(num_feat_xt, num_feat_out, dr=dr)
        self.out_attentions = LinkAttention(hidden_dim, n_heads)
        self.out_fc1 = Linear(hidden_dim * 3, 256 * 8)
        self.out_fc2 = Linear(256 * 8, hidden_dim * 2)
        self.out_fc3 = Linear(hidden_dim * 2, 1)
        self.layer_norm = LayerNorm(128 * 2)
        self.dropout = Dropout(dr)
        self.relu = ReLU()

    def forward(self, data, target):
        xd, edge_index, batch, target = data.x, data.edge_index, data.batch, target
        xt_length, xt_out, xt_masks, xt_cat, xd_masks = self.proteinLSTM(target)
        xd_length, xd_out, xd_masks, xd_cat, xd_attn = self.drugGNN(xd, edge_index, batch)
        h = torch.cat((xd_out, xt_out), dim=1)
        out_masks = generate_out_masks(xd_length, xd_masks, xt_masks, xt_length, n_heads)
        out_cat, out_attn = self.out_attentions(h, out_masks)

        out = torch.cat([xd_cat, xt_cat, out_cat], dim=1)
        d_block = self.dropout(self.relu(self.out_fc1(out)))
        out = self.dropout(self.relu(self.out_fc2(d_block)))
        out = self.out_fc3(out).squeeze()
        return d_block, out
