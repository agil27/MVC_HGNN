import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.parameter import Parameter

class HGNN_conv(nn.Module):
    def __init__(self, in_ft, out_ft, bias=True):
        super(HGNN_conv, self).__init__()

        self.weight = Parameter(torch.Tensor(in_ft, out_ft))
        if bias:
            self.bias = Parameter(torch.Tensor(out_ft))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()

    def reset_parameters(self):
        stdv = 1. / math.sqrt(self.weight.size(1))
        self.weight.data.uniform_(-stdv, stdv)
        if self.bias is not None:
            self.bias.data.uniform_(-stdv, stdv)

    def forward(self, x: torch.Tensor, G: torch.Tensor):
        x = x.matmul(self.weight)
        if self.bias is not None:
            x = x + self.bias
        x = G.matmul(x)
        return x

class TSHGNN_conv(nn.Module):
    def __init__(self, in_ft, out_ft, bias=True):
        super(TSHGNN_conv, self).__init__()
        self.weight = Parameter(torch.Tensor(in_ft, out_ft))
        if bias:
            self.bias = Parameter(torch.Tensor(out_ft))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()

    def reset_parameters(self):
        stdv = 1. / math.sqrt(self.weight.size(1))
        self.weight.data.uniform_(-stdv, stdv)
        if self.bias is not None:
            self.bias.data.uniform_(-stdv, stdv)

    def forward(self, x: torch.Tensor, EP: torch.Tensor, NP: torch.Tensor):
        x = x.matmul(self.weight)
        if self.bias is not None:
            x = x + self.bias
        x = torch.matmul(EP, x)
        x = F.relu(x)
        x = torch.matmul(NP, x)
        x = F.relu(x)
        return x


class HGNN(nn.Module):
    def __init__(self, in_ch, n_class, n_hid, dropout=0.5):
        super(HGNN, self).__init__()
        self.dropout = dropout
        self.hgc1 = HGNN_conv(in_ch, n_hid)
        self.hgc2 = HGNN_conv(n_hid, n_class)

    def forward(self, x, G):
        x = F.relu(self.hgc1(x, G))
        x = F.dropout(x, self.dropout)
        x = self.hgc2(x, G)
        return x


# 两步HGNN
class TSHGNN(nn.Module):
    def __init__(self, in_ch, n_class, n_hid, dropout=0.5):
        super(TSHGNN, self).__init__()
        self.dropout = dropout
        self.tshgc1 = TSHGNN_conv(in_ch, n_hid)
        self.tshgc2 = TSHGNN_conv(n_hid, n_class)

    def forward(self, x, Params):
        EP, NP = Params
        x = self.tshgc1(x, EP, NP)
        x = F.dropout(x, self.dropout)
        x = self.tshgc2(x, EP, NP)
        return x


class MVHGNN(nn.Module):
    def __init__(self, n_class, n_hid, in_ft_list, dropout = 0.5):
        super(MVHGNN, self).__init__()
        assert(len(in_ft_list) >= 1)
        self.n_modal = len(in_ft_list)
        self.n_class = n_class
        self.dropout = dropout
        self.nets = [HGNN(in_ft_list[i], n_class, n_hid, dropout) for i in range(self.n_modal)]
        self.fc = nn.Sequential(
            nn.Linear(self.n_modal * n_class, n_class),
            nn.Dropout(dropout)
        )

    def forward(self, xs, G_list):
        assert(len(xs) == self.n_modal and len(G_list) == self.n_modal)
        x = [self.nets[i].forward(xs[i], G_list[i]) for i in range(self.n_modal)]
        x = torch.cat(tuple(x), dim = 1)
        x = self.fc(x)
        x = F.log_softmax(x, dim = 1)
        return x

    def cuda_(self):
        self.nets = [net.cuda() for net in self.nets]


class MVTSHGNN(nn.Module):
    def __init__(self, n_class, n_hid, in_ft_list, dropout = 0.5):
        super(MVTSHGNN, self).__init__()
        assert(len(in_ft_list) >= 1)
        self.n_modal = len(in_ft_list)
        self.n_class = n_class
        self.dropout = dropout
        self.nets = [TSHGNN(in_ft_list[i], n_class, n_hid, dropout) for i in range(self.n_modal)]
        self.fc = nn.Sequential(
            nn.Linear(self.n_modal * n_class, n_class),
            nn.Dropout(dropout)
        )

    def forward(self, xs, ParamList):
        assert(len(xs) == self.n_modal and len(ParamList) == self.n_modal)
        x = [self.nets[i].forward(xs[i], Params=ParamList[i]) for i in range(self.n_modal)]
        x = torch.cat(tuple(x), dim = 1)
        x = self.fc(x)
        x = F.log_softmax(x, dim = 1)
        return x

    def cuda_(self):
        self.nets = [net.cuda() for net in self.nets]


class MVCHGNN(nn.Module):
    def __init__(self, n_class, n_hid, in_ft_mat, dropout = 0.5):
        super(MVCHGNN, self).__init__()
        assert(len(in_ft_mat) >= 1)
        self.n_modal = len(in_ft_mat)
        # n_cascade 指的是有多少个k来构图
        self.n_cascade = len(in_ft_mat[0])
        self.n_class = n_class
        self.dropout = dropout
        self.nets = [[HGNN(in_ft_mat[i][j], n_class, n_hid, dropout) for j in range(self.n_cascade)] for i in range(self.n_modal)]
        self.fc1 = [ nn.Sequential(
            nn.Linear(self.n_cascade * n_class, n_class),
            nn.Dropout(dropout)
        ) for i in range(self.n_modal)]
        self.fc2 = nn.Sequential(
            nn.Linear(self.n_modal * n_class, n_class),
            nn.Dropout(dropout)
        )

    def forward(self, xs, G_mat):
        assert(len(xs) == self.n_modal)
        for i in range(self.n_modal):
            assert(len(xs[i]) == self.n_cascade)
        x_mat = [[self.nets[i][j].forward(xs[i][j], G_mat[i][j]) for j in range(self.n_cascade)] for i in range(self.n_modal)]
        x_vec = [torch.cat(tuple(v), dim = 1) for v in x_mat]
        x_cat = [self.fc1[i](v) for (i, v) in enumerate(x_vec)]
        x = torch.cat(tuple(x_cat), dim = 1)
        x = self.fc2(x)
        x = F.log_softmax(x, dim=1)
        return x

    def cuda_(self):
        self.nets = [[net.cuda() for net in net_modal] for net_modal in self.nets]
        self.fc1 = [fc.cuda() for fc in self.fc1]