import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from .utils.attention import JCA as attention_module
from .utils.init_method import conv_init


def import_class(name):
    components = name.split('.')
    mod = __import__(components[0])
    for comp in components[1:]:
        mod = getattr(mod, comp)
    return mod

class unit_tcn(nn.Module):
    def __init__(self,
                 D_in,
                 D_out,
                 kernel_size,
                 stride=1,
                 dropout=0,
                 bias=True):
        super(unit_tcn, self).__init__()
        pad = int((kernel_size - 1) / 2)
        self.conv = nn.Conv2d(D_in, D_out, kernel_size=(kernel_size, 1), padding=(pad, 0), stride=(stride, 1), bias=bias)
        self.bn = nn.BatchNorm2d(D_out)
        self.dropout = nn.Dropout(dropout)
        # initialize
        conv_init(self.conv)

    def forward(self, x):
        x = self.dropout(x)
        x = self.bn(self.conv(x))
        return x


class unit_gcn(nn.Module):
    def __init__(self, in_channels, out_channels, A, kernel_size=1, stride=1, mask_learning=False,
                 attention=False):
        super(unit_gcn, self).__init__()
        self.V = A.shape[-1]
        self.register_buffer('A', torch.from_numpy(A).float())
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.mask_learning = mask_learning
        self.attention = attention
        self.num_A = A.shape[0]

        self.conv_list = nn.ModuleList([
            nn.Conv2d(
                self.in_channels,
                self.out_channels,
                kernel_size=(kernel_size, 1),
                padding=(int((kernel_size - 1) / 2), 0),
                stride=(stride, 1)) for i in range(self.num_A)
        ])

        if mask_learning:
            self.mask = nn.Parameter(torch.ones(self.A.size()), requires_grad=True)
        if self.attention:
            self.jca = attention_module(self.out_channels)
        self.bn = nn.BatchNorm2d(self.out_channels)

        self.relu = nn.ReLU()
        # initialize
        for conv in self.conv_list:
            conv_init(conv)

    def forward(self, x):
        N, C, T, V = x.size()
        A = self.A

        if self.mask_learning:
            A = A * self.mask

        for i, a in enumerate(A):
            xa = x.view(-1, V).mm(a).view(N, C, T, V)

            if i == 0:
                y = self.conv_list[i](xa)
            else:
                y = y + self.conv_list[i](xa)

        y = self.bn(y)
        y = self.relu(y)
        if self.attention:
            y = self.jca(y) + y

        return y


class TCN_GCN_unit(nn.Module):
    def __init__(self, in_channel, out_channel, A, kernel_size=9, stride=1, dropout=0.5, mask_learning=False,
                 attention=False, residual=True):
        super(TCN_GCN_unit, self).__init__()
        self.gcn1 = unit_gcn(in_channel, out_channel, A, mask_learning=mask_learning, attention=attention)
        self.tcn1 = unit_tcn(out_channel, out_channel, kernel_size=kernel_size, dropout=dropout, stride=stride)
        self.relu = nn.ReLU(inplace=True)

        if not residual:
            self.down = lambda x: 0
        elif (in_channel == out_channel) and (stride == 1):
            self.down = lambda x: x
        else:
            self.down = unit_tcn(in_channel, out_channel, kernel_size=1, stride=stride)

    def forward(self, x):
        x = self.relu(self.tcn1(self.gcn1(x)) + self.down(x))
        return x


class Model(nn.Module):
    def __init__(self, in_channels, num_class, num_point, num_person=1, graph=None, graph_args=dict(),
                 mask_learning=False, temporal_kernel_size=9, dropout=0.5, attention=True):
        super(Model, self).__init__()
        if graph is None:
            raise ValueError()
        Graph = import_class(graph)
        self.graph = Graph(**graph_args)
        self.register_buffer('SGP1', torch.from_numpy(self.graph.SGP1).float())
        self.register_buffer('SGP2', torch.from_numpy(self.graph.SGP2).float())
        self.register_buffer('SGP3', torch.from_numpy(self.graph.SGP3).float())
        self.SGP1_mask = nn.Parameter(torch.ones(self.graph.SGP1.shape), requires_grad=True)
        self.SGP2_mask = nn.Parameter(torch.ones(self.graph.SGP2.shape), requires_grad=True)
        self.SGP3_mask = nn.Parameter(torch.ones(self.graph.SGP3.shape), requires_grad=True)

        self.data_bn = nn.BatchNorm1d(num_person * in_channels * num_point)
        self.l0 = TCN_GCN_unit(in_channels, 64, self.graph.A0, kernel_size=temporal_kernel_size,
                               mask_learning=mask_learning, dropout=dropout, attention=attention, residual=False)
        self.l1 = TCN_GCN_unit(64, 128, self.graph.A0, kernel_size=temporal_kernel_size,
                               mask_learning=mask_learning, dropout=dropout, attention=attention)
        self.l2 = TCN_GCN_unit(128, 256, self.graph.A1, kernel_size=temporal_kernel_size,
                               mask_learning=mask_learning, dropout=dropout, attention=attention)
        self.l3 = TCN_GCN_unit(256, 256, self.graph.A2,kernel_size=temporal_kernel_size,
                               mask_learning=mask_learning, dropout=dropout, attention=attention)
        self.l4 = TCN_GCN_unit(256, 256, self.graph.A3, kernel_size=temporal_kernel_size,
                               mask_learning=mask_learning, dropout=dropout, attention=attention)
        self.fc = nn.Linear(256, num_class)
        nn.init.normal_(self.fc.weight, 0, math.sqrt(2. / num_class))

    def forward(self, x):
        N, C, T, V, M = x.size()
        x = x.permute(0, 4, 3, 1, 2).contiguous().view(N, M * V * C, T)
        x = self.data_bn(x)
        x = x.view(N, M, V, C, T).permute(0, 1, 3, 4, 2).contiguous().view(N * M, C, T, V)
        x = self.l0(x)
        x = self.l1(x)
        x = torch.einsum('nctv, vu->nctu', x, self.SGP1*self.SGP1_mask)
        x = F.avg_pool2d(x, kernel_size=(2, 1))

        x = self.l2(x)
        x = torch.einsum('nctv, vu->nctu', x, self.SGP2 * self.SGP2_mask)
        x = F.avg_pool2d(x, kernel_size=(2, 1))

        x = self.l3(x)
        x = torch.einsum('nctv, vu->nctu', x, self.SGP3 * self.SGP3_mask)
        x = F.avg_pool2d(x, kernel_size=(2, 1))

        x = self.l4(x)

        c_new = x.size(1)
        x = x.view(N, M, c_new, -1)
        x = x.mean(3).mean(1)
        return self.fc(x)


