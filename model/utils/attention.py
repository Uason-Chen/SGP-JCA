import torch
import torch.nn as nn

class JCA(nn.Module):
    def __init__(self, in_channel, reduction1=2, reduction2=4, temporal_kernel_size=5, temporal_stride=2):
        super(JCA, self).__init__()
        time_pad = temporal_kernel_size // 2
        mid_channel1 = in_channel//reduction1
        mid_channel2 = in_channel//reduction2
        self.conv1 = nn.Conv2d(in_channel, mid_channel1, kernel_size=(temporal_kernel_size, 1), padding=(time_pad, 0),
                               stride=(temporal_stride, 1))
        self.conv2 = nn.Conv2d(mid_channel1, mid_channel2, kernel_size=(temporal_kernel_size, 1), padding=(time_pad, 0))
        self.conv3 = nn.Conv2d(in_channel, mid_channel2, kernel_size=(temporal_kernel_size, 1), padding=(time_pad, 0),
                               stride=(temporal_stride, 1))
        self.conv4 = nn.Conv2d(mid_channel2, in_channel, kernel_size=1)
        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x1 = self.relu(self.conv1(x)).mean(-1, keepdim=True)  # N,C,T,V -> N,C',T',1
        x1 = self.relu(self.conv2(x1)).mean(-2, keepdim=True)  # N,C',T',1->N,C'',1,1
        x2 = self.relu(self.conv3(x)).mean(-2, keepdim=True)  # N,C,T,V -> N,C'',1,V
        x2 = x1 + x2
        x2 = self.sigmoid(self.conv4(x2))
        return x * x2