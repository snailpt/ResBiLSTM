'''
Zhao W, Wang W F, Patnaik L M, et al. Residual and bidirectional LSTM for epileptic seizure detection[J]. 
    Frontiers in Computational Neuroscience, 2024, 18: 1415967.
    
Author: zhaowei701@163.com
'''

import torch
import torch.nn as nn

class ResnetBasicBlock(nn.Module):
    '''
    In the basic convolutional blocks of the convolutional network, 
    if the number of channels is inconsistent, 
    the 1x1 dimension is transformed to be consistent before the addition operation is performed.
    
        卷积网络里的基本卷积块，通道数如果不一致，则通过1x1维度变换到一致后再做加法运算
    '''
    def __init__(self, in_channels, out_channels, stride=1):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.stride = stride
        
        self.conv1 = nn.Conv1d(in_channels, out_channels, kernel_size=5, padding=2, stride=stride, bias=False)
        self.bn1 = nn.BatchNorm1d(out_channels)
        
        self.conv2 = nn.Conv1d(out_channels, out_channels, kernel_size=5, padding=2, stride=1, bias=False)
        self.bn2 = nn.BatchNorm1d(out_channels)
        
        # 恒等映射
        self.conv1x1 = nn.Conv1d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False)
        self.bn1x1 = nn.BatchNorm1d(out_channels)
        
    def forward(self, x):
        out = torch.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        if self.stride != 1 or self.in_channels != self.out_channels:
            residual = self.bn1x1(self.conv1x1(x))
        else:
            residual = x
        out += residual
        
        return torch.relu(out)
        
        
class ResBiLSTMNet(nn.Module):
    def __init__(self, classify_number, number_RnnCell, number_fc1):
        super().__init__()

        
        self.block_1 = nn.Sequential(ResnetBasicBlock(1, 64, 2),
                                     nn.Dropout(p=0.2)
                                    )
        
        self.block_2 = nn.Sequential(ResnetBasicBlock(64, 64, 1),
                                     nn.Dropout(p=0.2)
                                    )
        
        self.block_3 = nn.Sequential(ResnetBasicBlock(64, 128, 2),
                                     nn.Dropout(p=0.2)
                                    )

        self.lstm = nn.LSTM(128, number_RnnCell, batch_first=True, num_layers=1, bidirectional=True)
        self.fc1 = nn.Linear(number_RnnCell*2, number_fc1)
        self.fc2 = nn.Linear(number_fc1, classify_number)

    def forward(self, x):
        x = self.block_1(x)
        x = self.block_2(x)
        x = self.block_3(x)
        
        # Here permute transforms the dimension position to facilitate LSTM calculation
        x = x.permute(0, 2, 1)  
        # BiLSTM
        x_out, (h, c) = self.lstm(x) 
        # Take the last state of the bidirectional LSTM
        x = torch.concat([h[0], h[1]], dim=1)
        x = F.dropout(x, p=0.2)
        x = F.dropout(F.relu(self.fc1(x)), p=0.5) 
        x = self.fc2(x)  
        return x
