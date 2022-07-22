from turtle import forward
import torch
import torch.nn as nn
from torch.nn import init
from torch.nn.parameter import Parameter

class SelfAtt(nn.Module):
    '''
    N, C, H, W
    '''
    def __init__(self, in_channels, out_channels, mode='spatial'):
        super().__init__()
        #self.in_channels = in_channels
        self.out_channels = out_channels
        self.mode = mode

        self.conv_query = nn.Conv2d(in_channels, out_channels, kernel_size=1) # W^(dim_in x dim_k)
        self.conv_key = nn.Conv2d(in_channels, out_channels, kernel_size=1) # W^(dim_in x dim_k)
        self.conv_value = nn.Conv2d(in_channels, out_channels, kernel_size=1) # W^(dim_in x dim_v)
        self.gamma = Parameter(torch.zeros(1))

        # redidual layer
        if in_channels == out_channels:
            self.residual = lambda x: x
        else:
            self.residual = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False),
                nn.BatchNorm2d(out_channels)
            )

        self.softmax = nn.Softmax(dim=-1)

        self.init_weights()

    def forward(self, input):
        '''
        shape:
            input: (N, in_channels, H, W
            output: (N, out_channels, H, W)
        '''
        N, C, H, W = input.shape
        residual = self.residual(input)

        # step 1: embed
        ## query: 第i行表示第i个像素位置上所有通道的值; key: 第i行表示第i个通道上所有像素的值
        query = self.conv_query(input).view(N, self.out_channels, H*W).permute(0, 2, 1).contiguous() # (N, H*W, out_channels) # 20220128 QW
        key = self.conv_key(input).view(N, self.out_channels, H*W) # (N, -1, H*W) # 20220128 KW
        value = self.conv_value(input).view(N, self.out_channels, H*W) # (N, -1, H*W) # QW

        # step 2: matmul
        if self.mode == 'spatial':
            # 第(i,j)位置的元素是指输入特征图(input)第j个元素对第i个元素的影响
            energy = self.softmax(torch.bmm(query, key)) # (N, H*W, H*W)
            energy_temp = energy.permute(0, 2, 1).contiguous()
            att = torch.bmm(value, energy_temp) # (N, out_channels, H*W)

        elif self.mode == 'channel':
            energy = self.softmax(torch.bmm(key, query)) # (N, out_channels, out_channels)
            att = torch.bmm(energy, value) # (N, out_channels, H*W)

        # step 3: skip connection
        att = att.view(N, self.out_channels, H, W)
        # out = residual + self.gamma * self.conv_out(att) # for spatial
        out = residual + self.gamma * att # for channel

        return out

    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                init.kaiming_normal_(m.weight, mode='fan_out')
                if m.bias is not None:
                    init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                init.constant_(m.weight, 1)
                init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                init.normal_(m.weight, std=0.001)
                if m.bias is not None:
                    init.constant_(m.bias, 0)




if __name__ == "__main__":
    input = torch.randn(4, 16, 20, 20)
    SelfAtt_layer = SelfAtt(in_channels=16, out_channels=20, mode='spatial')
    output = SelfAtt_layer(input)
    print (output.shape)