import torch
from torch import nn
from torch.autograd.variable import Variable
import numpy as np

def conv_init(conv):
    if conv.weight is not None:
        nn.init.kaiming_normal_(conv.weight, mode='fan_out')
    if conv.bias is not None:
        nn.init.constant_(conv.bias, 0)

def bn_init(bn, scale):
    nn.init.constant_(bn.weight, scale)
    nn.init.constant_(bn.bias, 0)

class unit_tcn(nn.Module):
    def __init__(self, in_channels, out_channels, bias=True, seg=20, mode='VT'):
        super(unit_tcn, self).__init__()

        if mode == 'VT':
            self.maxpool = nn.AdaptiveMaxPool2d((1, seg))
            self.conv_1 = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=(1, 3), padding=(0, 1), bias=bias),
                nn.BatchNorm2d(out_channels),
                nn.ReLU()
            )
        else: # TV
            self.maxpool = nn.AdaptiveMaxPool2d((seg, 1))
            self.conv_1 = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=(3, 1), padding=(1, 0), bias=bias),
                nn.BatchNorm2d(out_channels),
                nn.ReLU()
            )


        self.conv_2 = nn.Sequential(
            nn.Conv2d(out_channels, out_channels, kernel_size=1, bias=bias),
            nn.BatchNorm2d(out_channels),
            nn.ReLU()
        )

        self.dropout = nn.Dropout2d(0.2)

    def forward(self, x):
        r"""
        shape of x: (N, 256, V, T)
        """
        x = self.maxpool(x) # shape (N, 256, 1, 20)

        x = self.conv_1(x) # shape (N, 256, 1, 20)
        x = self.dropout(x)
        x = self.conv_2(x) # shape (N, 512, 1, 20)

        return x


class MultiScale_tcn(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=[3, 3], stride=1, dilations=[1, 2], residual=True, mode='joint'):
        super().__init__()

        assert out_channels % (len(dilations)) == 0

        self.num_branches = len(dilations)
        branch_channels = out_channels // self.num_branches

        self.branches = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(in_channels, branch_channels, kernel_size=1),
                nn.BatchNorm2d(branch_channels),
                nn.ReLU(inplace=True),
                nn.Conv2d(
                    branch_channels, branch_channels, kernel_size=(1, ks),
                    padding=(0, (ks + (ks -1) * (_dilation-1) - 1) //2),
                    dilation=(1, _dilation)
                ),
                nn.BatchNorm2d(branch_channels)
                #nn.ReLU()
                #module_selfAtt(branch_channels, branch_channels, mode='spatial')
                #SelfAtt(branch_channels, branch_channels, mode='spatial')
            ) for ks, _dilation in zip(kernel_size, dilations)
        ])

        '''
        # Additional Max & 1x1 branch; by gzb: from ctrgcn
        self.branches.append(nn.Sequential(
            nn.Conv2d(in_channels, branch_channels, kernel_size=1, padding=0),
            nn.BatchNorm2d(branch_channels),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=(1, 3), stride=(1, stride), padding=(0, 1)),
            nn.BatchNorm2d(branch_channels)  # 为什么还要加bn
        ))

        self.branches.append(nn.Sequential(
            nn.Conv2d(in_channels, branch_channels, kernel_size=1, padding=0, stride=(1, stride)),
            nn.BatchNorm2d(branch_channels)
        ))
        '''

        #self.gcn1 = unit_gcn(in_channels, in_channels, mode=mode)
        #self.gcn2 = unit_gcn(in_channels, in_channels, mode=mode)

        if residual  == False:
            self.residual = lambda x : 0
        elif in_channels == out_channels:
            self.residual = lambda x: x
        else:
            self.residual = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1),
                nn.BatchNorm2d(out_channels)
            )

        # 20211116 lstm or gru

        #self.convlstm = ConvLSTM(128, hidden_dim=[] 32, kernel_size=3, num_layers=1)

    def forward(self, input, adaA):
        '''
        shape:
            input: N, in_channels, V, T
            adaA: (N, in_C, V, V) or (N, in_C, T, T)
        
        if adaA.shape[2] != input.shape[2]: # (N, -1, T, T) and (N, -1, V, T)
            #residual = self.residual(torch.matmul(input, adaA))
            residual = self.residual(torch.matmul(adaA, input.permute(0, 1, 3, 2).contiguous())) # (N, -1, T, V)
            residual = residual.permute(0, 1, 3, 2).contiguous() # (N, -1, V, T)
        else: # (N, -1, V, V) and (N, -1, V, T)
            residual = self.residual(torch.matmul(adaA, input))
        '''
        
        residual = self.residual(input)
        #residual = self.residual(self.gcn1(input, adaA))
        #residual = self.residual(self.gcn2(self.gcn1(input, adaA), adaA))

        #input_1 = input.permute(0, 3, 2, 1).contiguous().view(64, 20, 25 * 128) #(N, T, V*C)
        #input_tcn = input.permute(0, 3, 2, 1).contiguous() # (N, T, V, 128)
        #lstm_input = self.get_lstm_input(input) # (N, T, V, 256, 256)

        branch_outs = []
        for _, tempConv in enumerate(self.branches):
            out = tempConv(input)
            branch_outs.append(out)

        out = torch.cat(branch_outs, dim=1)
        out += residual
        return out

class CascadeNet(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv_1 = nn.Conv2d(3, 16, kernel_size=1)
        self.conv_2_1 = nn.Conv2d(3, 16, kernel_size=1)
        self.conv_3_1 = nn.Conv2d(16, 32, kernel_size=1)
        self.conv_4_1 = nn.Conv2d(32, 64, kernel_size=1)

        self.conv_2_2 = nn.Conv2d(32, 32, kernel_size=1)
        self.conv_3_2 = nn.Conv2d(64, 64, kernel_size=1)
        self.conv_4_2 = nn.Conv2d(128, 128, kernel_size=1)

        self.conv_5 = nn.Conv2d(128, 128,kernel_size=1)
        
    def forward(self, input):
        out_1 = self.conv_1(input) # (-1, 16, -1, -1)
        out_2_1 = self.conv_2_1(input) # (-1, 16, -1, -1)
        out_3_1 = self.conv_3_1(out_2_1) # (-1, 32, -1, -1)
        out_4_1 = self.conv_4_1(out_3_1) # (-1, 64, -1, -1)
        out_5 = self.conv_5(out_4_1) # (-1, 64, -1, -1)

        # stage 1

        # stage 2
        out_2_2 = self.conv_2_2(torch.cat([out_1, out_2_1], 1)) # (-1, 32, -1, -1)
        
        # stage 3
        out_3_2 = self.conv_3_2(torch.cat[out_3_1, out_2_2], 1) # (-1, 64, -1, -1)

        # stage 4
        out_4_2 = self.conv_4_2(torch.cat[out_4_1, out_3_2], 1) # (-1, 64, -1, -1)

        output = out_5 + out_4_2
        return output


class unit_gcn(nn.Module):
    def __init__(self, in_channels, out_channels, mode='joint'):
        super(unit_gcn, self).__init__()
        self.mode = mode
        self.residual = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=1),
            nn.BatchNorm2d(out_channels)
        )

        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=1)
        self.bn = nn.BatchNorm2d(out_channels)

        self.relu = nn.ReLU()

        conv_init(self.conv1)
        bn_init(self.bn, 1)

    def forward_bak(self, input, A):
        '''
        shape:
            input: (N, in_channels, V, T)
            A: (N, T, V, V) or (N, V, T, T)
            output: (N, out_channels, V, T)
        '''
        if self.mode == 'joint':
            out = input.permute(0, 3, 2, 1).contiguous() # (N, T, V, C)
            out = torch.matmul(A, out) # (N, 1, V, V) * (N, T, V, C) = (N, T, V, C)
            out = out.permute(0, 3, 2, 1).contiguous() # (N, C, V, T)
        else:
            out = input.permute(0, 2, 3, 1).contiguous() # (N, V, T, C)
            out = torch.matmul(A, out) # (N, V, T, T) * (N, V, T, C) = (N, V, T, C)
            out = out.permute(0, 3, 1, 2).contiguous() # (N, C, T, V)      

        output = self.bn(self.conv1(out)) + self.residual(input)
        output = self.relu(output)
        return output

    def forward(self, input, A):
        ''' spatial conv? 
        shape:
            input: (N, C, V, T)
            A: (N, C, V, V) or (N, C, T, T)
        '''
        if self.mode == 'joint':
            out = torch.matmul(A, input) # (N, C, V, T)
        else:
            A = A.permute(0, 1, 3, 2).contiguous() # (N, C, T, T)
            out = torch.matmul(input, A) # (N, C, V, T)
        
        output = self.bn(self.conv1(out)) + self.residual(input)
        output = self.relu(output)
        return output

class module_gcn(nn.Module):
    def __init__(self, in_channels, out_channels, mode='joint'):
        super(module_gcn, self).__init__()
        self.gcn1 = unit_gcn(in_channels, out_channels, mode=mode)
        self.gcn2 = unit_gcn(out_channels, out_channels, mode=mode)
        self.gcn3 = unit_gcn(out_channels, out_channels, mode=mode)

        #self.att = SelfAtt(out_channels, out_channels, mode='spatial')

        if in_channels != out_channels:
            self.conv_A2 = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1),
                nn.BatchNorm2d(out_channels) #,
                #nn.ReLU()
            )
        else:
            self.conv_A2 = lambda x: x
        
    def forward(self, input, A):
        '''
        shape:
            input: (N, C, V, T)
            A1: (N, 128, V, V) for joint; (N, C, T, T) for frame
            A2: (N, 256, -1, -1) 20211109
            output: (N, C, V, T)
        '''
        out = self.gcn1(input, A)
        A = self.conv_A2(A) # fron (N, in_C, -1, -1) to (N, out_C, -1, -1)
        out = self.gcn2(out, A)
        out = self.gcn3(out, A)
        
        #out = self.att(out)

        return out




class ada_gcn(nn.Module):
    def __init__(self, in_channels, out_channels, rel_reduction=8):
        super(ada_gcn, self).__init__()

        self.conv3 = nn.Conv2d(in_channels, out_channels, kernel_size=1)
        
        if in_channels != out_channels:
            self.conv_ada = nn.Conv2d(in_channels, out_channels, kernel_size=1)
        else:
            self.conv_ada = lambda x: x

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                conv_init(m)
            elif isinstance(m, nn.BatchNorm2d):
                bn_init(m, 1)

    def forward(self, input, A=None, ada_A=1):
        input = self.conv3(input) # (N, out_c, V, T)
        ada_A = self.conv_ada(ada_A)
        
        A = ada_A * (A.unsqueeze(0).unsqueeze(0) if A is not None else 0)  # N,C,V,V
        output = torch.matmul(A, input) # (N, C, V, T)
        return output




# by gzb: refer to ctrgcn (unit_gcn)
class MultiScale_gcn(nn.Module):
    def __init__(self, in_channels, out_channels, A, adaptive=True, residual=True):
        super(MultiScale_gcn, self).__init__()
        self.in_c = in_channels
        self.out_c = out_channels
        self.adaptive = adaptive
        self.num_subset = A.shape[0] # 3

        self.convs_list = nn.ModuleList()
        for idx in range(self.num_subset):
            self.convs_list.append(ada_gcn(in_channels, out_channels))

        # adajacent
        if self.adaptive:
            self.PA = nn.parameter.Parameter(torch.from_numpy(A.astype(np.float32)))
        else:
            self.A = Variable(torch.from_numpy(A.astype(np.float32)), requires_grad=False)

        # residual
        if residual:
            if in_channels != out_channels:
                self.down = nn.Sequential(
                    nn.Conv2d(in_channels, out_channels, 1),
                    nn.BatchNorm2d(out_channels)
                )
            else:
                self.down = lambda x: x
        else:
            self.down = lambda x: 0
            
        self.alpha = nn.Parameter(torch.zeros(1))
        self.bn = nn.BatchNorm2d(out_channels)
        self.soft = nn.Softmax(-2)
        self.relu = nn.ReLU(inplace=True)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                conv_init(m)
            elif isinstance(m, nn.BatchNorm2d):
                bn_init(m, 1)
        bn_init(self.bn, 1e-6)

    def forward(self, input, ada_A):
        output = None
        A = self.PA if self.adaptive else self.A.cuda(input.get_device())

        # multi scale gcn
        for idx in range(self.num_subset):
            out = self.convs_list[idx](input, A[idx], ada_A)
            output = out + output if output is not None else out

        output = self.bn(output)
        output += self.down(input)
        output = self.relu(output)
        
        return output


'''
class CTRGC(nn.Module):
    def __init__(self, in_channels, out_channels, rel_reduction=8, mid_reduction=1):
        super(CTRGC, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        if in_channels == 3 or in_channels == 9:
            self.rel_channels = 8
            self.mid_channels = 16
        else:
            self.rel_channels = in_channels // rel_reduction
            self.mid_channels = in_channels // mid_reduction
        self.conv1 = nn.Conv2d(self.in_channels, self.rel_channels, kernel_size=1)
        self.conv2 = nn.Conv2d(self.in_channels, self.rel_channels, kernel_size=1)
        self.conv3 = nn.Conv2d(self.in_channels, self.out_channels, kernel_size=1)
        self.conv4 = nn.Conv2d(self.rel_channels, self.out_channels, kernel_size=1)
        self.tanh = nn.Tanh()
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                conv_init(m)
            elif isinstance(m, nn.BatchNorm2d):
                bn_init(m, 1)

    def forward(self, x, A=None, alpha=1):
        x1, x2, x3 = self.conv1(x).mean(-2), self.conv2(x).mean(-2), self.conv3(x)
        x1 = self.tanh(x1.unsqueeze(-1) - x2.unsqueeze(-2)) # (N, C, V, 1) - (N, C, 1, V)
        x1 = self.conv4(x1) * alpha + (A.unsqueeze(0).unsqueeze(0) if A is not None else 0)  # N,C,V,V
        x1 = torch.einsum('ncuv,nctv->nctu', x1, x3)
        return x1

class unit_gcn(nn.Module):
    def __init__(self, in_channels, out_channels, A, coff_embedding=4, adaptive=True, residual=True):
        super(unit_gcn, self).__init__()
        inter_channels = out_channels // coff_embedding
        self.inter_c = inter_channels
        self.out_c = out_channels
        self.in_c = in_channels
        self.adaptive = adaptive
        self.num_subset = A.shape[0]
        self.convs = nn.ModuleList()
        for i in range(self.num_subset):
            self.convs.append(CTRGC(in_channels, out_channels))

        if residual:
            if in_channels != out_channels:
                self.down = nn.Sequential(
                    nn.Conv2d(in_channels, out_channels, 1),
                    nn.BatchNorm2d(out_channels)
                )
            else:
                self.down = lambda x: x
        else:
            self.down = lambda x: 0
        if self.adaptive:
            self.PA = nn.Parameter(torch.from_numpy(A.astype(np.float32)))
        else:
            self.A = Variable(torch.from_numpy(A.astype(np.float32)), requires_grad=False)
        self.alpha = nn.Parameter(torch.zeros(1))
        self.bn = nn.BatchNorm2d(out_channels)
        self.soft = nn.Softmax(-2)
        self.relu = nn.ReLU(inplace=True)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                conv_init(m)
            elif isinstance(m, nn.BatchNorm2d):
                bn_init(m, 1)
        bn_init(self.bn, 1e-6)

    def forward(self, x):
        y = None
        if self.adaptive:
            A = self.PA
        else:
            A = self.A.cuda(x.get_device())
        for i in range(self.num_subset):
            z = self.convs[i](x, A[i], self.alpha)
            y = z + y if y is not None else z
        y = self.bn(y)
        y += self.down(x)
        y = self.relu(y)


        return y
'''