import torch
import torch.nn as nn
from mmcv.cnn import normal_init

from ..builder import HEADS
from .base import BaseHead, BaseHeadCom


# by gzb: for parallel gtcn
from ...utils import Graph, GraphMulti
from ..gcns import GCNBlock, CNNBlock
from ..gcns import unit_tcn2

@HEADS.register_module()
class SimpleHead(BaseHead):
    """ A simple classification head.

    Args:
        num_classes (int): Number of classes to be classified.
        in_channels (int): Number of channels in input feature.
        loss_cls (dict): Config for building loss. Default: dict(type='CrossEntropyLoss')
        dropout (float): Probability of dropout layer. Default: 0.5.
        init_std (float): Std value for Initiation. Default: 0.01.
        kwargs (dict, optional): Any keyword argument to be used to initialize
            the head.
    """

    def __init__(self,
                 num_classes,
                 in_channels,
                 loss_cls=dict(type='CrossEntropyLoss'),
                 dropout=0.5,
                 init_std=0.01,
                 mode='3D',
                 **kwargs):
        super().__init__(num_classes, in_channels, loss_cls, **kwargs)

        self.dropout_ratio = dropout
        self.init_std = init_std
        if self.dropout_ratio != 0:
            self.dropout = nn.Dropout(p=self.dropout_ratio)
        else:
            self.dropout = None
        assert mode in ['3D', 'GCN', '2D', 'GTCN']
        self.mode = mode

        self.in_c = in_channels
        self.fc_cls = nn.Linear(self.in_c, num_classes)

    def init_weights(self):
        """Initiate the parameters from scratch."""
        normal_init(self.fc_cls, std=self.init_std)

    def forward(self, x):
        """Defines the computation performed at every call.

        Args:
            x (torch.Tensor): The input data.

        Returns:
            torch.Tensor: The classification scores for input samples.
        """

        if isinstance(x, list):
            for item in x:
                assert len(item.shape) == 2
            x = [item.mean(dim=0) for item in x]
            x = torch.stack(x)

        if len(x.shape) != 2:
            if self.mode == '2D':
                assert len(x.shape) == 5
                N, S, C, H, W = x.shape
                pool = nn.AdaptiveAvgPool2d(1)
                x = x.reshape(N * S, C, H, W)
                x = pool(x)
                x = x.reshape(N, S, C)
                x = x.mean(dim=1)
            if self.mode == '3D':
                pool = nn.AdaptiveAvgPool3d(1)
                if isinstance(x, tuple) or isinstance(x, list):
                    x = torch.cat(x, dim=1)
                x = pool(x)
                x = x.view(x.shape[:2])
            if self.mode == 'GCN':
                pool = nn.AdaptiveAvgPool2d(1)
                N, M, C, T, V = x.shape
                x = x.reshape(N * M, C, T, V) # by gzb: N*M 256 100/4 17

                x = pool(x) # by gzb: N*M 256 1 1
                x = x.reshape(N, M, C) # by gzb: N M 256
                x = x.mean(dim=1) # by gzb: N 256
                
            if self.mode == 'GTCN':
                pool = nn.AdaptiveMaxPool2d(1)
                N, M, C, T, V = x.shape
                x = x.reshape(N * M, C, T, V) # by gzb: N*M 256 100/4 17

                x = pool(x) # by gzb: N*M 256 1 1
                x = x.reshape(N, M, C) # by gzb: N M 256
                x = x.mean(dim=1) # by gzb: N 256

        assert x.shape[1] == self.in_c
        if self.dropout is not None:
            x = self.dropout(x)

        cls_score = self.fc_cls(x)
        return cls_score


@HEADS.register_module()
class I3DHead(SimpleHead):

    def __init__(self,
                 num_classes,
                 in_channels,
                 loss_cls=dict(type='CrossEntropyLoss'),
                 dropout=0.5,
                 init_std=0.01,
                 **kwargs):
        super().__init__(num_classes,
                         in_channels,
                         loss_cls=loss_cls,
                         dropout=dropout,
                         init_std=init_std,
                         mode='3D',
                         **kwargs)


@HEADS.register_module()
class SlowFastHead(I3DHead):
    pass


@HEADS.register_module()
class GCNHead(SimpleHead):

    def __init__(self,
                 num_classes,
                 in_channels,
                 loss_cls=dict(type='CrossEntropyLoss'),
                 dropout=0.,
                 init_std=0.01,
                 **kwargs):
        super().__init__(num_classes,
                         in_channels,
                         loss_cls=loss_cls,
                         dropout=dropout,
                         init_std=init_std,
                         mode='GCN',
                         **kwargs)


@HEADS.register_module()
class TSNHead(BaseHead):

    def __init__(self,
                 num_classes,
                 in_channels,
                 loss_cls=dict(type='CrossEntropyLoss'),
                 dropout=0.5,
                 init_std=0.01,
                 **kwargs):
        super().__init__(num_classes,
                         in_channels,
                         loss_cls=loss_cls,
                         dropout=dropout,
                         init_std=init_std,
                         mode='2D',
                         **kwargs)




@HEADS.register_module()
class SimpleHeadCom(BaseHeadCom):
    """ A simple classification head.

    Args:
        num_classes (int): Number of classes to be classified.
        in_channels (int): Number of channels in input feature.
        loss_cls (dict): Config for building loss. Default: dict(type='CrossEntropyLoss')
        dropout (float): Probability of dropout layer. Default: 0.5.
        init_std (float): Std value for Initiation. Default: 0.01.
        kwargs (dict, optional): Any keyword argument to be used to initialize
            the head.
    """

    def __init__(self,
                 num_classes,
                 in_channels,
                 loss_cls=dict(type='CrossEntropyLoss'),
                 dropout=0.5,
                 init_std=0.01,
                 mode='3D',
                 **kwargs):
        super().__init__(num_classes, in_channels, loss_cls, **kwargs)

        self.dropout_ratio = dropout
        self.init_std = init_std
        if self.dropout_ratio != 0:
            self.dropout = nn.Dropout(p=self.dropout_ratio)
        else:
            self.dropout = None
        assert mode in ['3D', 'GCN', '2D']
        self.mode = mode

        self.in_c = in_channels
        self.fc_cls = nn.Linear(self.in_c, num_classes)

        self.fc_cls1 = nn.Linear(self.in_c, num_classes)

    def init_weights(self):
        """Initiate the parameters from scratch."""
        normal_init(self.fc_cls, std=self.init_std)

    def forward(self, x, x1):
        """Defines the computation performed at every call.

        Args:
            x (torch.Tensor): The input data.

        Returns:
            torch.Tensor: The classification scores for input samples.
        """

        if isinstance(x, list):
            for item in x:
                assert len(item.shape) == 2
            x = [item.mean(dim=0) for item in x]
            x = torch.stack(x)

        if len(x.shape) != 2:
            if self.mode == '2D':
                assert len(x.shape) == 5
                N, S, C, H, W = x.shape
                pool = nn.AdaptiveAvgPool2d(1)
                x = x.reshape(N * S, C, H, W)
                x = pool(x)
                x = x.reshape(N, S, C)
                x = x.mean(dim=1)
            if self.mode == '3D':
                pool = nn.AdaptiveAvgPool3d(1)
                if isinstance(x, tuple) or isinstance(x, list):
                    x = torch.cat(x, dim=1)
                x = pool(x)
                x = x.view(x.shape[:2])
            if self.mode == 'GCN':
                pool = nn.AdaptiveAvgPool2d(1)
                N, M, C, T, V = x.shape
                x = x.reshape(N * M, C, T, V)

                x = pool(x)
                x = x.reshape(N, M, C)
                x = x.mean(dim=1)

                maxpool = nn.AdaptiveMaxPool2d(1)
                _, _, _, T1, _ = x1.shape
                x1 = x1.reshape(N * M, C, T1, V)

                x1 = maxpool(x1)
                x1 = x1.reshape(N, M, C)
                x1 = x1.mean(dim=1)


        assert x.shape[1] == self.in_c
        if self.dropout is not None:
            x = self.dropout(x)
            x1 = self.dropout(x1) # by gzb

        cls_score = self.fc_cls(x)
        cls_score1 = self.fc_cls1(x1)
        return cls_score + 0.8 * cls_score1


@HEADS.register_module()
class GCNHeadCom(SimpleHeadCom):

    def __init__(self,
                 num_classes,
                 in_channels,
                 loss_cls=dict(type='CrossEntropyLoss'),
                 dropout=0.,
                 init_std=0.01,
                 **kwargs):
        super().__init__(num_classes,
                         in_channels,
                         loss_cls=loss_cls,
                         dropout=dropout,
                         init_std=init_std,
                         mode='GCN',
                         **kwargs)


@HEADS.register_module()
class SimpleHeadGTCN(BaseHead):
    """ A simple classification head.

    Args:
        num_classes (int): Number of classes to be classified.
        in_channels (int): Number of channels in input feature.
        loss_cls (dict): Config for building loss. Default: dict(type='CrossEntropyLoss')
        dropout (float): Probability of dropout layer. Default: 0.5.
        init_std (float): Std value for Initiation. Default: 0.01.
        kwargs (dict, optional): Any keyword argument to be used to initialize
            the head.
    """

    def __init__(self,
                 num_classes,
                 in_channels,
                 graph_cfg=dict(layout='coco', mode='spatial'),
                 loss_cls=dict(type='CrossEntropyLoss'),
                 dropout=0.5,
                 init_std=0.01,
                 mode='3D',
                 **kwargs):
        super().__init__(num_classes, in_channels, loss_cls, **kwargs)

        self.dropout_ratio = dropout
        self.init_std = init_std
        if self.dropout_ratio != 0:
            self.dropout = nn.Dropout(p=self.dropout_ratio)
        else:
            self.dropout = None
        assert mode in ['3D', 'GCN', '2D', 'GTCN']
        self.mode = mode

        # by gzb: for parallel TCN GCN
        self.graph = Graph(**graph_cfg)
        #A = torch.tensor(self.graph.A, dtype=torch.float32, requires_grad=False)
        A = self.graph.A
        self.para1_tcn = CNNBlock(in_channels=256, out_channels=256, dilations=[1,2])
        self.para2_gcn = GCNBlock(in_channels=256, out_channels=256, A=A, adaptive=True)
        self.uni_tcn = unit_tcn2(256 * 2, 256 * 2, seg=25, mode='TV')

        self.in_c = in_channels
        self.fc_cls = nn.Linear(self.in_c, num_classes)

    def init_weights(self):
        """Initiate the parameters from scratch."""
        normal_init(self.fc_cls, std=self.init_std)

    def forward(self, x):
        """Defines the computation performed at every call.

        Args:
            x (torch.Tensor): The input data.

        Returns:
            torch.Tensor: The classification scores for input samples.
        """

        if isinstance(x, list):
            for item in x:
                assert len(item.shape) == 2
            x = [item.mean(dim=0) for item in x]
            x = torch.stack(x)

        if len(x.shape) != 2:
            if self.mode == '2D':
                assert len(x.shape) == 5
                N, S, C, H, W = x.shape
                pool = nn.AdaptiveAvgPool2d(1)
                x = x.reshape(N * S, C, H, W)
                x = pool(x)
                x = x.reshape(N, S, C)
                x = x.mean(dim=1)
            if self.mode == '3D':
                pool = nn.AdaptiveAvgPool3d(1)
                if isinstance(x, tuple) or isinstance(x, list):
                    x = torch.cat(x, dim=1)
                x = pool(x)
                x = x.view(x.shape[:2])
            if self.mode == 'GCN':
                pool = nn.AdaptiveAvgPool2d(1)
                N, M, C, T, V = x.shape
                x = x.reshape(N * M, C, T, V) # by gzb: N*M 256 100/4 17

                x = pool(x) # by gzb: N*M 256 1 1
                x = x.reshape(N, M, C) # by gzb: N M 256
                x = x.mean(dim=1) # by gzb: N 256
                
            if self.mode == 'GTCN':
                pool = nn.AdaptiveMaxPool2d(1)
                N, M, C, T, V = x.shape
                x = x.reshape(N * M, C, T, V) # by gzb: N*M 256 100/4 17

                x_tcn = self.para1_tcn(x) # -1 256 25 17
                x_gcn = self.para2_gcn(x) # -1 256 25 17
                x = torch.cat([x_tcn, x_gcn], dim=1) # -1 512 25 17
                x = self.uni_tcn(x) # -1 512 25 1

                x = pool(x) # by gzb: N*M 512 1 1
                x = x.reshape(N, M, C*2) # by gzb: N M 512
                x = x.mean(dim=1) # by gzb: N 512

        assert x.shape[1] == self.in_c
        if self.dropout is not None:
            x = self.dropout(x)

        cls_score = self.fc_cls(x)
        return cls_score

@HEADS.register_module()
class ParaGTCNHead(SimpleHeadGTCN):

    def __init__(self,
                 num_classes,
                 in_channels,
                 loss_cls=dict(type='CrossEntropyLoss'),
                 dropout=0.,
                 init_std=0.01,
                 **kwargs):
        super().__init__(num_classes,
                         in_channels,
                         loss_cls=loss_cls,
                         dropout=dropout,
                         init_std=init_std,
                         mode='GTCN',
                         **kwargs)


@HEADS.register_module()
class SimpleHeadTCN(BaseHead):
    """ A simple classification head.

    Args:
        num_classes (int): Number of classes to be classified.
        in_channels (int): Number of channels in input feature.
        loss_cls (dict): Config for building loss. Default: dict(type='CrossEntropyLoss')
        dropout (float): Probability of dropout layer. Default: 0.5.
        init_std (float): Std value for Initiation. Default: 0.01.
        kwargs (dict, optional): Any keyword argument to be used to initialize
            the head.
    """

    def __init__(self,
                 num_classes,
                 in_channels,
                 graph_cfg=dict(layout='coco', mode='spatial'),
                 loss_cls=dict(type='CrossEntropyLoss'),
                 dropout=0.5,
                 init_std=0.01,
                 mode='3D',
                 **kwargs):
        super().__init__(num_classes, in_channels, loss_cls, **kwargs)

        self.dropout_ratio = dropout
        self.init_std = init_std
        if self.dropout_ratio != 0:
            self.dropout = nn.Dropout(p=self.dropout_ratio)
        else:
            self.dropout = None
        assert mode in ['3D', 'GCN', '2D', 'GTCN']
        self.mode = mode

        # by gzb: for parallel TCN GCN
        self.graph = Graph(**graph_cfg)
        #A = torch.tensor(self.graph.A, dtype=torch.float32, requires_grad=False)
        A = self.graph.A
        #self.para1_tcn = CNNBlock(in_channels=256, out_channels=256, dilations=[1,2])
        #self.para2_gcn = GCNBlock(in_channels=256, out_channels=256, A=A, adaptive=True)
        self.uni_tcn = unit_tcn2(256, 256, seg=25, mode='TV')

        self.in_c = in_channels
        self.fc_cls = nn.Linear(self.in_c, num_classes)

    def init_weights(self):
        """Initiate the parameters from scratch."""
        normal_init(self.fc_cls, std=self.init_std)

    def forward(self, x):
        """Defines the computation performed at every call.

        Args:
            x (torch.Tensor): The input data.

        Returns:
            torch.Tensor: The classification scores for input samples.
        """

        if isinstance(x, list):
            for item in x:
                assert len(item.shape) == 2
            x = [item.mean(dim=0) for item in x]
            x = torch.stack(x)

        if len(x.shape) != 2:
            if self.mode == '2D':
                assert len(x.shape) == 5
                N, S, C, H, W = x.shape
                pool = nn.AdaptiveAvgPool2d(1)
                x = x.reshape(N * S, C, H, W)
                x = pool(x)
                x = x.reshape(N, S, C)
                x = x.mean(dim=1)
            if self.mode == '3D':
                pool = nn.AdaptiveAvgPool3d(1)
                if isinstance(x, tuple) or isinstance(x, list):
                    x = torch.cat(x, dim=1)
                x = pool(x)
                x = x.view(x.shape[:2])
            if self.mode == 'GCN':
                pool = nn.AdaptiveAvgPool2d(1)
                N, M, C, T, V = x.shape
                x = x.reshape(N * M, C, T, V) # by gzb: N*M 256 100/4 17

                x = pool(x) # by gzb: N*M 256 1 1
                x = x.reshape(N, M, C) # by gzb: N M 256
                x = x.mean(dim=1) # by gzb: N 256
                
            if self.mode == 'GTCN':
                pool = nn.AdaptiveMaxPool2d(1)
                N, M, C, T, V = x.shape
                x = x.reshape(N * M, C, T, V) # by gzb: N*M 256 100/4 17

                #x_tcn = self.para1_tcn(x) # -1 256 25 17
                #x_gcn = self.para2_gcn(x) # -1 256 25 17
                #x = torch.cat([x_tcn, x_gcn], dim=1) # -1 512 25 17
                x = self.uni_tcn(x) # -1 512 25 1

                x = pool(x) # by gzb: N*M 512 1 1
                x = x.reshape(N, M, C) # by gzb: N M 512
                x = x.mean(dim=1) # by gzb: N 512

        assert x.shape[1] == self.in_c
        if self.dropout is not None:
            x = self.dropout(x)

        cls_score = self.fc_cls(x)
        return cls_score

@HEADS.register_module()
class ParaTCNHead(SimpleHeadTCN):

    def __init__(self,
                 num_classes,
                 in_channels,
                 loss_cls=dict(type='CrossEntropyLoss'),
                 dropout=0.,
                 init_std=0.01,
                 **kwargs):
        super().__init__(num_classes,
                         in_channels,
                         loss_cls=loss_cls,
                         dropout=dropout,
                         init_std=init_std,
                         mode='GTCN',
                         **kwargs)





@HEADS.register_module()
class SimpleHeadMultiA(BaseHead):
    """ A simple classification head.

    Args:
        num_classes (int): Number of classes to be classified.
        in_channels (int): Number of channels in input feature.
        loss_cls (dict): Config for building loss. Default: dict(type='CrossEntropyLoss')
        dropout (float): Probability of dropout layer. Default: 0.5.
        init_std (float): Std value for Initiation. Default: 0.01.
        kwargs (dict, optional): Any keyword argument to be used to initialize
            the head.
    """

    def __init__(self,
                 num_classes,
                 in_channels,
                 graph_cfg=dict(layout='cocolr', mode='spatial_lr3a'),
                 loss_cls=dict(type='CrossEntropyLoss'),
                 dropout=0.5,
                 init_std=0.01,
                 mode='seqGTCN',
                 **kwargs):
        super().__init__(num_classes, in_channels, loss_cls, **kwargs)

        self.dropout_ratio = dropout
        self.init_std = init_std
        if self.dropout_ratio != 0:
            self.dropout = nn.Dropout(p=self.dropout_ratio)
        else:
            self.dropout = None
        assert mode in ['3D', 'GCN', '2D', 'GTCN', 'seqGTCN']
        self.mode = mode

        # by gzb: for parallel TCN GCN
        #self.graph = Graph(**graph_cfg)
        self.graph = GraphMulti(**graph_cfg)
        #A = torch.tensor(self.graph.A, dtype=torch.float32, requires_grad=False)
        A5 = self.graph.A5
        self.para1_tcn = CNNBlock(in_channels=256, out_channels=256, dilations=[1,2])
        self.para2_gcn = GCNBlock(in_channels=256, out_channels=256, A=A5, adaptive=False)
        self.uni_tcn = unit_tcn2(256, 256, seg=25, mode='TV')

        self.in_c = in_channels
        self.fc_cls = nn.Linear(self.in_c, num_classes)

    def init_weights(self):
        """Initiate the parameters from scratch."""
        normal_init(self.fc_cls, std=self.init_std)

    def forward(self, x):
        """Defines the computation performed at every call.

        Args:
            x (torch.Tensor): The input data.

        Returns:
            torch.Tensor: The classification scores for input samples.
        """

        if isinstance(x, list):
            for item in x:
                assert len(item.shape) == 2
            x = [item.mean(dim=0) for item in x]
            x = torch.stack(x)

        if len(x.shape) != 2:
            if self.mode == '2D':
                assert len(x.shape) == 5
                N, S, C, H, W = x.shape
                pool = nn.AdaptiveAvgPool2d(1)
                x = x.reshape(N * S, C, H, W)
                x = pool(x)
                x = x.reshape(N, S, C)
                x = x.mean(dim=1)
            if self.mode == '3D':
                pool = nn.AdaptiveAvgPool3d(1)
                if isinstance(x, tuple) or isinstance(x, list):
                    x = torch.cat(x, dim=1)
                x = pool(x)
                x = x.view(x.shape[:2])
            if self.mode == 'GCN':
                pool = nn.AdaptiveAvgPool2d(1)
                N, M, C, T, V = x.shape
                x = x.reshape(N * M, C, T, V) # by gzb: N*M 256 100/4 17

                x = pool(x) # by gzb: N*M 256 1 1
                x = x.reshape(N, M, C) # by gzb: N M 256
                x = x.mean(dim=1) # by gzb: N 256
                
            if self.mode == 'GTCN':
                pool = nn.AdaptiveMaxPool2d(1)
                N, M, C, T, V = x.shape
                x = x.reshape(N * M, C, T, V) # by gzb: N*M 256 100/4 17

                x_tcn = self.para1_tcn(x) # -1 256 25 17
                x_gcn = self.para2_gcn(x) # -1 256 25 17
                x = torch.cat([x_tcn, x_gcn], dim=1) # -1 512 25 17
                x = self.uni_tcn(x) # -1 512 25 1

                x = pool(x) # by gzb: N*M 512 1 1
                x = x.reshape(N, M, C*2) # by gzb: N M 512
                x = x.mean(dim=1) # by gzb: N 512

            if self.mode == 'seqGTCN':
                pool = nn.AdaptiveMaxPool2d(1)
                N, M, C, T, V = x.shape
                x = x.reshape(N * M, C, T, V) # by gzb: N*M 256 100/4 17

                x = self.para2_gcn(x) # -1 256 25 17
                x = self.para1_tcn(x) # -1 256 25 17
                #x = torch.cat([x_tcn, x_gcn], dim=1) # -1 512 25 17
                x = self.uni_tcn(x) # -1 256 25 1

                x = pool(x) # by gzb: N*M 512 1 1
                x = x.reshape(N, M, C) # by gzb: N M 256
                x = x.mean(dim=1) # by gzb: N 256

        assert x.shape[1] == self.in_c
        if self.dropout is not None:
            x = self.dropout(x)

        cls_score = self.fc_cls(x)
        return cls_score

@HEADS.register_module()
class SeqGTCNHeadMultiA(SimpleHeadMultiA):

    def __init__(self,
                 num_classes,
                 in_channels,
                 loss_cls=dict(type='CrossEntropyLoss'),
                 dropout=0.,
                 init_std=0.01,
                 **kwargs):
        super().__init__(num_classes,
                         in_channels,
                         loss_cls=loss_cls,
                         dropout=dropout,
                         init_std=init_std,
                         mode='seqGTCN',
                         **kwargs)
