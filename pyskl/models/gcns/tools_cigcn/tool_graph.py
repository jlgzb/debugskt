import torch
from torch import nn
import numpy as np

def edge2mat(link, num_node=25):
    A = np.zeros((num_node, num_node))
    for i, j in link:
        A[j, i] = 1
    return A
    
def normalize_digraph(A):  # 除以每列的和 # by gzb: the element value of A divided by sum of its column
    Dl = np.sum(A, 0) # by gzb: shape: (25,)
    rol, col = A.shape # by gzb: shape: 25 x 25
    Dn = np.zeros((col, col))
    for i in range(col):
        if Dl[i] > 0:
            Dn[i, i] = Dl[i] ** (-1) # by gzb: calculate the diagonal line fo Dn
    AD = np.dot(A, Dn)
    return AD

def normalize_undigraph(A):
    Dl = np.sum(A, 0)
    num_node = A.shape[0]
    Dn = np.zeros((num_node, num_node))
    for i in range(num_node):
        if Dl[i] > 0:
            Dn[i, i] = Dl[i]**(-0.5)
    DAD = np.dot(np.dot(Dn, A), Dn)
    return DAD


def conv_init(conv):
    if conv.weight is not None:
        nn.init.kaiming_normal_(conv.weight, mode='fan_out')
    if conv.bias is not None:
        nn.init.constant_(conv.bias, 0)

def bn_init(bn, scale):
    nn.init.constant_(bn.weight, scale)
    nn.init.constant_(bn.bias, 0)

class unit_adjacent(nn.Module):
    def __init__(self, in_channels, out_channels=64, mode='joint'):
        super(unit_adjacent, self).__init__()
        self.mode = mode
        self.conv_query = nn.Conv2d(in_channels, out_channels, kernel_size=1)
        self.conv_key = nn.Conv2d(in_channels, out_channels, kernel_size=1)

        self.conv_pool = nn.Conv2d(2, 1, kernel_size=1, bias=False)

        self.softmax = nn.Softmax(-1)

        if mode == 'joint':
            self.conv_A = nn.Sequential(
                nn.Conv2d(20, out_channels, kernel_size=1),
                nn.BatchNorm2d(out_channels),
                nn.ReLU()
            )
        elif mode == 'frame':
            self.conv_A = nn.Sequential(
                nn.Conv2d(25, out_channels, kernel_size=1),
                nn.BatchNorm2d(out_channels),
                nn.ReLU()
            )

        conv_init(self.conv_query)
        conv_init(self.conv_key)

    def forward(self, input):
        '''
        shape:
            input: (N, C, V, T)
            output: (N, C, V, T)
        '''
        query = self.conv_query(input) # (N, C, V, T)
        key = self.conv_key(input).permute(0, 1, 3, 2).contiguous() # (N, C, T, V)
        if self.mode == 'joint':
            A = torch.matmul(query, key) # (N, C, V, V)
        else:
            A = torch.matmul(key, query) # (N, C, T, T)

        A = self.softmax(A)
        return A

class Graph:
    def __init__(self, graph_mode='spatial'):

        
        self.get_pysicial_edge()
        #self.get_sub_edge()
        '''

        #self.A = self.get_adjacency_by_mode(graph_mode)
        self.A_global = self.get_spatial_adjacency()
        self.A_local = self.get_partial_adjacency()
        '''
        #self.A = self.get_adj_for_ctrgcn()

        self.link_part = self.get_right_part()
        self.A = self.get_right_adjacentcy()

    def get_right_adjacentcy(self):
        I = edge2mat(self.self_link)
        A = edge2mat(self.link_part)
        A = A + I
        A = normalize_digraph(A)
        return A

    def get_spatial_adjacency(self):
        # global
        I = edge2mat(self.self_link)
        A = edge2mat(self.inOut_link)
        A = A + I
        A = normalize_digraph(A)
        #A = normalize_undigraph(A) # normalize
        A_global = np.stack((I, A))

        return A_global # shape (2, 25, 25)

    def get_adj_for_ctrgcn(self):
        I = edge2mat(self.self_link)
        In = normalize_digraph(edge2mat(self.in_link))
        Out = normalize_digraph(edge2mat(self.out_link))
        A = np.stack((I, In, Out))
        return A

    def get_partial_adjacency(self):
        # local
        A_arm_left = normalize_digraph(edge2mat(self.link_arm_left))
        A_arm_right = normalize_digraph(edge2mat(self.link_arm_right))
        A_leg_left = normalize_digraph(edge2mat(self.link_leg_left))
        A_leg_right = normalize_digraph(edge2mat(self.link_leg_right))
        #A_torso = normalize_digraph(edge2mat(self.link_torso))
        A_local = np.stack((A_arm_left, A_arm_right, A_leg_left, A_leg_right)) #, A_torso))

        return A_local # shape (5, 25, 25)

    def get_pysicial_edge(self):
        num_node = 25
        link_index = [
            (1, 2), (2, 21), (3, 21), (4, 3), (5, 21), (6, 5), (7, 6),
            (8, 7), (9, 21), (10, 9), (11, 10), (12, 11), (13, 1),
            (14, 13), (15, 14), (16, 15), (17, 1), (18, 17), (19, 18),
            (20, 19), (22, 23), (23, 8), (24, 25), (25, 12)
        ]

        in_link = [(i - 1, j - 1) for (i, j) in link_index] # 0-based
        out_link = [(j, i) for (i, j) in in_link]
        self_link = [(i, i) for i in range(num_node)]
        inOut_link = in_link + out_link
        #neighbor = self_link + inOut_link

        self.self_link = self_link
        self.inOut_link = out_link# + self_link
        self.out_link = out_link
        self.in_link = in_link

    @staticmethod
    def get_right_part():
        link_arm_right = [
            (9, 10), (10, 11), (11, 12), (12, 24), (12, 25), (25, 24), #(9, 9), 
            (10, 10), (11, 11), (12, 12), (24, 24), (25, 25)
        ]

        link_leg_right = [(17, 18), (18, 19), (19, 20), #(17, 17), 
        (18, 18), (19, 19), (20, 20)]

        link_torso = [
            (1, 2), (1, 13), (1, 17), 
            (21, 2), (21, 3), (21, 5), (21, 9), 
            (3, 4),
            (1, 1), (2, 2), (3, 3), (4, 4), (5, 5), (9, 9), (13, 13), (17, 17), (21, 21)
        ]

        
        link_part = link_arm_right + link_leg_right + link_torso
        link_part = [(i - 1, j - 1) for (i, j) in link_part]

        return link_part

    def get_sub_edge(self):
        link_arm_left = [
            (5, 6), (6, 7), (7, 8), (8, 22), (8, 23), (23, 22), #(5, 5), 
            (6, 6), (7, 7), (8, 8), (22, 22), (23, 23) # self link
        ]
        link_arm_right = [
            (9, 10), (10, 11), (11, 12), (12, 24), (12, 25), (25, 24), #(9, 9), 
            (10, 10), (11, 11), (12, 12), (24, 24), (25, 25)
        ]

        link_leg_left = [(13, 14), (14, 15), (15, 16), #(13, 13), 
        (14, 14), (15, 15), (16, 16)]

        link_leg_right = [(17, 18), (18, 19), (19, 20), #(17, 17), 
        (18, 18), (19, 19), (20, 20)]

        link_torso = [
            (1, 2), (1, 13), (1, 17), 
            (21, 2), (21, 3), (21, 5), (21, 9), 
            (3, 4),
            (1, 1), (2, 2), (3, 3), (4, 4), (5, 5), (9, 9), (13, 13), (17, 17), (21, 21)
        ]

        # translate to 0-base index
        link_torso = [(i - 1, j - 1) for (i, j) in link_torso]
        self.link_arm_left = [(i - 1, j - 1) for (i, j) in link_arm_left] + link_torso
        self.link_arm_right = [(i - 1, j - 1) for (i, j) in link_arm_right] + link_torso
        self.link_leg_left = [(i - 1, j - 1) for (i, j) in link_leg_left] + link_torso
        self.link_leg_right = [(i - 1, j - 1) for (i, j) in link_leg_right] + link_torso


