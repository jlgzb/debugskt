import numpy as np

def get_sgp_mat(num_in, num_out, link):
    A = np.zeros((num_in, num_out))
    for i, j in link:
        A[i, j] = 1
    A_norm = A / np.sum(A, axis=0, keepdims=True)
    return A_norm

def edge2mat(link, num_node):
    A = np.zeros((num_node, num_node))
    for i, j in link:
        A[j, i] = 1
    return A

def get_k_scale_graph(scale, A):
    if scale == 1:
        return A
    An = np.zeros_like(A)
    A_power = np.eye(A.shape[0])
    for k in range(scale):
        A_power = A_power @ A
        An += A_power
    An[An > 0] = 1
    return An

def normalize_digraph(A):
    Dl = np.sum(A, 0)
    h, w = A.shape
    Dn = np.zeros((w, w))
    for i in range(w):
        if Dl[i] > 0:
            Dn[i, i] = Dl[i] ** (-1)
    AD = np.dot(A, Dn)
    return AD


def get_spatial_graph(num_node, self_link, inward, outward):
    I = edge2mat(self_link, num_node)
    In = normalize_digraph(edge2mat(inward, num_node))
    Out = normalize_digraph(edge2mat(outward, num_node))
    A = np.stack((I, In, Out))
    return A

def gzb_get_spatial_graph(num_node, self_link, inward, outward, neighbor_left, neighbor_right):
    I = edge2mat(self_link, num_node)
    In = normalize_digraph(edge2mat(inward, num_node))
    Out = normalize_digraph(edge2mat(outward, num_node))
    left = normalize_digraph(edge2mat(neighbor_left, num_node))
    right = normalize_digraph(edge2mat(neighbor_right, num_node))
    A = np.stack((I, In, Out, left, right))
    return A

def get_spatial_graph_6(num_node, self_link, inward, outward, neighbor_left, neighbor_right, neighbor_abs):
    I = edge2mat(self_link, num_node)
    In = normalize_digraph(edge2mat(inward, num_node))
    Out = normalize_digraph(edge2mat(outward, num_node))
    left = normalize_digraph(edge2mat(neighbor_left, num_node))
    right = normalize_digraph(edge2mat(neighbor_right, num_node))
    abs17 = normalize_digraph(edge2mat(neighbor_abs, num_node))
    A = np.stack((I, In, Out, left, right, abs17))
    return A

def get_graph_1(num_node, self_link, inward, outward, neighbor_left, neighbor_right):
    I = edge2mat(self_link, num_node)
    in_out = normalize_digraph(edge2mat(inward + outward, num_node))
    left_right = normalize_digraph(edge2mat(neighbor_left + neighbor_right, num_node))
    A = np.stack((I, in_out, left_right))
    return A

def normalize_adjacency_matrix(A):
    node_degrees = A.sum(-1)
    degs_inv_sqrt = np.power(node_degrees, -0.5)
    norm_degs_matrix = np.eye(len(node_degrees)) * degs_inv_sqrt
    return (norm_degs_matrix @ A @ norm_degs_matrix).astype(np.float32)


def k_adjacency(A, k, with_self=False, self_factor=1):
    assert isinstance(A, np.ndarray)
    I = np.eye(len(A), dtype=A.dtype)
    if k == 0:
        return I
    Ak = np.minimum(np.linalg.matrix_power(A + I, k), 1) \
       - np.minimum(np.linalg.matrix_power(A + I, k - 1), 1)
    if with_self:
        Ak += (self_factor * I)
    return Ak

def get_multiscale_spatial_graph(num_node, self_link, inward, outward):
    I = edge2mat(self_link, num_node)
    A1 = edge2mat(inward, num_node)
    A2 = edge2mat(outward, num_node)
    A3 = k_adjacency(A1, 2)
    A4 = k_adjacency(A2, 2)
    A1 = normalize_digraph(A1)
    A2 = normalize_digraph(A2)
    A3 = normalize_digraph(A3)
    A4 = normalize_digraph(A4)
    A = np.stack((I, A1, A2, A3, A4))
    return A



def get_uniform_graph(num_node, self_link, neighbor):
    A = normalize_digraph(edge2mat(neighbor + self_link, num_node))
    return A

num_node = 25
self_link = [(i, i) for i in range(num_node)]
inward_ori_index = [(1, 2), (2, 21), (3, 21), (4, 3), (5, 21), (6, 5), (7, 6),
                    (8, 7), (9, 21), (10, 9), (11, 10), (12, 11), (13, 1),
                    (14, 13), (15, 14), (16, 15), (17, 1), (18, 17), (19, 18),
                    (20, 19), (22, 23), (23, 8), (24, 25), (25, 12)]
inward = [(i - 1, j - 1) for (i, j) in inward_ori_index]
outward = [(j, i) for (i, j) in inward]
neighbor = inward + outward

# by gzb: 20220706, 2 center
'''
left_right_index_in = [(13, 2), (14, 13), (15, 14), (15, 16), # left leg
                 (5, 2), (6, 5), (7, 6), (8, 7), (23, 8), (22, 23), # left arm
                 (17, 2), (18, 17), (19, 18), (20, 19), # right leg
                 (9, 2), (10, 9), (11, 10), (12, 11), (25, 12), (24, 25)], # right arm
'''
left_index_in = [(13, 2), (14, 13), (15, 14), (15, 16), # left leg
                 (5, 2), (6, 5), (7, 6), (8, 7), (22, 8), (22, 7)] # left arm

right_index_in = [(17, 2), (18, 17), (19, 18), (20, 19), # right leg
                 (9, 2), (10, 9), (11, 10), (12, 11), (24, 12), (24, 11)] # right arm

inward_left = [(i - 1, j - 1) for (i, j) in left_index_in]
inward_right = [(i - 1, j - 1) for (i, j) in right_index_in]
outward_left = [(j, i) for (i, j) in inward_left]
outward_right = [(j, i) for (i, j) in inward_right]
neighbor_left = inward_left + outward_left
neighbor_right = inward_right + outward_right

# by gzb: 20220706,
# 18 nodes:  [2, 4, 5, 6, 7, 9, 10, 11, 13, 14, 16, 17, 18, 20, 21, 22, 24]
# new index: [1, 2, 3, 4, 5, 6, 7,  8,  9,  10, 11, 12, 13, 14, 15, 16, 17] 
num_node_18 = 18 # deleted nodes: [1, 3, 15, 19, 8, 23, 12, 25]
'''
index_18 = [(13, 2), (14, 13), (16, 14),
            (17, 2), (18, 17), (20, 18),
            (5, 2), (6, 5), (7, 6), (22, 7),
            (9, 2), (10, 9), (11, 10), (24, 11),
            (3, 2), (4, 2), (21, 2),
            (22, 21), (24, 21), (22, 24)]
'''
index_17 = [(9, 1), (10, 9), (11, 10),
            (12, 1), (13, 12), (14, 13),
            (3, 1), (4, 3), (5, 4), (16, 5),
            (6, 1), (7, 6), (8, 7), (17, 8),
            (15, 1), (2, 1),
            (16, 15), (17, 15), (16, 17), # abstract link add on 20220709
            (16, 2), (17, 2), # abstract link add on 20220711
            (16, 11), (17, 11),
            (14, 11)]
        
index_abs_17 = [(4, 1), (4, 2), (7, 1), (7, 2),
                (16, 15), (17, 15), (16, 17),
                (16, 1), (16, 2), (17, 1), (17, 2),
                (16, 11), (17, 11),
                (11, 9), (14, 9), (14, 11),
                (8, 10), (5, 13)]

#inward_17 = [(i - 1, j - 1) for (i, j) in index_17]
inward_17 = [(i - 1, j - 1) for (i, j) in (index_17 + index_abs_17)]
outward_17 = [(j, i) for (i, j) in inward_17]
link_17 = [(i, i) for i in range(17)]
left_idx_17 = [(9, 1), (10, 9), (11, 10), (3, 1), (4, 3), (5, 4), (16, 5),
               (15, 1), (2, 1),
               (16, 15), (16, 17),
               (16, 2), (16, 11)]

right_idx_17 = [(12, 1), (13, 12), (14, 13), (6, 1), (7, 6), (8, 7), (17, 8),
                (2, 1),
                (17, 15),
                (17, 2), (17, 11), (14, 11)]

# 20220711
up_idx_17 = [(3, 1), (4, 3), (5, 4), (16, 15),
             (6, 1), (7, 6), (8, 7), (17, 8),
             (15, 1), (2, 1),
             (16, 2), (17, 2),
             (16, 17),
             (5, 8), (4, 7)]

# 20220711
down_idx_17 = [(9, 1), (10, 9), (11, 10),
               (12, 1), (13, 12), (14, 13),
               (13, 10), (14, 10)]

inward_left_17 = [(i - 1, j - 1) for (i, j) in left_idx_17]
inward_right_17 = [(i - 1, j - 1) for (i, j) in right_idx_17]
outward_left_17 = [(j, i) for (i, j) in inward_left_17]
outward_right_17 = [(j, i) for (i, j) in inward_right_17]
neighbor_left_17 = inward_left_17 + outward_left_17
neighbor_right_17 = inward_right_17 + outward_right_17

inward_abs_17 = [(i - 1, j - 1) for (i, j) in index_abs_17]
outward_abs_17 = [(j, i) for (i, j) in inward_abs_17]
neighbor_abs_17 = inward_abs_17 + outward_abs_17


# 20220711
inward_up_17 = [(i - 1, j - 1) for (i, j) in up_idx_17]
inward_down_17 = [(i - 1, j - 1) for (i, j) in down_idx_17]
outward_up_17 = [(j, i) for (i, j) in inward_up_17]
outward_down_17 = [(j, i) for (i, j) in inward_down_17]
neighbor_up_17 = inward_up_17 + outward_up_17
neighbor_down_17 = inward_down_17 + outward_down_17





# other dataset
num_node_1 = 11
indices_1 = [0, 3, 5, 7, 9, 11, 13, 15, 17, 19, 20]
self_link_1 = [(i, i) for i in range(num_node_1)]
inward_ori_index_1 = [(1, 11), (2, 11), (3, 11), (4, 3), (5, 11), (6, 5), (7, 1), (8, 7), (9, 1), (10, 9)]
inward_1 = [(i - 1, j - 1) for (i, j) in inward_ori_index_1]
outward_1 = [(j, i) for (i, j) in inward_1]
neighbor_1 = inward_1 + outward_1

num_node_2 = 5
indices_2 = [3, 5, 6, 8, 10]
self_link_2 = [(i ,i) for i in range(num_node_2)]
inward_ori_index_2 = [(0, 4), (1, 4), (2, 4), (3, 4), (0, 1), (2, 3)]
inward_2 = [(i - 1, j - 1) for (i, j) in inward_ori_index_2]
outward_2 = [(j, i) for (i, j) in inward_2]
neighbor_2 = inward_2 + outward_2





class Graph:
    def __init__(self, labeling_mode='spatial', scale=1):
        self.num_node = num_node
        self.self_link = self_link
        self.inward = inward
        self.outward = outward
        self.neighbor = neighbor
        self.A = self.get_adjacency_matrix(labeling_mode)
        self.A1 = get_spatial_graph(num_node_1, self_link_1, inward_1, outward_1)
        self.A2 = get_spatial_graph(num_node_2, self_link_2, inward_2, outward_2)
        self.A_binary = edge2mat(neighbor, num_node)
        self.A_norm = normalize_adjacency_matrix(self.A_binary + 2*np.eye(num_node))
        self.A_binary_K = get_k_scale_graph(scale, self.A_binary)

        self.A_A1 = ((self.A_binary + np.eye(num_node)) / np.sum(self.A_binary + np.eye(self.A_binary.shape[0]), axis=1, keepdims=True))[indices_1]
        self.A1_A2 = edge2mat(neighbor_1, num_node_1) + np.eye(num_node_1)
        self.A1_A2 = (self.A1_A2 / np.sum(self.A1_A2, axis=1, keepdims=True))[indices_2]

        # by gzb: 20220706
        self.A_lr = self.gzb_get_adjacency_matrix(labeling_mode)
        self.A_inout_lr = self.get_A_inout_lr(labeling_mode)
        self.A_17 = get_spatial_graph(17, link_17, inward_17, outward_17)
        self.A_17_lr = gzb_get_spatial_graph(17, link_17, inward_17, outward_17, neighbor_left_17, neighbor_right_17)
        self.A_17_ud = gzb_get_spatial_graph(17, link_17, inward_17, outward_17, neighbor_up_17, neighbor_down_17)
        self.A_17_lrud = gzb_get_spatial_graph(17, link_17, neighbor_left_17, neighbor_right_17, neighbor_up_17, neighbor_down_17)
        self.A_17_lrabs = get_spatial_graph_6(17, link_17, inward_17, outward_17, neighbor_left_17, neighbor_right_17, neighbor_abs_17)


    def get_adjacency_matrix(self, labeling_mode=None):
        if labeling_mode is None:
            return self.A
        if labeling_mode == 'spatial':
            A = get_spatial_graph(num_node, self_link, inward, outward)
        else:
            raise ValueError()
        return A
    

    def gzb_get_adjacency_matrix(self, labeling_mode=None):
        if labeling_mode is None:
            return self.A
        if labeling_mode == 'spatial':
            A = gzb_get_spatial_graph(num_node, self_link, inward, outward, neighbor_left, neighbor_right)
        else:
            raise ValueError()
        return A

    def get_A_inout_lr(self, labeling_mode=None):
        if labeling_mode is None:
            return self.A
        if labeling_mode == 'spatial':
            A = get_graph_1(num_node, self_link, inward, outward, neighbor_left, neighbor_right)
        else:
            raise ValueError()
        return A
