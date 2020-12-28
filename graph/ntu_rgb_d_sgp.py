import numpy as np
from . import tools


num_node_0 = 25
self_link_0 = [(i, i) for i in range(num_node_0)]
inward_ori_index_0 = [(1, 2), (2, 21), (3, 21), (4, 3), (5, 21), (6, 5), (7, 6),
                    (8, 7), (9, 21), (10, 9), (11, 10), (12, 11), (13, 1),
                    (14, 13), (15, 14), (16, 15), (17, 1), (18, 17), (19, 18),
                    (20, 19), (22, 23), (23, 8), (24, 25), (25, 12)]
inward_0 = [(i - 1, j - 1) for (i, j) in inward_ori_index_0]
outward_0 = [(j, i) for (i, j) in inward_0]
neighbor_0 = inward_0 + outward_0

num_node_1 = 10
self_link_1 = [(i, i) for i in range(num_node_1)]
inward_ori_index_1 = [(2,1), (3,1), (4,3), (5,1), (6,5), (7,1), (8,7), (9,1), (10,9)]
inward_1 = [(i - 1, j - 1) for (i, j) in inward_ori_index_1]
outward_1 = [(j, i) for (i, j) in inward_1]
neighbor_1 = inward_1 + outward_1

num_node_2 = 5
self_link_2 = [(i, i) for i in range(num_node_2)]
inward_ori_index_2 = [(2,1),(3,1),(4,1),(5,1)]
inward_2 = [(i - 1, j - 1) for (i, j) in inward_ori_index_2]
outward_2 = [(j, i) for (i, j) in inward_2]
neighbor_2 = inward_2 + outward_2

num_node_3 = 2
self_link_3 = [(i, i) for i in range(num_node_3)]
inward_ori_index_3 = [(2,1)]
inward_3 = [(i - 1, j - 1) for (i, j) in inward_ori_index_3]
outward_3 = [(j, i) for (i, j) in inward_3]
neighbor_3 = inward_3 + outward_3


SGP1_index = [(1,1), (2,1), (21,1),
              (3,2), (4,2), (21,2),
              (5,3), (6,3), (7,3),
              (8,4), (22,4), (23,4),
              (9,5), (10,5), (11,5),
              (12,6), (24,6), (25,6),
              (13,7), (14,7),
              (15,8), (16,8),
              (17,9), (18,9),
              (19,10), (20,10)]
SGP1_index = [(i - 1, j - 1) for (i, j) in SGP1_index]

SGP2_index = [(1,1), (2,1),
              (3,2), (4,2),
              (5,3), (6,3),
              (7,4), (8,4),
              (9,5), (10,5)]
SGP2_index = [(i - 1, j - 1) for (i, j) in SGP2_index]

SGP3_index = [(1,1), (2,1), (3,1),
              (4,2), (5,2)]
SGP3_index = [(i - 1, j - 1) for (i, j) in SGP3_index]

class Graph():
    """ The Graph to model the skeletons in NTU RGB+D

    Arguments:
        labeling_mode: must be one of the follow candidates
            uniform: Uniform Labeling
            spatial: Spatial Configuration
    """

    def __init__(self, labeling_mode='uniform'):
        self.labeling_mode = labeling_mode
        self.A0 = self.get_adjacency_matrix(num_node_0, self_link_0, neighbor_0, inward_0, outward_0)
        self.A1 = self.get_adjacency_matrix(num_node_1, self_link_1, neighbor_1, inward_1, outward_1)
        self.A2 = self.get_adjacency_matrix(num_node_2, self_link_2, neighbor_2, inward_2, outward_2)
        self.A3 = self.get_adjacency_matrix(num_node_3, self_link_3, neighbor_3, inward_3, outward_3)
        self.SGP1 = tools.get_sgp_mat(num_node_0, num_node_1, SGP1_index)
        self.SGP2 = tools.get_sgp_mat(num_node_1, num_node_2, SGP2_index)
        self.SGP3 = tools.get_sgp_mat(num_node_2, num_node_3, SGP3_index)

    def get_adjacency_matrix(self, num_node, self_link, neighbor, inward, outward):
        if self.labeling_mode == 'uniform':
            A = tools.get_uniform_graph(num_node, self_link, neighbor)
        elif self.labeling_mode == 'spatial':
            A = tools.get_spatial_graph(num_node, self_link, inward, outward)
        else:
            raise ValueError()
        return A