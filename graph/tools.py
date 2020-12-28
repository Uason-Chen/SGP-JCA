import numpy as np


def get_hierarchy_graph(num_node, A, num_graph):
    A_hierarchy = np.zeros([num_graph, num_node, num_node])
    for i in range(num_node):
        index = np.where(A[:, i] > 0)[0]
        N = len(index)
        for j in range(N):
            A_hierarchy[j, index[j], i] = A[index[j], i]
    return A_hierarchy


def edge2mat(link, num_node):
    A = np.zeros((num_node, num_node))
    for i, j in link:
        A[j, i] = 1
    return A

def get_sgp_mat(num_in, num_out, link):
    A = np.zeros((num_in, num_out))
    for i, j in link:
        A[i, j] = 1
    A_norm = A / np.sum(A, axis=0, keepdims=True)
    return A_norm


def normalize_digraph(A):
    Dl = np.sum(A, 0)
    num_node = A.shape[0]
    Dn = np.zeros((num_node, num_node))
    for i in range(num_node):
        if Dl[i] > 0:
            Dn[i, i] = Dl[i]**(-1)
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


def get_uniform_graph(num_node, self_link, neighbor):
    A = normalize_digraph(edge2mat(neighbor + self_link, num_node))
    return A


def get_uniform_distance_graph(num_node, self_link, neighbor):
    I = edge2mat(self_link, num_node)
    N = normalize_digraph(edge2mat(neighbor, num_node))
    A = I - N
    return A


def get_distance_graph(num_node, self_link, neighbor):
    I = edge2mat(self_link, num_node)
    N = normalize_digraph(edge2mat(neighbor, num_node))
    A = np.stack((I, N))
    return A


def get_spatial_graph(num_node, self_link, inward, outward):
    I = edge2mat(self_link, num_node)
    In = normalize_digraph(edge2mat(inward, num_node))
    Out = normalize_digraph(edge2mat(outward, num_node))
    A = np.stack((I, In, Out))
    return A


def get_DAD_graph(num_node, self_link, neighbor):
    A = normalize_undigraph(edge2mat(neighbor + self_link, num_node))
    return A


def get_DLD_graph(num_node, self_link, neighbor):
    I = edge2mat(self_link, num_node)
    A = I - normalize_undigraph(edge2mat(neighbor, num_node))
    return A


def get_new_graph(num_node, self_link, inward, outward):
    I = edge2mat(self_link, num_node)
    I = I[np.newaxis, :]
    In = normalize_digraph(edge2mat(inward, num_node))
    In = In[np.newaxis, :]
    Out = normalize_digraph(edge2mat(outward, num_node))
    index = np.where(Out > 0)
    num_connection = np.bincount(index[1])
    N = max(num_connection)
    Out_hierarchy = get_hierarchy_graph(num_node, Out, N)
    A = np.concatenate([I, In, Out_hierarchy], axis=0)
    return A


if __name__ == '__main__':
    A = np.array([[1, 1, 1, 1], [0, 1, 1, 1], [0, 0, 1, 1], [0, 0, 0, 1]], dtype=np.float)
    A_norm = normalize_digraph(A)
    print(A_norm)
