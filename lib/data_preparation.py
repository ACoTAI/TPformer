# -*- coding:utf-8 -*-
import numpy as np
import torch
import torch.utils.data
import torch.utils.data as Data


def normalization(x):
    mean = np.mean(x, axis=1, keepdims=True)
    std = np.std(x, axis=1, keepdims=True)
    return (x - mean) / std


def get_simples(data_seq, input_num, pre_num ):
    simple=[]
    i, j = 0, input_num + pre_num

    while j < data_seq.shape[0]:
        input_data = data_seq[i: i + input_num,:,0]
        input_data = normalization(input_data)

        traget = data_seq[i + input_num:j,:,0]
        traget = normalization(traget)

        simple.append([input_data, traget])

        i, j = i + input_num + pre_num, j + input_num + pre_num
    return simple


class TPDataset(torch.utils.data.Dataset):
    """ 交通流量数据集

        Returns:
            [sample, label]
        """
    def __init__(self,
                 input_num=None,  # 输入数据时间间隔
                 pre_num=None,  # 预测数据时间间隔
                 data=None):  # 是否使用邻接矩阵表示链路序列):

        self.input_num = input_num
        self.pre_num = pre_num
        self.data = data
        self.nodes = data.shape[1]
        self.samples, self.labels = self.__getsamples()

    def __getsamples(self):
        simples = get_simples(self.data,self.input_num,self.pre_num)
        self.sample_num = simples.__len__()

        X = torch.zeros((self.sample_num, self.nodes, self.input_num))
        Y = torch.zeros((self.sample_num, self.nodes, self.pre_num))

        for i in range(self.sample_num):
            input_data = simples[i][0]
            input_data = torch.from_numpy(input_data).t()
            X[i,:,:] = input_data
            target = simples[i][1]
            target = torch.from_numpy(target).t()
            Y[i,:,:] = target

        return X, Y

    def __len__(self):
        return self.sample_num

    def __getitem__(self, idx):
        return self.samples[idx, :, :], self.labels[idx, :, :]


def links_to_sim(links):
    l1 = links.shape[1]  # link number
    node_num = int(links.shape[2] / 2)
    link_sim_mats = torch.zeros([links.shape[0], l1, l1]).cuda()

    for s in range(links.shape[0]):
        links_of_samlpe = links[s, :, :]
        N1 = torch.where(links_of_samlpe[:, :node_num])[1] + 1  # 出发节点
        N2 = torch.where(links_of_samlpe[:, node_num:])[1] + 1  # 到达节点
        link_sim_mat = torch.zeros([l1, l1]).cuda()

        for i in range(l1):
            print('link sim calculate step:', s, 'total steps:', links.shape[0], 'link:', i)
            v1 = N1[i]  # link1出发节点
            v2 = N2[i]  # link1达到节点
            n12 = torch.where(N1 == v1, N2, torch.zeros_like(N1))
            n12 = n12[torch.where(n12)]  # v1节点的所有到达节点
            n11 = torch.where(N2 == v1, N1, torch.zeros_like(N1))
            n11 = n11[torch.where(n11)]  # v1节点的所有出发节点
            n1 = torch.cat((n11, n12)).unique()
            n21 = torch.where(N1 == v2, N2, torch.zeros_like(N1))
            n21 = n21[torch.where(n21)]  # v2节点的所有到达节点
            n22 = torch.where(N2 == v2, N1, torch.zeros_like(N1))
            n22 = n22[torch.where(n22)]  # v2节点的所有出发节点
            n2 = torch.cat((n21, n22)).unique()

            for j in range(l1):
                if j > i:
                    v3 = N1[j]  # link2出发节点
                    v4 = N2[j]  # link2达到节点
                    n32 = torch.where(N1 == v3, N2, torch.zeros_like(N1))
                    n32 = n32[torch.where(n32)]  # v3节点的所有到达节点
                    n31 = torch.where(N2 == v3, N1, torch.zeros_like(N1))
                    n31 = n31[torch.where(n31)]  # v3节点的所有出发节点
                    n3 = torch.cat((n31, n32)).unique()
                    n41 = torch.where(N1 == v4, N2, torch.zeros_like(N1))
                    n41 = n41[torch.where(n41)]  # v4节点的所有到达节点
                    n42 = torch.where(N2 == v4, N1, torch.zeros_like(N1))
                    n42 = n42[torch.where(n42)]  # v4节点的所有出发节点
                    n4 = torch.cat((n41, n42)).unique()
                    # print(n1, '\n', n2, '\n', n3, '\n', n4)

                    s1 = 0
                    s2 = 0
                    for v5 in n1:
                        if torch.where(n3 == v5)[0].shape[0] > 0:
                            s1 += 1

                    for v6 in n2:
                        if torch.where(n4 == v6)[0].shape[0] > 0:
                            s2 += 1

                    sim = s1 / (len(n1) + len(n3)) + s2 / (len(n2) + len(n4))

                    link_sim_mat[i, j] = sim
                    link_sim_mat[j, i] = sim

        link_sim_mats[s, :, :] = link_sim_mat
    return link_sim_mats


def adj_to_triplets(adj):
    # 输入邻接矩阵
    # 输出每个节点的三元组向量
    node_vectors1 = np.zeros_like(adj)
    # 所有样本的链路的开集三元组特征向量
    node_vectors2 = np.zeros_like(adj)
    # 所有样本的链路的闭集三元组特征向量

    for i in range(adj.shape[0]):
        N1 = np.where(adj[i,:]>0)[0]  # 节点i的1阶邻居节点位置
        if N1.shape[0]>0:
            for j in N1:
                N2 = np.where(adj[j,:]>0)[0] # 节点j的1阶邻居节点位置
                # N2 = N2.numpy()
                p = np.where(N2==i)[0]
                if p.shape[0]>0:
                    N2 = np.delete(N2,p)
                if N2.shape[0]>0:
                    node_vectors1[i,j] += 1
                    node_vectors1[i,N2] += 1
                    for k in N2:
                        if adj[k,i]>0:
                            node_vectors2[i,j] +=1
                            node_vectors2[i,k] +=1

    return node_vectors1, node_vectors2


def adj_to_normal_Laplace(adj):
    # 增加自环
    adj = adj + np.diag(np.ones(adj.shape[0]))

    d = np.sum(adj, axis=1)
    d_sqrt = 1 / np.sqrt(d)
    d_sqrt = np.where(d > 0, d_sqrt, d)
    D_sqrt = np.diag(d_sqrt)
    I = np.ones_like(d)
    I = np.where(d > 0, I, d)
    I = np.diag(I)
    L1 = np.matmul(adj, D_sqrt)
    L2 = np.matmul(D_sqrt, L1)

    Lps = L2 - I
    return Lps


# adj = np.random.randn(5,5)
# # l = adj_to_normal_Laplace(adj)
# # print(adj)
# # print(l)
# batch = 2
# normal_lps = torch.from_numpy(adj)
# print(normal_lps)
# normal_lps = normal_lps.expand(batch, normal_lps.shape[0], normal_lps.shape[1])
# print(normal_lps.shape)
# a = np.array([[0,1,1,0],[1,0,1,0],[1,0,0,1],[0,1,0,0]])
# l = adj_to_normal_Laplace(a)
# print(a)
# print(l)

