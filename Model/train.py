import torch.nn as nn
import torch
import numpy as np
import math
from model import TPformer
import torch.utils.data as Data
from lib import utils
from lib import data_preparation
device = 'cuda:0' if torch.cuda.is_available() else 'cpu'

# ===========================================================================================================
# 数据参数
adj_filename = '../DATA/PEMS08/distance.csv'
graph_signal_matrix_filename = '../DATA/PEMS08/pems08.npz'
num_of_vertices = 170
points_per_hour = 12
num_for_predict = 12
# ===========================================================================================================
# 训练参数
learning_rate = 0.001
epochs = 1000
batch_size = 50
input_num = 12
pre_num = 12
# ===========================================================================================================
# model超参数
d_ff = 512  # FeedForward 512->2048->512 做特征提取的
d_k = d_v = 64  # K(K=Q)和V的维度 Q和K的维度需要相同，这里为了方便让K=V
n_layers = 6  # Encoder and Decoder Layer Block的个数 6
n_heads = 6
d_node_feat = 128
d_model = input_num + d_node_feat
# ===========================================================================================================
# 流量数据加载
all_data = np.load(graph_signal_matrix_filename)['data']
all_data = all_data[:100,:,:]

train_data = all_data[:int(all_data.shape[0]*0.6),:,:]
val_data = all_data[int(all_data.shape[0]*0.6):int(all_data.shape[0]*0.8),:,:]
test_data = all_data[int(all_data.shape[0]*0.8):,:,:]

train_DATA = data_preparation.TPDataset(input_num=input_num,pre_num=pre_num,data=train_data)
train_loader = Data.DataLoader(train_DATA,batch_size=2,shuffle=True,drop_last=False)

val_DATA = data_preparation.TPDataset(input_num=input_num,pre_num=pre_num,data=val_data)
val_loader = Data.DataLoader(val_DATA,batch_size=2,shuffle=True,drop_last=False)

test_DATA = data_preparation.TPDataset(input_num=input_num,pre_num=pre_num,data=test_data)
test_loader = Data.DataLoader(test_DATA,batch_size=2,shuffle=True,drop_last=False)
# 邻接矩阵数据加载
adj = utils.get_adjacency_matrix(adj_filename,num_of_vertices)
node_triplet1, node_triplet2 = data_preparation.adj_to_triplets(adj)
normal_lps = data_preparation.adj_to_normal_Laplace(adj)

node_triplet1 = torch.from_numpy(node_triplet1).to(device)
node_triplet2 = torch.from_numpy(node_triplet2).to(device)
normal_lps = torch.from_numpy(normal_lps).to(device)
print('数据加载完成！')


# ===========================================================================================================
# 训练过程
def train():
    # ============================================train===============================================================
    # 实例化模型 损失函数和优化器
    model = TPformer.TPformer(node_num=num_of_vertices,
                              d_model=d_model,
                              n_layers=n_layers,
                              d_k=d_k,
                              d_v=d_v,
                              n_heads=n_heads,
                              d_ff=d_ff,
                              r=input_num,
                              d_node_feat=d_node_feat).to(device)

    loss_func = nn.SmoothL1Loss()
    # optimizer = optim.SGD(model.parameters(), lr=1e-3, momentum=0.99)  # 用adam的话效果不好
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    for epoch in range(epochs):
        Loss1=[]
        for step, batch in enumerate(train_loader):
            samples = batch[0].to(device)
            labels = batch[1].to(device)
            outputs, enc_self_attns, dec_self_attns, dec_enc_attns = \
                model(samples, labels, normal_lps,node_triplet1, node_triplet2)

            loss = loss_func(outputs[:,:,:labels.shape[2]],labels)
            optimizer.zero_grad()
            loss.backward(retain_graph=True)
            optimizer.step()

            Loss1.append(loss.detach().cpu().numpy())

        loss_1 = np.array(Loss1)
        train_loss = np.mean(loss_1)
        print('Epoch:', '%01d' % (epoch), 'train_loss =', '{:.6f}'.format(train_loss))
        # ============================================val===============================================================
        # 验证集效果
        with torch.no_grad():
            Loss2 = []
            for step, batch in enumerate(val_loader):
                samples = batch[0].to(device)
                labels = batch[1].to(device)
                outputs, enc_self_attns, dec_self_attns, dec_enc_attns = \
                    model(samples, labels, normal_lps,node_triplet1, node_triplet2)

                loss2 = loss_func(outputs[:,:,:labels.shape[2]],labels)
                Loss2.append(loss2.detach().cpu().numpy())

            val_loss = np.mean(np.array(Loss2))
            print('Epoch:', '%01d' % (epoch), 'val_loss =', '{:.6f}'.format(val_loss))
    # 保存模型
    torch.save(model, 'TPformer.pkl')  # 保存整个模型


# ===========================================================================================================
# 验证模型性能
def eval():
    with torch.no_grad():
        model = torch.load('TPformer.pkl').to(device)
        RMSE = []
        MAE = []
        for step, batch in enumerate(test_loader):
            samples = batch[0].to(device)
            labels = batch[1].to(device)
            outputs, enc_self_attns, dec_self_attns, dec_enc_attns = \
                model(samples, labels, normal_lps, node_triplet1, node_triplet2)

            rmse = torch.sum(torch.pow(outputs[:,:,:labels.shape[2]]-labels,2),dim=(0,1,2))
            mae =  torch.sum(torch.abs(outputs[:,:,:labels.shape[2]]-labels),dim=(0,1,2))
            RMSE.append(rmse.cpu().numpy())
            MAE.append(mae.cpu().numpy())
        RMSE = np.array(RMSE)
        MAE = np.array(MAE)
        final_rmse = np.mean(RMSE)
        final_mae = np.mean(MAE)
        print(final_rmse)
        print(final_mae)


if __name__ == '__main__':
    train()
    # eval()



