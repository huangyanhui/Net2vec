from input_data import InputData
import numpy
from model import NetModel
from torch.autograd import Variable
import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
import sys



class Net2vec:
    def __init__(self,
                 input_user_file_name,
                 input_links_file_name,
                 output_file_name,
                 emb_dimension=100,
                 num_batch=30000,
                 batch_size=100,
                 initial_lr=0.025):
        """Initilize class parameters.

        Args:
            input_user_file_name: 用户数据文件
            input_links_file_name: 关系数据文件
            output_file_name:保存文件
            emb_dimention: 向量维度
            num_batch:处理次数
            batch_size:批处理大小
            initial_lr: 初始学习率


        Returns:
            None.
        """
        ##处理数据
        self.data = InputData(input_user_file_name,input_links_file_name)
        self.output_file_name = output_file_name
        ##emb_size为embed的大小，等于顶点个数
        self.emb_size = self.data.vertex_count
        self.emb_dimension = emb_dimension
        ##batch_size是每次更新时的数据规模
        self.batch_size = batch_size
        self.initial_lr = initial_lr
        self.num_batch=num_batch
        ##调用模型，+1的原因是顶点是从1开始的，所以我们把0位置的向量保存下来，但其实没啥意思
        self.NetModel = NetModel(self.emb_size+1, self.emb_dimension)
        ##是否使用cuda加速
        self.use_cuda = torch.cuda.is_available()
        if self.use_cuda:
            self.NetModel.cuda()

        ##使用随机梯度下降的方法来更新参数
        self.optimizer = optim.SGD(
            self.NetModel.parameters(), lr=self.initial_lr)

    def train(self):
        """Multiple training.

        Returns:
            None.
        """

        ##设置进度条
        process_bar = tqdm(range(self.num_batch))

        lr=self.initial_lr
        for i in process_bar:
            ##返回正样本集的ui，uj和负样本集的ui和uj，5为一个正样本对应的负样本的个数
            u_i,u_j,neg_u,neg_v = self.data.get_pairs(self.batch_size,5)


            pos_u = Variable(torch.LongTensor(u_i))
            pos_v = Variable(torch.LongTensor(u_j))
            neg_u = Variable(torch.LongTensor(neg_u))
            neg_v=Variable(torch.LongTensor(neg_v))
            if self.use_cuda:
                pos_u = pos_u.cuda()
                pos_v = pos_v.cuda()
                neg_u = neg_u.cuda()
                neg_v=neg_v.cuda()
            ##将正样本集和负样本集传入模型计算，2表示选择second-order proximities
            loss = self.NetModel.forward(pos_u, pos_v, neg_u,neg_v,2)
            ##清空梯度
            self.optimizer.zero_grad()

            loss.backward()

            self.optimizer.step()

            process_bar.set_description("Loss: %0.8f, lr: %0.6f" %
                                        (loss.data[0],
                                         self.optimizer.param_groups[0]['lr']))

            ##调整学习率
            if i  % 1500000 == 0:
                lr=0.5*lr
                for param_group in self.optimizer.param_groups:
                    param_group['lr'] = lr
        ##将学习的参数保存下来
        self.NetModel.save_embedding(self.output_file_name, self.use_cuda)




if __name__ == '__main__':
    w2v = Net2vec(input_user_file_name="release-youtube-users.txt", input_links_file_name="release-youtube-links.txt",output_file_name="vec.txt")
    w2v.train()
