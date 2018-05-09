import torch
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F


class NetModel(nn.Module):
    """

    Attributes:
        emb_size: Embedding的规模.
        emb_dimention: Embedding 维度
        u_embedding: 代表ui
        v_embedding: 代表ui'
    """

    def __init__(self, emb_size, emb_dimension):
        """Initialize model parameters.

        Apply for two embedding layers.
        Initialize layer weight

        Args:
            emb_size: Embedding size.
            emb_dimention: Embedding dimention, typically from 50 to 500.

        Returns:
            None
        """
        super(NetModel, self).__init__()
        self.emb_size = emb_size
        self.emb_dimension = emb_dimension
        self.u_embeddings = nn.Embedding(emb_size, emb_dimension, sparse=True)
        self.v_embeddings = nn.Embedding(emb_size, emb_dimension, sparse=True)
        self.init_emb()

    def init_emb(self):
        """Initialize embedding weight like word2vec.

        将ui和ui'均衡化

        Returns:
            None
        """
        initrange = 0.5 / self.emb_dimension
        self.u_embeddings.weight.data.uniform_(-0.5/self.emb_size, 0.5/self.emb_size)
        self.v_embeddings.weight.data.uniform_(-0.5/self.emb_size, 0.5/self.emb_size)

    def forward(self, pos_u, pos_v,neg_u,neg_v,order):
        """Forward process.


        Args:
            pos_u: 代表正样本集的ui
            pos_v: 代表正样本集的ui'
            neg_u: 代表负样本集的ui
            neg_v: 代表负样本集的un'

        Returns:
            Loss of this process, a pytorch variable.
        """

        ##取出对应正样本集的向量
        emb_u = self.u_embeddings(pos_u)
        ##order代表选择first还是second order proximity
        if order==1:
            emb_v = self.u_embeddings(pos_v)
        elif order==2:
            emb_v = self.v_embeddings(pos_v)
        ##计算公式前部分正样本的数值
        score = torch.mul(emb_u, emb_v).squeeze()
        score = torch.sum(score, dim=1)
        score = F.logsigmoid(score)
        ##取出负样本集对应的向量
        neg_emb_u = self.u_embeddings(neg_u)
        neg_emb_v = self.v_embeddings(neg_v)
        ##计算负样本集的数值
        neg_score = torch.mul(neg_emb_u, neg_emb_v).squeeze()
        neg_score = torch.sum(neg_score, dim=1)
        neg_score = F.logsigmoid(-1 * neg_score)
        return -1 * (torch.sum(score)+torch.sum(neg_score))

    def save_embedding(self, file_name, use_cuda):
        """Save all embeddings to file.

        As this class only record word id, so the map from id to word has to be transfered from outside.

        Args:
            id2word: map from word id to word.
            file_name: file name.
        Returns:
            None.
        """
        if use_cuda:
            embedding = self.u_embeddings.weight.cpu().data.numpy()
        else:
            embedding = self.u_embeddings.weight.data.numpy()
        fout = open(file_name, 'w',encoding="utf-8")
        for i in range(self.emb_size) :
            e = embedding[i]
            e = ' '.join(map(lambda x: str(x), e))
            fout.write('%s %s\n' % (i, e))


def test():
    model = NetModel(100, 100)
    id2word = dict()
    for i in range(100):
        id2word[i] = str(i)
    model.save_embedding(id2word)


if __name__ == '__main__':
    test()
