import numpy
from collections import deque
numpy.random.seed(12345)


class InputData:

    def __init__(self, input_user_file_name,input_links_file_name):
        self.input_user_file_name = input_user_file_name
        self.input_links_file_name=input_links_file_name
        self.get_vertex_edge()
        self.word_pair_catch = deque()
        self.init_sample_table()
        print('Vertex Count: %d' % self.vertex_count)
        print('Edge Count: %d' % (self.edge_count))


    ##处理数据，edge_count为边的个数，vertex_edge保存为每个顶点对应的边，vertex_edge保存为每个顶点对应的边的个数，vertex_count为顶点个数
    def get_vertex_edge(self):
        self.input_file = open(self.input_links_file_name,encoding="utf-8")
        readlines=self.input_file.readlines()
        ##代表边数
        self.edge_count = len(readlines)
        ##顶点和边统计
        self.vertex_edge = dict()
        self.vertex_edge_count=dict()
        for line in readlines:
            pair = line.strip('\n').split('\t')
            try:
                self.vertex_edge[int(pair[0])].append(int(pair[1]))
                self.vertex_edge_count[int(pair[0])]+=1
            except:
                self.vertex_edge[int(pair[0])] = []
                self.vertex_edge[int(pair[0])].append(int(pair[1]))
                self.vertex_edge_count[int(pair[0])] = 1
        self.vertex_count=len(open(self.input_user_file_name,encoding="gbk").readlines())

    #构造映射样本集，根据顶点的个数按比例扩充了。
    def init_sample_table(self):
        self.sample_table = []
        sample_table_size = 1e8
        pow_frequency = numpy.array(list(self.vertex_edge_count.values()))**0.75
        words_pow = sum(pow_frequency)
        ratio = pow_frequency / words_pow


        count = numpy.round(ratio * sample_table_size)


        for wid, c in enumerate(count):
            self.sample_table += [wid+1] * int(c)
        self.sample_table = numpy.array(self.sample_table)

    # @profile





    ##收集正样本集和负样本集
    def get_pairs(self,batch_size,K):
        u_i=[]
        u_j=[]
        neg_i=[]
        neg_j=[]
        while len(u_i) < batch_size:
            sentence = self.input_file.readline()
            if sentence is None or sentence == '':
                self.input_file = open(self.input_links_file_name,encoding="gbk")
                sentence = self.input_file.readline()
            pairs=sentence.strip('\n').split('\t')
            pairs[0]=int(pairs[0])
            pairs[1]=int(pairs[1])
            u_i.append(pairs[0])
            u_j.append(pairs[1])
            i=0
            while i<K:
                neg_v=numpy.random.choice(self.sample_table)
                if neg_v in self.vertex_edge[pairs[0]]:
                    continue
                else:
                    i+=1
                    neg_i.append(pairs[0])
                    neg_j.append(neg_v)

        return u_i,u_j,neg_i,neg_j









def test():
    a = InputData('release-youtube-users.txt','release-youtube-links.txt')
    u_i,u_j,neg_i,neg_j=a.get_pairs(50,5)
    print(a)




if __name__ == '__main__':
    test()
    print()
