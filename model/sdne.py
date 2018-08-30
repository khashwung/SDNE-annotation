import numpy as np
import tensorflow as tf
import time
import copy
import random
from .rbm import *

class SDNE:
    def __init__(self, config):
    
        self.is_variables_init = False
        self.config = config 
        ######### not running out gpu sources ##########
        # gpu配置与session设置
        tf_config = tf.ConfigProto()
        tf_config.gpu_options.allow_growth = True
        self.sess = tf.Session(config =  tf_config)

        ############ define variables ##################
        # struct存的是每层的维度？struct确实是这个
        self.layers = len(config.struct)
        self.struct = config.struct
        self.sparse_dot = config.sparse_dot
        self.W = {}
        self.b = {}
        struct = self.struct
        for i in range(self.layers - 1):
            name = "encoder" + str(i)
            self.W[name] = tf.Variable(tf.random_normal([struct[i], struct[i+1]]), name = name)
            self.b[name] = tf.Variable(tf.zeros([struct[i+1]]), name = name)
        struct.reverse()
        for i in range(self.layers - 1):
            name = "decoder" + str(i)
            self.W[name] = tf.Variable(tf.random_normal([struct[i], struct[i+1]]), name = name)
            self.b[name] = tf.Variable(tf.zeros([struct[i+1]]), name = name)
        self.struct.reverse()
        ###############################################
        ############## define input ###################
                
        self.adjacent_matriX = tf.placeholder("float", [None, None])
        # these variables are for sparse_dot
        self.X_sp_indices = tf.placeholder(tf.int64)
        self.X_sp_ids_val = tf.placeholder(tf.float32)
        self.X_sp_shape = tf.placeholder(tf.int64)
        # tf稀疏矩阵三要素，位置，值与原矩阵形状
        self.X_sp = tf.SparseTensor(self.X_sp_indices, self.X_sp_ids_val, self.X_sp_shape)
        #
        self.X = tf.placeholder("float", [None, config.struct[0]])
        
        ###############################################
        self.__make_compute_graph()
        self.loss = self.__make_loss(config)
        self.optimizer = tf.train.RMSPropOptimizer(config.learning_rate).minimize(self.loss)
        

    
    def __make_compute_graph(self):
        ''' 需要强调，这里的输入X并非点的特征，而是节点相邻矩阵！！ '''
        # 稀疏矩阵和普通矩阵两种初始化方法
        def encoder(X):
            for i in range(self.layers - 1):
                name = "encoder" + str(i)
                X = tf.nn.sigmoid(tf.matmul(X, self.W[name]) + self.b[name])
            return X

        def encoder_sp(X):
            # 可以存在多层
            for i in range(self.layers - 1):
                name = "encoder" + str(i)
                if i == 0:
                    # 稀疏矩阵第一层用稀疏矩阵相乘
                    X = tf.nn.sigmoid(tf.sparse_tensor_dense_matmul(X, self.W[name]) + self.b[name])
                else:
                    X = tf.nn.sigmoid(tf.matmul(X, self.W[name]) + self.b[name])
            return X
            
        def decoder(X):
            for i in range(self.layers - 1):
                name = "decoder" + str(i)
                X = tf.nn.sigmoid(tf.matmul(X, self.W[name]) + self.b[name])
            return X

        # 是否使用稀疏点积
        if self.sparse_dot:
            self.H = encoder_sp(self.X_sp)
        else:
            self.H = encoder(self.X)
        # self.H会被用于计算一阶损失，即是否相连
        # self.X_reconstruct会被用于计算二阶损失，即是否可以重建出来
        self.X_reconstruct = decoder(self.H)
    

        
    def __make_loss(self, config):
        def get_1st_loss_link_sample(self, Y1, Y2):
            return tf.reduce_sum(tf.pow(Y1 - Y2, 2))
        def get_1st_loss(H, adj_mini_batch):
            # 每个节点的度数对角矩阵
            D = tf.diag(tf.reduce_sum(adj_mini_batch,1))
            # 图的拉普拉斯矩阵
            L = D - adj_mini_batch ## L is laplation-matriX
            # tr(H^T*L*H)作为目标函数，这其实来自于Rayleigh熵，和谱聚类中的目标函数一致，
            # 本质上是一个空间中的点映射到另一个空间
            # 为何乘以2？因为拉普拉斯矩阵有这样的一个性质：任意f^T*L*f = 1/2 * sigma_i,j(w_i,j * (f_i - f_j)^2)
            # 并且RatioCut(A1,A2,..Ak) = tr(H^T*L*H)，所以本质上等价于最小化RatioCut，在流形上进行最优分割
            return 2*tf.trace(tf.matmul(tf.matmul(tf.transpose(H),L),H))

        def get_2nd_loss(X, newX, beta):
            # B的目的是解决稀疏性的问题，因为邻接矩阵中存在大量的0元素，而我们更多是想重构更加精确的非零元素，因此对于
            # 非0元素的惩罚要大于0元素，通过B经过哈达玛积可以产生不同的惩罚项，加大对不准确非0元素的惩罚
            B = X * (beta - 1) + 1
            return tf.reduce_sum(tf.pow((newX - X)* B, 2))

        def get_reg_loss(weight, biases):
            ret = tf.add_n([tf.nn.l2_loss(w) for w in weight.itervalues()])
            ret = ret + tf.add_n([tf.nn.l2_loss(b) for b in biases.itervalues()])
            return ret
            
        #Loss function
        self.loss_2nd = get_2nd_loss(self.X, self.X_reconstruct, config.beta)
        self.loss_1st = get_1st_loss(self.H, self.adjacent_matriX)
        # 正则化损失
        self.loss_xxx = tf.reduce_sum(tf.pow(self.X_reconstruct,2)) 
        # we don't need the regularizer term, since we have nagetive sampling.
        #self.loss_reg = get_reg_loss(self.W, self.b) 
        #return config.gamma * self.loss_1st + config.alpha * self.loss_2nd + config.reg * self.loss_reg
        
        return config.gamma * self.loss_1st + config.alpha * self.loss_2nd +self.loss_xxx

    def save_model(self, path):
        saver = tf.train.Saver(self.b.values() + self.W.values())
        saver.save(self.sess, path)

    def restore_model(self, path):
        # restore的就是权重这些东西
        saver = tf.train.Saver(self.b.values() + self.W.values())
        saver.restore(self.sess, path)
        # 这里一般用于初始化
        self.is_Init = True
    
    def do_variables_init(self, data):
        def assign(a, b):
            op = a.assign(b)
            self.sess.run(op)
        # 所有权重的初始化
        init = tf.global_variables_initializer()       
        self.sess.run(init)
        if self.config.restore_model:
            # 如果restore model，即从之前训练好的部分开始训练，调用restore
            self.restore_model(self.config.restore_model)
            print("restore model" + self.config.restore_model)
        elif self.config.DBN_init:
            # SDNE算法的初始化采用的RBM初始化，论文中有说
            shape = self.struct
            myRBMs = []
            for i in range(len(shape) - 1):
                myRBM = rbm([shape[i], shape[i+1]], {"batch_size": self.config.dbn_batch_size, "learning_rate":self.config.dbn_learning_rate})
                myRBMs.append(myRBM)
                for epoch in range(self.config.dbn_epochs):
                    error = 0
                    for batch in range(0, data.N, self.config.dbn_batch_size):
                        mini_batch = data.sample(self.config.dbn_batch_size).X
                        for k in range(len(myRBMs) - 1):
                            mini_batch = myRBMs[k].getH(mini_batch)
                        error += myRBM.fit(mini_batch)
                    print("rbm epochs:", epoch, "error : ", error)

                W, bv, bh = myRBM.getWb()
                name = "encoder" + str(i)
                assign(self.W[name], W)
                assign(self.b[name], bh)
                name = "decoder" + str(self.layers - i - 2)
                assign(self.W[name], W.transpose())
                assign(self.b[name], bv)
        self.is_Init = True

    def __get_feed_dict(self, data):
        # X即为节点相连矩阵
        X = data.X
        if self.sparse_dot:
            # 如果第一层采用稀疏点积的方式，那么会获取 index，value，original size 这三个数据
            X_ind = np.vstack(np.where(X)).astype(np.int64).T
            X_shape = np.array(X.shape).astype(np.int64)
            X_val = X[np.where(X)]
            return {self.X : data.X, self.X_sp_indices: X_ind, self.X_sp_shape:X_shape, self.X_sp_ids_val: X_val, self.adjacent_matriX : data.adjacent_matriX}
        else:
            return {self.X: data.X, self.adjacent_matriX: data.adjacent_matriX}
            
    def fit(self, data):
        feed_dict = self.__get_feed_dict(data)
        ret, _ = self.sess.run((self.loss, self.optimizer), feed_dict = feed_dict)
        return ret
    
    def get_loss(self, data):
        feed_dict = self.__get_feed_dict(data)
        return self.sess.run(self.loss, feed_dict = feed_dict)

    def get_embedding(self, data):
        # 最终的embedding结果就是最后一层encode的输出
        return self.sess.run(self.H, feed_dict = self.__get_feed_dict(data))

    def get_W(self):
        return self.sess.run(self.W)
        
    def get_B(self):
        return self.sess.run(self.b)
        
    def close(self):
        self.sess.close()

    

